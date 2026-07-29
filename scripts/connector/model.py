"""Task-specific connector over frozen Gemma hidden states (see docs in this folder).

Inputs (precomputed by extract_features.py):
  V : [P, L, D]  hidden states of visual patch positions (L = cached layers)
  H : [T, L, D]  hidden states of the review-copy answer tokens

Connector: learnable layer mix -> Q/K/V projections -> N cross-attention blocks
(residual + LayerNorm) -> fusion MLP over [H, C, H-C, H*C] -> token features Z.
Char refiner: token features scattered to characters + char position embedding ->
1D CNN -> heads: q_c (hard-span), p_c (soft prob), p_ck (5 types), gate (pooled).
"""
import torch
import torch.nn as nn

CATS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]


class LayerMix(nn.Module):
    """Learnable softmax mix over the cached backbone layers."""

    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):                     # [*, L, D]
        a = torch.softmax(self.w, dim=0)
        return torch.einsum("l,...ld->...d", a, x)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.n2 = nn.LayerNorm(dim)

    def forward(self, h, kv, kv_mask=None):
        a, _ = self.attn(h, kv, kv, key_padding_mask=kv_mask)
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        return h


class Connector(nn.Module):
    def __init__(self, d_backbone, n_layers, dim=768, heads=8, blocks=3, dropout=0.1,
                 arch="connector", use_gru=True):
        super().__init__()
        self.arch = arch
        self.mix_h = LayerMix(n_layers)
        self.mix_v = LayerMix(n_layers)
        self.proj_h = nn.Linear(d_backbone, dim)
        self.proj_v = nn.Linear(d_backbone, dim)
        if arch == "connector":
            self.blocks = nn.ModuleList(CrossBlock(dim, heads, dropout) for _ in range(blocks))
            self.fuse = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(),
                                      nn.Dropout(dropout), nn.LayerNorm(dim))
        # char refiner: CNN + BiGRU (sequence context for contiguous spans)
        self.char_pos = nn.Embedding(64, 32)                   # position inside token (capped)
        self.refine = nn.Sequential(
            nn.Conv1d(dim + 32, dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=2), nn.GELU())
        self.use_gru = use_gru
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.head_q = nn.Linear(dim, 1)
        self.head_bio = nn.Linear(dim, 3)                      # O / B / I per char
        self.head_p = nn.Linear(dim, 1)
        self.head_t = nn.Linear(dim, len(CATS))
        self.head_gate = nn.Linear(dim, 1)

    def forward(self, H, V, tok2char, char_inpos, v_mask=None):
        """H:[B,T,L,D] V:[B,P,L,D] tok2char:[B,Cmax] (token index per char, -1 pad)
        char_inpos:[B,Cmax] (char position inside its token). Returns per-char logits."""
        h = self.proj_h(self.mix_h(H))                          # [B,T,dim]
        if self.arch == "connector":
            v = self.proj_v(self.mix_v(V))                      # [B,P,dim]
            c = h
            for blk in self.blocks:
                c = blk(c, v, v_mask)
            z = self.fuse(torch.cat([h, c, h - c, h * c], dim=-1))
        else:                                                   # "linear" readout baseline
            z = h
        # scatter token features to chars
        B, Cmax = tok2char.shape
        idx = tok2char.clamp(min=0).unsqueeze(-1).expand(-1, -1, z.shape[-1])
        zc = torch.gather(z, 1, idx)                            # [B,Cmax,dim]
        pos = self.char_pos(char_inpos.clamp(min=0, max=63))
        x = torch.cat([zc, pos], dim=-1).transpose(1, 2)        # [B,dim+32,Cmax]
        x = self.refine(x).transpose(1, 2)                      # [B,Cmax,dim]
        if self.use_gru:
            g, _ = self.gru(x)
            x = x + g                                               # residual BiGRU context
        valid = (tok2char >= 0)
        pooled = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return (self.head_q(x).squeeze(-1), self.head_p(x).squeeze(-1),
                self.head_t(x), self.head_gate(pooled).squeeze(-1),
                self.head_bio(x), x)


class SetDecoder(nn.Module):
    """DETR-style set-of-spans head: K learnable queries cross-attend to char features;
    each query predicts (start dist, end dist, confidence/null, type, soft prob)."""

    def __init__(self, dim, K=12, n_cats=len(CATS), layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.K = K
        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)
        dl = nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout=dropout,
                                        batch_first=True)
        self.dec = nn.TransformerDecoder(dl, layers)
        self.px = nn.Linear(dim, dim)
        self.qs = nn.Linear(dim, dim)
        self.qe = nn.Linear(dim, dim)
        self.conf = nn.Linear(dim, 1)
        self.typ = nn.Linear(dim, n_cats)
        self.sprob = nn.Linear(dim, 1)

    def forward(self, x, pad_mask):
        """x [B,C,dim]; pad_mask [B,C] True where padded. Returns
        start/end logits [B,K,C], conf [B,K], type [B,K,cats], sprob [B,K]."""
        B = x.shape[0]
        q = self.dec(self.queries.unsqueeze(0).expand(B, -1, -1), x,
                     memory_key_padding_mask=pad_mask)
        xk = self.px(x)
        sl = torch.einsum("bkd,bcd->bkc", self.qs(q), xk)
        el = torch.einsum("bkd,bcd->bkc", self.qe(q), xk)
        neg = torch.finfo(sl.dtype).min
        sl = sl.masked_fill(pad_mask.unsqueeze(1), neg)
        el = el.masked_fill(pad_mask.unsqueeze(1), neg)
        return sl, el, self.conf(q).squeeze(-1), self.typ(q), self.sprob(q).squeeze(-1)


def set_decode(sl, el, conf, typ, tau=0.5, max_len=150):
    """Greedy NMS decode for ONE example (numpy in, list of spans out)."""
    import numpy as np
    order = np.argsort(-conf)
    spans = []
    for k in order:
        if conf[k] < tau:
            break
        a = int(sl[k].argmax())
        e_l = el[k].copy()
        e_l[:a] = -1e30
        e_l[a + max_len:] = -1e30
        b = int(e_l.argmax()) + 1
        if any(not (b <= s["start"] or a >= s["end"]) for s in spans):
            continue                                      # overlap -> NMS drop
        spans.append({"start": a, "end": b, "prob": float(conf[k]),
                      "label": CATS[int(typ[k].argmax())]})
    return sorted(spans, key=lambda s: s["start"])


class SegmentScorer(nn.Module):
    """Semi-Markov-style segment head: scores WHOLE candidate segments [a,b) jointly
    (mean-pool + boundary feats + length bucket) -> (span score, type logits)."""

    def __init__(self, dim, n_cats=len(CATS)):
        super().__init__()
        self.len_emb = nn.Embedding(24, 32)
        self.mlp = nn.Sequential(nn.Linear(dim * 3 + 32, 256), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(256, 1 + n_cats))

    @staticmethod
    def _bucket(L):
        import math
        return min(23, int(math.log2(max(1, L)) * 3))

    def forward(self, x, cands):
        """x [C,dim] (single example); cands list of (a,b). -> scores [N], type [N,cats]."""
        if not cands:
            return x.new_zeros(0), x.new_zeros(0, 5)
        pooled = torch.stack([x[a:b].mean(0) for a, b in cands])
        first = torch.stack([x[a] for a, _ in cands])
        last = torch.stack([x[b - 1] for _, b in cands])
        lens = torch.tensor([self._bucket(b - a) for a, b in cands], device=x.device)
        out = self.mlp(torch.cat([pooled, first, last, self.len_emb(lens)], -1))
        return out[:, 0], out[:, 1:]


def dp_select(cands, scores, theta):
    """Weighted-interval-scheduling DP: pick non-overlapping candidates maximizing
    sum(score - theta). cands: [(a,b,type_idx)], scores: list[float]."""
    idx = sorted(range(len(cands)), key=lambda i: cands[i][1])
    best, chosen = {0: (0.0, [])}, None
    ends = [0]
    dp = [(0.0, [])]
    for i in idx:
        a, b = cands[i][0], cands[i][1]
        w = scores[i] - theta
        # best dp state ending <= a
        j = max(k for k in range(len(ends)) if ends[k] <= a)
        take = (dp[j][0] + w, dp[j][1] + [i])
        keep = dp[-1]
        if take[0] > keep[0]:
            dp.append(take); ends.append(b)
        else:
            dp.append(keep); ends.append(ends[-1])
    return dp[-1][1]


class SemiCRF(nn.Module):
    """TRUE semi-Markov CRF over segmentations: NLL = logZ - score(gold), with the
    partition function over ALL valid segmentations via batched forward DP.
    Factorized segment score phi(a,b) = s_start[a] + s_end[b-1] + mean(u[a:b]) + w_len(b-a)
    keeps it O(n*Lmax) with cheap scalar ops (no per-candidate MLP)."""

    def __init__(self, dim, max_len=120, n_len_buckets=24):
        super().__init__()
        self.max_len = max_len
        self.st = nn.Linear(dim, 1)
        self.en = nn.Linear(dim, 1)
        self.um = nn.Linear(dim, 1)
        self.oo = nn.Linear(dim, 1)
        self.wlen = nn.Parameter(torch.zeros(n_len_buckets))

    @staticmethod
    def _lb(L):
        import math
        return min(23, int(math.log2(max(1, L)) * 3))

    def scores(self, x):
        """x [B,C,dim] -> st,en,u,o [B,C]; U prefix sums [B,C+1]."""
        st = self.st(x).squeeze(-1)
        en = self.en(x).squeeze(-1)
        u = self.um(x).squeeze(-1)
        o = self.oo(x).squeeze(-1)
        U = torch.cat([torch.zeros_like(u[:, :1]), u.cumsum(1)], 1)
        return st, en, u, o, U

    def seg_score_vec(self, st, en, U, j, Ls):
        """phi(j-L, j) for a vector of L values (torch tensor Ls)."""
        a = j - Ls
        lb = torch.tensor([self._lb(int(L)) for L in Ls], device=st.device)
        return (st[:, a] if st.dim() == 2 else st[a]) +                (en[:, j - 1:j] if en.dim() == 2 else en[j - 1]) +                ((U[:, j:j + 1] - U[:, a]) / Ls if U.dim() == 2 else (U[j] - U[a]) / Ls) +                self.wlen[lb]

    def nll(self, x, valid, segs_list):
        """Batched NLL. x [B,C,dim]; valid [B,C]; segs_list: per-example [(a,b), ...]."""
        B, C, _ = x.shape
        st, en, u, o, U = self.scores(x)
        n_i = valid.sum(1).long()
        alphas = [x.new_zeros(B)]                                  # alpha[0] = 0
        for j in range(1, C + 1):
            Lmax = min(j, self.max_len)
            Ls = torch.arange(1, Lmax + 1, device=x.device)
            a_idx = j - Ls                                          # [Lmax]
            seg = (st[:, a_idx] + en[:, j - 1].unsqueeze(1)
                   + (U[:, j].unsqueeze(1) - U[:, a_idx]) / Ls.float()
                   + self.wlen[torch.clamp((torch.log2(Ls.float()) * 3).long(), max=23)])
            hist = torch.stack([alphas[int(a)] for a in a_idx], 1)  # [B, Lmax]
            cand = torch.cat([(alphas[j - 1] + o[:, j - 1]).unsqueeze(1), hist + seg], 1)
            alphas.append(torch.logsumexp(cand, 1))
        # gold scores + pick logZ at each example's length
        nll = x.new_zeros(())
        cnt = 0
        for b in range(B):
            n = int(n_i[b])
            if n < 2:
                continue
            logZ = alphas[n][b]
            gold = x.new_zeros(())
            inseg = torch.zeros(n, dtype=torch.bool)
            for (a, e) in segs_list[b]:
                e = min(e, n)
                if e - a < 1 or a >= n:
                    continue
                Lg = torch.tensor([e - a], device=x.device)
                gold = gold + (st[b, a] + en[b, e - 1]
                               + (U[b, e] - U[b, a]) / float(e - a)
                               + self.wlen[self._lb(e - a)])
                inseg[a:e] = True
            gold = gold + o[b, :n][~inseg.to(x.device)].sum()
            nll = nll + (logZ - gold) / n          # per-char normalization: keeps the
            cnt += 1                                # CRF term on the same scale as BCE
        return nll / max(cnt, 1)

    @torch.no_grad()
    def log_p_empty(self, x1, n):
        """Exact log P(Y = all-O | X) = Score(all-O) - logZ, single example."""
        st, en, u, o, U = self.scores(x1.unsqueeze(0))
        st, en, o, U = st[0], en[0], o[0], U[0]
        alphas = [torch.tensor(0.0, device=x1.device)]
        for j in range(1, n + 1):
            Lmax = min(j, self.max_len)
            Ls = torch.arange(1, Lmax + 1, device=x1.device)
            a_idx = j - Ls
            seg = (st[a_idx] + en[j - 1] + (U[j] - U[a_idx]) / Ls.float()
                   + self.wlen[torch.clamp((torch.log2(Ls.float()) * 3).long(), max=23)])
            hist = torch.stack([alphas[int(a)] for a in a_idx])
            cand = torch.cat([(alphas[j - 1] + o[j - 1]).unsqueeze(0), hist + seg])
            alphas.append(torch.logsumexp(cand, 0))
        score_empty = o[:n].sum()
        return float(score_empty - alphas[n])

    @torch.no_grad()
    def viterbi(self, x1, n, bias=0.0):
        """Exact best segmentation for ONE example. x1 [C,dim]; returns [(a,b), ...]."""
        st, en, u, o, U = self.scores(x1.unsqueeze(0))
        st, en, o, U = st[0], en[0], o[0], U[0]
        NEG = -1e30
        alpha = [0.0] + [NEG] * n
        back = [None] * (n + 1)
        for j in range(1, n + 1):
            best, arg = alpha[j - 1] + float(o[j - 1]), ("O", j - 1)
            Lmax = min(j, self.max_len)
            for L in range(1, Lmax + 1):
                a = j - L
                sc = (alpha[a] + float(st[a]) + float(en[j - 1])
                      + float(U[j] - U[a]) / L + float(self.wlen[self._lb(L)]) + bias)
                if sc > best:
                    best, arg = sc, ("S", a)
            alpha[j] = best
            back[j] = arg
        segs, j = [], n
        while j > 0:
            kind, a = back[j]
            if kind == "S":
                segs.append((a, j))
            j = a if kind == "S" else j - 1
        return segs[::-1]


def tversky_loss(q_logit, m, valid, alpha=0.7, beta=0.3):
    """Precision-weighted soft Tversky: alpha>beta penalizes FP harder than FN."""
    q = torch.sigmoid(q_logit) * valid
    tp = (q * m).sum(1)
    fp = (q * (1 - m) * valid).sum(1)
    fn = ((1 - q) * m).sum(1)
    return (1 - tp / (tp + alpha * fp + beta * fn + 1e-6)).mean()


def soft_jaccard_loss(q_logit, m, valid):
    """1 - soft Jaccard between sigmoid(q) and binary mask m (per example, then mean)."""
    q = torch.sigmoid(q_logit) * valid
    inter = (q * m).sum(1)
    union = (q + m - q * m).sum(1).clamp(min=1e-6)
    return (1 - inter / union).mean()


def ranking_loss(p_logit, y, valid, margin=0.1, n_pairs=256):
    """Pairwise margin ranking: chars with higher gold prob should score higher."""
    B = p_logit.shape[0]
    losses = []
    for b in range(B):
        v = valid[b].bool()
        pb, yb = p_logit[b][v], y[b][v]
        pos = (yb > 0).nonzero(as_tuple=True)[0]
        neg = (yb == 0).nonzero(as_tuple=True)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(n_pairs, len(pos) * len(neg))
        pi = pos[torch.randint(len(pos), (k,), device=pb.device)]
        ni = neg[torch.randint(len(neg), (k,), device=pb.device)]
        losses.append(torch.relu(margin - (pb[pi] - pb[ni])).mean())
    return torch.stack(losses).mean() if losses else p_logit.new_zeros(())
