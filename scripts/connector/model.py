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
                self.head_bio(x))


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
