import json

import numpy as np
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - fallback for lean cluster environments
    def _rankdata(a):
        a = np.asarray(a, dtype=float)
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty(len(a), dtype=int)
        inv[sorter] = np.arange(len(a))
        a_sorted = a[sorter]
        obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
        dense = obs.cumsum()[inv]
        count = np.r_[np.flatnonzero(obs), len(a)]
        return 0.5 * (count[dense] + count[dense - 1] + 1)

    class _SpearmanResult:
        def __init__(self, correlation):
            self.correlation = correlation

    def spearmanr(x, y):
        rx, ry = _rankdata(x), _rankdata(y)
        if rx.size < 2 or rx.std() == 0 or ry.std() == 0:
            return _SpearmanResult(0.0)
        return _SpearmanResult(float(np.corrcoef(rx, ry)[0, 1]))
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except Exception:  # pragma: no cover - fallback for inference-only environments
    def accuracy_score(refs, preds):
        refs = list(refs)
        preds = list(preds)
        return sum(r == p for r, p in zip(refs, preds)) / max(1, len(refs))

    def precision_recall_fscore_support(refs, preds, average="binary", zero_division=0,
                                        pos_label=True):
        tp = sum(r == pos_label and p == pos_label for r, p in zip(refs, preds))
        fp = sum(r != pos_label and p == pos_label for r, p in zip(refs, preds))
        fn = sum(r == pos_label and p != pos_label for r, p in zip(refs, preds))
        precision = tp / (tp + fp) if tp + fp else zero_division
        recall = tp / (tp + fn) if tp + fn else zero_division
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else zero_division)
        return precision, recall, f1, None
import argparse as ap
from collections import Counter

def __check_one(label):
    if type(label) is not dict: 
        return False
    expected_keys = {'start':int, 'end':int, 'prob':float, 'label':str}
    if any(key not in label for key in expected_keys):
        return False
    if any(type(label[key]) is not type_ for key, type_ in expected_keys.items()):
        return False
    if label['label'] not in {'mischaracterization', 'OCR', 'miscounting', 'invention', 'other'}:
        return False
    return True

def load_jsonl_file_to_records(filename, is_ref=True, do_check=True):
    """read data from a JSONL file and format that as scorer records.
    Performs minor format checks (ensures that some labels are present,
    optionally compute missing labels on the fly)."""
    records = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not is_ref:
                assert 'labels' in row, f'File {filename} contains no predicted label!'
            labels = row.get('labels') or []
            if do_check:
                assert all(map(__check_one, labels)), f'Some labels in {filename} are malformed!'
            rec = {'id': row['id'], 'labels': labels}
            if is_ref:
                rec['text_len'] = len(row.get('response', ''))
                rec['raw_annots'] = row.get('raw_annots', {})
            records.append(rec)
    return sorted(records, key=lambda rec: rec['id'])


def score_cor(ref_dict, pred_dict, label_filtered_=None):
    """computes Spearman correlation between predicted and reference soft labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    - label_filtered_: if set, limits computation to spans marked with this label
    returns:
    the Spearman correlation, or a binarized exact match (0.0 or 1.0) if the reference or prediction contains no variation
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to vectors of observations
    ref_vec = [0.] * ref_dict['text_len']
    pred_vec = [0.] * ref_dict['text_len']
    ref_labels = (
        ref_dict['labels'] if label_filtered_ is None 
        else [span for span in ref_dict['labels'] if span['label'] == label_filtered_]
    ) 
    pred_labels = (
        pred_dict['labels'] if label_filtered_ is None 
        else [span for span in pred_dict['labels'] if span['label'] == label_filtered_]
    ) 
    for span in ref_labels:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] += span['prob']
    for span in pred_labels:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
    # constant series (i.e., no hallucination) => cor is undef
    ref_cmps = {round(flt, 8) for flt in ref_vec}
    pred_cmps = {round(flt, 8) for flt in pred_vec}
    if len(pred_cmps) == 1 or len(ref_cmps) == 1 : 
        if len(pred_cmps) != len(ref_cmps):
            return 0.0
        if ref_cmps == {0.0}:
            return float(pred_cmps == {0.0})
        return float(pred_cmps != {0.0})
    # otherwise compute Spearman's rho
    return spearmanr(ref_vec, pred_vec).correlation


def score_cor_lbl(ref_dict, pred_dict):
    all_labels = {
        span['label'] 
        for dict_ in [ref_dict, pred_dict] 
        for span in dict_['labels']
    }
    if all_labels:
        return sum(
            score_cor(
                ref_dict,
                pred_dict,
                label_filtered_=label,
            )
            for label in all_labels
        ) / len(all_labels)
    else: 
        # neither the pred nor the ref included labels, which is what we want
        return 1.0


def score_iou(ref_dict, pred_dict):
    """computes intersection-over-union between reference and predicted hard labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to sets of indices
    ref_indices = {idx for span in ref_dict['labels'] for idx in range(span['start'], span['end'])}
    pred_indices = {idx for span in pred_dict['labels'] for idx in range(span['start'], span['end'])}
    # avoid division by zero
    if not pred_indices and not ref_indices: return 1.
    # otherwise compute & return IoU
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

def scores_classif(ref_dicts, pred_dicts):
    refs = [
        sum(v != '|' for v in ref_dict['raw_annots'].values()) > (len(ref_dict['raw_annots']) / 2)
        for ref_dict in ref_dicts
    ]
    preds = [
        bool({idx for span in pred_dict['labels'] for idx in range(span['start'], span['end'])})
        for pred_dict in pred_dicts
    ]
    print('distribs:\n- pred', Counter(preds), '\n- ref ', Counter(refs))
    prec, rec, f1, _ = precision_recall_fscore_support(
        refs, preds,
        average="binary",
        zero_division=0,
        pos_label=True,
    )
    acc = accuracy_score(refs, preds)
    return {'acc ': acc, 'prec': prec, 'rec ': rec, 'f1  ': f1}


def main(ref_dicts, pred_dicts, output_file=None, verbose=False):
    print(len(ref_dicts),len(pred_dicts), len(ref_dicts) == len(pred_dicts))
    #print(ref_dicts)
    pred_ids = {item['id'] for item in pred_dicts}
    gold_subset = [item for item in ref_dicts if item['id'] in pred_ids]  
    ref_dicts =   gold_subset
    print("updated lists:", len(ref_dicts),len(pred_dicts))
    assert len(ref_dicts) == len(pred_dicts)
    cors = np.array([score_cor(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    ious = np.array([score_iou(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    cors_lbl = np.array([score_cor_lbl(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    clf_out_ = scores_classif(ref_dicts, pred_dicts)
    if output_file is not None:
        with open(output_file, 'w') as ostr:
            print(f'Cor    : {cors.mean():.8f}', file=ostr)
            print(f'Cor lbl: {cors_lbl.mean():.8f}', file=ostr)
    if verbose:
        print(f'Cor    : {cors.mean():.8f}')
        print(f'IoU    : {ious.mean():.8f}')
        print(f'Cor lbl: {cors_lbl.mean():.8f}')
        for k, v in sorted(clf_out_.items()):
            print(f'{k}   : {float(v):.8f}')
    return cors, cors_lbl

if __name__ == '__main__':
    p = ap.ArgumentParser()
    p.add_argument('ref_file', type=lambda fname: load_jsonl_file_to_records(fname, do_check=False))
    p.add_argument('pred_file', type=lambda fname: load_jsonl_file_to_records(fname, is_ref=False))
    p.add_argument('output_file', type=str)
    a = p.parse_args()
    _ = main(a.ref_file, a.pred_file, a.output_file, verbose=True)
