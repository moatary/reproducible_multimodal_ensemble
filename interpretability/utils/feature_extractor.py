
import numpy as np

def extract_multimodal_features(dt, seed=42):
    """
    Extract concatenated text+audio+video feature matrices.
    Tries the full preprocessor pipeline first; falls back to
    a simple binary bag-of-words if the preprocessor is absent.
    """
    def _labels(split):
        labs = dt[split]["labels"]
        return {k: labs[:, i, 1].astype(int)
                for i, k in enumerate(["hp","sd","ng"])}
    tr_l = _labels("train"); vl_l = _labels("valid"); ts_l = _labels("test")
    try:
        from _extract_full import _full_extract
        x_tr, x_vl, x_ts, vocab = _full_extract(dt, seed)
    except ImportError:
        x_tr, x_vl, x_ts, vocab = _simple_bow(
            dt["train"]["text"], dt["valid"]["text"], dt["test"]["text"],
            dt["train"]["audio"], dt["valid"]["audio"], dt["test"]["audio"],
            dt["train"]["vision"], dt["valid"]["vision"], dt["test"]["vision"])
    out = {"x_tr": x_tr, "x_vl": x_vl, "x_ts": x_ts, "text_vocab": vocab}
    for split, labels in [("tr", tr_l), ("vl", vl_l), ("ts", ts_l)]:
        for emotion, vec in labels.items():
            out[f"y_{split}_{emotion}"] = vec
    return out

def _simple_bow(tr_tx, vl_tx, ts_tx, tr_au, vl_au, ts_au,
                tr_vd, vl_vd, ts_vd):
    vocab_set = {}
    for seq in tr_tx:
        for tok in seq:
            t = str(tok).lower()
            if t not in vocab_set:
                vocab_set[t] = len(vocab_set)
    vocab = list(vocab_set.keys()); V = len(vocab)
    def _bow(splits):
        rows = []
        for seq in splits:
            v = np.zeros(V, dtype=np.float32)
            for tok in seq:
                t = str(tok).lower()
                if t in vocab_set: v[vocab_set[t]] = 1.0
            rows.append(v)
        return np.array(rows)
    def _pool(splits):
        return np.array([np.mean(np.array(s),0) if s else np.zeros(len(splits[0][0]))
                         for s in splits], dtype=np.float32)
    x_tr = np.concatenate([_bow(tr_tx), _pool(tr_au), _pool(tr_vd)], 1)
    x_vl = np.concatenate([_bow(vl_tx), _pool(vl_au), _pool(vl_vd)], 1)
    x_ts = np.concatenate([_bow(ts_tx), _pool(ts_au), _pool(ts_vd)], 1)
    return x_tr, x_vl, x_ts, vocab
