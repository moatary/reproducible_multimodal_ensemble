
"""
run_ldt.py
=============
Run LDT with paper-optimal hyperparameters, averaged over 10 seeds.
"""
import argparse, random, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, "..")
from utils.data_loader import load_dataset
from utils.feature_extractor import extract_multimodal_features
from utils.undersampling import apply_undersampling
from utils.metrics import compute_f1mod

SEEDS = list(range(10))   # 10 seeds for averaging, as in the paper

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


LDA_COMP=5; LDA_SHRINK="auto"; BASE_D=10; CRIT="entropy"; DISC="lda"

def build_model(seed):
    import sys; sys.path.insert(0, "..")
    from model.ldt_models import LinearDiscriminantTree
    return LinearDiscriminantTree(positive_label_ind=1, tree_criterion=CRIT,
                                   max_depth=BASE_D, discriminator=DISC)


def run_seed(seed, dataset, datapath):
    set_seeds(seed)
    dt   = load_dataset(dataset, datapath)
    feat = extract_multimodal_features(dt, seed=seed)
    x_tr, x_ts = feat["x_tr"], feat["x_ts"]
    out = {}
    for t in ["hp", "sd"]:
        y_tr = feat[f"y_tr_{t}"]; y_ts = feat[f"y_ts_{t}"]
        x_b, y_b = apply_undersampling(x_tr, y_tr, "NoUndersampling", seed)
        clf = build_model(seed); clf.fit(x_b, y_b)
        yh = clf.predict(x_ts)
        out[t] = dict(acc=accuracy_score(y_ts, yh),
                      f1=f1_score(y_ts, yh, average="weighted", zero_division=0),
                      f1mod=compute_f1mod(y_ts, yh))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="iemocap")
    p.add_argument("--datapath", default="../../data/")
    args = p.parse_args()
    all_r = []
    for s in SEEDS:
        r = run_seed(s, args.dataset, args.datapath); all_r.append(r)
        print(f"seed={s:2d}  acc_hp={r['hp']['acc']:.4f}  "
              f"f1_hp={r['hp']['f1']:.4f}")
    for t in ["hp","sd"]:
        for m in ["acc","f1","f1mod"]:
            v = [r[t][m] for r in all_r]
            print(f"AVG {t} {m:6s}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

if __name__ == "__main__":
    main()
