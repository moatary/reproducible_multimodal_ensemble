
\"\"\"
run_interpretability.py
=======================
Computes global and local modified feature importance (FI) for the
multimodal emotion-recognition models.

The modified FI (Equation 1 of the paper) up-weights tree nodes where
positive-class instances dominate the right branch, yielding more
interpretable, fault-diagnosable explanations than standard sklearn FI.

Usage
-----
    python run_interpretability.py --dataset iemocap --seed 42 --mode global
    python run_interpretability.py --dataset iemocap --seed 42 --mode local
    python run_interpretability.py --dataset iemocap --seed 42 --mode both
\"\"\"

import argparse, random, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(GLOBAL_SEED)

# ── Optimal hyperparameters for interpretability experiments ──────────────────
# Validated via the fine-tuning pipeline (see finetuning/).
INTERP_CONFIG = {
    "_classifier":              "RandomForest",
    "_clustermode":             "nothing",  # use raw vocabulary
    "_representationmode":      "bin",      # binary presence/absence
    "_restricting_mxlen":       14,
    "_t_multimodal_embed_nclusters": None,  # full vocab — no k-means reduction
    "_v_multimodal_embed_nclusters": 200,
    "_a_multimodal_embed_nclusters": 200,
    "_undersampling":           "NoUndersampling",
    "_tree_criterion":          "entropy",
    "_max_tree_depth":          None,
    "_min_tree_samples_leaf":   2,
    "_min_tree_samples_split":  2,
    "_ensemble_n_estimators":   200,
    "_discriminator_component_analyser": "lda",
    "_positive_label_ind":      1,
    "_neg_features_enabled":    False,
}


def modified_fi_single_tree(tree_obj, spec=5):
    \"\"\"
    Compute modified feature importance for a single decision tree.

    Modified gain at node j (Equation 1 of paper):
        G'_j = G_j * (N_jrp - N_jlp) / (N_jrp + N_jln) * N_j

    where N_jrp = pos in right, N_jln = neg in left, N_jlp = pos in left.

    spec selects among the 14 FI variants evaluated in Table 6 (0=default).
    \"\"\"
    t = tree_obj.tree_
    fi = np.zeros(t.n_features)
    for j in range(t.node_count):
        if t.children_left[j] == -1:
            continue
        feat  = t.feature[j]; n_tot = t.n_node_samples[j]
        g0    = t.impurity[j]
        l, r  = t.children_left[j], t.children_right[j]
        nl, nr = t.n_node_samples[l], t.n_node_samples[r]
        gain  = g0 - nl/n_tot*t.impurity[l] - nr/n_tot*t.impurity[r]
        vr = t.value[r][0]; vl = t.value[l][0]
        n_rp = vr[1] if len(vr)>1 else 0
        n_rn = vr[0]
        n_lp = vl[1] if len(vl)>1 else 0
        n_ln = vl[0]
        denom = (n_rp + n_ln) + 1e-10
        if spec == 5:
            # Best-performing variant: positive density × coverage
            w = max(n_rp - n_lp, 0) / denom * n_tot
        elif spec == 12:
            # F-beta style with positive recall emphasis
            n_p = (t.value[j][0][1] if len(t.value[j][0])>1 else 1) + 1e-7
            n_n = t.value[j][0][0] + 1e-7
            b2  = max(n_n / n_p, 2)
            w   = (1+b2)*n_rp / ((1+b2)*n_rp + b2*n_lp + n_rn + 1e-10) * n_tot
        else:
            w = max(n_rp - n_lp, 0) / denom * (n_tot ** (0.5 + spec*0.05))
        fi[feat] += gain * w
    m = fi.max()
    return fi / m if m > 0 else fi


def global_fi(model, X, y, text_vocab, n_specs=14):
    \"\"\"Compute separability scores across all FI variants.\"\"\"
    results = {}
    ests = getattr(model, "estimators_", [model])
    for spec in range(n_specs):
        fi_sum = np.zeros(X.shape[1])
        for est in ests:
            fi_sum += modified_fi_single_tree(est, spec)
        fi_sum /= len(ests)
        n_t = min(len(text_vocab), X.shape[1])
        top = np.argsort(fi_sum[:n_t])[::-1][:20]
        results[spec] = {
            "fi": fi_sum,
            "top_words": [text_vocab[i] for i in top if i < len(text_vocab)],
        }
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="iemocap")
    p.add_argument("--datapath", default="../data/")
    p.add_argument("--seed",     type=int, default=GLOBAL_SEED)
    p.add_argument("--mode",     default="global",
                   choices=["global","local","both"])
    p.add_argument("--n_specs",  type=int, default=14)
    p.add_argument("--output",   default="results/fi_results.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    import sys; sys.path.insert(0, "..")
    from utils.data_loader import load_dataset
    from utils.feature_extractor import extract_multimodal_features
    from utils.undersampling import apply_undersampling
    from sklearn.ensemble import RandomForestClassifier

    print(f"Loading {args.dataset}...")
    dt   = load_dataset(args.dataset, args.datapath)
    feat = extract_multimodal_features(dt, seed=args.seed)
    x_tr, x_ts  = feat["x_tr"], feat["x_ts"]
    y_hp, y_sd  = feat["y_tr_hp"], feat["y_tr_sd"]
    vocab       = feat.get("text_vocab", [f"feat_{i}" for i in range(x_tr.shape[1])])

    x_b, y_b = apply_undersampling(x_tr, y_hp, INTERP_CONFIG["_undersampling"],
                                   args.seed)
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, criterion="entropy",
                                  max_depth=None, min_samples_leaf=2,
                                  random_state=args.seed, n_jobs=-1)
    clf.fit(x_b, y_b)

    if args.mode in ("global","both"):
        print(f"Computing global FI ({args.n_specs} variants)...")
        res = global_fi(clf, x_b, y_b, vocab, args.n_specs)
        print("\nTop-5 words per spec (0=default, 5=paper best):")
        for s in (0, 5, 12):
            print(f"  spec {s:2d}: {res[s]['top_words'][:5]}")
        with open(args.output, "wb") as f:
            pickle.dump(res, f)
        print(f"Results saved to {args.output}")

    if args.mode in ("local","both"):
        print("Local FI: see utils/local_fi.py for per-sample visualisation.")

if __name__ == "__main__":
    main()
"""