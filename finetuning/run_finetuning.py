
"""
run_finetuning.py
=================
Entry point for the sequential coordinate-wise hyperparameter search.
Corresponds to the fine-tuning procedure described in Section 4 and
summarised in Table 1 of the manuscript.

Usage
-----
    python run_finetuning.py --dataset iemocap --criterion f1 --seed 42
"""

import argparse, pickle, time, random, os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_all_seeds(GLOBAL_SEED)

# ── Hyperparameter search space ───────────────────────────────────────────────
HYPERPARAMETER_SPACE = {
    "_classifier":                  ["DecisionTree", "RandomForest", "LDT", "LDF"],
    "_bow_mode":                    ["normal"],
    "_undersampling":               ["eigenbased_hierbinarycluster",
                                     "CondensedNearestNeighbour", "NearMiss",
                                     "OneSidedSelection", "RandomUndersampling",
                                     "NoUndersampling"],
    "_max_tree_depth":              [None, 10, 15, 20, 25, 30, 40, 50, 75, 100],
    "_min_tree_samples_leaf":       [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    "_min_tree_samples_split":      [2, 3, 4, 5, 6, 7, 8],
    "_ensemble_n_estimators":       [35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135],
    # [t_1g, v_1g, a_1g, t_2g, v_2g, a_2g] — cluster counts for each modality
    "nclusters_compilation_1g2g":   [[200, 200, 200, 625, 900, 225],
                                     [400, 200, 200, 625, 900, 225],
                                     [200, 100, 100, 625, 900, 225]],
}

# Optimal values from paper Table 1 (used as reference)
PAPER_OPTIMAL = {
    "DecisionTree": {"max_depth": 20, "min_samples_split": 5,
                     "criterion": "entropy"},
    "RandomForest": {"n_estimators": 200, "max_depth": None,
                     "max_features": "sqrt", "min_samples_leaf": 1},
    "AdaBoost":     {"n_estimators": 100, "learning_rate": 0.1,
                     "base_depth": 2, "algorithm": "SAMME.R"},
    "XGBoost":      {"n_estimators": 300, "max_depth": 6,
                     "learning_rate": 0.05, "subsample": 0.8,
                     "colsample_bytree": 0.8},
    "LDT":          {"lda_components": 5, "lda_shrinkage": "auto",
                     "base_tree_depth": 10},
}


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter search for LDTE")
    p.add_argument("--dataset",  default="iemocap",
                   choices=["iemocap", "cmu_mosi"])
    p.add_argument("--datapath", default="../data/")
    p.add_argument("--criterion",default="f1",
                   choices=["f1", "acc", "fi", "f1mod"])
    p.add_argument("--seed",     type=int, default=GLOBAL_SEED)
    p.add_argument("--n_runs",   type=int, default=3)
    p.add_argument("--output",   default="results/tuning_log.txt")
    return p.parse_args()


def _build_clf(config, seed):
    """Instantiate a classifier from the current config."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from xgboost import XGBClassifier

    name   = config.get("_classifier", "RandomForest")
    depth  = config.get("_max_tree_depth", None)
    crit   = config.get("_tree_criterion", "entropy")
    n_est  = config.get("_ensemble_n_estimators", 100)
    lr     = config.get("_ensemble_learning_rate", 0.1) or 0.1
    leaf   = config.get("_min_tree_samples_leaf", 2)
    split  = config.get("_min_tree_samples_split", 2)

    if name == "DecisionTree":
        return DecisionTreeClassifier(criterion=crit, max_depth=depth,
                                      min_samples_leaf=leaf,
                                      min_samples_split=split,
                                      random_state=seed)
    elif name == "RandomForest":
        return RandomForestClassifier(n_estimators=n_est, criterion=crit,
                                      max_depth=depth, min_samples_leaf=leaf,
                                      max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    elif name == "AdaBoost":
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                   learning_rate=lr, algorithm="SAMME",
                                   random_state=seed)
    elif name == "XGBoost":
        return XGBClassifier(n_estimators=n_est,
                              max_depth=depth if depth else 6,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss",
                              random_state=seed, n_jobs=-1)
    else:
        # LDT / LDF: import custom implementation
        import sys; sys.path.insert(0, ".")
        from model.ldt_models import build_ldt_classifier
        return build_ldt_classifier(config, seed)


def _evaluate(config, x_tr, y_tr, x_vl, y_vl, n_runs, seed):
    """Average F1 / accuracy over n_runs seeds."""
    f1s, accs = [], []
    for run in range(n_runs):
        s = seed + run * 100
        set_all_seeds(s)
        rng = np.random.default_rng(s)
        # Simple random undersampling as default balancer in search
        pos = np.where(y_tr == 1)[0]; neg = np.where(y_tr == 0)[0]
        n = min(len(pos), len(neg))
        idx = np.concatenate([pos[:n],
                               rng.choice(neg, n, replace=False)])
        rng.shuffle(idx)
        clf = _build_clf(config, s)
        clf.fit(x_tr[idx], y_tr[idx])
        yh = clf.predict(x_vl)
        f1s.append(f1_score(y_vl, yh, average="weighted", zero_division=0))
        accs.append(accuracy_score(y_vl, yh))
    return float(np.mean(f1s)), float(np.mean(accs))


def run_coordinate_search(args, x_tr, y_tr_hp, y_tr_sd,
                           x_vl, y_vl_hp, y_vl_sd):
    axes  = list(HYPERPARAMETER_SPACE.keys())
    best  = {k: v[0] for k, v in HYPERPARAMETER_SPACE.items()}
    best_score = 0.0; log = []
    # Sweep order matching the original paper
    order = [1, 3, 4, 5, 6, 7, 1, 3, 2, 5, 4, 7, 6, 1, 6, 5, 4]
    for step, oi in enumerate(order):
        axis = axes[oi % len(axes)]
        step_best, step_val = best_score, best[axis]
        for candidate in HYPERPARAMETER_SPACE[axis]:
            cfg = dict(best); cfg[axis] = candidate
            f1h, _ = _evaluate(cfg, x_tr, y_tr_hp, x_vl, y_vl_hp,
                                args.n_runs, args.seed)
            f1s, _ = _evaluate(cfg, x_tr, y_tr_sd, x_vl, y_vl_sd,
                                args.n_runs, args.seed)
            score = (f1h + f1s) / 2
            line = (f"step={step:02d} axis={axis:<35s} "
                    f"val={str(candidate):<25s} score={score:.4f}")
            log.append(line); print(line)
            if score > step_best:
                step_best = score; step_val = candidate
        best[axis] = step_val; best_score = step_best
        log.append(f"  >>> best {axis}: {step_val} (score={step_best:.4f})")
    return best, best_score, log


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs("results", exist_ok=True)
    print(f"Loading {args.dataset}...")
    import sys; sys.path.insert(0, "..")
    from utils.data_loader import load_dataset
    from utils.feature_extractor import extract_multimodal_features
    dt   = load_dataset(args.dataset, args.datapath)
    feat = extract_multimodal_features(dt, seed=args.seed)
    x_tr, x_vl    = feat["x_tr"], feat["x_vl"]
    y_tr_hp, y_vl_hp = feat["y_tr_hp"], feat["y_vl_hp"]
    y_tr_sd, y_vl_sd = feat["y_tr_sd"], feat["y_vl_sd"]
    print(f"Starting coordinate search (criterion={args.criterion})...")
    t0 = time.time()
    best, score, log = run_coordinate_search(
        args, x_tr, y_tr_hp, y_tr_sd, x_vl, y_vl_hp, y_vl_sd)
    elapsed = time.time() - t0
    summary = (f"
{'='*60}
Done in {elapsed/60:.1f} min
"
               f"Best score ({args.criterion}): {score:.4f}
"
               + "
".join(f"  {k}: {v}" for k, v in best.items())
               + f"
{'='*60}
")
    print(summary); log.append(summary)
    with open(args.output, "w") as f:
        f.write("
".join(log))
    print(f"Log saved to {args.output}")

if __name__ == "__main__":
    main()
