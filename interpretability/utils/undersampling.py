
import numpy as np

def apply_undersampling(X, y, method="NoUndersampling", seed=42):
    """Apply the specified class-balancing strategy to (X, y).

    Strategies: NoUndersampling, RandomUndersampling, NearMiss,
    CondensedNearestNeighbour, OneSidedSelection,
    eigenbased_hierbinarycluster."""
    rng = np.random.default_rng(seed)
    if method == "NoUndersampling":
        return X, y
    elif method == "RandomUndersampling":
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        n   = len(pos)
        idx = np.concatenate([pos, rng.choice(neg, n, replace=False)])
        rng.shuffle(idx)
        return X[idx], y[idx]
    elif method in ("NearMiss","CondensedNearestNeighbour","OneSidedSelection"):
        try:
            from imblearn.under_sampling import (NearMiss,
                CondensedNearestNeighbour, OneSidedSelection)
            cls = {"NearMiss": NearMiss,
                   "CondensedNearestNeighbour": CondensedNearestNeighbour,
                   "OneSidedSelection": OneSidedSelection}[method]
            return cls(random_state=seed).fit_resample(X, y)
        except ImportError:
            return apply_undersampling(X, y, "RandomUndersampling", seed)
    elif "eigenbased" in method:
        # Eigenvalue hierarchical clustering undersampler:
        # projects the majority class onto the between-class scatter direction
        # and retains the cluster nearest to the minority-class centroid.
        pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
        if len(neg_idx) <= len(pos_idx):
            return X, y
        mu_p = X[pos_idx].mean(0); mu_n = X[neg_idx].mean(0)
        d = mu_n - mu_p; n = np.linalg.norm(d)
        if n < 1e-10:
            d = rng.standard_normal(X.shape[1]); n = np.linalg.norm(d)
        d /= n
        proj = X[neg_idx] @ d
        med  = np.median(proj)
        ca   = neg_idx[proj <= med]; cb = neg_idx[proj > med]
        dist_a = np.linalg.norm(X[ca].mean(0) - mu_p)
        dist_b = np.linalg.norm(X[cb].mean(0) - mu_p)
        kept   = ca if dist_a < dist_b else cb
        n_keep = min(len(pos_idx), len(kept))
        chosen = rng.choice(kept, n_keep, replace=False)
        idx    = np.concatenate([pos_idx, chosen]); rng.shuffle(idx)
        return X[idx], y[idx]
    else:
        raise ValueError(f"Unknown undersampling method: {method}")
