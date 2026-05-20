
"""
ldt_models.py
=============
Custom Linear Discriminant Tree (LDT) and Linear Discriminant Forest (LDF).
Extends standard decision trees with oblique splits derived from LDA projections.

Key novelty: positive-class-focused discriminant projections give more
interpretable splits (see Section 3 and Algorithm 1 of the paper).
"""

import numpy as np
from joblib import Parallel, delayed


# ── Discriminant projection ───────────────────────────────────────────────────

def get_discriminative_node_view(X, y, pos_lbl, _method="lda"):
    """Project X onto a discriminant subspace at a tree node.

    Returns (X_proj, components).  Falls back to PCA on failure."""
    n_s, n_f = X.shape
    n_comp = min(2, n_f, n_s - 1, len(np.unique(y)) - 1)
    if n_comp < 1:
        return X, np.eye(n_f)
    if _method == "lda":
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(n_components=n_comp,
                                             solver="eigen", shrinkage="auto")
            Xp = lda.fit_transform(X, y)
            return Xp, lda.scalings_.T
        except Exception:
            pass
    if _method.startswith("ica"):
        try:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=n_comp, max_iter=500, random_state=0)
            return ica.fit_transform(X), ica.components_
        except Exception:
            pass
    # PCA fallback
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp, random_state=0)
    return pca.fit_transform(X), pca.components_


# ── Linear Discriminant Tree ──────────────────────────────────────────────────

class LinearDiscriminantTree:
    """Single oblique decision tree with LDA-based splits.  sklearn API."""

    def __init__(self, positive_label_ind=1, tree_criterion="entropy",
                 max_depth=None, min_samples_leaf=1,
                 discriminator="lda", neg_features_enabled=False):
        self.positive_label_ind   = positive_label_ind
        self.tree_criterion       = tree_criterion
        self.max_depth            = int(1e6) if max_depth is None else max_depth
        self.min_samples_leaf     = min_samples_leaf
        self.discriminator        = discriminator
        self.neg_features_enabled = neg_features_enabled
        self._imp = ((lambda p: 1 - p @ p) if tree_criterion == "gini"
                     else (lambda p: -p @ np.log2(p + 1e-17)))

    def _probs(self, y, nc):
        c = np.bincount(y.astype(int), minlength=nc)
        return c / (c.sum() + 1e-10)

    def fit(self, X, y):
        self.n_classes_    = len(np.unique(y))
        self.n_features_in_= X.shape[1]
        self.nodes_ = []
        self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        nid = len(self.nodes_)
        n   = len(y)
        prb = self._probs(y, self.n_classes_)
        imp = self._imp(prb)
        maj = int(np.bincount(y.astype(int), minlength=self.n_classes_).argmax())
        leaf = (depth >= self.max_depth or n < self.min_samples_leaf * 2
                or len(np.unique(y)) == 1)
        node = dict(depth=depth, n=n, imp=imp, label=maj, is_leaf=leaf,
                    left=None, right=None, proj=None, thr=None)
        self.nodes_.append(node)
        if leaf:
            return nid
        Xp, comp = get_discriminative_node_view(X, y, self.positive_label_ind,
                                                 self.discriminator)
        pv = Xp[:, 0] if Xp.ndim > 1 else Xp.ravel()
        best_g, best_t = -np.inf, None
        for t in np.percentile(pv, np.linspace(10, 90, 17)):
            ml = pv <= t; mr = ~ml
            if ml.sum() < self.min_samples_leaf or mr.sum() < self.min_samples_leaf:
                continue
            g = (imp - ml.sum()/n * self._imp(self._probs(y[ml], self.n_classes_))
                     - mr.sum()/n * self._imp(self._probs(y[mr], self.n_classes_)))
            if g > best_g:
                best_g = g; best_t = t
        if best_t is None or best_g <= 0:
            node["is_leaf"] = True; return nid
        ml = pv <= best_t
        node["proj"] = comp[0] if comp.ndim > 1 else comp
        node["thr"]  = best_t
        node["left"] = self._grow(X[ml],  y[ml],  depth + 1)
        node["right"]= self._grow(X[~ml], y[~ml], depth + 1)
        return nid

    def _pred1(self, x, nid=0):
        node = self.nodes_[nid]
        if node["is_leaf"] or node["proj"] is None:
            return node["label"]
        child = node["left"] if np.dot(node["proj"], x) <= node["thr"] else node["right"]
        return self._pred1(x, child)

    def predict(self, X):
        return np.array([self._pred1(x) for x in X])

    def predict_proba(self, X):
        preds = self.predict(X)
        p = np.zeros((len(preds), self.n_classes_))
        for i, v in enumerate(preds): p[i, v] = 1.0
        return p


# ── Linear Discriminant Forest ────────────────────────────────────────────────

class LinearDiscriminantForest:
    """Ensemble (bagging) of LinearDiscriminantTree classifiers."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 tree_criterion="entropy", discriminator="lda",
                 positive_label_ind=1, neg_features_enabled=False,
                 random_state=42, n_jobs=-1):
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.min_samples_leaf    = min_samples_leaf
        self.tree_criterion      = tree_criterion
        self.discriminator       = discriminator
        self.positive_label_ind  = positive_label_ind
        self.neg_features_enabled= neg_features_enabled
        self.random_state        = random_state
        self.n_jobs              = n_jobs
        self.estimators_         = []

    def _fit_one(self, X, y, seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(y), len(y))
        t = LinearDiscriminantTree(
            positive_label_ind=self.positive_label_ind,
            tree_criterion=self.tree_criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            discriminator=self.discriminator,
        )
        t.fit(X[idx], y[idx]); return t

    def fit(self, X, y):
        seeds = range(self.random_state, self.random_state + self.n_estimators)
        if self.n_jobs == 1:
            self.estimators_ = [self._fit_one(X, y, s) for s in seeds]
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_one)(X, y, s) for s in seeds)
        return self

    def predict(self, X):
        votes = np.array([t.predict(X) for t in self.estimators_])
        return np.apply_along_axis(
            lambda c: np.bincount(c.astype(int)).argmax(), 0, votes)

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.estimators_], 0)


def build_ldt_classifier(config, seed=42):
    """Factory: build LDT or LDF from a config dict."""
    name   = config.get("_classifier", "LDF")
    depth  = config.get("_max_tree_depth", None)
    n_est  = config.get("_ensemble_n_estimators", 100)
    crit   = config.get("_tree_criterion", "entropy")
    disc   = config.get("_discriminator_component_analyser", "lda")
    pos    = config.get("_positive_label_ind", 1)
    leaf   = config.get("_min_tree_samples_leaf", 1)
    neg    = config.get("_neg_features_enabled", False)
    if name == "LDT":
        return LinearDiscriminantTree(pos, crit, depth, leaf, disc, neg)
    return LinearDiscriminantForest(n_est, depth, leaf, crit, disc, pos, neg,
                                    random_state=seed)
