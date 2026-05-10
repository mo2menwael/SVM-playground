"""
Compare our simplified-SMO SVM (a NumPy port of js/svm.js) against
scikit-learn's SVC on the same datasets and hyperparameters.

This script mirrors the website's quick demos (binary + multi-class) and
compares both OvA and OvO strategies where applicable.

Reports per test:
        - training accuracy (ours vs sklearn)
        - support vector count (unique points) (ours vs sklearn)
        - prediction agreement on a 60x60 grid (% of cells where we and
            sklearn assign the same class)
        - iteration counts (ours vs sklearn's `n_iter_`)
        - bias `b` (binary cases only)

Run: python tests/sklearn_compare.py
"""

from __future__ import annotations
import math
import random
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# ----------- our simplified-SMO port (mirrors js/svm.js) -----------

def make_kernel(name: str, gamma: float, degree: int, coef0: float):
    if name == "linear":
        return lambda a, b: float(np.dot(a, b))
    if name == "poly":
        return lambda a, b: (gamma * float(np.dot(a, b)) + coef0) ** degree
    if name == "rbf":
        return lambda a, b: math.exp(-gamma * float(np.sum((a - b) ** 2)))
    raise ValueError(name)


class SimplifiedSMO:
    """Pure NumPy port of js/svm.js — same code path, same constants."""

    def __init__(self, kernel="rbf", C=1.0, gamma=0.5, degree=3, coef0=1.0,
                 tol=1e-3, max_passes=8, max_iter=4000, seed=None):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.rng = random.Random(seed)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = len(X)
        kfn = make_kernel(self.kernel_name, self.gamma, self.degree, self.coef0)
        K = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = kfn(X[i], X[j])

        alphas = np.zeros(m)
        b = 0.0
        E = np.zeros(m)

        def f(k):
            s = 0.0
            for i in range(m):
                if alphas[i] != 0:
                    s += alphas[i] * y[i] * K[i, k]
            return s + b

        # Closure-captured state lets _take_step / _optimize_pair mutate
        # alphas, b, E in lock-step with js/svm.js.
        state = {"b": 0.0}

        def take_step(i, j, Ei):
            Ej = f(j) - y[j]
            ai_old, aj_old = alphas[i], alphas[j]
            if y[i] != y[j]:
                L = max(0.0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)
            if L == H: return False
            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0: return False
            aj = aj_old - (y[j] * (Ei - Ej)) / eta
            aj = min(H, max(L, aj))
            if abs(aj - aj_old) < 1e-5: return False
            ai = ai_old + y[i] * y[j] * (aj_old - aj)
            alphas[i], alphas[j] = ai, aj
            b_cur = state["b"]
            b1 = b_cur - Ei - y[i]*(ai - ai_old)*K[i,i] - y[j]*(aj - aj_old)*K[i,j]
            b2 = b_cur - Ej - y[i]*(ai - ai_old)*K[i,j] - y[j]*(aj - aj_old)*K[j,j]
            if 0 < ai < self.C:    state["b"] = b1
            elif 0 < aj < self.C:  state["b"] = b2
            else:                  state["b"] = (b1 + b2) / 2
            nonlocal b
            b = state["b"]
            E[i] = f(i) - y[i]
            E[j] = f(j) - y[j]
            return True

        def optimize_pair(i, Ei):
            TAU = 1e-12
            j_best, best_gain = -1, -1.0
            Kii = K[i, i]
            for k in range(m):
                if k == i: continue
                denom = Kii + K[k, k] - 2 * K[i, k]
                if denom <= TAU: continue
                diff = Ei - E[k]
                gain = (diff * diff) / denom
                if gain > best_gain:
                    best_gain = gain; j_best = k
            if j_best != -1 and take_step(i, j_best, Ei):
                return True
            start_nb = self.rng.randrange(m)
            for off in range(m):
                k = (start_nb + off) % m
                if k == i or k == j_best: continue
                ak = alphas[k]
                if ak <= 0 or ak >= self.C: continue
                if take_step(i, k, Ei):
                    return True
            start_all = self.rng.randrange(m)
            for off in range(m):
                k = (start_all + off) % m
                if k == i: continue
                if take_step(i, k, Ei):
                    return True
            return False

        passes = 0
        iters = 0
        total_pair_updates = 0
        C, tol = self.C, self.tol
        while passes < self.max_passes and iters < self.max_iter:
            state["b"] = b
            for k in range(m):
                E[k] = f(k) - y[k]
            num_changed = 0
            for i in range(m):
                Ei = f(i) - y[i]
                if (y[i] * Ei < -tol and alphas[i] < C) or \
                   (y[i] * Ei >  tol and alphas[i] > 0):
                    if optimize_pair(i, Ei):
                        num_changed += 1
            iters += 1
            total_pair_updates += num_changed
            passes = passes + 1 if num_changed == 0 else 0

        self.alphas_ = alphas
        self.b_ = b
        self.X_ = X
        self.y_ = y
        self.kfn_ = kfn
        self.iterations_ = iters
        self.pair_updates_ = total_pair_updates
        self.converged_ = passes >= self.max_passes
        return self

    def decision(self, x):
        s = 0.0
        for i in range(len(self.X_)):
            if self.alphas_[i] > 1e-9:
                s += self.alphas_[i] * self.y_[i] * self.kfn_(self.X_[i], x)
        return s + self.b_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([1 if self.decision(x) >= 0 else -1 for x in X])

    def n_support_vectors(self):
        return int(np.sum(self.alphas_ > 1e-6))

    def support_vector_indices(self):
        return np.where(self.alphas_ > 1e-6)[0].tolist()


class MultiClassSMO:
    """Wrapper that mirrors the UI's OvA/OvO strategies for multi-class."""

    def __init__(self, kernel="rbf", C=1.0, gamma=0.5, degree=3, coef0=1.0,
                 tol=1e-3, max_passes=8, max_iter=4000, seed=None, mode="auto"):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.seed = seed
        self.mode = mode

    def _new_model(self, seed):
        return SimplifiedSMO(
            kernel=self.kernel_name,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            tol=self.tol,
            max_passes=self.max_passes,
            max_iter=self.max_iter,
            seed=seed,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(X)
        classes = sorted(set(y.tolist()))
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes")

        self.classes_ = classes
        self.classifiers_ = []

        if len(classes) == 2:
            self.mode_ = "binary"
        else:
            self.mode_ = self.mode if self.mode in ("ovo", "ova") else "ova"

        seed_base = None if self.seed is None else int(self.seed)
        next_seed = 0

        if self.mode_ == "binary":
            pos, neg = classes[0], classes[1]
            y_bin = np.where(y == pos, 1, -1)
            model = self._new_model(seed_base)
            model.fit(X, y_bin)
            self.classifiers_.append({
                "model": model,
                "pos_class": pos,
                "neg_class": neg,
                "train_idx": list(range(n)),
            })
        elif self.mode_ == "ovo":
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    ci, cj = classes[i], classes[j]
                    mask = (y == ci) | (y == cj)
                    idx = np.where(mask)[0]
                    X_pair = X[idx]
                    y_pair = np.where(y[idx] == ci, 1, -1)
                    seed = None if seed_base is None else seed_base + next_seed
                    model = self._new_model(seed)
                    next_seed += 1
                    model.fit(X_pair, y_pair)
                    self.classifiers_.append({
                        "model": model,
                        "pos_class": ci,
                        "neg_class": cj,
                        "train_idx": idx.tolist(),
                    })
        else:
            for ci in classes:
                y_bin = np.where(y == ci, 1, -1)
                seed = None if seed_base is None else seed_base + next_seed
                model = self._new_model(seed)
                next_seed += 1
                model.fit(X, y_bin)
                self.classifiers_.append({
                    "model": model,
                    "pos_class": ci,
                    "neg_class": None,
                    "train_idx": list(range(n)),
                })

        self.iterations_ = max(c["model"].iterations_ for c in self.classifiers_)
        self.pair_updates_ = sum(c["model"].pair_updates_ for c in self.classifiers_)
        return self

    def _predict_one(self, x):
        if len(self.classifiers_) == 1:
            c = self.classifiers_[0]
            other = c["neg_class"]
            if other is None:
                other = self.classes_[1] if self.classes_[0] == c["pos_class"] else self.classes_[0]
            return c["pos_class"] if c["model"].decision(x) >= 0 else other

        if self.mode_ == "ovo":
            votes = {}
            for c in self.classifiers_:
                winner = c["pos_class"] if c["model"].decision(x) >= 0 else c["neg_class"]
                votes[winner] = votes.get(winner, 0) + 1
            return max(votes.items(), key=lambda kv: kv[1])[0]

        best_class = self.classifiers_[0]["pos_class"]
        best_f = -float("inf")
        for c in self.classifiers_:
            f = c["model"].decision(x)
            if f > best_f:
                best_f = f
                best_class = c["pos_class"]
        return best_class

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def support_vector_indices(self):
        out = set()
        for c in self.classifiers_:
            idx = c["model"].support_vector_indices()
            train_idx = c["train_idx"]
            for i in idx:
                out.add(train_idx[i])
        return out

    def support_vector_total(self):
        return sum(c["model"].n_support_vectors() for c in self.classifiers_)

    def biases(self):
        return [c["model"].b_ for c in self.classifiers_]


# ----------- shared synthetic datasets (match demos.js) -----------

def gauss(rng, mu, sigma, n):
    return rng.normal(mu, sigma, n)

def dataset_linear(rng):
    n = 25
    X1 = np.column_stack([gauss(rng, -1.8, 0.7, n), gauss(rng,  1.5, 0.7, n)])
    X2 = np.column_stack([gauss(rng,  1.8, 0.7, n), gauss(rng, -1.5, 0.7, n)])
    X = np.vstack([X1, X2])
    y = np.array([1]*n + [2]*n)
    return X, y

def dataset_xor(rng):
    n = 18
    X = np.vstack([
        np.column_stack([gauss(rng,  2, 0.55, n), gauss(rng,  2, 0.55, n)]),
        np.column_stack([gauss(rng, -2, 0.55, n), gauss(rng, -2, 0.55, n)]),
        np.column_stack([gauss(rng, -2, 0.55, n), gauss(rng,  2, 0.55, n)]),
        np.column_stack([gauss(rng,  2, 0.55, n), gauss(rng, -2, 0.55, n)]),
    ])
    y = np.array([1]*n + [1]*n + [2]*n + [2]*n)
    return X, y

def dataset_circles(rng):
    n_in, n_out = 30, 36
    a1 = rng.random(n_in) * 2 * np.pi
    r1 = gauss(rng, 1.0, 0.18, n_in)
    Xin = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    a2 = rng.random(n_out) * 2 * np.pi
    r2 = gauss(rng, 3.0, 0.22, n_out)
    Xout = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    X = np.vstack([Xin, Xout])
    y = np.array([1]*n_in + [2]*n_out)
    return X, y

def dataset_moons(rng):
    n = 30
    t = rng.random(n) * np.pi
    X1 = np.column_stack([
        2.2 * np.cos(t) - 1 + rng.normal(0, 0.18, n),
        2.2 * np.sin(t) + 0.2 + rng.normal(0, 0.18, n),
    ])
    t2 = rng.random(n) * np.pi
    X2 = np.column_stack([
        2.2 * np.cos(t2) + 1 + rng.normal(0, 0.18, n),
        -2.2 * np.sin(t2) - 0.2 + rng.normal(0, 0.18, n),
    ])
    X = np.vstack([X1, X2])
    y = np.array([1]*n + [2]*n)
    return X, y

def dataset_spiral(rng):
    n = 50
    t = np.arange(n) * (4 * np.pi / n)
    r = 0.4 + 0.45 * t
    X1 = np.column_stack([
        r * np.cos(t) + rng.normal(0, 0.10, n),
        r * np.sin(t) + rng.normal(0, 0.10, n),
    ])
    X2 = np.column_stack([
        r * np.cos(t + np.pi) + rng.normal(0, 0.10, n),
        r * np.sin(t + np.pi) + rng.normal(0, 0.10, n),
    ])
    X = np.vstack([X1, X2])
    y = np.array([1]*n + [2]*n)
    return X, y

def dataset_three_blobs(rng):
    centers = [(-2.5, 1.8), (2.5, 1.8), (0, -2.5)]
    n = 22
    pts = []
    for cx, cy in centers:
        pts.append(np.column_stack([gauss(rng, cx, 0.55, n), gauss(rng, cy, 0.55, n)]))
    X = np.vstack(pts)
    y = np.array([1]*n + [2]*n + [3]*n)
    return X, y

def dataset_four_blobs(rng):
    centers = [(-2.5, 2.5), (2.5, 2.5), (-2.5, -2.5), (2.5, -2.5)]
    n = 18
    pts = []
    for cx, cy in centers:
        pts.append(np.column_stack([gauss(rng, cx, 0.55, n), gauss(rng, cy, 0.55, n)]))
    X = np.vstack(pts)
    y = np.array([1]*n + [2]*n + [3]*n + [4]*n)
    return X, y

def dataset_three_rings(rng):
    radii = [0.9, 2.2, 3.6]
    pts = []
    labels = []
    for k, r in enumerate(radii):
        n = 20 + k * 8
        a = rng.random(n) * 2 * np.pi
        rr = gauss(rng, r, 0.18, n)
        pts.append(np.column_stack([rr * np.cos(a), rr * np.sin(a)]))
        labels.extend([k + 1] * n)
    X = np.vstack(pts)
    y = np.array(labels)
    return X, y

def dataset_five_blobs(rng):
    centers = [(0, 0), (-3, 0), (3, 0), (0, 3), (0, -3)]
    n = 18
    pts = []
    for cx, cy in centers:
        pts.append(np.column_stack([gauss(rng, cx, 0.55, n), gauss(rng, cy, 0.55, n)]))
    X = np.vstack(pts)
    y = np.array([1]*n + [2]*n + [3]*n + [4]*n + [5]*n)
    return X, y

def dataset_ten_blobs(rng):
    R = 3.5
    n = 8
    pts = []
    labels = []
    for c in range(1, 11):
        angle = ((c - 1) / 10) * 2 * np.pi
        cx = R * np.cos(angle)
        cy = R * np.sin(angle)
        pts.append(np.column_stack([gauss(rng, cx, 0.35, n), gauss(rng, cy, 0.35, n)]))
        labels.extend([c] * n)
    X = np.vstack(pts)
    y = np.array(labels)
    return X, y


# ----------- compare on a grid -----------

def grid_agreement(ours, sk, xMin=-5, xMax=5, yMin=-5, yMax=5, n=60):
    xs = np.linspace(xMin, xMax, n)
    ys = np.linspace(yMin, yMax, n)
    grid = np.array([[x, y] for y in ys for x in xs])
    ours_pred = ours.predict(grid)
    sk_pred = sk.predict(grid)
    return float(np.mean(ours_pred == sk_pred))


def sklearn_support_vector_indices(model):
    if isinstance(model, OneVsRestClassifier):
        idx = set()
        for est in model.estimators_:
            if hasattr(est, "support_"):
                idx.update(est.support_.tolist())
        return idx
    if hasattr(model, "support_"):
        return set(model.support_.tolist())
    return set()


def sklearn_pair_updates(model):
    def iter_total(est):
        arr = np.atleast_1d(est.n_iter_)
        return int(np.sum(arr)), int(np.max(arr))

    if isinstance(model, OneVsRestClassifier):
        total = 0
        max_iter = 0
        for est in model.estimators_:
            t, m = iter_total(est)
            total += t
            max_iter = max(max_iter, m)
        return total, max_iter

    return iter_total(model)


def sklearn_biases(model):
    if isinstance(model, OneVsRestClassifier):
        vals = []
        for est in model.estimators_:
            if hasattr(est, "intercept_"):
                vals.extend(est.intercept_.tolist())
        return vals
    if hasattr(model, "intercept_"):
        return model.intercept_.tolist()
    return []


def build_sklearn_model(kernel, params, mode, n_classes):
    sk_kwargs = dict(C=params.get("C", 1.0))
    if kernel == "rbf":
        sk_kwargs["gamma"] = params.get("gamma", 0.5)
    elif kernel == "poly":
        sk_kwargs["gamma"] = params.get("gamma", 0.5)
        sk_kwargs["degree"] = params.get("degree", 3)
        sk_kwargs["coef0"] = params.get("coef0", 1.0)

    base = SVC(kernel=kernel, **sk_kwargs)
    if mode == "ova" and n_classes > 2:
        return OneVsRestClassifier(base)
    return base


# ----------- run cases -----------

def run_case(name, X, y, kernel, mode="auto", **params):
    n_classes = len(set(y))
    ours = MultiClassSMO(kernel=kernel, seed=1, mode=mode, **params).fit(X, y)
    ours_acc = float(np.mean(ours.predict(X) == y))

    sk = build_sklearn_model(kernel, params, ours.mode_, n_classes).fit(X, y)
    sk_acc = float(np.mean(sk.predict(X) == y))

    agree = grid_agreement(ours, sk)
    sk_pairs, _ = sklearn_pair_updates(sk)
    sk_sv = len(sklearn_support_vector_indices(sk))
    ours_sv = len(ours.support_vector_indices())

    mode_label = "Binary" if ours.mode_ == "binary" else ("OvO" if ours.mode_ == "ovo" else "OvA")
    header = f"\n=== {name}  |  kernel={kernel}  mode={mode_label}  {params} ==="
    print(header)

    if ours.mode_ == "binary":
        ours_b = ours.biases()[0]
        sk_b = sklearn_biases(sk)[0]
        print(
            f"  ours    : acc={ours_acc:6.2%}  SVs={ours_sv:3d}  iters={ours.iterations_:4d}  pair_updates={ours.pair_updates_:5d}  b={ours_b:+.4f}"
        )
        print(
            f"  sklearn : acc={sk_acc:6.2%}  SVs={sk_sv:3d}  pair_updates={sk_pairs:5d}  b={sk_b:+.4f}"
        )
    else:
        print(
            f"  ours    : acc={ours_acc:6.2%}  SVs={ours_sv:3d}  iters(max)={ours.iterations_:4d}  pair_updates={ours.pair_updates_:5d}"
        )
        print(
            f"  sklearn : acc={sk_acc:6.2%}  SVs={sk_sv:3d}  pair_updates={sk_pairs:5d}"
        )

    print(f"  -> grid prediction agreement: {agree:6.2%}")
    return {
        "name": name,
        "kernel": kernel,
        "mode": mode_label,
        "ours_acc": ours_acc,
        "sk_acc": sk_acc,
        "ours_sv": ours_sv,
        "sk_sv": sk_sv,
        "ours_iters": ours.iterations_,
        "ours_pairs": ours.pair_updates_,
        "sk_pairs": sk_pairs,
        "agree": agree,
    }


def main():
    rng = np.random.default_rng(42)
    Xlin, ylin       = dataset_linear(rng)
    Xxor, yxor       = dataset_xor(rng)
    Xmoons, ymoons   = dataset_moons(rng)
    Xcirc, ycirc     = dataset_circles(rng)
    Xspiral, yspiral = dataset_spiral(rng)

    X3, y3        = dataset_three_blobs(rng)
    X4, y4        = dataset_four_blobs(rng)
    Xrings, yrings = dataset_three_rings(rng)
    X5, y5        = dataset_five_blobs(rng)
    X10, y10      = dataset_ten_blobs(rng)

    rows = []
    rows.append(run_case("Linear data",   Xlin, ylin,   "linear",  C=1.0))
    rows.append(run_case("Linear data",   Xlin, ylin,   "rbf",     C=1.0, gamma=0.5))
    rows.append(run_case("XOR",           Xxor, yxor,   "rbf",     C=1.0, gamma=0.8))
    rows.append(run_case("Two moons",     Xmoons, ymoons, "rbf",   C=1.0, gamma=0.5))
    rows.append(run_case("Two moons",     Xmoons, ymoons, "rbf",   C=1.0, gamma=2.0))
    rows.append(run_case("Two moons",     Xmoons, ymoons, "poly",  C=1.0, gamma=0.5, degree=3, coef0=1.0))
    rows.append(run_case("Concentric circles", Xcirc, ycirc, "rbf", C=1.0, gamma=0.5))
    rows.append(run_case("Concentric circles", Xcirc, ycirc, "poly",C=1.0, gamma=0.5, degree=2, coef0=1.0))
    rows.append(run_case("Two spirals",   Xspiral, yspiral, "rbf", C=1.0, gamma=1.2))

    rows.append(run_case("Three blobs",   X3, y3, "rbf", mode="ova", C=1.0, gamma=0.5))
    rows.append(run_case("Three blobs",   X3, y3, "rbf", mode="ovo", C=1.0, gamma=0.5))
    rows.append(run_case("Four blobs",    X4, y4, "rbf", mode="ova", C=1.0, gamma=0.5))
    rows.append(run_case("Four blobs",    X4, y4, "rbf", mode="ovo", C=1.0, gamma=0.5))
    rows.append(run_case("Three rings",   Xrings, yrings, "rbf", mode="ova", C=1.0, gamma=0.5))
    rows.append(run_case("Three rings",   Xrings, yrings, "rbf", mode="ovo", C=1.0, gamma=0.5))
    rows.append(run_case("Five blobs",    X5, y5, "rbf", mode="ova", C=1.0, gamma=0.5))
    rows.append(run_case("Five blobs",    X5, y5, "rbf", mode="ovo", C=1.0, gamma=0.5))
    rows.append(run_case("Ten blobs",     X10, y10, "rbf", mode="ova", C=1.0, gamma=0.5))
    rows.append(run_case("Ten blobs",     X10, y10, "rbf", mode="ovo", C=1.0, gamma=0.5))

    print("\n" + "="*70)
    print("Summary  (worst grid-agreement across all cases):")
    worst = min(rows, key=lambda r: r["agree"])
    print(f"  worst case = {worst['name']} / {worst['kernel']} / {worst['mode']}  ->  {worst['agree']:.2%}")
    print(f"  median grid agreement = {sorted(r['agree'] for r in rows)[len(rows)//2]:.2%}")
    print(f"  mean   grid agreement = {sum(r['agree'] for r in rows)/len(rows):.2%}")

    print("\nApples-to-apples PAIR UPDATES  (sklearn n_iter_ is per-pair):")
    for r in rows:
        ratio = r["ours_pairs"] / max(r["sk_pairs"], 1)
        print(
            f"  {r['name']:20s} {r['kernel']:6s} {r['mode']:4s}  ours_pairs={r['ours_pairs']:5d}  sklearn_pairs={r['sk_pairs']:5d}  ours_iters={r['ours_iters']:3d}  ratio={ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
