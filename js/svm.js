/* =========================================================
   svm.js — Support Vector Machine trained with simplified SMO
   (Sequential Minimal Optimization, Platt 1998 — simplified
    version from Stanford CS229 lecture notes).

   Two ways to train:
       1. Synchronous:
              const r = model.train(X, y);

       2. Step-by-step (used to drive the animated UI):
              model.beginTrain(X, y);
              while (!model.trainStep().done) { ... render ... }
   ========================================================= */

import { makeKernel } from './kernels.js';

export class SVM {
    constructor(opts = {}) {
        this.kernel = opts.kernel ?? 'rbf';
        this.C = opts.C ?? 1.0;
        this.gamma = opts.gamma ?? 0.5;
        this.degree = opts.degree ?? 3;
        this.coef0 = opts.coef0 ?? 1.0;
        this.tol = opts.tol ?? 1e-3;
        this.maxPasses = opts.maxPasses ?? 15;
        this.maxIter = opts.maxIter ?? 6000;

        this.alphas = null;
        this.b = 0;
        this.X = null;
        this.y = null;
        this.K = null;            // cached Gram matrix
        this._kFn = null;

        this._iter = 0;
        this._passes = 0;
        this._done = true;
    }

    _kernelParams() {
        return { gamma: this.gamma, degree: this.degree, coef0: this.coef0 };
    }

    /** Decision value at training sample k (using cached Gram-matrix row). */
    _decisionAt(k) {
        const a = this.alphas, y = this.y, K = this.K, m = this.X.length;
        let s = 0;
        for (let i = 0; i < m; i++) {
            if (a[i] !== 0) s += a[i] * y[i] * K[i][k];
        }
        return s + this.b;
    }

    /** Initialise step-training: alphas, b, kernel, Gram matrix, error cache. */
    beginTrain(X, y) {
        const m = X.length;
        this.X = X;
        this.y = y;
        this.alphas = new Float64Array(m);
        this.b = 0;
        this._kFn = makeKernel(this.kernel, this._kernelParams());

        const K = new Array(m);
        for (let i = 0; i < m; i++) {
            K[i] = new Float64Array(m);
            for (let j = 0; j < m; j++) {
                K[i][j] = this._kFn(X[i], X[j]);
            }
        }
        this.K = K;

        // Error cache E[k] = f(x_k) - y_k, used by Platt's second-choice
        // heuristic to pick the j that maximises |E_i - E_j|.
        this._E = new Float64Array(m);

        this._iter = 0;
        this._passes = 0;
        this._done = (m === 0);
    }

    /** One pair-update on (i, j). Returns true if alphas actually changed. */
    _takeStep(i, j, Ei) {
        const y = this.y, K = this.K, alphas = this.alphas, E = this._E;
        const C = this.C;

        const Ej = this._decisionAt(j) - y[j];
        const aiOld = alphas[i];
        const ajOld = alphas[j];

        let L, H;
        if (y[i] !== y[j]) {
            L = Math.max(0, ajOld - aiOld);
            H = Math.min(C, C + ajOld - aiOld);
        } else {
            L = Math.max(0, aiOld + ajOld - C);
            H = Math.min(C, aiOld + ajOld);
        }
        if (L === H) return false;

        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= 0) return false;

        let aj = ajOld - (y[j] * (Ei - Ej)) / eta;
        if (aj > H) aj = H;
        else if (aj < L) aj = L;
        if (Math.abs(aj - ajOld) < 1e-5) return false;

        const ai = aiOld + y[i] * y[j] * (ajOld - aj);
        alphas[i] = ai;
        alphas[j] = aj;

        const b1 = this.b - Ei
            - y[i] * (ai - aiOld) * K[i][i]
            - y[j] * (aj - ajOld) * K[i][j];
        const b2 = this.b - Ej
            - y[i] * (ai - aiOld) * K[i][j]
            - y[j] * (aj - ajOld) * K[j][j];

        if (ai > 0 && ai < C) this.b = b1;
        else if (aj > 0 && aj < C) this.b = b2;
        else this.b = (b1 + b2) / 2;

        E[i] = this._decisionAt(i) - y[i];
        E[j] = this._decisionAt(j) - y[j];
        return true;
    }

    /** Pick j for the working set using a three-tier heuristic:
        1) WSS3 second-order gain  argmax (E_i - E_j)^2 / (K_ii + K_jj - 2 K_ij),
        2) non-bound alphas in random order,
        3) all alphas in random order. */
    _optimizePair(i, Ei) {
        const m = this.X.length;
        const E = this._E;
        const K = this.K;
        const alphas = this.alphas;
        const C = this.C;
        const TAU = 1e-12;

        let jBest = -1, bestGain = -1;
        const Kii = K[i][i];
        for (let k = 0; k < m; k++) {
            if (k === i) continue;
            const denom = Kii + K[k][k] - 2 * K[i][k];
            if (denom <= TAU) continue;
            const diff = Ei - E[k];
            const gain = (diff * diff) / denom;
            if (gain > bestGain) { bestGain = gain; jBest = k; }
        }
        if (jBest !== -1 && this._takeStep(i, jBest, Ei)) return true;

        const startNB = Math.floor(Math.random() * m);
        for (let off = 0; off < m; off++) {
            const k = (startNB + off) % m;
            if (k === i || k === jBest) continue;
            const ak = alphas[k];
            if (ak <= 0 || ak >= C) continue;
            if (this._takeStep(i, k, Ei)) return true;
        }

        const startAll = Math.floor(Math.random() * m);
        for (let off = 0; off < m; off++) {
            const k = (startAll + off) % m;
            if (k === i) continue;
            if (this._takeStep(i, k, Ei)) return true;
        }
        return false;
    }

    /** One simplified-SMO pass over all samples. */
    trainStep() {
        if (this._done) {
            return { numChanged: 0, iter: this._iter, passes: this._passes, done: true };
        }
        const y = this.y, alphas = this.alphas, E = this._E;
        const m = this.X.length;
        const C = this.C, tol = this.tol;

        for (let k = 0; k < m; k++) E[k] = this._decisionAt(k) - y[k];

        let numChanged = 0;
        for (let i = 0; i < m; i++) {
            const Ei = this._decisionAt(i) - y[i];
            if ((y[i] * Ei < -tol && alphas[i] < C) ||
                (y[i] * Ei > tol && alphas[i] > 0)) {
                if (this._optimizePair(i, Ei)) numChanged++;
            }
        }

        this._iter++;
        if (numChanged === 0) this._passes++;
        else this._passes = 0;

        if (this._passes >= this.maxPasses || this._iter >= this.maxIter) {
            this._done = true;
        }

        return {
            numChanged,
            iter: this._iter,
            passes: this._passes,
            done: this._done,
        };
    }

    /** Synchronous training. */
    train(X, y) {
        this.beginTrain(X, y);
        while (!this._done) this.trainStep();
        return {
            iterations: this._iter,
            supportVectors: this.supportVectorCount(),
            converged: this._passes >= this.maxPasses,
        };
    }

    supportVectorCount() {
        if (!this.alphas) return 0;
        let n = 0;
        for (let i = 0; i < this.alphas.length; i++) if (this.alphas[i] > 1e-6) n++;
        return n;
    }

    supportVectorIndices() {
        const out = [];
        if (!this.alphas) return out;
        for (let i = 0; i < this.alphas.length; i++) {
            if (this.alphas[i] > 1e-6) out.push(i);
        }
        return out;
    }

    /** Geometric margin width  =  1 / ||w||  (in feature space).
        ||w||^2 = sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j). */
    marginWidth() {
        if (!this.alphas || !this.K) return 0;
        const a = this.alphas, y = this.y, K = this.K;
        const m = a.length;
        let w2 = 0;
        for (let i = 0; i < m; i++) {
            if (a[i] <= 0) continue;
            for (let j = 0; j < m; j++) {
                if (a[j] <= 0) continue;
                w2 += a[i] * a[j] * y[i] * y[j] * K[i][j];
            }
        }
        return w2 > 0 ? 1 / Math.sqrt(w2) : 0;
    }

    /** Number of training points still violating the KKT conditions
        within tolerance — drops to 0 at convergence. */
    kktViolationCount(tol) {
        if (!this.alphas) return 0;
        const a = this.alphas, y = this.y, C = this.C;
        const t = tol ?? this.tol;
        let count = 0;
        for (let i = 0; i < a.length; i++) {
            const Ei = this._decisionAt(i) - y[i];
            const yEi = y[i] * Ei;
            if ((yEi < -t && a[i] < C) || (yEi > t && a[i] > 0)) count++;
        }
        return count;
    }

    /** Raw decision function f(x) at an arbitrary 2D point. */
    decision(x) {
        if (!this.X) return 0;
        const m = this.X.length;
        const a = this.alphas;
        const y = this.y;
        const k = this._kFn;
        let s = 0;
        for (let i = 0; i < m; i++) {
            if (a[i] > 1e-9) s += a[i] * y[i] * k(this.X[i], x);
        }
        return s + this.b;
    }

    predict(x) {
        return this.decision(x) >= 0 ? 1 : -1;
    }

    /** Evaluate the decision function on an `nx × ny` grid covering
        [xMin..xMax] × [yMin..yMax]. Row-major; row 0 is data y = yMin. */
    decisionGrid(xMin, xMax, yMin, yMax, nx, ny) {
        const out = new Float32Array(nx * ny);
        if (!this.X || this.X.length === 0) return out;

        const dx = (xMax - xMin) / (nx - 1);
        const dy = (yMax - yMin) / (ny - 1);
        const a = this.alphas;
        const y = this.y;
        const X = this.X;
        const k = this._kFn;
        const b = this.b;
        const m = X.length;

        const svIdx = [];
        for (let i = 0; i < m; i++) if (a[i] > 1e-9) svIdx.push(i);

        const point = [0, 0];
        for (let iy = 0; iy < ny; iy++) {
            point[1] = yMin + iy * dy;
            const rowOff = iy * nx;
            for (let ix = 0; ix < nx; ix++) {
                point[0] = xMin + ix * dx;
                let s = 0;
                for (let s_i = 0; s_i < svIdx.length; s_i++) {
                    const i = svIdx[s_i];
                    s += a[i] * y[i] * k(X[i], point);
                }
                out[rowOff + ix] = s + b;
            }
        }
        return out;
    }
}

/* ---------------------------------------------------------
   Multi-class prediction helper.

   `classifiers`: array of { model: SVM, posClass, negClass }.
    OvA entries have negClass === null (one classifier per class,
       trained on "this class" vs "all others"). Predict by argmax of
       raw decision values.
       OvO entries have negClass set (one classifier per class pair).
       Predict by majority vote across all pairwise classifiers.
   `classes`: sorted array of all unique class labels.
   --------------------------------------------------------- */
export function predictMulti(x, classifiers, classes) {
    if (classifiers.length === 0) return null;

    /* Binary fast path — single classifier, OvA or OvO degenerate to the same thing. */
    if (classifiers.length === 1) {
        const c = classifiers[0];
        const otherClass = c.negClass != null
            ? c.negClass
            : (classes[0] === c.posClass ? classes[1] : classes[0]);
        return c.model.decision(x) >= 0 ? c.posClass : otherClass;
    }

    /* OvO: each classifier carries negClass — vote. */
    if (classifiers[0].negClass != null) {
        const votes = new Map();
        for (const c of classifiers) {
            const winner = c.model.decision(x) >= 0 ? c.posClass : c.negClass;
            votes.set(winner, (votes.get(winner) ?? 0) + 1);
        }
        let bestC = classes[0], bestV = -1;
        for (const [cls, v] of votes) {
            if (v > bestV) { bestV = v; bestC = cls; }
        }
        return bestC;
    }

    /* OvA: argmax raw decision value. */
    let bestC = classifiers[0].posClass;
    let bestF = -Infinity;
    for (const c of classifiers) {
        const f = c.model.decision(x);
        if (f > bestF) { bestF = f; bestC = c.posClass; }
    }
    return bestC;
}
