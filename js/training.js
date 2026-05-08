/* =========================================================
   training.js — build classifiers and run SMO, optionally
                 animated frame-by-frame.

   Public entry point: `trainModel()`. It looks at `state` and
   either trains synchronously or schedules a requestAnimationFrame
   loop with a per-frame time budget. A monotonically-increasing
   `state.trainingId` is used to cancel any in-flight loop when
   training is restarted (e.g. user clicks Train again, loads a
   demo, changes the kernel mid-animation).
   ========================================================= */

import { state } from './state.js';
import { VIEW, GRID_RES_FULL, GRID_RES_ANIM } from './config.js';
import { SVM } from './svm.js';
import { drawScene, updateLegend, updateStats, setStatus } from './ui.js';

/* ---------- Build classifiers from current points ---------- */
function buildClassifiers() {
    const labels = [...new Set(state.points.map(p => p.label))].sort((a, b) => a - b);
    state.classes = labels;
    if (labels.length < 2) return [];

    const makeModel = () => new SVM({
        kernel: state.kernel,
        C: state.params.C,
        gamma: state.params.gamma,
        degree: state.params.degree,
        coef0: state.params.coef0,
    });

    /* Binary case is identical for OvA and OvO: a single binary SVM. */
    if (labels.length === 2) {
        const X = state.points.map(p => [p.x, p.y]);
        const posClass = labels[0];
        const negClass = labels[1];
        const y = state.points.map(p => p.label === posClass ? 1 : -1);
        const m = makeModel();
        m.beginTrain(X, y);
        return [{ model: m, posClass, negClass, done: false }];
    }

    /* OvO: one binary SVM per class *pair*, each trained only on points
       belonging to those two classes. K(K-1)/2 classifiers total. */
    if (state.multiClassMode === 'ovo') {
        const out = [];
        for (let i = 0; i < labels.length; i++) {
            for (let j = i + 1; j < labels.length; j++) {
                const ci = labels[i], cj = labels[j];
                const pair = state.points.filter(p => p.label === ci || p.label === cj);
                const X = pair.map(p => [p.x, p.y]);
                const y = pair.map(p => p.label === ci ? 1 : -1);
                const m = makeModel();
                m.beginTrain(X, y);
                out.push({ model: m, posClass: ci, negClass: cj, done: false });
            }
        }
        return out;
    }

    /* OvA: one binary SVM per class — "this class" vs "all others". */
    const X = state.points.map(p => [p.x, p.y]);
    return labels.map(c => {
        const y = state.points.map(p => p.label === c ? 1 : -1);
        const m = makeModel();
        m.beginTrain(X, y);
        return { model: m, posClass: c, negClass: null, done: false };
    });
}

/* ---------- Recompute the decision-grid cache ---------- */
/* For both OvA and OvO we produce one grid per class label so the existing
   multi-class renderer (which argmaxes over `gridCaches[k]` per pixel) keeps
   working unchanged.

    OvA: gridCaches[k] = raw decision values from the class-k vs rest classifier.
   OvO: gridCaches[k] = vote counts (how many pairwise classifiers vote for
        class `classes[k]` at each pixel).
*/
function recomputeGrids(N) {
    state.gridN = N;
    const isOvO = state.classifiers.length > 0
        && state.classifiers[0].negClass != null
        && state.classifiers.length !== 1;   // binary fast-path stays OvA-shaped

    if (!isOvO) {
        state.gridCaches = state.classifiers.map(c =>
            c.model.decisionGrid(VIEW.xMin, VIEW.xMax, VIEW.yMin, VIEW.yMax, N, N)
        );
        return;
    }

    /* OvO: build vote-count grids, one per class. */
    const K = state.classes.length;
    const grids = Array.from({ length: K }, () => new Float32Array(N * N));
    const labelIdx = new Map(state.classes.map((c, i) => [c, i]));
    for (const c of state.classifiers) {
        const dec = c.model.decisionGrid(VIEW.xMin, VIEW.xMax, VIEW.yMin, VIEW.yMax, N, N);
        const posIdx = labelIdx.get(c.posClass);
        const negIdx = labelIdx.get(c.negClass);
        for (let p = 0; p < dec.length; p++) {
            if (dec[p] >= 0) grids[posIdx][p]++;
            else             grids[negIdx][p]++;
        }
    }
    state.gridCaches = grids;
}

/* ---------- Pre-flight checks ---------- */
function preflightFail(msg) {
    state.classifiers = [];
    state.classes = [...new Set(state.points.map(p => p.label))].sort((a, b) => a - b);
    state.gridCaches = [];
    setStatus(msg);
    updateStats();
    updateLegend();
    drawScene();
}

/* ---------- Public entry point ---------- */
export function trainModel() {
    /* Bump the training id so any in-flight animation loop sees a stale id
       and exits without rendering. */
    state.trainingId++;
    const myId = state.trainingId;

    if (state.points.length < 2) {
        preflightFail('Need ≥ 2 points');
        return;
    }
    const uniq = new Set(state.points.map(p => p.label));
    if (uniq.size < 2) {
        preflightFail('Need ≥ 2 classes');
        return;
    }

    state.classifiers = buildClassifiers();
    updateLegend();
    runAnimated(myId);
}

/* ---------- Animated path (requestAnimationFrame) ---------- */
function runAnimated(myId) {
    state.isTraining = true;
    setStatus('Training…');
    const t0 = performance.now();

    function tick() {
        if (myId !== state.trainingId) return;     // cancelled

        /* Exactly one SMO pass per classifier per frame so every iteration
           is visible. (For typical datasets one pass is well under 1 ms;
           a time-budget loop would otherwise pack many passes into each
           frame and finish before the user sees anything.) */
        for (const c of state.classifiers) {
            if (!c.done) {
                const r = c.model.trainStep();
                if (r.done) c.done = true;
            }
        }

        /* Coarse grid during animation, full resolution at the end. */
        recomputeGrids(GRID_RES_ANIM);
        updateStats();
        drawScene();

        const allDone = state.classifiers.every(c => c.done);
        if (!allDone) {
            requestAnimationFrame(tick);
        } else {
            state.isTraining = false;
            recomputeGrids(GRID_RES_FULL);
            const elapsed = performance.now() - t0;
            updateStats({ status: `Trained (${elapsed.toFixed(0)} ms)` });
            drawScene();
        }
    }
    requestAnimationFrame(tick);
}

/* ---------- Invalidate the current model ----------
   Called after any user action that changes data or hyperparameters
   (point add/remove, slider release, kernel change, demo load).
   Discards the previous training output and redraws so stale heatmaps
   don't linger; training itself only happens when the user clicks
   Step or Train SVM. */
export function invalidateModel() {
    state.trainingId++;                      // cancel any in-flight animation
    state.isTraining = false;
    state.classifiers = [];
    state.classes = [...new Set(state.points.map(p => p.label))].sort((a, b) => a - b);
    state.gridCaches = [];
    setStatus('Idle');
    updateStats();
    updateLegend();
    drawScene();
}

/* ---------- Manual step (one SMO pass per click) ---------- */

/* True if the existing classifiers were built with parameters that no longer
   match `state` (kernel/sliders changed, or points added/removed). */
function classifiersStale() {
    if (state.classifiers.length === 0) return true;
    const m = state.classifiers[0].model;
    if (m.kernel !== state.kernel
        || m.C       !== state.params.C
        || m.gamma   !== state.params.gamma
        || m.degree  !== state.params.degree
        || m.coef0   !== state.params.coef0) return true;
    /* OvA classifiers are trained on all m points; OvO classifiers are trained
       on only the points of two classes. Detect strategy mismatch by comparing
       classifier count to what each strategy would currently produce. */
    const labels = [...new Set(state.points.map(p => p.label))];
    const expected = labels.length < 2 ? 0
        : labels.length === 2 ? 1
        : (state.multiClassMode === 'ovo'
            ? labels.length * (labels.length - 1) / 2
            : labels.length);
    return state.classifiers.length !== expected;
}

export function stepOnce() {
    /* Pre-flight (same gates as `trainModel`). */
    if (state.points.length < 2) { preflightFail('Need ≥ 2 points'); return; }
    const uniq = new Set(state.points.map(p => p.label));
    if (uniq.size < 2)           { preflightFail('Need ≥ 2 classes'); return; }

    /* Cancel any in-flight animation. */
    state.trainingId++;
    state.isTraining = false;

    /* If the current model is stale (or absent), rebuild from scratch. The
       very first Step click after a fresh dataset takes this path. */
    if (classifiersStale()) {
        state.classifiers = buildClassifiers();
        updateLegend();
    }

    /* If the model has already converged, just report it. */
    if (state.classifiers.every(c => c.done)) {
        const iters = Math.max(...state.classifiers.map(c => c.model._iter));
        updateStats({ status: `Converged at iter ${iters}` });
        return;
    }

    /* One SMO pass per classifier. */
    for (const c of state.classifiers) {
        if (!c.done) {
            const r = c.model.trainStep();
            if (r.done) c.done = true;
        }
    }

    const allDone = state.classifiers.every(c => c.done);
    recomputeGrids(allDone ? GRID_RES_FULL : GRID_RES_ANIM);

    const iters = Math.max(...state.classifiers.map(c => c.model._iter));
    updateStats({
        status: allDone ? `Converged at iter ${iters}` : `Iter ${iters}`,
    });
    drawScene();
}
