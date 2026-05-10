/* =========================================================
   ui.js — DOM references, view orchestration, status panel.

   This module:
     • Caches DOM elements once.
     • Owns the `drawScene()` orchestrator that combines render
       primitives based on current `state`.
     • Exposes small pure-ish updaters (legend, stats, kernel
       formula, active class toggle).
   ========================================================= */

import { state } from './state.js';
import { CLASS_PALETTE, KERNEL_FORMULAS } from './config.js';
import { predictMulti } from './svm.js';
import {
    clearCanvas, drawAxes, drawHeatmapBinary, drawHeatmapMulti,
    drawContour, drawMultiClassBoundaries, drawPoints,
} from './render.js';

/** Cached DOM references — populated by `initDom()`. */
export const ui = {
    canvas: null, ctx: null,
    classButtons: null,
    kernel: null, formula: null, demo: null,
    multiClassMode: null,
    clear: null, undo: null, train: null, step: null,
    export: null, theme: null,
    c: null, outC: null,
    gamma: null, outGamma: null,
    degree: null, outDegree: null,
    coef0: null, outCoef0: null,
    statStatus: null, statAcc: null, statSv: null,
    statIter: null, statMode: null, statClasses: null,
    statMargin: null, statKKT: null,
    legend: null,
    rowGamma: null, rowDegree: null, rowCoef0: null,
};

const $ = (id) => document.getElementById(id);

export function initDom() {
    ui.canvas = $('canvas');
    ui.ctx    = ui.canvas.getContext('2d');

    ui.classButtons = Array.from(document.querySelectorAll('.class-btn'));

    ui.kernel         = $('kernel-select');
    ui.formula        = $('kernel-formula');
    ui.demo           = $('demo-select');
    ui.multiClassMode = $('multiclass-mode');

    ui.clear  = $('btn-clear');
    ui.undo   = $('btn-undo');
    ui.train  = $('btn-train');
    ui.step   = $('btn-step');
    ui.theme  = $('btn-theme');

    ui.export    = $('btn-export');

    ui.c      = $('param-c');      ui.outC      = $('out-c');
    ui.gamma  = $('param-gamma');  ui.outGamma  = $('out-gamma');
    ui.degree = $('param-degree'); ui.outDegree = $('out-degree');
    ui.coef0  = $('param-coef0');  ui.outCoef0  = $('out-coef0');

    ui.statStatus  = $('stat-status');
    ui.statAcc     = $('stat-acc');
    ui.statSv      = $('stat-sv');
    ui.statIter    = $('stat-iter');
    ui.statMode    = $('stat-mode');
    ui.statClasses = $('stat-classes');
    ui.statMargin  = $('stat-margin');
    ui.statKKT     = $('stat-kkt');
    ui.legend      = $('legend');

    ui.rowGamma  = document.querySelector('[data-param="gamma"]');
    ui.rowDegree = document.querySelector('[data-param="degree"]');
    ui.rowCoef0  = document.querySelector('[data-param="coef0"]');
}

/* ---------- Active-class toggle ---------- */

export function setActiveClass(label) {
    state.activeClass = label;
    for (const btn of ui.classButtons) {
        btn.classList.toggle('active', parseInt(btn.dataset.class, 10) === label);
    }
    /* When no points exist, the legend's fallback is [state.activeClass]; refresh it
       so the swatch tracks the toggle live. With points present the call is a no-op
       semantically (state.classes drives the legend). */
    if (ui.legend) updateLegend();
}

export function nextClass(label) {
    return (label % 10) + 1;
}

/* ---------- Kernel selector + formula ---------- */

export function updateKernelUI() {
    ui.formula.textContent = KERNEL_FORMULAS[state.kernel];
    const isLinear = state.kernel === 'linear';
    const isPoly   = state.kernel === 'poly';
    ui.rowGamma .classList.toggle('disabled', isLinear);
    ui.rowDegree.classList.toggle('disabled', !isPoly);
    ui.rowCoef0 .classList.toggle('disabled', !isPoly);
}

/* ---------- Legend ---------- */

export function updateLegend() {
    const items = [];
    const classes = state.classes.length ? state.classes : [state.activeClass];
    for (const c of classes) {
        items.push(`<span><i class="sw sw-c${c}"></i> ${CLASS_PALETTE[c].name}</span>`);
    }
    items.push('<span><i class="sw sw-sv"></i> Support vector</span>');
    if (state.classifiers.length === 1) {
        items.push('<span><i class="sw sw-bnd"></i> Decision boundary</span>');
        items.push('<span><i class="sw sw-mar"></i> Margin (f = &plusmn;1)</span>');
    } else if (state.classifiers.length > 1) {
        items.push('<span><i class="sw sw-bnd"></i> Class boundaries</span>');
    }
    ui.legend.innerHTML = items.join('');
}

/* ---------- Stats panel ---------- */

export function updateStats(extra) {
    if (state.classifiers.length === 0) {
        ui.statAcc.textContent     = '—';
        ui.statSv.textContent      = '—';
        ui.statIter.textContent    = '—';
        ui.statMode.textContent    = '—';
        ui.statClasses.textContent = '—';
        ui.statMargin.textContent  = '—';
        if (ui.statKKT) ui.statKKT.textContent = '—';
        if (extra && extra.status !== undefined) ui.statStatus.textContent = extra.status;
        return;
    }

    let correct = 0;
    for (const p of state.points) {
        const pred = predictMulti([p.x, p.y], state.classifiers, state.classes);
        if (pred === p.label) correct++;
    }
    const acc = (100 * correct / state.points.length).toFixed(1);

    const svUnion = collectSupportVectorIndices().size;
    const maxIter = Math.max(...state.classifiers.map(c => c.model._iter || 0));

    /* Margin: tightest classifier's margin (the worst-case corridor).
       KKT: total violations across all classifiers — drops to 0 at convergence. */
    let minMargin = Infinity;
    let kktTotal = 0;
    for (const c of state.classifiers) {
        const w = c.model.marginWidth();
        if (w > 0 && w < minMargin) minMargin = w;
        kktTotal += c.model.kktViolationCount();
    }
    const marginText = isFinite(minMargin) ? minMargin.toFixed(3) : '—';

    ui.statAcc.textContent    = acc + '%';
    ui.statSv.textContent     = String(svUnion);
    ui.statIter.textContent   = String(maxIter);
    ui.statMargin.textContent = marginText;
    if (ui.statKKT) ui.statKKT.textContent = kktTotal === 0 ? '0  ✓' : String(kktTotal);
    if (state.classifiers.length === 1) {
        ui.statMode.textContent = 'Binary';
    } else {
        const tag = state.multiClassMode === 'ovo' ? 'OvO' : 'OvA';
        ui.statMode.textContent = `${tag} (${state.classifiers.length})`;
    }
    ui.statClasses.textContent = state.classes.join(', ');

    if (extra && extra.status !== undefined) ui.statStatus.textContent = extra.status;
}

export function setStatus(text) {
    ui.statStatus.textContent = text;
}

/** Union of support vectors across all classifiers (for ring drawing). */
export function collectSupportVectorIndices() {
    const set = new Set();
    for (const c of state.classifiers) {
        const idx = c.model.supportVectorIndices();
        for (const i of idx) set.add(i);
    }
    return set;
}

/* ---------- Scene draw ---------- */

export function drawScene() {
    const { ctx, canvas } = ui;
    clearCanvas(ctx, canvas);

    if (state.gridCaches.length === 1 && state.classifiers.length === 1) {
        const posClass = state.classifiers[0].posClass;
        const negClass = state.classes[0] === posClass ? state.classes[1] : state.classes[0];
        const grid = state.gridCaches[0];
        const N = state.gridN;

        drawHeatmapBinary(ctx, canvas, grid, N,
            CLASS_PALETTE[posClass].soft, CLASS_PALETTE[negClass].soft);
        drawContour(ctx, canvas, grid, N,  1, { color: 'rgba(107, 114, 128, 0.85)', width: 1.2, dash: [6, 5] });
        drawContour(ctx, canvas, grid, N, -1, { color: 'rgba(107, 114, 128, 0.85)', width: 1.2, dash: [6, 5] });
        drawContour(ctx, canvas, grid, N,  0, { color: '#111827', width: 2.2 });
    } else if (state.gridCaches.length > 1) {
        drawHeatmapMulti(ctx, canvas, state.gridCaches, state.gridN, state.classes);
        drawMultiClassBoundaries(ctx, canvas, state.gridCaches, state.gridN);
    }

    drawAxes(ctx, canvas);
    drawPoints(ctx, canvas, state.points, collectSupportVectorIndices());
}
