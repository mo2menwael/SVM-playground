/* =========================================================
   main.js — application entry point.

   Responsibilities:
     • Cache DOM references.
     • Wire user events (canvas clicks, buttons, sliders, dropdowns).
     • Kick off the initial demo and first training run.

   Loaded as an ES module via <script type="module"> so it runs
   after the DOM is parsed.
   ========================================================= */

import { state } from './state.js';
import { VIEW } from './config.js';
import { demos } from './demos.js';
import { eventToData } from './render.js';
import {
    ui, initDom, setActiveClass, nextClass,
    updateKernelUI, updateLegend, updateStats, drawScene, setStatus,
} from './ui.js';
import { trainModel, stepOnce, invalidateModel } from './training.js';

/* ---------- Canvas point placement ---------- */
function addPointFromEvent(ev, label) {
    ev.preventDefault();
    const p = eventToData(ev, ui.canvas);
    if (p.x < VIEW.xMin || p.x > VIEW.xMax || p.y < VIEW.yMin || p.y > VIEW.yMax) return;
    state.points.push({ x: p.x, y: p.y, label });
    invalidateModel();
}

function wireCanvas() {
    ui.canvas.addEventListener('click', (ev) => {
        addPointFromEvent(ev, state.activeClass);
    });
    ui.canvas.addEventListener('contextmenu', (ev) => {
        const next = nextClass(state.activeClass);
        setActiveClass(next);
        addPointFromEvent(ev, next);
    });
}

/* ---------- Sidebar controls ---------- */
function wireClassToggle() {
    ui.classButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            setActiveClass(parseInt(btn.dataset.class, 10));
        });
    });
}

function wireKernelAndDemo() {
    ui.kernel.addEventListener('change', () => {
        state.kernel = ui.kernel.value;
        updateKernelUI();
        invalidateModel();
    });

    ui.demo.addEventListener('change', () => {
        const k = ui.demo.value;
        if (!k) return;
        state.points = demos[k]();
        invalidateModel();
    });

    ui.multiClassMode.addEventListener('change', () => {
        state.multiClassMode = ui.multiClassMode.value;
        invalidateModel();
    });
}

function wireActionButtons() {
    ui.clear.addEventListener('click', () => {
        state.trainingId++;                // cancel any animation
        state.points = [];
        state.classifiers = [];
        state.classes = [];
        state.gridCaches = [];
        ui.demo.value = '';                // reset demo dropdown to "Choose"
        setStatus('Idle');
        updateStats();
        updateLegend();
        drawScene();
    });

    ui.undo.addEventListener('click', () => {
        if (state.points.length === 0) return;
        state.points.pop();
        invalidateModel();
    });

    ui.train.addEventListener('click', trainModel);
    ui.step.addEventListener('click', stepOnce);

    ui.export.addEventListener('click', () => {
        ui.canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `svm-${state.kernel}-${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 'image/png');
    });

    /* Theme toggle: light <-> dark, persisted in localStorage. The initial
       attribute is set by the inline <head> script before first paint. */
    const reflectTheme = () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        ui.theme.textContent = isDark ? 'Light' : 'Dark';
        ui.theme.setAttribute('aria-pressed', isDark ? 'true' : 'false');
    };
    reflectTheme();
    ui.theme.addEventListener('click', () => {
        const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        if (next === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
        else                 document.documentElement.removeAttribute('data-theme');
        try { localStorage.setItem('svm-theme', next); } catch (_) { /* ignore */ }
        reflectTheme();
    });
}

/* ---------- Hyperparameter sliders + number inputs ---------- */
/* Bidirectional sync: dragging the slider updates the number input and the
   reverse. Direct editing accepts free typing (no clamping until commit) so
   users can type "0.05" without the leading "0" getting stomped by min. */
function bindSlider(input, output, key, fmt) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const clamp = v => Math.min(max, Math.max(min, v));

    const fromSlider = () => {
        const v = parseFloat(input.value);
        state.params[key] = v;
        output.value = fmt(v);
    };
    const fromOutput = (commit) => {
        const raw = parseFloat(output.value);
        if (!isFinite(raw)) return;
        const v = clamp(raw);
        state.params[key] = v;
        input.value = v;
        if (commit) output.value = fmt(v);
    };

    input.addEventListener('input',  fromSlider);
    input.addEventListener('change', () => { fromSlider(); invalidateModel(); });
    output.addEventListener('input',  () => fromOutput(false));
    output.addEventListener('change', () => { fromOutput(true);  invalidateModel(); });

    fromSlider();
}

function wireSliders() {
    bindSlider(ui.c,      ui.outC,      'C',      v => v.toFixed(2));
    bindSlider(ui.gamma,  ui.outGamma,  'gamma',  v => v.toFixed(2));
    bindSlider(ui.degree, ui.outDegree, 'degree', v => String(v | 0));
    bindSlider(ui.coef0,  ui.outCoef0,  'coef0',  v => v.toFixed(2));
}

/* ---------- Bootstrap ---------- */
function init() {
    initDom();

    /* Expose state and helpers for tests/debugging. Local-only app. */
    if (typeof window !== 'undefined') {
        window.__svm = {
            state,
            ui,
            setActiveClass,
            updateLegend,
            trainModel,
        };
    }

    wireCanvas();
    wireClassToggle();
    wireKernelAndDemo();
    wireActionButtons();
    wireSliders();

    setActiveClass(1);
    updateKernelUI();

    /* Boot with a demo so the user has data to play with — but don't train
       until they click Step or Train SVM. */
    state.points = demos.moons();
    invalidateModel();
}

init();
