# SVM Playground

An interactive, browser-based simulation of **Support Vector Machines** built
from scratch in HTML, CSS, and vanilla JavaScript (ES modules). No frameworks,
no build step.

The app is split into two pages:

1. **Learn** (`index.html`) — an educational landing page covering SVM theory,
   kernel types, and multi-class strategies.
2. **Playground** (`simulation.html`) — scatter labelled points on a canvas,
   choose a kernel (`linear`, `polynomial`, `RBF`), tweak
   hyperparameters, and watch the SVM train **pass-by-pass** via simplified
   SMO. Supports both **binary** and **multi-class (one-vs-all / one-vs-one)**
   classification.

---

## Features

- **Three kernels**: linear, polynomial, RBF (Gaussian).
- **Binary** SVM with decision boundary, margins (`f = ±1`), and support-vector
  highlighting.
- **Multi-class**: up to 10 classes via **one-vs-all** (OvA) or **one-vs-one**
  (OvO) — user selectable.
- **Animated training**: SMO runs inside `requestAnimationFrame` with one pass
  per frame so you can watch the boundary form.
- **Manual stepping**: click **Step** to advance one SMO pass at a time.
- **Live hyperparameters**: sliders for `C`, `γ`, `degree`, `coef0` retrain
  on release.
- **Demo datasets**: linearly separable, XOR, concentric circles, two moons,
  two spirals, 3-blobs, 4-blobs, 5-blobs, 10-blobs, 3-rings.
- **Export PNG**: snapshot the current visualization.
- **Dark / Light theme**: persisted in `localStorage`.
- **Educational content**: landing page explains SVM fundamentals, the kernel
  trick, and OvA vs OvO strategies with interactive cards.

---

## Project structure

```
Project/
├── index.html              # Educational landing page
├── simulation.html         # Interactive SVM playground
├── README.md               # This file
├── css/
│   ├── base.css            # Theme variables, reset, typography
│   ├── layout.css          # App shell grid, info sections, responsive
│   ├── components.css      # Cards, buttons, sliders, toggles, stats, canvas
│   └── landing.css         # Hero, sticky nav, kernel cards, strategy grid
└── js/
    ├── config.js           # Constants, class palette, kernel formulas
    ├── kernels.js          # Kernel functions + factory
    ├── svm.js              # SVM class (simplified SMO) + predictMulti helper
    ├── demos.js            # Demo dataset generators
    ├── state.js            # Mutable application state (single source of truth)
    ├── render.js           # Pure canvas drawing primitives (no state import)
    ├── ui.js               # DOM refs + view orchestration (drawScene, stats, legend)
    ├── training.js         # buildClassifiers + animated/sync training loop
    └── main.js             # Entry point: event wiring + bootstrap
```

### Module dependency graph

```
config ─────► kernels   demos
   │             │
   ▼             ▼
state ◄──── svm ◄────── render
   │         │             │
   └────► ui (uses state, svm, render, config)
              ▲
              │
         training (uses state, svm, ui, config)
              ▲
              │
            main (wires everything; uses demos, ui, training, render)
```

No cyclic imports.

---

## Running

ES modules are blocked from `file://` by browser CORS, so you need to serve
the folder over HTTP. Pick whichever you have:

### Python (probably already installed)

```bash
cd Project
python -m http.server 8000
```

Open <http://localhost:8000>.

### Node

```bash
npx serve Project
# or
npx http-server Project
```

### VS Code "Live Server"

Right-click `index.html` → **Open with Live Server**.

---

## How it works

### Simplified SMO (`js/svm.js`)

The SVM is trained with the simplified Sequential Minimal Optimization
algorithm from the [Stanford CS229 lecture notes][cs229]. The algorithm is
exposed in two flavors:

- `model.train(X, y)` — synchronous, runs to convergence.
- `model.beginTrain(X, y)` + `model.trainStep()` — one SMO pass per call;
  used by the animation loop in `training.js`.

The Gram matrix is precomputed once per training run, so each pass is
*O(m²)* point-evaluations using cached kernel values.

### Multi-class

When ≥ 3 classes are present, `training.js` builds binary classifiers based
on the selected strategy:

| Strategy | Classifiers | Prediction |
|----------|-------------|------------|
| **OvA** (one-vs-all) | K (one per class vs rest) | argmax of raw decision values |
| **OvO** (one-vs-one) | K(K−1)/2 (one per pair) | majority vote |

Boundaries are drawn via marching squares on the auxiliary field
`f_c(x) − max_{d≠c} f_d(x) = 0`.

### Rendering

The decision function is evaluated on a coarse grid (80×80 during animation,
180×180 after training finishes). The grid is rendered as an `ImageData`
heat-map blended with the per-class soft tints, then upscaled with bilinear
filtering onto the visible canvas. Margin lines (`f = ±1`) and the boundary
(`f = 0`) are drawn as marching-squares contours over the heat-map.

[cs229]: http://cs229.stanford.edu/materials/smo.pdf

---

## Pages

### Landing page (`index.html`)

The educational page covers:

- **What is a Support Vector Machine?** — margin maximization and the kernel trick.
- **Kernel types** — linear, polynomial, RBF with formulas and descriptions.
- **Multi-class strategies** — OvA vs OvO with pros / cons.
- Animated hero canvas showing a 4-cluster demo.

### Simulation page (`simulation.html`)

The interactive playground with:

- 3-column app shell (controls | canvas | live stats).
- **How to use** guide below the simulation.
- **How to read the live stats** section explaining accuracy, margin, and KKT violations.

---

## Controls

| Action               | Effect                                                      |
|----------------------|-------------------------------------------------------------|
| Left-click canvas    | Add point of the active class                                |
| Right-click canvas   | Cycle to next class (1→2→…→10→1) and add a point            |
| Class toggle         | Pick which class new points get                              |
| Quick-demo dropdown  | Replace points with a preset dataset                         |
| Multi-class dropdown | Switch between OvA and OvO strategies                        |
| Kernel dropdown      | Switch kernel (irrelevant sliders dim out)                   |
| Sliders              | Adjust C, γ, degree, coef0 (auto-invalidates model)         |
| Train SVM            | Start animated training                                      |
| Step                 | Advance one SMO pass manually                                |
| Export PNG           | Save the current canvas as a PNG file                        |
| Clear / Undo         | Remove all / last point                                      |
| Dark / Light         | Toggle theme (persisted)                                     |

---

## License

Educational project — use freely.
