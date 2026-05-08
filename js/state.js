/* =========================================================
   state.js — single source of truth for mutable app state.

   Other modules import this object and mutate it in place.
   Keeping all live state in one place makes cancellation,
   training pauses, and rendering easy to reason about.
   ========================================================= */

import { GRID_RES_FULL } from './config.js';

export const state = {
    /* User input */
    points: [],                       // [{ x, y, label: 1..10 }]
    activeClass: 1,

    /* Hyperparameters */
    kernel: 'rbf',
    params: { C: 1, gamma: 0.5, degree: 3, coef0: 1 },

    /* Multi-class strategy: 'ova' (one-vs-all, K classifiers)
       or 'ovo' (one-vs-one, K(K-1)/2 classifiers w/ voting). */
    multiClassMode: 'ova',

   /* Training output. For OvA each entry is { model, posClass, negClass=null }.
       For OvO each entry is { model, posClass, negClass } where negClass is set. */
    classifiers: [],
    classes: [],                      // sorted unique labels in `points`

    /* Rendering caches */
    gridN: GRID_RES_FULL,             // current resolution of decision grids
    gridCaches: [],                   // Float32Array per classifier

    /* Cancellation */
    trainingId: 0,                    // bumped when training is (re)started
    isTraining: false,
};
