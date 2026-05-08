/* =========================================================
   config.js — application-wide constants and palette
   ========================================================= */

/** Coordinate range shown on the canvas (data units). */
export const VIEW = { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };

/** Decision-grid resolution (NxN) used for the heat-map after training finishes. */
export const GRID_RES_FULL = 180;
/** Lower resolution used during animated training to keep frames cheap. */
export const GRID_RES_ANIM = 80;

/** Visual sizes (pixels). */
export const POINT_RADIUS = 6;
export const SV_RING_RADIUS = 11;

/** Per-class palette. `soft` is the heat-map tint (Tailwind *-100 RGB). */
export const CLASS_PALETTE = {
    1:  { fill: '#ef4444', soft: [254, 226, 226], name: 'Class 1'  },
    2:  { fill: '#2563eb', soft: [221, 231, 251], name: 'Class 2'  },
    3:  { fill: '#10b981', soft: [209, 250, 229], name: 'Class 3'  },
    4:  { fill: '#a855f7', soft: [243, 232, 255], name: 'Class 4'  },
    5:  { fill: '#f97316', soft: [255, 237, 213], name: 'Class 5'  },
    6:  { fill: '#ec4899', soft: [252, 231, 243], name: 'Class 6'  },
    7:  { fill: '#06b6d4', soft: [207, 250, 254], name: 'Class 7'  },
    8:  { fill: '#eab308', soft: [254, 249, 195], name: 'Class 8'  },
    9:  { fill: '#b45309', soft: [254, 243, 199], name: 'Class 9'  },
    10: { fill: '#84cc16', soft: [236, 252, 203], name: 'Class 10' },
};
export const ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/** Human-readable kernel formulas (rendered next to the kernel selector). */
export const KERNEL_FORMULAS = {
    linear:  'K(x, y) = x · y',
    poly:    'K(x, y) = (γ · x · y + coef0)^degree',
    rbf:     'K(x, y) = exp(−γ · ‖x − y‖²)',
    sigmoid: 'K(x, y) = tanh(γ · x · y + coef0)',
};
