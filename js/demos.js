/* =========================================================
   demos.js — preset datasets that showcase different kernels.

   Each generator returns an array of { x, y, label } points,
   where label ∈ {1, ..., 10}.
   ========================================================= */

/** Standard normal sample via Box-Muller. */
function gauss(mu, sigma) {
    const u = 1 - Math.random();
    const v = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const demos = {
    /* ---------- Binary ---------- */

    linear() {
        const pts = [];
        for (let i = 0; i < 25; i++) pts.push({ x: gauss(-1.8, 0.7), y: gauss( 1.5, 0.7), label: 1 });
        for (let i = 0; i < 25; i++) pts.push({ x: gauss( 1.8, 0.7), y: gauss(-1.5, 0.7), label: 2 });
        return pts;
    },

    xor() {
        const pts = [];
        for (let i = 0; i < 18; i++) pts.push({ x: gauss( 2, 0.55), y: gauss( 2, 0.55), label: 1 });
        for (let i = 0; i < 18; i++) pts.push({ x: gauss(-2, 0.55), y: gauss(-2, 0.55), label: 1 });
        for (let i = 0; i < 18; i++) pts.push({ x: gauss(-2, 0.55), y: gauss( 2, 0.55), label: 2 });
        for (let i = 0; i < 18; i++) pts.push({ x: gauss( 2, 0.55), y: gauss(-2, 0.55), label: 2 });
        return pts;
    },

    circles() {
        const pts = [];
        for (let i = 0; i < 30; i++) {
            const a = Math.random() * 2 * Math.PI;
            const r = gauss(1.0, 0.18);
            pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 1 });
        }
        for (let i = 0; i < 36; i++) {
            const a = Math.random() * 2 * Math.PI;
            const r = gauss(3.0, 0.22);
            pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 2 });
        }
        return pts;
    },

    moons() {
        const pts = [];
        for (let i = 0; i < 30; i++) {
            const t = Math.random() * Math.PI;
            pts.push({
                x: 2.2 * Math.cos(t)  + gauss(0, 0.18) - 1,
                y: 2.2 * Math.sin(t)  + gauss(0, 0.18) + 0.2,
                label: 1,
            });
        }
        for (let i = 0; i < 30; i++) {
            const t = Math.random() * Math.PI;
            pts.push({
                x: 2.2 * Math.cos(t)  + gauss(0, 0.18) + 1,
                y: -2.2 * Math.sin(t) + gauss(0, 0.18) - 0.2,
                label: 2,
            });
        }
        return pts;
    },

    /* ---------- Multi-class ---------- */

    'three-blobs'() {
        const centers = [[-2.5, 1.8], [2.5, 1.8], [0, -2.5]];
        const pts = [];
        centers.forEach((cen, k) => {
            for (let i = 0; i < 22; i++) {
                pts.push({ x: gauss(cen[0], 0.55), y: gauss(cen[1], 0.55), label: k + 1 });
            }
        });
        return pts;
    },

    'four-blobs'() {
        const centers = [[-2.5, 2.5], [2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]];
        const pts = [];
        centers.forEach((cen, k) => {
            for (let i = 0; i < 18; i++) {
                pts.push({ x: gauss(cen[0], 0.55), y: gauss(cen[1], 0.55), label: k + 1 });
            }
        });
        return pts;
    },

    'three-rings'() {
        const radii = [0.9, 2.2, 3.6];
        const pts = [];
        radii.forEach((r, k) => {
            const n = 20 + k * 8;
            for (let i = 0; i < n; i++) {
                const a = Math.random() * 2 * Math.PI;
                pts.push({
                    x: gauss(r, 0.18) * Math.cos(a),
                    y: gauss(r, 0.18) * Math.sin(a),
                    label: k + 1,
                });
            }
        });
        return pts;
    },

    'five-blobs'() {
        /* One central blob surrounded by four cardinal-direction blobs. */
        const centers = [[0, 0], [-3, 0], [3, 0], [0, 3], [0, -3]];
        const pts = [];
        centers.forEach((cen, k) => {
            for (let i = 0; i < 18; i++) {
                pts.push({ x: gauss(cen[0], 0.55), y: gauss(cen[1], 0.55), label: k + 1 });
            }
        });
        return pts;
    },

    'ten-blobs'() {
        /* Ten tight Gaussian blobs evenly arranged on a circle. */
        const pts = [];
        const R = 3.5;
        for (let c = 1; c <= 10; c++) {
            const angle = ((c - 1) / 10) * 2 * Math.PI;
            const cx = R * Math.cos(angle);
            const cy = R * Math.sin(angle);
            for (let i = 0; i < 8; i++) {
                pts.push({ x: gauss(cx, 0.35), y: gauss(cy, 0.35), label: c });
            }
        }
        return pts;
    },

    spiral() {
        /* Two interleaved spirals — classic non-linearly-separable challenge.
           Linear/poly fail, RBF with γ ≳ 1 nails it. */
        const pts = [];
        const N = 50;
        for (let i = 0; i < N; i++) {
            const t = (i / N) * 4 * Math.PI;
            const r = 0.4 + 0.45 * t;
            pts.push({
                x: r * Math.cos(t)            + gauss(0, 0.10),
                y: r * Math.sin(t)            + gauss(0, 0.10),
                label: 1,
            });
            pts.push({
                x: r * Math.cos(t + Math.PI)  + gauss(0, 0.10),
                y: r * Math.sin(t + Math.PI)  + gauss(0, 0.10),
                label: 2,
            });
        }
        return pts;
    },
};
