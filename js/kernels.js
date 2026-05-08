/* =========================================================
   kernels.js — SVM kernel functions and a factory.

   Each kernel takes two 2D points `a` and `b` (number arrays
   of length 2). Parameterised kernels also receive a `params`
   object: { gamma, degree, coef0 }.
   ========================================================= */

export const Kernels = {
    linear(a, b) {
        return a[0] * b[0] + a[1] * b[1];
    },

    poly(a, b, p) {
        const dot = a[0] * b[0] + a[1] * b[1];
        return Math.pow(p.gamma * dot + p.coef0, p.degree);
    },

    rbf(a, b, p) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        return Math.exp(-p.gamma * (dx * dx + dy * dy));
    },

    sigmoid(a, b, p) {
        const dot = a[0] * b[0] + a[1] * b[1];
        return Math.tanh(p.gamma * dot + p.coef0);
    },
};

/** Returns a `(a, b) -> number` kernel function bound to the given params. */
export function makeKernel(name, params) {
    switch (name) {
        case 'linear':  return (a, b) => Kernels.linear(a, b);
        case 'poly':    return (a, b) => Kernels.poly(a, b, params);
        case 'rbf':     return (a, b) => Kernels.rbf(a, b, params);
        case 'sigmoid': return (a, b) => Kernels.sigmoid(a, b, params);
        default: throw new Error('Unknown kernel: ' + name);
    }
}
