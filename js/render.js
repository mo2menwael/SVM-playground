/* =========================================================
   render.js — pure canvas drawing primitives.

   These functions know nothing about app state; they take the
   canvas, the data they need to draw, and config parameters.
   Higher-level orchestration lives in ui.js (drawScene).
   ========================================================= */

import { VIEW, CLASS_PALETTE, POINT_RADIUS, SV_RING_RADIUS } from './config.js';

/* ---------- Coordinate transforms ---------- */

export function dataToPx(p, canvas) {
    const W = canvas.width, H = canvas.height;
    const x = ((p.x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * W;
    const y = H - ((p.y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * H;
    return { x, y };
}

export function pxToData(px, py, canvas) {
    const W = canvas.width, H = canvas.height;
    const x = VIEW.xMin + (px / W) * (VIEW.xMax - VIEW.xMin);
    const y = VIEW.yMin + ((H - py) / H) * (VIEW.yMax - VIEW.yMin);
    return { x, y };
}

export function eventToData(ev, canvas) {
    const rect = canvas.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width)  * canvas.width;
    const py = ((ev.clientY - rect.top)  / rect.height) * canvas.height;
    return pxToData(px, py, canvas);
}

/* ---------- Color helpers ---------- */

const WHITE = [255, 255, 255];

/** Blend WHITE → rgb by `intensity` ∈ [0, 1]. */
export function blendTowards(rgb, intensity) {
    return [
        (WHITE[0] + (rgb[0] - WHITE[0]) * intensity) | 0,
        (WHITE[1] + (rgb[1] - WHITE[1]) * intensity) | 0,
        (WHITE[2] + (rgb[2] - WHITE[2]) * intensity) | 0,
    ];
}

/* ---------- Background / axes ---------- */

export function clearCanvas(ctx, canvas) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

export function drawAxes(ctx, canvas) {
    const W = canvas.width, H = canvas.height;
    ctx.save();
    ctx.strokeStyle = 'rgba(120, 130, 150, 0.18)';
    ctx.lineWidth = 1;
    for (let v = Math.ceil(VIEW.xMin); v <= VIEW.xMax; v++) {
        const { x } = dataToPx({ x: v, y: 0 }, canvas);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let v = Math.ceil(VIEW.yMin); v <= VIEW.yMax; v++) {
        const { y } = dataToPx({ x: 0, y: v }, canvas);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(60, 70, 90, 0.35)';
    ctx.lineWidth = 1.2;
    const origin = dataToPx({ x: 0, y: 0 }, canvas);
    ctx.beginPath(); ctx.moveTo(0, origin.y); ctx.lineTo(W, origin.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(origin.x, 0); ctx.lineTo(origin.x, H); ctx.stroke();
    ctx.restore();
}

/* ---------- Heat-maps ---------- */

/** Binary mode: tint between two soft colors based on sign of f. */
export function drawHeatmapBinary(ctx, canvas, grid, N, posSoft, negSoft) {
    const W = canvas.width, H = canvas.height;
    const tmp = document.createElement('canvas');
    tmp.width = N; tmp.height = N;
    const tctx = tmp.getContext('2d');
    const img = tctx.createImageData(N, N);
    const data = img.data;

    for (let iy = 0; iy < N; iy++) {
        const srcRow = (N - 1 - iy) * N;     // flip y for ImageData
        const dstRow = iy * N * 4;
        for (let ix = 0; ix < N; ix++) {
            const f = grid[srcRow + ix];
            const t = Math.tanh(f * 0.7);
            const intensity = Math.abs(t);
            const c = blendTowards(t >= 0 ? posSoft : negSoft, intensity);
            const o = dstRow + ix * 4;
            data[o] = c[0]; data[o+1] = c[1]; data[o+2] = c[2]; data[o+3] = 255;
        }
    }
    tctx.putImageData(img, 0, 0);

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, W, H);
    ctx.restore();
}

/** Multi-class mode: argmax classifier wins, intensity = tanh(margin). */
export function drawHeatmapMulti(ctx, canvas, grids, N, classes) {
    const W = canvas.width, H = canvas.height;
    const K = grids.length;

    const tmp = document.createElement('canvas');
    tmp.width = N; tmp.height = N;
    const tctx = tmp.getContext('2d');
    const img = tctx.createImageData(N, N);
    const data = img.data;

    const softs = classes.map(c => CLASS_PALETTE[c].soft);

    for (let iy = 0; iy < N; iy++) {
        const srcRow = (N - 1 - iy) * N;
        const dstRow = iy * N * 4;
        for (let ix = 0; ix < N; ix++) {
            const idx = srcRow + ix;

            let best = -Infinity, second = -Infinity, bestK = 0;
            for (let k = 0; k < K; k++) {
                const v = grids[k][idx];
                if (v > best) { second = best; best = v; bestK = k; }
                else if (v > second) { second = v; }
            }

            const margin = best - second;
            const intensity = Math.tanh(margin * 0.9);
            const c = blendTowards(softs[bestK], intensity);
            const o = dstRow + ix * 4;
            data[o] = c[0]; data[o+1] = c[1]; data[o+2] = c[2]; data[o+3] = 255;
        }
    }
    tctx.putImageData(img, 0, 0);

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, W, H);
    ctx.restore();
}

/* ---------- Contours ---------- */

/** Marching-squares contour at one level on an N×N grid. */
export function drawContour(ctx, canvas, grid, N, level, style) {
    const W = canvas.width, H = canvas.height;
    const cellW = W / (N - 1);
    const cellH = H / (N - 1);

    ctx.save();
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.width || 1.5;
    if (style.dash) ctx.setLineDash(style.dash);
    ctx.beginPath();

    const lerp = (a, b) => (level - a) / (b - a);

    for (let j = 0; j < N - 1; j++) {
        for (let i = 0; i < N - 1; i++) {
            const v00 = grid[j*N + i];
            const v10 = grid[j*N + i + 1];
            const v11 = grid[(j+1)*N + i + 1];
            const v01 = grid[(j+1)*N + i];

            let idx = 0;
            if (v00 > level) idx |= 1;
            if (v10 > level) idx |= 2;
            if (v11 > level) idx |= 4;
            if (v01 > level) idx |= 8;
            if (idx === 0 || idx === 15) continue;

            const x0 = i * cellW;
            const x1 = (i + 1) * cellW;
            const yBot = H - j * cellH;
            const yTop = H - (j + 1) * cellH;

            const eBottom = () => ({ x: x0 + lerp(v00, v10) * cellW, y: yBot });
            const eRight  = () => ({ x: x1, y: yBot + lerp(v10, v11) * (yTop - yBot) });
            const eTop    = () => ({ x: x0 + lerp(v01, v11) * cellW, y: yTop });
            const eLeft   = () => ({ x: x0, y: yBot + lerp(v00, v01) * (yTop - yBot) });

            const draw = (a, b) => { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); };

            switch (idx) {
                case 1:  case 14: draw(eLeft(),   eBottom()); break;
                case 2:  case 13: draw(eBottom(), eRight());  break;
                case 3:  case 12: draw(eLeft(),   eRight());  break;
                case 4:  case 11: draw(eTop(),    eRight());  break;
                case 6:  case 9:  draw(eBottom(), eTop());    break;
                case 7:  case 8:  draw(eLeft(),   eTop());    break;
                case 5:
                    draw(eLeft(), eTop());
                    draw(eBottom(), eRight());
                    break;
                case 10:
                    draw(eLeft(), eBottom());
                    draw(eTop(),  eRight());
                    break;
            }
        }
    }
    ctx.stroke();
    ctx.restore();
}

/** Draw all class-vs-rest boundaries (one contour per class, level 0). */
export function drawMultiClassBoundaries(ctx, canvas, grids, N) {
    const K = grids.length;
    const aux = new Float32Array(N * N);

    for (let k = 0; k < K; k++) {
        for (let i = 0; i < N * N; i++) {
            let maxOther = -Infinity;
            for (let d = 0; d < K; d++) {
                if (d === k) continue;
                const v = grids[d][i];
                if (v > maxOther) maxOther = v;
            }
            aux[i] = grids[k][i] - maxOther;
        }
        drawContour(ctx, canvas, aux, N, 0, { color: '#111827', width: 2 });
    }
}

/* ---------- Points & support-vector rings ---------- */

export function drawPoints(ctx, canvas, points, svIdxSet) {
    /* SV rings first so the dot draws over them. */
    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#f59e0b';
    points.forEach((p, i) => {
        if (svIdxSet.has(i)) {
            const { x, y } = dataToPx(p, canvas);
            ctx.beginPath();
            ctx.arc(x, y, SV_RING_RADIUS, 0, Math.PI * 2);
            ctx.stroke();
        }
    });
    ctx.restore();

    for (const p of points) {
        const { x, y } = dataToPx(p, canvas);
        ctx.beginPath();
        ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = CLASS_PALETTE[p.label].fill;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
    }
}
