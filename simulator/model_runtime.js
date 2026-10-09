/*
 * model_runtime.js
 * Browser/Node runtime for the exported Histogram Gradient Boosting model.
 * Reproduces model.predict() of the Python pipeline exactly:
 *   x_scaled = x * xScale + xMin            (MinMaxScaler on the 5 parameters)
 *   y_scaled = baseline + sum(tree leaves)  (one boosted ensemble per strength)
 *   y_MPa    = (y_scaled - yMin) / yScale   (inverse MinMaxScaler on the targets)
 * Also provides: worst-case prediction over the process tolerance band, exact
 * interventional Shapley values (all 32 coalitions of the 5 parameters), the
 * print-time index, and a step-wise differential-evolution optimiser.
 */
(function (root) {
  'use strict';

  function buildModel(M) {
    const targets = M.targets.map(t => ({
      base: t.base,
      roots: Int32Array.from(t.roots),
      feat: Int8Array.from(t.feat),
      thr: Float64Array.from(t.thr),
      left: Int32Array.from(t.left),
      right: Int32Array.from(t.right),
      val: Float64Array.from(t.val)
    }));
    const xScale = M.xScale, xMin = M.xMin, yScale = M.yScale, yMin = M.yMin;
    const xs = new Float64Array(5);

    function predictScaled(t, x) {
      let s = t.base;
      const roots = t.roots, feat = t.feat, thr = t.thr, left = t.left, right = t.right, val = t.val;
      for (let k = 0; k < roots.length; k++) {
        let n = roots[k];
        while (feat[n] >= 0) n = x[feat[n]] <= thr[n] ? left[n] : right[n];
        s += val[n];
      }
      return s;
    }

    // x: [nozzle, bed, speed, layer, infill] in physical units -> [tensile, compressive] MPa
    function predict(x) {
      for (let i = 0; i < 5; i++) xs[i] = x[i] * xScale[i] + xMin[i];
      return [
        (predictScaled(targets[0], xs) - yMin[0]) / yScale[0],
        (predictScaled(targets[1], xs) - yMin[1]) / yScale[1]
      ];
    }

    // ---------------- robust (worst-case) prediction -----------------
    const tol = M.tolerance;               // [5,5,5,0,5]
    const lo = M.bounds.map(b => b[0]), hi = M.bounds.map(b => b[1]);
    const stencil = [];
    (function gen(d, cur) {
      if (d === 5) { stencil.push(cur.slice()); return; }
      const steps = tol[d] > 0 ? [-1, 0, 1] : [0];
      for (const s of steps) { cur[d] = s * tol[d]; gen(d + 1, cur); }
    })(0, [0, 0, 0, 0, 0]);
    // Fast stencil for the live search: the 16 corners of the tolerance box + its centre.
    // The final recipe is always re-checked with the full 3^4 = 81-point stencil.
    const corners = [[0, 0, 0, 0, 0]];
    for (let m = 0; m < 16; m++) {
      const c = [0, 0, 0, 0, 0];
      [0, 1, 2, 4].forEach((d, k) => { c[d] = ((m >> k) & 1 ? 1 : -1) * tol[d]; });
      corners.push(c);
    }
    const tmp = new Float64Array(5);

    function worstCase(x, fast) {
      let wt = Infinity, wc = Infinity;
      for (const st of (fast ? corners : stencil)) {
        for (let i = 0; i < 5; i++) tmp[i] = Math.min(hi[i], Math.max(lo[i], x[i] + st[i]));
        const p = predict(tmp);
        if (p[0] < wt) wt = p[0];
        if (p[1] < wc) wc = p[1];
      }
      return [wt, wc];
    }

    // ---------------- exact interventional Shapley values ----------------
    // phi_i = sum_S |S|!(n-|S|-1)!/n! [v(S+i) - v(S)],  v(S) = mean_b f(x_S, b_notS)
    const bg = M.background;               // array of [5] recipes (training data)
    const FACT = [1, 1, 2, 6, 24, 120];
    function shapley(x) {
      const v = [];                         // v[mask] = [tensile, compressive]
      const z = new Float64Array(5);
      for (let mask = 0; mask < 32; mask++) {
        let st = 0, sc = 0;
        for (const b of bg) {
          for (let i = 0; i < 5; i++) z[i] = (mask >> i) & 1 ? x[i] : b[i];
          const p = predict(z);
          st += p[0]; sc += p[1];
        }
        v.push([st / bg.length, sc / bg.length]);
      }
      const phi = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
      for (let i = 0; i < 5; i++) {
        for (let mask = 0; mask < 32; mask++) {
          if ((mask >> i) & 1) continue;
          let size = 0;
          for (let j = 0; j < 5; j++) size += (mask >> j) & 1;
          const w = FACT[size] * FACT[5 - size - 1] / FACT[5];
          const withI = mask | (1 << i);
          phi[0][i] += w * (v[withI][0] - v[mask][0]);
          phi[1][i] += w * (v[withI][1] - v[mask][1]);
        }
      }
      return { base: v[0], phi: phi, pred: v[31] };
    }

    function printTime(speed, layer, infill) {
      const vol = M.shellFraction + (1 - M.shellFraction) * infill / 100;
      return 1000 * vol / (speed * layer);
    }

    // ---------------- differential evolution (step-wise) ----------------
    // genes: 4 continuous (nozzle, bed, speed, infill) + 1 layer selector in [0,1].
    // opts.layer fixes the layer height (one population per tested layer, as in the Python optimiser).
    function createDE(objective, opts) {
      opts = opts || {};
      const NP = opts.pop || 36;
      const F_LO = 0.5, F_HI = 1.0, CR = 0.7;
      let seed = opts.seed || 42;
      const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
      const cont = [0, 1, 2, 4];
      const layers = M.layerLevels;
      const decode = g => {
        const x = [0, 0, 0, 0, 0];
        cont.forEach((d, k) => { x[d] = lo[d] + g[k] * (hi[d] - lo[d]); });
        x[3] = opts.layer !== undefined ? opts.layer : (g[4] < 0.5 ? layers[0] : layers[layers.length - 1]);
        return x;
      };
      const pop = [], fit = [], info = [];
      for (let i = 0; i < NP; i++) {
        const g = [rand(), rand(), rand(), rand(), rand()];
        pop.push(g);
        const r = objective(decode(g));
        fit.push(r.f); info.push(r);
      }
      let gen = 0;
      function step() {
        for (let i = 0; i < NP; i++) {
          let a, b, c;
          do { a = (rand() * NP) | 0; } while (a === i);
          do { b = (rand() * NP) | 0; } while (b === i || b === a);
          do { c = (rand() * NP) | 0; } while (c === i || c === a || c === b);
          const F = F_LO + rand() * (F_HI - F_LO);
          const jr = (rand() * 5) | 0;
          const trial = pop[i].map((v, j) => {
            if (rand() < CR || j === jr) {
              let t = pop[a][j] + F * (pop[b][j] - pop[c][j]);
              if (t < 0 || t > 1) t = rand();
              return t;
            }
            return v;
          });
          const r = objective(decode(trial));
          if (r.f <= fit[i]) { pop[i] = trial; fit[i] = r.f; info[i] = r; }
        }
        gen++;
        let best = 0;
        for (let i = 1; i < NP; i++) if (fit[i] < fit[best]) best = i;
        return { gen: gen, best: decode(pop[best]), bestInfo: info[best],
                 population: pop.map(decode), infos: info.slice() };
      }
      return { step: step, decode: decode };
    }

    return { predict, worstCase, shapley, printTime, createDE, stencilSize: stencil.length };
  }

  root.ABSModel = { buildModel };
  if (typeof module !== 'undefined') module.exports = { buildModel };
})(typeof window !== 'undefined' ? window : globalThis);
