// Verifies that the JavaScript model in ABS_Print_Lab.html reproduces the Python model.
// Run from the project root after build_simulator.py:  node simulator/test_runtime.js
const fs = require('fs');
const path = require('path');
const { buildModel } = require('./model_runtime.js');

const html = fs.readFileSync(path.join(__dirname, 'ABS_Print_Lab.html'), 'utf8');
const m = html.match(/const MODEL_DATA = (\{.*?\});\r?\n/s);
if (!m) throw new Error('MODEL_DATA not found in ABS_Print_Lab.html');
const data = JSON.parse(m[1]);
const model = buildModel(data);
const tv = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_vectors.json'), 'utf8'));

let maxErr = 0;
tv.x.forEach((x, i) => {
  const p = model.predict(x);
  maxErr = Math.max(maxErr, Math.abs(p[0] - tv.pred[i][0]), Math.abs(p[1] - tv.pred[i][1]));
});
let maxWc = 0;
tv.wc_x.forEach((x, i) => {
  const w = model.worstCase(x);
  maxWc = Math.max(maxWc, Math.abs(w[0] - tv.wc[i][0]), Math.abs(w[1] - tv.wc[i][1]));
});

const t0 = Date.now();
const s = model.shapley(data.presets.optimum.x);
const shapMs = Date.now() - t0;
const add = Math.max(...[0, 1].map(k => Math.abs(s.base[k] + s.phi[k].reduce((a, b) => a + b, 0) - s.pred[k])));

const de = model.createDE(x => {
  const w = model.worstCase(x);
  return { f: -(0.5 * w[0] + 0.5 * w[1]) + 1e-6 * model.printTime(x[2], x[3], x[4]) };
});
const t1 = Date.now();
let r;
for (let g = 0; g < 60; g++) r = de.step();
const deMs = Date.now() - t1;
const wcBest = model.worstCase(r.best);

console.log(`predict  : ${tv.x.length} recipes, max |JS - Python| = ${maxErr.toExponential(2)} MPa`);
console.log(`worstCase: ${tv.wc_x.length} recipes, max |JS - Python| = ${maxWc.toExponential(2)} MPa`);
console.log(`shapley  : additivity error ${add.toExponential(2)} MPa, ${shapMs} ms, base ${s.base[0].toFixed(2)} MPa`);
console.log(`DE (60 generations, ${deMs} ms): best ${r.best.map(v => v.toFixed(1)).join(', ')} -> worst case ${wcBest.map(v => v.toFixed(2)).join(' / ')} MPa`);
const ok = maxErr < 1e-9 && maxWc < 1e-9 && add < 1e-9;
console.log(ok ? 'ALL CHECKS PASSED' : 'CHECKS FAILED');
process.exit(ok ? 0 : 1);
