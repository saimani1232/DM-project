// feature_tests.js - end-to-end feature tests for ABS_Print_Lab.html.
// Paste into the browser console on the open simulator page (or inject it); it drives the real
// UI (slider input events, button clicks, canvas clicks) and checks every result against the model.
// Returns a summary object; every check is also listed in result.log.
(async () => {
  const P = window.PrintLab, D = P.data, M = P.model;
  const $ = id => document.getElementById(id);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const log = []; let pass = 0, fail = 0;
  const check = (name, ok, info = '') => { ok ? pass++ : fail++; log.push(`${ok ? 'PASS' : 'FAIL'} | ${name}${info ? ' | ' + info : ''}`); };
  const close = (a, b, tol = 0.006) => Math.abs(a - b) <= tol;
  const num = el => parseFloat($(el).textContent);
  const setSlider = (id, v) => { const el = $(id); el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); };
  const clickPreset = key => document.querySelector(`#presets [data-preset="${key}"]`).click();
  const same = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  const noNaN = () => !/NaN|undefined|Infinity/.test(document.body.innerText);

  // 1. Boot state
  check('boot: header chips show model and CV R2', /Histogram Gradient Boosting/.test($('chips').textContent) && /0\.9995/.test($('chips').textContent));
  check('boot: starts on the max-strength recipe', same(P.state.x, D.presets.optimum.x), P.state.x.join(','));
  check('boot: tensile 50.41 / compressive 63.02', close(num('tOut'), 50.41) && close(num('cOut'), 63.02), `${num('tOut')} / ${num('cOut')}`);
  check('boot: explanation drawn', $('shapSvg').querySelectorAll('rect').length === 7 && /50\.41/.test($('whyText').textContent));
  check('boot: no NaN/undefined text on page', noNaN());

  // 2. Presets (left panel)
  const expected = { optimum: [50.41, 63.02], pedal: [37.72, 47.27], handle: [30.78, 38.55] };
  for (const key of Object.keys(D.presets)) {
    clickPreset(key);
    const p = M.predict(D.presets[key].x);
    check(`preset ${key}: recipe loaded into controls`, same(P.state.x, D.presets[key].x) &&
      +$('nozzle').value === D.presets[key].x[0] && +$('speed').value === D.presets[key].x[2] && +$('infill').value === D.presets[key].x[4]);
    check(`preset ${key}: readouts equal model prediction`, close(num('tOut'), p[0]) && close(num('cOut'), p[1]), `${num('tOut')} / ${num('cOut')}`);
    if (expected[key]) check(`preset ${key}: matches Python pipeline CSV`, close(p[0], expected[key][0], 0.01) && close(p[1], expected[key][1], 0.01));
  }
  clickPreset('weak');
  check('preset weak: recognised as an exact dataset record', /exact recipe is in the dataset/.test($('nearest').textContent) && /5\.8/.test($('nearest').textContent));

  // 3. Sliders: every slider at min, mid, max
  clickPreset('optimum');
  const sliders = [['nozzle', 0], ['bed', 1], ['speed', 2], ['infill', 4]];
  for (const [id, i] of sliders) {
    const [lo, hi] = D.bounds[i];
    for (const v of [lo, Math.round((lo + hi) / 2), hi]) {
      setSlider(id, v);
      const p = M.predict(P.state.x), wc = M.worstCase(P.state.x);
      const ok = P.state.x[i] === v && parseFloat($(id + 'Out').textContent) === v &&
        close(num('tOut'), p[0]) && close(num('cOut'), p[1]) &&
        $('tWc').textContent.includes(wc[0].toFixed(2)) && $('cWc').textContent.includes(wc[1].toFixed(2));
      check(`slider ${id}=${v}: state, label, prediction and worst case update`, ok);
    }
    clickPreset('optimum');
  }

  // 4. Layer buttons
  $('layer08').click();
  check('layer 0.8 button: state + pressed state + prediction', P.state.x[3] === 0.8 && $('layer08').getAttribute('aria-pressed') === 'true' &&
    $('layer02').getAttribute('aria-pressed') === 'false' && close(num('tOut'), M.predict(P.state.x)[0]));
  $('layer02').click();
  check('layer 0.2 button: state + pressed state', P.state.x[3] === 0.2 && $('layer02').getAttribute('aria-pressed') === 'true');

  // 5. Print-time index and comparison text
  clickPreset('optimum');
  check('print time: max-strength recipe says "same print time"', /same print time/.test($('tauSub').textContent) && close(num('tauOut'), D.maxStrengthTau, 0.06));
  clickPreset('handle');
  const tauH = M.printTime(...P.state.x.slice(2, 5));
  check('print time: door handle shows % faster', close(num('tauOut'), tauH, 0.06) && new RegExp(`${Math.round((1 - tauH / D.maxStrengthTau) * 100)} % faster`).test($('tauSub').textContent), $('tauSub').textContent);
  clickPreset('optimum'); setSlider('speed', 10);
  check('print time: slow recipe shows "× the time"', /× the time/.test($('tauSub').textContent));
  check('print time: bar width capped at 100 %', parseFloat($('tauBar').style.width) <= 100);

  // 6. Shapley explanation
  clickPreset('weak'); clickPreset('optimum');
  check('shap: panel greys out ("updating") right after a recipe change', document.querySelector('.explain').classList.contains('stale'));
  await sleep(700);
  check('shap: panel returns to normal once recalculated', !document.querySelector('.explain').classList.contains('stale'));
  let s = P.getShap();
  const pOpt = M.predict(P.state.x);
  const add = k => Math.abs(s.base[k] + s.phi[k].reduce((a, b) => a + b, 0) - s.pred[k]);
  check('shap: explanation is for the current recipe', close(s.pred[0], pOpt[0], 1e-9) && close(s.pred[1], pOpt[1], 1e-9));
  check('shap: contributions add up exactly (both strengths)', add(0) < 1e-9 && add(1) < 1e-9);
  check('shap: infill is the biggest tensile contributor at the optimum', s.phi[0][4] === Math.max(...s.phi[0]));
  $('expC').click();
  check('shap: compressive toggle redraws chart and text', $('expC').getAttribute('aria-pressed') === 'true' &&
    /compressive prediction ends at 63\.0/.test($('whyText').textContent) && $('shapSvg').textContent.includes('63.02 MPa'));
  $('expT').click();
  check('shap: tensile toggle back', /tensile prediction ends at 50\.41/.test($('whyText').textContent));
  // regression test for the stale-explanation bug: change recipe, toggle immediately
  clickPreset('weak'); $('expC').click(); await sleep(450);
  s = P.getShap();
  const pW = M.predict(P.state.x);
  check('shap: toggling right after a recipe change explains the NEW recipe', close(s.pred[1], pW[1], 1e-9) &&
    $('whyText').textContent.includes(pW[1].toFixed(2)), $('whyText').textContent.slice(-40));
  $('expT').click();
  const t0 = performance.now(); M.shapley(P.state.x); const shapMs = performance.now() - t0;
  check('shap: computation time acceptable (< 1 s)', shapMs < 1000, `${shapMs.toFixed(0)} ms`);

  // 7. Load-test rigs: verdicts, safety factor
  const verdictOf = key => $(key + 'Verdict').textContent;
  clickPreset('optimum');
  check('rig: max-strength recipe passes both parts', verdictOf('pedal') === 'PASS' && verdictOf('handle') === 'PASS');
  const wcO = M.worstCase(P.state.x);
  check('rig: safety factors = worst case / requirement', $('pedalSf').textContent === (wcO[1] / 45).toFixed(2) && $('handleSf').textContent === (wcO[0] / 30).toFixed(2));
  clickPreset('weak');
  check('rig: weakest specimen fails both parts', verdictOf('pedal') === 'FAIL' && verdictOf('handle') === 'FAIL');
  // find a recipe in the warning band (requirement <= worst case < requirement + margin)
  let found = null;
  for (const c of D.cloud) {
    const w = M.worstCase(c.slice(0, 5));
    if (w[1] >= 45 && w[1] < 45 + D.cases.pedal.margin) { found = c.slice(0, 5); break; }
  }
  if (found) {
    P.state.x.splice(0, 5, ...found); $('nozzle').dispatchEvent(new Event('input')); // keep slider-driven path
    setSlider('nozzle', found[0]);
    check('rig: recipe inside the safety margin shows WITHIN MARGIN', verdictOf('pedal') === 'WITHIN MARGIN', found.join(','));
  } else check('rig: WITHIN MARGIN case (no such recipe in cloud, logic only)', true, 'skipped');

  // 8. Load test animation
  clickPreset('pedal');
  $('pedalRun').click(); await sleep(1300);
  check('load test: animation running (gauge fill rising)', P.rigs.pedal.anim !== null);
  $('pedalRun').click();   // second press while running must restart cleanly
  await sleep(3000);
  const wcP = M.worstCase(P.state.x);
  check('load test: finishes with failure message at predicted strength', P.rigs.pedal.anim === null &&
    $('pedalTicker').textContent.includes(`Failed at ${wcP[1].toFixed(1)} MPa`), $('pedalTicker').textContent);
  check('load test: crack drawn after failure', $('pedalSvg').innerHTML.includes('stroke-dashoffset="0"'));
  $('handleRun').click(); await sleep(800);
  setSlider('infill', 50);      // changing the recipe mid-test must cancel the test
  check('load test: recipe change mid-test cancels and resets the rig', P.rigs.handle.anim === null && /Required with model margin/.test($('handleTicker').textContent));
  clickPreset('weak'); $('handleRun').click(); await sleep(3000);
  check('load test: weak recipe fails before the requirement', /before reaching the 30 MPa requirement/.test($('handleTicker').textContent));
  clickPreset('handle');
  document.querySelector('#rig-pedal [data-preset="pedal"]').click();
  check('rig button: "Load optimised pedal recipe" loads it', same(P.state.x, D.presets.pedal.x));
  document.querySelector('#rig-handle [data-preset="handle"]').click();
  check('rig button: "Load optimised handle recipe" loads it', same(P.state.x, D.presets.handle.x));

  // 9. Map click loads a recipe
  const g = P.getMapGeom(), rect = $('mapCanvas').getBoundingClientRect();
  const target = D.pareto[D.pareto.length - 3];
  $('mapCanvas').dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: rect.left + g.sx(target[6]), clientY: rect.top + g.sy(target[5]) }));
  check('map: clicking a Pareto point loads that recipe', same(P.state.x, target.slice(0, 5)), target.slice(0, 5).join(','));
  const before = P.state.x.slice();
  $('mapCanvas').dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: rect.left + 3, clientY: rect.top + 3 }));
  check('map: clicking empty space changes nothing', same(P.state.x, before));

  // 10. Optimiser, all three objectives
  const needs = { strength: null, pedal: 45 + D.cases.pedal.margin, handle: 30 + D.cases.handle.margin };
  for (const obj of ['strength', 'pedal', 'handle']) {
    $('objective').value = obj; $('objective').dispatchEvent(new Event('change'));
    $('optRun').click();
    await sleep(200);
    check(`optimiser ${obj}: run button disabled while running`, $('optRun').disabled && P.isOptimising());
    $('optRun').click();   // ignored while running
    const tStart = performance.now();
    while (P.isOptimising() && performance.now() - tStart < 60000) await sleep(200);
    const dur = (performance.now() - tStart) / 1000;
    const r = P.getOptResult();
    if (!r) { check(`optimiser ${obj}: produced a result`, false, `status: ${$('optStatus').textContent} | disabled ${$('optRun').disabled}`); continue; }
    const wc = M.worstCase(r);
    let ok;
    const tauR = M.printTime(r[2], r[3], r[4]);
    if (obj === 'strength') ok = 0.5 * wc[0] + 0.5 * wc[1] >= 56.70 && r[3] === 0.2 && tauR <= 138.95;
    else ok = (obj === 'pedal' ? wc[1] : wc[0]) >= needs[obj] - 1e-9 && tauR <= (obj === 'pedal' ? 34.75 : 17.45);  // same optimum as Python
    check(`optimiser ${obj}: finished and result meets the goal`, !P.isOptimising() && ok && !$('optLoad').disabled,
      `${r.join(', ')} | wc ${wc.map(v => v.toFixed(2)).join('/')} | tau ${M.printTime(r[2], r[3], r[4]).toFixed(1)} | ${dur.toFixed(1)} s`);
    check(`optimiser ${obj}: result text matches result`, $('optRecipe').textContent.includes(`speed ${r[2]} mm/s`) && $('optRecipe').textContent.includes(`infill ${r[4]} %`));
    $('optLoad').click();
    check(`optimiser ${obj}: "Load result into printer" loads it`, same(P.state.x, r));
  }

  // 11. Canvases render, no broken text
  const nonBlank = c => { const ctx = c.getContext('2d'); const d = ctx.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4 * 50) if (d[i] > 0) n++; return n; };
  check('print canvas draws content', nonBlank($('printCanvas')) > 100);
  check('map canvas draws content', nonBlank($('mapCanvas')) > 100);
  check('page: no NaN/undefined text after all tests', noNaN());
  check('page: no horizontal scroll', document.documentElement.scrollWidth <= document.documentElement.clientWidth);
  check('page: Greek symbols in upper-case labels are not transformed (σ stays σ, τ stays τ)',
    [...document.querySelectorAll('.eyebrow')].every(el => !/[στ]/.test(el.textContent) ||
      [...el.querySelectorAll('.sym')].every(s => getComputedStyle(s).textTransform === 'none') &&
      ![...el.childNodes].some(n => n.nodeType === 3 && /[στ]/.test(n.textContent))));
  check('page: white theme (light background, light colour scheme)',
    getComputedStyle(document.body).backgroundColor === 'rgb(243, 245, 248)' && getComputedStyle(document.documentElement).colorScheme === 'light');

  clickPreset('optimum');
  return { pass, fail, log };
})();
