/* KAIROS review station.
 *
 * Reads the KAIROS_DATA bundle inlined at build time. Every figure on screen
 * comes from a committed artifact; nothing is computed from a live model.
 */

'use strict';

const DATA = window.KAIROS_DATA || {};
const el = (id) => document.getElementById(id);

function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits === undefined ? 3 : digits);
}

function esc(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* ---------------------------------------------------------------- browser */

let viewer = null;
let designs = [];
let selected = 0;

function buildTree() {
  designs = DATA.designs || [];
  const byFamily = new Map();
  designs.forEach((design, index) => {
    if (!byFamily.has(design.family)) byFamily.set(design.family, []);
    byFamily.get(design.family).push({ design, index });
  });

  el('rail-count').textContent = designs.length + ' parts';
  el('tree').innerHTML = [...byFamily.entries()].map(([family, rows]) => `
    <div class="family">
      <div class="family-name">${esc(family)}<span class="count">${rows.length}</span></div>
      ${rows.map(({ design, index }) => `
        <button class="tree-item" data-index="${index}" aria-current="false">
          <span class="dot ${design.all_satisfied ? '' : 'partial'}"></span>
          <span class="tree-id">${esc(design.design_id.replace('design_', ''))}</span>
        </button>`).join('')}
    </div>`).join('');

  el('tree').querySelectorAll('.tree-item').forEach((button) => {
    button.addEventListener('click', () => select(Number(button.dataset.index)));
  });
}

function select(index) {
  const design = designs[index];
  if (!design) return;
  selected = index;

  el('tree').querySelectorAll('.tree-item').forEach((button) => {
    button.setAttribute('aria-current', String(Number(button.dataset.index) === index));
  });

  if (viewer && design.mesh) {
    try {
      viewer.load(design.mesh);
      el('viewer-error').hidden = true;
      const extent = viewer.extentMm();
      el('scale-label').textContent = extent ? extent.toFixed(1) + ' mm' : '—';
    } catch (err) {
      el('viewer-error').hidden = false;
      el('viewer-error').textContent = 'Viewer: ' + err.message;
    }
  }

  el('requirement').textContent = design.requirement || '(no requirement recorded)';

  const extentMm = design.extent_mm || [];
  const wall = design.min_wall_thickness_mm;
  el('props').innerHTML = `
    <dt>Mass</dt><dd>${fmt(design.mass_g, 2)} g</dd>
    <dt>Volume</dt><dd>${fmt(design.volume_mm3, 0)} mm³</dd>
    <dt>Surface area</dt><dd>${fmt(design.surface_area_mm2, 0)} mm²</dd>
    <dt>Extent X</dt><dd>${fmt(extentMm[0], 2)}</dd>
    <dt>Extent Y</dt><dd>${fmt(extentMm[1], 2)}</dd>
    <dt>Extent Z</dt><dd>${fmt(extentMm[2], 2)}</dd>
    <dt>Faces</dt><dd>${design.faces === null || design.faces === undefined ? '—' : design.faces}</dd>
    <dt>Holes</dt><dd>${design.hole_count === null || design.hole_count === undefined ? '—' : design.hole_count}</dd>
    <dt>Min wall</dt><dd>${wall === null || wall === undefined ? 'not measured' : fmt(wall, 3) + ' mm'}</dd>`;

  const checks = design.constraints || [];
  el('checks').innerHTML = checks.length ? checks.map((check) => `
    <div class="check">
      <span class="flag ${esc(check.status)}"></span>
      <div>
        <div class="kind">${esc(check.kind)}</div>
        <div class="detail">${esc(check.detail)}</div>
      </div>
    </div>`).join('') : '<p class="empty">No constraints recorded.</p>';

  const ops = design.operations || [];
  el('ops').innerHTML = ops.length
    ? ops.map((op) => `<span class="${op === 'FINISH_DESIGN' ? 'terminal' : ''}">${esc(op)}</span>`).join('')
    : '<span class="empty">No trajectory.</span>';

  const met = design.satisfaction_rate === null || design.satisfaction_rate === undefined
    ? '—' : fmt(design.satisfaction_rate * 100, 0) + '%';
  el('titleblock').innerHTML = `
    <div><span class="k">PART</span><span class="v">${esc(design.design_id)}</span></div>
    <div><span class="k">FAMILY</span><span class="v">${esc(design.family)}</span></div>
    <div><span class="k">MATERIAL</span><span class="v">${esc(design.material || '—')}</span></div>
    <div><span class="k">UNITS</span><span class="v">mm / g</span></div>
    <div><span class="k">EXPERT STEPS</span><span class="v">${design.steps === null || design.steps === undefined ? '—' : design.steps}</span></div>
    <div><span class="k">CONSTRAINTS MET</span><span class="v">${met}</span></div>`;

  const mesh = design.mesh;
  el('status-part').textContent = design.design_id.toUpperCase();
  el('status-mesh').textContent = mesh
    ? `${mesh.triangle_count} TRI / ${mesh.vertex_count} VTX`
    : 'NO MESH';
}

/* ---------------------------------------------------------------- benchmark */

function policies() {
  return ((DATA.benchmark || {}).leaderboard || {}).policies || [];
}

function renderBenchmark() {
  const rows = policies();
  if (!rows.length) {
    el('leaderboard').innerHTML = '<p class="empty">No benchmark run in this bundle.</p>';
    return;
  }
  const best = Math.max(...rows.map((r) => r.progress_mean || 0));
  const sorted = rows.slice().sort((a, b) => (b.progress_mean || 0) - (a.progress_mean || 0));
  const oracle = rows.find((r) => /oracle/.test(r.policy || ''));
  const learned = sorted.filter((r) => /^(bc|ppo)$/.test(r.policy || ''));

  el('benchmark-stats').innerHTML = `
    <div class="stat">
      <div class="stat-value ${oracle && oracle.progress_mean >= 0.999 ? 'good' : ''}">${fmt(oracle && oracle.progress_mean)}</div>
      <span class="label">Ceiling</span>
      <div class="stat-sub">oracle-replay</div>
    </div>
    <div class="stat">
      <div class="stat-value">${fmt(learned[0] && learned[0].progress_mean)}</div>
      <span class="label">Best learned</span>
      <div class="stat-sub">${esc(learned[0] ? learned[0].policy : '—')}</div>
    </div>
    <div class="stat">
      <div class="stat-value">${DATA.benchmark.tasks || '—'}</div>
      <span class="label">Tasks</span>
      <div class="stat-sub">${esc(DATA.benchmark.preset || '')} preset</div>
    </div>
    <div class="stat">
      <div class="stat-value">${rows.length}</div>
      <span class="label">Policies</span>
      <div class="stat-sub">incl. 3 baselines</div>
    </div>`;

  el('leaderboard').innerHTML = `
    <table>
      <thead><tr>
        <th>Policy</th><th>Progress</th><th style="width:96px"></th>
        <th>Success</th><th>Validity</th><th>Constraints</th><th>Episodes</th>
      </tr></thead>
      <tbody>${sorted.map((row) => {
        const width = best > 0 ? ((row.progress_mean || 0) / best) * 100 : 0;
        const isCeiling = /oracle/.test(row.policy || '');
        return `<tr class="${isCeiling ? 'ceiling' : ''}">
          <td>${esc(row.policy)}</td>
          <td>${fmt(row.progress_mean)}</td>
          <td><span class="meter ${isCeiling ? '' : 'muted'}"><i style="width:${width.toFixed(1)}%"></i></span></td>
          <td>${fmt(row.success_rate)}</td>
          <td>${fmt(row.validity)}</td>
          <td>${fmt(row.satisfaction)}</td>
          <td>${row.tasks === null || row.tasks === undefined ? '—' : row.tasks}</td>
        </tr>`;
      }).join('')}</tbody>
    </table>`;
}

function renderSuccessCurve() {
  const curves = (DATA.benchmark || {}).success_curve || {};
  const names = Object.keys(curves);
  if (!names.length) {
    el('success-curve').innerHTML = '<p class="empty">No traces in this bundle.</p>';
    return;
  }
  const ks = [...new Set(names.flatMap((n) => Object.keys(curves[n])))]
    .filter((k) => k !== 'build').map(Number).sort((a, b) => a - b);

  const series = names.map((name) => ({
    name,
    points: ks.filter((k) => curves[name][String(k)] !== undefined)
      .map((k) => [k, curves[name][String(k)]]),
  })).filter((s) => s.points.length);

  el('success-curve').innerHTML = lineChart(series, {
    xLabel: 'k = trailing actions the policy must supply',
    yLabel: 'success rate',
    yMax: 1.0,
    xTickLabels: ks.map((k) => [k, String(k)]),
  }) + legend(series.map((s) => s.name));
}

function renderComparisons() {
  const rows = (DATA.comparisons || {}).rows || [];
  if (!rows.length) {
    el('comparisons').innerHTML = '<p class="empty">No paired comparisons in this bundle.</p>';
    return;
  }
  el('comparisons').innerHTML = `
    <table>
      <thead><tr>
        <th>Pair</th><th>Difference</th><th>95% interval</th>
        <th>W / L / T</th><th>Separates</th>
      </tr></thead>
      <tbody>${rows.map((row) => `
        <tr>
          <td>${esc(row.a)} − ${esc(row.b)}</td>
          <td>${(row.difference > 0 ? '+' : '') + fmt(row.difference)}</td>
          <td>[${fmt(row.low)}, ${fmt(row.high)}]</td>
          <td>${row.wins}/${row.losses}/${row.ties}</td>
          <td><span class="verdict ${row.separates ? 'yes' : 'no'}">${row.separates ? 'YES' : 'NO'}</span></td>
        </tr>`).join('')}</tbody>
    </table>`;
}

/* ---------------------------------------------------------------- training */

function renderTraining() {
  const training = DATA.training || {};
  const bc = training.bc || {};
  const ppo = training.ppo || {};
  const history = bc.history || [];

  el('training-stats').innerHTML = `
    <div class="stat">
      <div class="stat-value">${fmt(bc.best_held_out_accuracy)}</div>
      <span class="label">Held-out accuracy</span>
      <div class="stat-sub">next action</div>
    </div>
    <div class="stat">
      <div class="stat-value">${bc.parameters ? (bc.parameters / 1e6).toFixed(2) + 'M' : '—'}</div>
      <span class="label">Parameters</span>
      <div class="stat-sub">language + state</div>
    </div>
    <div class="stat">
      <div class="stat-value">${(bc.dataset || {}).steps_kept || '—'}</div>
      <span class="label">Training steps</span>
      <div class="stat-sub">coverage ${fmt((bc.dataset || {}).coverage, 3)}</div>
    </div>
    <div class="stat">
      <div class="stat-value">${history.length || '—'}</div>
      <span class="label">Epochs</span>
      <div class="stat-sub">best at ${bestEpoch(history)}</div>
    </div>`;

  if (!history.length) {
    el('bc-chart').innerHTML = '<p class="empty">No BC history in this bundle.</p>';
    el('bc-loss-chart').innerHTML = '';
  } else {
    const accuracy = [
      { name: 'train', points: history.map((r) => [r.epoch, r.train_accuracy]) },
      { name: 'held out', points: history.map((r) => [r.epoch, r.held_out_accuracy]) },
    ].filter((s) => s.points.every((p) => p[1] !== null && p[1] !== undefined));
    el('bc-chart').innerHTML =
      lineChart(accuracy, { xLabel: 'epoch', yLabel: 'next-action accuracy', yMax: 1.0 }) +
      legend(accuracy.map((s) => s.name));

    const loss = [
      { name: 'train', points: history.map((r) => [r.epoch, r.train_loss]) },
      { name: 'held out', points: history.map((r) => [r.epoch, r.val_loss]) },
    ].filter((s) => s.points.every((p) => p[1] !== null && p[1] !== undefined));
    el('bc-loss-chart').innerHTML =
      lineChart(loss, { xLabel: 'epoch', yLabel: 'loss' }) + legend(loss.map((s) => s.name));
  }

  const iterations = (ppo.history || []).filter((r) => r.iteration !== null && r.iteration !== undefined);
  if (!iterations.length) {
    el('ppo-reward-chart').innerHTML = '<p class="empty">No PPO history in this bundle.</p>';
    el('ppo-rate-chart').innerHTML = '';
    return;
  }
  el('ppo-reward-chart').innerHTML =
    lineChart([{ name: 'mean episode reward', points: iterations.map((r) => [r.iteration, r.reward_mean]) }],
      { xLabel: 'iteration', yLabel: 'reward' }) + legend(['mean episode reward']);

  const rates = [
    { name: 'success', points: iterations.map((r) => [r.iteration, r.success_rate]) },
    { name: 'invalid actions', points: iterations.map((r) => [r.iteration, r.invalid_action_rate]) },
  ].filter((s) => s.points.every((p) => p[1] !== null && p[1] !== undefined));
  el('ppo-rate-chart').innerHTML =
    lineChart(rates, { xLabel: 'iteration', yLabel: 'rate', yMax: 1.0 }) +
    legend(rates.map((s) => s.name));
}

function bestEpoch(history) {
  let best = null;
  for (const row of history) {
    if (row.held_out_accuracy === null || row.held_out_accuracy === undefined) continue;
    if (!best || row.held_out_accuracy > best.held_out_accuracy) best = row;
  }
  return best ? best.epoch : '—';
}

/* ---------------------------------------------------------------- ablations */

function renderAblations() {
  const rows = (DATA.ablations || {}).rows || [];
  if (!rows.length) {
    el('ablations').innerHTML = '<p class="empty">No ablations in this bundle.</p>';
    return;
  }
  el('ablations').innerHTML = `
    <table>
      <thead><tr>
        <th>Condition</th><th>Progress</th><th>Change</th>
        <th>Success</th><th>Validity</th>
      </tr></thead>
      <tbody>${rows.map((row) => `
        <tr class="${row.baseline ? 'ceiling' : ''}">
          <td>${esc(row.name)}</td>
          <td>${fmt(row.progress_mean)}</td>
          <td style="color:${(row.delta || 0) < 0 ? 'var(--fail)' : 'var(--ink-muted)'}">
            ${row.delta === null || row.delta === undefined ? '—' : (row.delta > 0 ? '+' : '') + fmt(row.delta * 100, 1) + '%'}
          </td>
          <td>${fmt(row.success_rate)}</td>
          <td>${fmt(row.validity)}</td>
        </tr>`).join('')}</tbody>
    </table>`;

  // State the finding rather than the hypothesis: identical success across the
  // intact and corrupted conditions is the result, not a caveat to it.
  const base = rows.find((r) => r.baseline);
  const shuffled = rows.find((r) => /shuffled/.test(r.name || ''));
  const same = base && shuffled && Math.abs(base.success_rate - shuffled.success_rate) < 1e-9;
  el('ablation-note').innerHTML = same
    ? '<strong>The policy does not read its requirement.</strong> Success is identical '
      + 'whether the requirement is intact, swapped for another task’s, or blank. '
      + 'With eight families and near-fixed recipes, a policy can score this well by '
      + 'learning what CAD builds look like, and these tasks do not separate that from '
      + 'requirement following.'
    : '<strong>Read these together.</strong> Requirement ablations test whether the '
      + 'policy is conditioned on the text at all; the mask ablation isolates how much '
      + 'of its action legality belongs to the environment rather than the policy.';
}

/* ---------------------------------------------------------------- shell */

function initViewer() {
  try {
    viewer = new Viewer(el('viewport'));
  } catch (err) {
    el('viewer-error').hidden = false;
    el('viewer-error').textContent = 'Viewer unavailable: ' + err.message
      + '. Metrics and tables are unaffected.';
    return;
  }
  // The viewport ground is a token, so it tracks the active theme.
  const ground = getComputedStyle(document.body).getPropertyValue('--viewport').trim();
  if (/^#[0-9a-f]{6}$/i.test(ground)) {
    viewer.background = [
      parseInt(ground.slice(1, 3), 16) / 255,
      parseInt(ground.slice(3, 5), 16) / 255,
      parseInt(ground.slice(5, 7), 16) / 255,
    ];
  }

  document.querySelectorAll('[data-view3d]').forEach((button) => {
    button.addEventListener('click', () => {
      viewer.setView(button.dataset.view3d);
      el('hud-orientation').textContent = button.dataset.view3d.toUpperCase();
    });
  });
  const gridButton = el('grid-toggle');
  gridButton.setAttribute('aria-pressed', 'true');
  gridButton.addEventListener('click', () => {
    viewer.toggleGrid();
    gridButton.setAttribute('aria-pressed', String(viewer.showGrid));
  });
}

function initTabs() {
  const buttons = document.querySelectorAll('.viewtabs button');
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      const view = button.dataset.view;
      buttons.forEach((other) => other.setAttribute('aria-selected', String(other === button)));

      const isModel = view === 'model';
      el('stage').hidden = !isModel;
      el('rail').hidden = !isModel;
      el('inspector').hidden = !isModel;
      ['benchmark', 'training', 'ablations'].forEach((name) => {
        el('sheet-' + name).hidden = view !== name;
      });
      // A canvas has no size while hidden, so the first draw into a zero-width
      // viewport produces nothing; redraw on reveal.
      if (isModel && viewer) viewer.render();
    });
  });
}

function init() {
  el('suite').textContent = (DATA.benchmark || {}).suite_version || DATA.generated_at || '';
  el('meta-designs').textContent = (DATA.designs || []).length;
  el('meta-policies').textContent = policies().length;
  el('meta-tasks').textContent = (DATA.benchmark || {}).tasks || '—';

  const oracle = policies().find((r) => /oracle/.test(r.policy || ''));
  el('status-invariant').textContent = oracle
    ? (oracle.progress_mean >= 0.999
      ? 'HARNESS OK / ORACLE 1.000'
      : 'ORACLE ' + fmt(oracle.progress_mean) + ' / BELOW CEILING')
    : 'NO ORACLE RUN';
  if (oracle && oracle.progress_mean < 0.999) {
    el('status-invariant').classList.remove('ok');
  }

  buildTree();
  initViewer();
  initTabs();
  renderBenchmark();
  renderSuccessCurve();
  renderComparisons();
  renderTraining();
  renderAblations();
  if ((DATA.designs || []).length) select(0);
}

document.addEventListener('DOMContentLoaded', init);
