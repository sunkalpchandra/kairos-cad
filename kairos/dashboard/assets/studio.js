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

  el('browser-count').textContent = designs.length;
  el('tree').innerHTML = [...byFamily.entries()].map(([family, rows], groupIndex) => `
    <div class="family" data-group="${groupIndex}">
      <button class="node group-node" data-toggle="${groupIndex}" aria-expanded="true">
        <span class="twisty">&#9660;</span>
        <svg class="icon" viewBox="0 0 24 24"><use href="#i-folder"></use></svg>
        ${esc(family)}<span class="count">${rows.length}</span>
      </button>
      <div class="leaves" data-leaves="${groupIndex}">
      ${rows.map(({ design, index }) => `
        <button class="leaf" data-index="${index}" aria-current="false">
          <svg class="icon" viewBox="0 0 24 24"><use href="#i-body"></use></svg>
          <span class="name">${esc(design.design_id.replace('design_', ''))}</span>
          <span class="flag ${design.all_satisfied ? 'pass' : 'warn'}"
                title="${design.all_satisfied ? 'all constraints met' : 'constraints unmet'}"
                role="img" aria-label="${design.all_satisfied ? 'all constraints met' : 'constraints unmet'}"></span>
        </button>`).join('')}
      </div>
    </div>`).join('');

  el('tree').querySelectorAll('.leaf').forEach((button) => {
    button.addEventListener('click', () => select(Number(button.dataset.index)));
  });
  el('tree').querySelectorAll('[data-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const leaves = el('tree').querySelector(`[data-leaves="${button.dataset.toggle}"]`);
      const open = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', String(!open));
      button.querySelector('.twisty').innerHTML = open ? '&#9654;' : '&#9660;';
      leaves.hidden = open;
    });
  });
}

function select(index) {
  const design = designs[index];
  if (!design) return;
  selected = index;

  el('tree').querySelectorAll('.leaf').forEach((button) => {
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

  renderTimeline(design);
  el('dim-readout').hidden = true;
  if (measuring) el('status-measure').textContent = 'MEASURE: PICK A POINT';

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

/* ---------------------------------------------------------------- rollouts */

let rolloutTask = 0;
let rolloutPolicy = null;
let expertView = null;
let policyView = null;

/** The two comparison viewers, built the first time the workspace is shown.
 *
 * A canvas has no size while its sheet is hidden, so one built at load would
 * come up zero-width and draw nothing.
 */
function ensureRolloutViews() {
  if (expertView || typeof Viewer === 'undefined') return;
  try {
    expertView = new Viewer(el('rollout-expert'));
    policyView = new Viewer(el('rollout-policy'));
  } catch (err) {
    expertView = policyView = null;
    return;
  }
  const styles = getComputedStyle(document.documentElement);
  [expertView, policyView].forEach((view) => {
    view.setPalette(styles);
    view.showGrid = false;
    // Both cameras move together: a comparison between two framings is not a
    // comparison between two parts.
    view.onCamera = () => {
      const other = view === expertView ? policyView : expertView;
      if (!other) return;
      other.camera.yaw = view.camera.yaw;
      other.camera.pitch = view.camera.pitch;
      other.camera.distance = view.camera.distance;
      other.render();
    };
  });
}

/** Load the two solids for the selected task and policy. */
function showRolloutSolids(task) {
  ensureRolloutViews();
  const compare = el('rollout-compare');
  if (!expertView) { compare.hidden = true; return; }

  const episodes = task.episodes || [];
  const oracle = episodes.find((e) => /oracle/.test(e.policy));
  const chosen = episodes.find((e) => e.policy === rolloutPolicy)
    || episodes.find((e) => e.mesh && !/oracle/.test(e.policy))
    || episodes.find((e) => !/oracle/.test(e.policy));
  // Nothing to compare if the rebuild has not been run for this bundle.
  if (!oracle || !oracle.mesh) { compare.hidden = true; return; }
  compare.hidden = false;

  expertView.load(oracle.mesh);
  el('rollout-expert-note').textContent =
    oracle.mesh.triangle_count + ' tri';

  el('rollout-policy-name').textContent = chosen ? chosen.policy : 'Policy';
  if (chosen && chosen.mesh) {
    policyView.load(chosen.mesh);
    el('rollout-policy-note').textContent = chosen.mesh.triangle_count + ' tri';
  } else {
    policyView.mesh = null;
    policyView.render();
    el('rollout-policy-note').textContent = chosen
      ? 'left no solid' : 'no episode';
  }
  // Same camera on both, and the expert's framing on both, so a policy part
  // that is half the size reads as half the size.
  policyView.camera.yaw = expertView.camera.yaw;
  policyView.camera.pitch = expertView.camera.pitch;
  policyView.camera.distance = expertView.camera.distance;
  policyView.center = expertView.center;
  policyView.scale = expertView.scale;
  policyView.render();
}

/** One task, every policy's episode, laid against the expert's step count.
 *
 * Read as a strip per policy: one cell per action, accepted or rejected, with
 * the milestones it reached beneath. A policy that jams shows as a solid run
 * of rejected cells, which no aggregate on the leaderboard can show.
 */
function renderRollouts() {
  const data = DATA.rollouts || {};
  const tasks = data.tasks || [];
  const head = el('rollout-head');
  const body = el('rollout-tracks');
  if (!tasks.length) {
    head.innerHTML = '';
    body.innerHTML = '<p class="empty">No traces in this bundle.</p>';
    return;
  }
  rolloutTask = ((rolloutTask % tasks.length) + tasks.length) % tasks.length;
  const task = tasks[rolloutTask];
  const expert = (task.episodes.find((e) => /oracle/.test(e.policy)) || {}).expert_steps
    || (task.episodes[0] || {}).expert_steps || 0;

  head.innerHTML = `
    <div class="rollout-meta">
      <div><span class="k">TASK</span><span class="v">${esc(task.task_id)}</span></div>
      <div><span class="k">FAMILY</span><span class="v">${esc(task.family)}</span></div>
      <div><span class="k">EXPERT STEPS</span><span class="v">${expert}</span></div>
      <div><span class="k">SHOWING</span><span class="v">${rolloutTask + 1} / ${tasks.length}</span></div>
    </div>
    <p class="requirement-line">${esc(task.requirement)}</p>`;

  // Widest episode sets the strip scale, so the strips are comparable.
  const longest = Math.max(1, ...task.episodes.map((e) => (e.operations || []).length));
  const ordered = task.episodes.slice().sort(
    (a, b) => (b.progress_score || 0) - (a.progress_score || 0));

  if (!ordered.some((e) => e.policy === rolloutPolicy)) {
    const first = ordered.find((e) => !/oracle/.test(e.policy));
    rolloutPolicy = first ? first.policy : null;
  }

  body.innerHTML = ordered.map((episode) => {
    const ops = episode.operations || [];
    const accepted = episode.accepted || [];
    const known = accepted.length === ops.length && ops.length > 0;
    const cells = ops.map((op, i) => {
      const state = known ? (accepted[i] ? 'ok' : 'bad') : 'unknown';
      const why = known && !accepted[i] && episode.rejections
        ? ' - ' + (episode.rejections[i] || 'rejected') : '';
      return `<span class="cell ${state}" title="${i + 1}. ${esc(op)}${esc(why)}"></span>`;
    }).join('');
    const reached = (episode.milestones || []).length;
    const total = (data.milestones || []).length || 7;
    return `
      <div class="track" data-policy="${esc(episode.policy)}"
           aria-current="${episode.policy === rolloutPolicy}">
        <div class="track-head">
          <span class="name">${esc(episode.policy)}</span>
          <span class="score">${fmt(episode.progress_score)}</span>
        </div>
        <div class="strip" style="--slots:${longest}">${cells}</div>
        <div class="track-foot">
          <span>${ops.length} action${ops.length === 1 ? '' : 's'}</span>
          <span class="${episode.invalid_actions ? 'warn-text' : ''}">${
            known ? episode.invalid_actions + ' rejected'
                  : 'per-step record not in this trace'}</span>
          <span>${reached}/${total} milestones${
            reached ? ': ' + esc((episode.milestones || []).join(', ')) : ''}</span>
          ${episode.mesh ? '<span>solid rebuilt from the trace</span>' : ''}
          ${episode.aborted ? `<span class="warn-text">aborted: ${esc(episode.abort_reason)}</span>` : ''}
        </div>
      </div>`;
  }).join('');

  body.querySelectorAll('.track').forEach((track) => {
    track.addEventListener('click', () => {
      rolloutPolicy = track.dataset.policy;
      renderRollouts();
    });
  });
  showRolloutSolids(task);
}

/* ---------------------------------------------------------------- shell */

/** Feature icons by operation, matching the ribbon's vocabulary. */
const OP_ICON = {
  CREATE_SKETCH: 'i-sketch', ADD_LINE: 'i-line', ADD_CIRCLE: 'i-circle',
  ADD_ARC: 'i-circle', ADD_RECTANGLE: 'i-rect', ADD_POLYGON: 'i-rect',
  PAD: 'i-pad', POCKET: 'i-pocket', REVOLVE: 'i-revolve',
  FILLET: 'i-fillet', CHAMFER: 'i-chamfer',
  LINEAR_PATTERN: 'i-pattern', CIRCULAR_PATTERN: 'i-pattern', MIRROR: 'i-pattern',
  FINISH_DESIGN: 'i-finish',
};

/* Rollback state for the design on screen. `stepMeshes` maps a step index to
 * the solid as it stood after that action; it is sparse, because a sketch
 * action changes no solid and several designs were not exported at all. */
let stepMeshes = {};
let builtSteps = [];
let atStep = null;
let playTimer = null;

/** The expert trajectory as a feature timeline.
 *
 * In a parametric CAD tool the timeline IS the ordered feature history that
 * produced the solid, and clicking a feature rolls the part back to it. The
 * recorded trajectory is the same object, and scripts/build_steps.py exports
 * the geometry, so this does the same thing rather than only highlighting.
 */
function renderTimeline(design) {
  const track = el('timeline-track');
  const ops = (design && design.operations) || [];
  stopPlayback();
  stepMeshes = (design && design.step_meshes) || {};
  builtSteps = Object.keys(stepMeshes).map(Number).sort((a, b) => a - b);
  atStep = null;
  el('tl-transport').hidden = builtSteps.length < 2;

  if (!ops.length) {
    track.innerHTML = '<span class="empty-track">No recorded trajectory.</span>';
    return;
  }
  track.innerHTML = ops.map((op, index) => {
    const icon = OP_ICON[op] || 'i-body';
    const classes = ['tl-node'];
    if (op === 'FINISH_DESIGN') classes.push('terminal');
    if (stepMeshes[index]) classes.push('built');
    const note = stepMeshes[index] ? ' - click to roll back here' : '';
    return `<button class="${classes.join(' ')}" data-step="${index}"
      title="${index + 1}. ${esc(op)}${note}" aria-current="false">
      <svg class="icon" viewBox="0 0 24 24"><use href="#${icon}"></use></svg>
    </button>`;
  }).join('');

  track.querySelectorAll('.tl-node').forEach((node) => {
    node.addEventListener('click', () => {
      stopPlayback();
      rollTo(Number(node.dataset.step));
    });
  });
  el('status-step').textContent = ops.length + ' FEATURES'
    + (builtSteps.length ? '  ' + builtSteps.length + ' SCRUBBABLE' : '');
}

/** Show the part as it stood after `step`, or the finished part for null.
 *
 * A step with no exported geometry falls back to the last one that has it, so
 * clicking a sketch action shows the solid that action was drawn on rather
 * than doing nothing.
 */
function rollTo(step) {
  const design = designs[selected];
  if (!design) return;
  const ops = design.operations || [];

  let source = null;
  if (step !== null) {
    for (const built of builtSteps) {
      if (built <= step) source = built; else break;
    }
  }
  const mesh = source === null ? design.mesh : stepMeshes[source];
  if (viewer && mesh) {
    try {
      // Keep the finished part's framing: an early step really is smaller.
      viewer.load(mesh, { frame: step === null });
      el('viewer-error').hidden = true;
    } catch (err) {
      el('viewer-error').hidden = false;
      el('viewer-error').textContent = 'Viewer: ' + err.message;
    }
  }

  atStep = step;
  el('dim-readout').hidden = true;
  el('timeline-track').querySelectorAll('.tl-node').forEach((node) => {
    const index = Number(node.dataset.step);
    node.setAttribute('aria-current', String(step !== null && index === step));
    node.classList.toggle('rolled', step !== null && index > step);
  });

  if (step === null) {
    el('status-step').textContent = ops.length + ' FEATURES'
      + (builtSteps.length ? '  ' + builtSteps.length + ' SCRUBBABLE' : '');
  } else {
    const rolled = source === null ? '  NO GEOMETRY YET'
      : (source === step ? '' : '  SHOWING STEP ' + (source + 1));
    el('status-step').textContent =
      'STEP ' + (step + 1) + '/' + ops.length + '  ' + ops[step] + rolled;
  }
  const shown = source === null ? design.mesh : stepMeshes[source];
  el('status-mesh').textContent = shown
    ? `${shown.triangle_count} TRI / ${shown.vertex_count} VTX`
    : 'NO MESH';
}

function stopPlayback() {
  if (playTimer !== null) clearInterval(playTimer);
  playTimer = null;
  const button = el('tl-play');
  if (!button) return;
  button.setAttribute('aria-pressed', 'false');
  button.querySelector('use').setAttribute('href', '#i-play');
}

/** Walk the exported steps in order, which is the build as it happened. */
function playBuild() {
  if (playTimer !== null) { stopPlayback(); return; }
  if (builtSteps.length < 2) return;
  const button = el('tl-play');
  button.setAttribute('aria-pressed', 'true');
  button.querySelector('use').setAttribute('href', '#i-pause');

  // Resume where the part is rolled to; from the top once it is finished.
  let cursor = 0;
  if (atStep !== null) {
    const next = builtSteps.findIndex((step) => step > atStep);
    cursor = next === -1 ? 0 : next;
  }
  rollTo(builtSteps[cursor]);
  playTimer = setInterval(() => {
    cursor += 1;
    if (cursor >= builtSteps.length) { stopPlayback(); rollTo(null); return; }
    rollTo(builtSteps[cursor]);
  }, 520);
}

/** Move one exported step along, stopping at the finished part. */
function stepBy(direction) {
  stopPlayback();
  if (!builtSteps.length) return;
  if (atStep === null) {
    if (direction < 0) rollTo(builtSteps[builtSteps.length - 1]);
    return;
  }
  const here = builtSteps.findIndex((step) => step >= atStep);
  const next = (here === -1 ? builtSteps.length : here) + direction;
  if (next < 0) rollTo(builtSteps[0]);
  else if (next >= builtSteps.length) rollTo(null);
  else rollTo(builtSteps[next]);
}

/** Rotate the ViewCube to match the orbit camera.
 *
 * The camera is yaw/pitch about the part; the cube shows the same orientation
 * from the outside, so its rotations are the inverse.
 */
function syncViewCube() {
  const cube = el('viewcube');
  if (!cube || !viewer) return;
  const pitch = (viewer.camera.pitch * 180) / Math.PI;
  const yaw = (viewer.camera.yaw * 180) / Math.PI;
  cube.style.transform = `rotateX(${pitch - 90}deg) rotateZ(${-yaw - 90}deg)`;
}

function initViewer() {
  try {
    viewer = new Viewer(el('viewport'));
  } catch (err) {
    el('viewer-error').hidden = false;
    el('viewer-error').textContent = 'Viewer unavailable: ' + err.message
      + '. Metrics and tables are unaffected.';
    return;
  }
  // The canvas composites over the CSS gradient, so only the grid tones need
  // handing to the renderer. They are tokens, so they track the active theme.
  viewer.setPalette(getComputedStyle(document.documentElement));

  // A theme can change after load: the host stamps data-theme when the viewer
  // switches. Re-read rather than keeping the tones from first paint.
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      viewer.setPalette(getComputedStyle(document.documentElement));
    });
  }
  new MutationObserver(() => viewer.setPalette(getComputedStyle(document.documentElement)))
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // Both the ribbon and the ViewCube drive the same setView.
  document.querySelectorAll('[data-view3d]').forEach((button) => {
    button.addEventListener('click', () => {
      const name = button.dataset.view3d;
      viewer.setView(name);
      el('hud-orientation').textContent = name.toUpperCase();
      document.querySelectorAll('.cmd[data-view3d]').forEach((cmd) => {
        cmd.setAttribute('aria-pressed', String(cmd.dataset.view3d === name));
      });
      syncViewCube();
    });
  });

  const grid = el('cmd-grid');
  grid.addEventListener('click', () => {
    viewer.toggleGrid();
    grid.setAttribute('aria-pressed', String(viewer.showGrid));
  });

  // Visual style, as a CAD tool cycles it. Shaded-with-edges first because it
  // is the one a part is normally read in.
  const STYLES = [
    { name: 'Edges', icon: 'i-edges', edges: true, wireframe: false },
    { name: 'Shaded', icon: 'i-shaded', edges: false, wireframe: false },
    // Wireframe draws the model edges, not the tessellation: the triangles are
    // an artifact of meshing, and drawing them buries the part in noise.
    { name: 'Wire', icon: 'i-wire', edges: true, wireframe: true },
  ];
  const shade = el('cmd-shade');
  let style = 0;
  shade.addEventListener('click', () => {
    style = (style + 1) % STYLES.length;
    const chosen = STYLES[style];
    viewer.showEdges = chosen.edges;
    viewer.wireframe = chosen.wireframe;
    shade.setAttribute('aria-pressed', String(!chosen.wireframe));
    shade.querySelector('use').setAttribute('href', '#' + chosen.icon);
    shade.lastChild.textContent = chosen.name;
    el('status-style').textContent = chosen.name.toUpperCase();
    viewer.render();
  });

  const fit = () => { viewer.setView('iso'); syncViewCube(); };
  el('cmd-fit').addEventListener('click', fit);
  el('nav-fit').addEventListener('click', fit);

  // Orbit/pan/zoom are modes on one pointer, so the buttons report which is
  // active rather than pretending to be separate tools.
  const modes = { 'nav-orbit': 'orbit', 'nav-pan': 'pan', 'nav-zoom': 'zoom' };
  Object.keys(modes).forEach((id) => {
    el(id).addEventListener('click', () => {
      viewer.mode = modes[id];
      Object.keys(modes).forEach((other) => {
        el(other).setAttribute('aria-pressed', String(other === id));
      });
    });
  });

  // ---- section view -------------------------------------------------------
  const axes = ['X', 'Y', 'Z'];
  const sectionState = () => {
    el('status-section').textContent = viewer.section.on
      ? `SECTION ${axes[viewer.section.axis]} @ ${(viewer.section.cut * 100).toFixed(0)}%`
      : '';
  };
  const sectionButton = el('cmd-section');
  sectionButton.addEventListener('click', () => {
    viewer.section.on = !viewer.section.on;
    sectionButton.setAttribute('aria-pressed', String(viewer.section.on));
    sectionState();
    viewer.render();
  });
  el('cmd-axis').addEventListener('click', () => {
    viewer.section.axis = (viewer.section.axis + 1) % 3;
    el('axis-label').textContent = axes[viewer.section.axis];
    sectionState();
    viewer.render();
  });
  el('section-cut').addEventListener('input', (event) => {
    viewer.section.cut = Number(event.target.value) / 100;
    sectionState();
    viewer.render();
  });

  // ---- export -------------------------------------------------------------
  el('cmd-export').addEventListener('click', () => {
    const design = designs[selected];
    if (!design || !viewer.mesh) return;
    const url = URL.createObjectURL(viewer.toStl(design.design_id));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${design.design_id}.stl`;
    link.click();
    // Revoke on the next tick: revoking synchronously can beat the download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  el('cmd-prev').addEventListener('click', () => select((selected - 1 + designs.length) % designs.length));
  el('cmd-next').addEventListener('click', () => select((selected + 1) % designs.length));
  el('cmd-props').addEventListener('click', () => el('panel-measure').scrollIntoView({ block: 'nearest' }));
  el('cmd-measure').addEventListener('click', toggleMeasure);
  el('cmd-checks').addEventListener('click', () => el('panel-checks').scrollIntoView({ block: 'nearest' }));

  document.querySelectorAll('[data-goto]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = el(button.dataset.goto);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  // The dimension label is positioned in screen space, so it has to follow the
  // camera; without this it detaches from the span the moment you orbit.
  viewer.onCamera = () => { syncViewCube(); if (viewer.measure.length === 2) placeDimension(); };
  syncViewCube();

  // Click, not drag: orbiting through the part must not drop a point.
  const viewport = el('viewport');
  let down = null;
  viewport.addEventListener('pointerdown', (event) => {
    down = { x: event.clientX, y: event.clientY };
  });
  viewport.addEventListener('pointerup', (event) => {
    if (!down) return;
    const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
    down = null;
    if (moved < 4) pickMeasurePoint(event);
  });
}

function initTabs() {
  const buttons = document.querySelectorAll('.workspaces button');
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      const view = button.dataset.view;
      buttons.forEach((other) => other.setAttribute('aria-selected', String(other === button)));

      const isModel = view === 'model';
      // Hiding a panel does not release its grid track: the browser and
      // inspector columns and the timeline row went on reserving their width
      // and left the sheets in a 970px slot with empty gutters either side.
      document.querySelector('.shell').classList.toggle('sheet-mode', !isModel);
      el('canvas').hidden = !isModel;
      el('browser').hidden = !isModel;
      el('inspector').hidden = !isModel;
      el('timeline').hidden = !isModel;
      // The ribbon carries each workspace's own commands, as a CAD ribbon does,
      // rather than going blank and leaving a dead band across the top.
      document.querySelectorAll('.group[data-workspace]').forEach((group) => {
        group.hidden = group.dataset.workspace !== view;
      });
      ['benchmark', 'training', 'rollouts', 'ablations'].forEach((name) => {
        el('sheet-' + name).hidden = view !== name;
      });
      // A canvas has no size while hidden, so the first draw into a zero-width
      // viewport produces nothing; redraw on reveal.
      if (isModel && viewer) viewer.render();
      // Same reason the model canvas is redrawn on reveal: these had no size
      // while the sheet was hidden, so the first draw went into nothing.
      if (view === 'rollouts') renderRollouts();
      // The part readouts describe something that is not on screen in a data
      // workspace, so they go quiet rather than reporting a stale part.
      ['status-part', 'status-mesh', 'status-style', 'status-step',
       'status-section', 'status-measure'].forEach((id) => {
        el(id).hidden = !isModel;
      });
    });
  });
}

/** Filter the browser tree, hiding families that end up empty. */
function initFilter() {
  const input = el('tree-filter');
  if (!input) return;
  input.addEventListener('input', () => {
    const needle = input.value.trim().toLowerCase();
    el('tree').querySelectorAll('.family').forEach((family) => {
      let shown = 0;
      family.querySelectorAll('.leaf').forEach((leaf) => {
        const design = designs[Number(leaf.dataset.index)] || {};
        const hay = `${design.design_id} ${design.family}`.toLowerCase();
        const match = !needle || hay.includes(needle);
        leaf.hidden = !match;
        if (match) shown += 1;
      });
      // A family with nothing left is noise, not an empty state.
      family.hidden = shown === 0;
    });
  });
}

/** Timeline transport: rewind, play the build, roll forward to the part. */
function initTransport() {
  el('tl-first').addEventListener('click', () => {
    stopPlayback();
    if (builtSteps.length) rollTo(builtSteps[0]);
  });
  el('tl-play').addEventListener('click', playBuild);
  el('tl-last').addEventListener('click', () => { stopPlayback(); rollTo(null); });
}

/* ---------------------------------------------------------------- measure */

let measuring = false;

/** Turn the measure tool on or off, clearing whatever was measured. */
function toggleMeasure() {
  measuring = !measuring;
  if (!viewer) return;
  viewer.measure = [];
  el('cmd-measure').setAttribute('aria-pressed', String(measuring));
  el('canvas').classList.toggle('measuring', measuring);
  el('dim-readout').hidden = true;
  el('status-measure').textContent = measuring ? 'MEASURE: PICK A POINT' : '';
  viewer.render();
}

/** Take a point off the part, and report the span once there are two.
 *
 * Two points then reset, rather than a growing chain: a chain reads as a
 * polyline and the question this answers is always between two features.
 */
function pickMeasurePoint(event) {
  if (!measuring || !viewer) return;
  if (viewer.measure.length >= 2) viewer.measure = [];
  const hit = viewer.pick(event.clientX, event.clientY);
  if (!hit) {
    el('status-measure').textContent = 'MEASURE: NO PART UNDER THE CURSOR';
    return;
  }
  viewer.measure.push(hit.point);
  viewer.render();
  placeDimension();
}

/** Put the readout at the midpoint of the span, and say what it spans. */
function placeDimension() {
  const readout = el('dim-readout');
  const points = viewer.measure;
  if (points.length < 2) {
    readout.hidden = true;
    el('status-measure').textContent = 'MEASURE: PICK THE SECOND POINT';
    return;
  }
  const [a, b] = points;
  const span = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const middle = viewer.project([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]);
  // Axis components alongside the span: on a machined part the useful number
  // is often one axis of it, not the diagonal.
  el('status-measure').textContent =
    'MEASURE  dX ' + Math.abs(b[0] - a[0]).toFixed(2)
    + '  dY ' + Math.abs(b[1] - a[1]).toFixed(2)
    + '  dZ ' + Math.abs(b[2] - a[2]).toFixed(2) + ' mm';
  if (!middle) { readout.hidden = true; return; }
  readout.hidden = false;
  readout.textContent = span.toFixed(2) + ' mm';
  readout.style.left = middle.x + 'px';
  readout.style.top = middle.y + 'px';
}

/** Which workspace tab is selected. */
function activeWorkspace() {
  const chosen = document.querySelector('.workspaces button[aria-selected="true"]');
  return chosen ? chosen.dataset.view : 'model';
}

/** Keyboard commands, as a CAD tool has. */
function initKeys() {
  const panel = el('keymap');
  const close = () => { panel.hidden = true; };
  el('keymap-close').addEventListener('click', close);
  panel.addEventListener('click', (event) => { if (event.target === panel) close(); });

  document.addEventListener('keydown', (event) => {
    // Never steal a key from a field the user is typing in.
    const tag = (event.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      if (event.key === 'Escape') event.target.blur();
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const press = (id) => { const node = el(id); if (node) node.click(); };
    switch (event.key) {
      // The arrows step whatever the active workspace lists: parts in Model,
      // tasks in Rollouts. One key, one meaning per context, as a CAD app does.
      case 'ArrowLeft': press(activeWorkspace() === 'rollouts' ? 'cmd-rollout-prev' : 'cmd-prev'); break;
      case 'ArrowRight': press(activeWorkspace() === 'rollouts' ? 'cmd-rollout-next' : 'cmd-next'); break;
      case '1': document.querySelector('.cmd[data-view3d="iso"]').click(); break;
      case '2': document.querySelector('.cmd[data-view3d="front"]').click(); break;
      case '3': document.querySelector('.cmd[data-view3d="top"]').click(); break;
      case '4': document.querySelector('.cmd[data-view3d="right"]').click(); break;
      case 'f': case 'F': press('cmd-fit'); break;
      case 'g': case 'G': press('cmd-grid'); break;
      case 'w': case 'W': press('cmd-shade'); break;
      case 's': case 'S': press('cmd-section'); break;
      case 'x': case 'X': press('cmd-axis'); break;
      case 'e': case 'E': press('cmd-export'); break;
      case 'm': case 'M': press('cmd-measure'); break;
      case ' ': event.preventDefault(); playBuild(); break;
      case ',': stepBy(-1); break;
      case '.': stepBy(1); break;
      case '/': event.preventDefault(); el('tree-filter').focus(); break;
      case '?': panel.hidden = !panel.hidden; break;
      case 'Escape': close(); break;
      default: return;
    }
  });
}

function init() {
  const suite = (DATA.benchmark || {}).suite_version || DATA.generated_at || '';
  el('suite').textContent = suite ? 'rev ' + suite.replace('kairos-cad-', '') : 'rev -';
  el('doctab-name').textContent = suite || 'kairos-cad';
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
  initFilter();
  initTransport();
  initKeys();
  renderBenchmark();
  renderSuccessCurve();
  renderComparisons();
  renderTraining();
  renderAblations();
  renderRollouts();
  el('cmd-rollout-prev').addEventListener('click', () => { rolloutTask -= 1; renderRollouts(); });
  el('cmd-rollout-next').addEventListener('click', () => { rolloutTask += 1; renderRollouts(); });
  if ((DATA.designs || []).length) select(0);
}

document.addEventListener('DOMContentLoaded', init);
