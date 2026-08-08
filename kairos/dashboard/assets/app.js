/* KAIROS dashboard application.
 *
 * Reads the `KAIROS_DATA` object inlined by build.py and renders four views.
 * Every number shown here comes from a committed artifact; nothing is computed
 * from a live model, and nothing is rounded up.
 */

'use strict';

const DATA = window.KAIROS_DATA;

/* ---------- helpers ---------- */

const el = (id) => document.getElementById(id);

function fmt(value, digits) {
  if (value == null || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits == null ? 3 : digits);
}

function esc(value) {
  return String(value == null ? '' : value).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

/* ---------- design explorer ---------- */

let viewer = null;
let currentDesign = null;

function renderDesignList() {
  const list = el('design-list');
  const designs = DATA.designs || [];
  if (!designs.length) {
    list.innerHTML = '<p class="empty" style="padding:14px">no designs in this bundle</p>';
    return;
  }
  list.innerHTML = designs.map((d, i) => `
    <button data-index="${i}" aria-current="${i === 0}">
      <div class="family">${esc(d.family)}</div>
      <div>${esc(d.design_id)}</div>
    </button>`).join('');
  list.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', () => selectDesign(Number(button.dataset.index)));
  });
}

function selectDesign(index) {
  const design = DATA.designs[index];
  if (!design) return;
  currentDesign = design;
  el('design-list').querySelectorAll('button').forEach((b, i) => {
    b.setAttribute('aria-current', String(i === index));
  });

  if (viewer && design.mesh) {
    try {
      viewer.load(design.mesh);
      el('viewer-error').hidden = true;
    } catch (err) {
      el('viewer-error').hidden = false;
      el('viewer-error').textContent = 'viewer: ' + err.message;
    }
  }

  const meshInfo = design.mesh
    ? `${design.mesh.triangle_count} triangles - ${design.mesh.vertex_count} vertices`
    : 'no mesh bundled';
  el('stage-info').textContent = `${design.design_id} - ${meshInfo}`;

  el('detail-requirement').textContent = design.requirement || '(no requirement text)';

  const extent = design.extent_mm || [];
  const dims = extent[0] != null
    ? `${fmt(extent[0], 1)} x ${fmt(extent[1], 1)} x ${fmt(extent[2], 1)}`
    : '-';
  const wall = design.min_wall_thickness_mm;
  el('detail-kv').innerHTML = `
    <dt>family</dt><dd>${esc(design.family)}</dd>
    <dt>material</dt><dd>${esc(design.material || '-')}</dd>
    <dt>mass</dt><dd>${fmt(design.mass_g, 2)} g</dd>
    <dt>volume</dt><dd>${fmt(design.volume_mm3, 0)} mm³</dd>
    <dt>surface area</dt><dd>${fmt(design.surface_area_mm2, 0)} mm²</dd>
    <dt>extent (mm)</dt><dd>${dims}</dd>
    <dt>faces</dt><dd>${design.faces == null ? '-' : design.faces}</dd>
    <dt>holes</dt><dd>${design.hole_count == null ? '-' : design.hole_count}</dd>
    <dt>min wall</dt><dd>${wall == null ? 'not measured' : fmt(wall, 2) + ' mm'}</dd>
    <dt>expert steps</dt><dd>${design.steps == null ? '-' : design.steps}</dd>
    <dt>constraints met</dt><dd>${design.satisfaction_rate == null ? '-' : fmt(design.satisfaction_rate * 100, 0) + '%'}</dd>`;

  const constraints = design.constraints || [];
  el('detail-constraints').innerHTML = constraints.length
    ? constraints.map((c) => `
        <div class="constraint">
          <span class="dot ${esc(c.status)}"></span>
          <span class="kind">${esc(c.kind)}</span>
          <span class="detail">${esc(c.detail)}</span>
        </div>`).join('')
    : '<p class="empty">no constraints recorded</p>';

  const ops = design.operations || [];
  el('detail-ops').innerHTML = ops.length
    ? ops.map((op) => `<span>${esc(op)}</span>`).join('')
    : '<span class="empty">no trajectory</span>';
}

function initViewer() {
  const canvas = el('viewer-canvas');
  try {
    viewer = new Viewer(canvas);
  } catch (err) {
    el('viewer-error').hidden = false;
    el('viewer-error').textContent = 'viewer unavailable: ' + err.message;
    return;
  }
  const picker = el('color-picker');
  if (picker) picker.addEventListener('input', () => viewer.setColor(picker.value));
}

/* ---------- benchmark ---------- */

function renderLeaderboard() {
  const board = (DATA.benchmark && DATA.benchmark.leaderboard) || {};
  const rows = board.policies || board.rows || [];
  if (!rows.length) {
    el('leaderboard').innerHTML = '<p class="empty">no benchmark run in this bundle</p>';
    return;
  }
  const best = Math.max(...rows.map((r) => r.progress_mean || 0));
  const sorted = rows.slice().sort((a, b) => (b.progress_mean || 0) - (a.progress_mean || 0));
  el('leaderboard').innerHTML = `
    <table>
      <thead><tr>
        <th>policy</th><th>progress</th><th></th>
        <th>success</th><th>validity</th><th>tasks</th>
      </tr></thead>
      <tbody>${sorted.map((r) => `
        <tr class="${/oracle|expert/.test(r.policy || '') ? 'highlight' : ''}">
          <td class="name">${esc(r.policy)}</td>
          <td class="num">${fmt(r.progress_mean)}</td>
          <td style="width:110px">${barRow(r.progress_mean || 0, best)}</td>
          <td class="num">${fmt(r.success_rate)}</td>
          <td class="num">${fmt(r.validity)}</td>
          <td class="num">${r.tasks == null ? '-' : r.tasks}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

function renderSuccessCurve() {
  const curves = (DATA.benchmark && DATA.benchmark.success_curve) || {};
  const names = Object.keys(curves);
  if (!names.length) {
    el('success-curve').innerHTML = '<p class="empty">no traces in this bundle</p>';
    return;
  }
  // COMPLETE(k) buckets are keyed by k; BUILD is the k = full-length case and
  // is charted separately because its x is not comparable.
  const ks = [...new Set(names.flatMap((n) => Object.keys(curves[n])))]
    .filter((k) => k !== 'build')
    .map(Number)
    .sort((a, b) => a - b);

  const series = names.map((name) => ({
    name,
    points: ks.filter((k) => curves[name][String(k)] != null)
      .map((k) => [k, curves[name][String(k)]]),
  })).filter((s) => s.points.length);

  // k is the number of trailing actions the POLICY must supply -- the expert
  // prefix is everything before it. Larger k is a harder task, so these curves
  // decay left to right; labelling k as "steps replayed" inverts the reading.
  el('success-curve').innerHTML =
    lineChart(series, {
      xLabel: 'k = trailing actions the policy must supply (larger is harder)',
      yLabel: 'success rate',
      yMax: 1.0,
      // Tick only where k was actually measured. Generic round-number ticks
      // put marks at 3, 5, 6 and 7, implying samples that do not exist.
      xTickLabels: ks.map((k) => [k, String(k)]),
    }) + legend(series.map((s) => s.name));
}

/* ---------- training ---------- */

function renderTraining() {
  const training = DATA.training || {};
  const bc = training.bc || {};
  const ppo = training.ppo || {};

  const bcHistory = bc.history || [];
  if (bcHistory.length) {
    const accuracy = [
      { name: 'train', points: bcHistory.map((r) => [r.epoch, r.train_accuracy]) },
      { name: 'held out', points: bcHistory.map((r) => [r.epoch, r.held_out_accuracy]) },
    ].filter((s) => s.points.every((p) => p[1] != null));
    el('bc-chart').innerHTML =
      lineChart(accuracy, { xLabel: 'epoch', yLabel: 'next-action accuracy', yMax: 1.0 }) +
      legend(accuracy.map((s) => s.name));

    const loss = [
      { name: 'train loss', points: bcHistory.map((r) => [r.epoch, r.train_loss]) },
      { name: 'held-out loss', points: bcHistory.map((r) => [r.epoch, r.val_loss]) },
    ].filter((s) => s.points.every((p) => p[1] != null));
    el('bc-loss-chart').innerHTML =
      lineChart(loss, { xLabel: 'epoch', yLabel: 'loss' }) + legend(loss.map((s) => s.name));

    const best = bc.best_held_out_accuracy;
    el('bc-summary').textContent = best == null ? '' :
      `best held-out next-action accuracy ${fmt(best)} over ${bcHistory.length} epochs` +
      (bc.parameters ? ` - ${(bc.parameters / 1e6).toFixed(2)}M parameters` : '');
  } else {
    el('bc-chart').innerHTML = '<p class="empty">no BC history in this bundle</p>';
    el('bc-loss-chart').innerHTML = '';
  }

  const ppoHistory = (ppo.history || []).filter((r) => r.iteration != null);
  if (ppoHistory.length) {
    const reward = [{ name: 'mean episode reward', points: ppoHistory.map((r) => [r.iteration, r.reward_mean]) }];
    el('ppo-reward-chart').innerHTML =
      lineChart(reward, { xLabel: 'iteration', yLabel: 'reward' }) + legend(['mean episode reward']);

    const rates = [
      { name: 'success rate', points: ppoHistory.map((r) => [r.iteration, r.success_rate]) },
      { name: 'invalid action rate', points: ppoHistory.map((r) => [r.iteration, r.invalid_action_rate]) },
    ].filter((s) => s.points.every((p) => p[1] != null));
    el('ppo-rate-chart').innerHTML =
      lineChart(rates, { xLabel: 'iteration', yLabel: 'rate', yMax: 1.0 }) +
      legend(rates.map((s) => s.name));
  } else {
    el('ppo-reward-chart').innerHTML = '<p class="empty">no PPO history in this bundle</p>';
    el('ppo-rate-chart').innerHTML = '';
  }
}

/* ---------- ablations ---------- */

function renderAblations() {
  const rows = (DATA.ablations && DATA.ablations.rows) || [];
  if (!rows.length) {
    el('ablations').innerHTML = '<p class="empty">no ablations in this bundle</p>';
    return;
  }
  el('ablations').innerHTML = `
    <table>
      <thead><tr>
        <th>condition</th><th>progress</th><th>&Delta; vs baseline</th><th>validity</th>
      </tr></thead>
      <tbody>${rows.map((r) => `
        <tr class="${r.baseline ? 'highlight' : ''}">
          <td class="name">${esc(r.name)}</td>
          <td class="num">${fmt(r.progress_mean)}</td>
          <td class="num" style="color:${(r.delta || 0) < 0 ? 'var(--fail)' : 'var(--muted)'}">
            ${r.delta == null ? '-' : (r.delta > 0 ? '+' : '') + fmt(r.delta * 100, 1) + '%'}
          </td>
          <td class="num">${fmt(r.validity)}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

/* ---------- comparisons ---------- */

function renderComparisons() {
  const rows = (DATA.comparisons && DATA.comparisons.rows) || [];
  if (!rows.length) {
    el('comparisons').innerHTML = '<p class="empty">no paired comparisons in this bundle</p>';
    return;
  }
  el('comparisons').innerHTML = `
    <table>
      <thead><tr>
        <th>pair</th><th>difference</th><th>95% CI</th><th>W / L / T</th><th>separates?</th>
      </tr></thead>
      <tbody>${rows.map((r) => {
        const separates = r.low != null && r.high != null && (r.low > 0 || r.high < 0);
        return `<tr>
          <td class="name">${esc(r.a)} - ${esc(r.b)}</td>
          <td class="num">${(r.difference > 0 ? '+' : '') + fmt(r.difference)}</td>
          <td class="num">[${fmt(r.low)}, ${fmt(r.high)}]</td>
          <td class="num">${r.wins}/${r.losses}/${r.ties}</td>
          <td class="num" style="color:${separates ? 'var(--pass)' : 'var(--muted)'}">
            ${separates ? 'yes' : 'no'}</td>
        </tr>`;
      }).join('')}</tbody>
    </table>`;
}

/* ---------- tabs ---------- */

function initTabs() {
  const buttons = document.querySelectorAll('nav button');
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      buttons.forEach((b) => b.setAttribute('aria-selected', String(b === button)));
      document.querySelectorAll('main section').forEach((section) => {
        section.hidden = section.id !== 'tab-' + button.dataset.tab;
      });
      // The canvas has no size while its tab is hidden, so the first render
      // into a zero-width viewport draws nothing; redraw on reveal.
      if (button.dataset.tab === 'designs' && viewer) viewer.render();
    });
  });
}

function init() {
  el('stamp').textContent = DATA.generated_at || '';
  renderDesignList();
  initViewer();
  initTabs();
  renderLeaderboard();
  renderSuccessCurve();
  renderTraining();
  renderAblations();
  renderComparisons();
  if ((DATA.designs || []).length) selectDesign(0);
}

document.addEventListener('DOMContentLoaded', init);
