/* Tiny SVG chart helpers.
 *
 * Same reasoning as the viewer: no charting library, because the dashboard has
 * to be one offline file. These draw into an SVG string rather than a canvas so
 * the curves stay crisp at any zoom and can be lifted straight into the paper.
 */

'use strict';

const SERIES_COLORS = [
  '#6b95dc', '#4bb87d', '#d2a24c', '#d9645f', '#9b7fd4', '#4bb3b8', '#c98bb5', '#8b94a8',
];

function escapeText(value) {
  return String(value).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

/** Nice round tick values covering [0, max]. */
/** Round tick values spanning [min, max].
 *
 * It used to walk up from zero, which is right only when the data is positive.
 * PPO reward is never positive, so every tick landed in the top few pixels of
 * the plot as an unreadable stack of labels, and the range the curve actually
 * occupied had no gridline at all.
 */
function ticks(min, max, count) {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || magnitude * 10;
  const out = [];
  const first = Math.ceil(min / step - 1e-9) * step;
  for (let value = first; value <= max + step * 1e-9; value += step) {
    out.push(Number(value.toFixed(10)));
  }
  return out.length ? out : [min, max];
}

/**
 * Line chart.
 * @param {Array<{name: string, points: Array<[number, number]>}>} series
 * @param {object} options - {xLabel, yLabel, yMax, width, height, xTickLabels}
 */
function lineChart(series, options) {
  const opts = Object.assign(
    { width: 720, height: 260, xLabel: '', yLabel: '', yMax: null, xTickLabels: null },
    options || {}
  );
  const pad = { top: 14, right: 18, bottom: 38, left: 52 };
  const plotWidth = opts.width - pad.left - pad.right;
  const plotHeight = opts.height - pad.top - pad.bottom;

  const live = series.filter((s) => s.points && s.points.length);
  if (!live.length) return '<p class="empty">no data</p>';

  const allX = live.flatMap((s) => s.points.map((p) => p[0]));
  const allY = live.flatMap((s) => s.points.map((p) => p[1]));
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  // Zero stays on the chart in both directions: it is the reference the reader
  // brings, and a reward curve that never reaches it should show how far off
  // it is rather than filling the frame.
  const dataMax = Math.max(...allY, 0);
  const yMax = opts.yMax != null ? opts.yMax : (dataMax > 0 ? dataMax * 1.08 : 0);
  const yMin = Math.min(0, ...allY);

  const sx = (x) => pad.left + (xMax === xMin ? plotWidth / 2 : ((x - xMin) / (xMax - xMin)) * plotWidth);
  const sy = (y) => pad.top + plotHeight - ((y - yMin) / (yMax - yMin || 1)) * plotHeight;

  let svg = `<svg viewBox="0 0 ${opts.width} ${opts.height}" role="img">`;

  for (const tick of ticks(yMin, yMax, 5)) {
    const y = sy(tick);
    svg += `<line class="grid" x1="${pad.left}" y1="${y}" x2="${pad.left + plotWidth}" y2="${y}"/>`;
    svg += `<text x="${pad.left - 8}" y="${y + 3}" text-anchor="end">${formatTick(tick)}</text>`;
  }

  svg += `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotHeight}"/>`;
  svg += `<line class="axis" x1="${pad.left}" y1="${sy(0)}" x2="${pad.left + plotWidth}" y2="${sy(0)}"/>`;

  const xTicks = opts.xTickLabels
    || ticks(xMin, xMax, Math.min(8, Math.max(2, allX.length)))
      .filter((t) => t >= xMin && t <= xMax)
      .map((t) => [t, formatTick(t)]);
  for (const [value, label] of xTicks) {
    svg += `<text x="${sx(value)}" y="${pad.top + plotHeight + 16}" text-anchor="middle">${escapeText(label)}</text>`;
  }

  live.forEach((s, index) => {
    const color = s.color || SERIES_COLORS[index % SERIES_COLORS.length];
    const path = s.points
      .slice()
      .sort((a, b) => a[0] - b[0])
      .map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`)
      .join(' ');
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
    for (const p of s.points) {
      svg += `<circle cx="${sx(p[0]).toFixed(2)}" cy="${sy(p[1]).toFixed(2)}" r="2.6" fill="${color}"/>`;
    }
  });

  if (opts.xLabel) {
    svg += `<text x="${pad.left + plotWidth / 2}" y="${opts.height - 4}" text-anchor="middle" class="series-label">${escapeText(opts.xLabel)}</text>`;
  }
  if (opts.yLabel) {
    const cy = pad.top + plotHeight / 2;
    svg += `<text transform="rotate(-90 12 ${cy})" x="12" y="${cy}" text-anchor="middle" class="series-label">${escapeText(opts.yLabel)}</text>`;
  }
  svg += '</svg>';
  return svg;
}

function formatTick(value) {
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) < 1) return value.toFixed(2);
  return value.toFixed(1);
}

/** Horizontal bar chart, used for the leaderboard's score column. */
function barRow(value, max, color) {
  const width = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return `<span class="bar" style="width:${width.toFixed(1)}px;background:${color || SERIES_COLORS[0]}"></span>`;
}

function legend(names) {
  return '<div class="legend">' + names.map((name, i) =>
    `<span><i style="background:${SERIES_COLORS[i % SERIES_COLORS.length]}"></i>${escapeText(name)}</span>`
  ).join('') + '</div>';
}
