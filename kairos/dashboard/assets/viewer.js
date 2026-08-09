/* Minimal WebGL solid viewer for KAIROS parts.
 *
 * The plan called for Three.js. This is a hand-written renderer instead, and
 * the reason is the one constraint that matters for this dashboard: it must be
 * ONE file that opens from disk with no network. Three.js is ~600 KB minified,
 * which would be six times the size of every part, metric and training curve in
 * the bundle combined. What the viewer actually needs -- orbit a static mesh,
 * shade it, fit it in frame -- is the ~250 lines below.
 *
 * Meshes arrive quantized (integer positions, scaled by `quantum`) and welded,
 * so vertex normals can be accumulated across shared faces for smooth shading.
 */

'use strict';

const VERTEX_SHADER = `
attribute vec3 aPosition;
attribute vec3 aNormal;
uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vModelPosition;
void main() {
  vModelPosition = aPosition;
  vec4 viewPosition = uModelView * vec4(aPosition, 1.0);
  vViewPosition = viewPosition.xyz;
  vNormal = normalize(uNormalMatrix * aNormal);
  gl_Position = uProjection * viewPosition;
}`;

/* Two-light rig in view space: a key light over the camera's shoulder and a
 * dim fill from below, so faces turned away from the key are still readable
 * rather than solid black. Plus a rim term to separate the part from the
 * background at grazing angles.
 *
 * Shading is FLAT, from screen-space derivatives of the view position, not from
 * the interpolated vertex normal. Welding is what makes the mesh compact, but
 * averaging normals across a welded 90-degree edge is exactly wrong for a
 * machined part: every corner shades as if it were filleted, and the
 * tessellation's diagonals show up as creases across faces that are dead flat.
 * A per-fragment face normal renders the part the way the geometry actually is.
 *
 * `vNormal` stays as the fallback for contexts without OES_standard_derivatives.
 */
const FRAGMENT_SHADER = `
#ifdef GL_OES_standard_derivatives
#extension GL_OES_standard_derivatives : enable
#endif
precision mediump float;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vModelPosition;
uniform vec3 uColor;
uniform bool uFlat;
// Section plane in model space: xyz is the axis mask, w the cut position.
// Fragments on the far side are discarded, which is what opens a solid up.
uniform vec4 uSection;
uniform bool uSectioning;
// The stencil pass that caps a section counts the material the cut removed,
// which is the same plane read the other way round.
uniform bool uSectionInvert;
void main() {
  if (uSectioning) {
    float side = dot(vModelPosition, uSection.xyz) - uSection.w;
    if (uSectionInvert ? (side <= 0.0) : (side > 0.0)) discard;
  }
  vec3 viewDir = normalize(-vViewPosition);
  vec3 normal = normalize(vNormal);
  if (!gl_FrontFacing) normal = -normal;
#ifdef GL_OES_standard_derivatives
  if (uFlat) {
    normal = normalize(cross(dFdx(vViewPosition), dFdy(vViewPosition)));
    // The derivative cross product's sign follows screen orientation, not
    // winding, so gl_FrontFacing cannot correct it. Every visible fragment of
    // a closed solid faces the camera, so orient it against the view ray.
    if (dot(normal, viewDir) < 0.0) normal = -normal;
  }
#endif
  vec3 keyDir = normalize(vec3(0.4, 0.7, 1.0));
  vec3 fillDir = normalize(vec3(-0.5, -0.6, 0.4));

  float key = max(dot(normal, keyDir), 0.0);
  float fill = max(dot(normal, fillDir), 0.0) * 0.28;
  float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.5) * 0.30;
  vec3 halfway = normalize(keyDir + viewDir);
  float spec = pow(max(dot(normal, halfway), 0.0), 48.0) * 0.35;

  vec3 color = uColor * (0.22 + 0.85 * key + fill) + vec3(rim * 0.55) + vec3(spec);
  gl_FragColor = vec4(color, 1.0);
}`;

/* The ground shadow: geometry flattened onto the floor plane, no shading. It
 * is the cheapest thing that puts a part in contact with the ground instead of
 * hovering over it, and a viewport that omits it reads as a model viewer. */
const SHADOW_VERTEX_SHADER = `
attribute vec3 aPosition;
uniform mat4 uModelView;
uniform mat4 uProjection;
void main() { gl_Position = uProjection * uModelView * vec4(aPosition, 1.0); }`;

const SHADOW_FRAGMENT_SHADER = `
precision mediump float;
uniform vec4 uShade;
void main() { gl_FragColor = uShade; }`;

const LINE_VERTEX_SHADER = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uModelView;
uniform mat4 uProjection;
varying vec3 vColor;
void main() {
  vColor = aColor;
  gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}`;

const LINE_FRAGMENT_SHADER = `
precision mediump float;
varying vec3 vColor;
void main() { gl_FragColor = vec4(vColor, 1.0); }`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error('shader compile failed: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

function linkProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource || VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource || FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('program link failed: ' + gl.getProgramInfoLog(program));
  }
  return program;
}

/* ---- matrices (column-major, WebGL order) ---- */

function perspective(fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * range, -1,
    0, 0, 2 * near * far * range, 0,
  ]);
}

function normalize3(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function lookAt(eye, target, up) {
  const forward = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  let right = cross3(forward, up);
  // Looking straight down the up axis makes the cross product vanish; pick any
  // perpendicular so the camera keeps working at the poles instead of blanking.
  if (Math.hypot(right[0], right[1], right[2]) < 1e-6) right = [1, 0, 0];
  right = normalize3(right);
  const trueUp = cross3(right, forward);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return new Float32Array([
    right[0], trueUp[0], -forward[0], 0,
    right[1], trueUp[1], -forward[1], 0,
    right[2], trueUp[2], -forward[2], 0,
    -dot(right, eye), -dot(trueUp, eye), dot(forward, eye), 1,
  ]);
}

/* The model transform is translation + uniform scale only, so the normal
 * matrix is just the rotation part of the model-view -- no inverse-transpose
 * needed, and no chance of the shading flipping when a part is very thin. */
function upperLeft3x3(m) {
  return new Float32Array([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
}

/** Area-weighted vertex normals. Welded meshes share vertices across faces,
 * which is what makes the averaging produce smooth curved surfaces at all. */
function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    // Un-normalized cross product: its magnitude is twice the triangle area,
    // so large faces dominate the average, which is what we want.
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const offset of [a, b, c]) {
      normals[offset] += nx;
      normals[offset + 1] += ny;
      normals[offset + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    if (length > 0) {
      normals[i] /= length;
      normals[i + 1] /= length;
      normals[i + 2] /= length;
    }
  }
  return normals;
}

/** Decode the quantized bundle mesh into float positions in mm. */
function decodeMesh(mesh) {
  const quantum = mesh.quantum;
  const positions = new Float32Array(mesh.positions.length);
  for (let i = 0; i < positions.length; i++) positions[i] = mesh.positions[i] * quantum;
  const indices = mesh.indices;
  const useUint32 = mesh.vertex_count > 65535;
  return {
    positions,
    normals: computeNormals(positions, indices),
    indices: useUint32 ? new Uint32Array(indices) : new Uint16Array(indices),
    useUint32,
    count: indices.length,
  };
}

/** Ground grid and origin axes, in the same normalized space as the part.
 *
 * The part is scaled into a roughly 2-unit box before drawing, so a grid built
 * here reads the same whether the part is a 6 mm spacer or a 200 mm rail. That
 * is what makes one camera distance work for every design.
 */
function buildGrid(divisions, half, faint, axisX, axisY) {
  const positions = [];
  const colors = [];
  const step = (half * 2) / divisions;

  const push = (x1, y1, z1, x2, y2, z2, c) => {
    positions.push(x1, y1, z1, x2, y2, z2);
    colors.push(c[0], c[1], c[2], c[0], c[1], c[2]);
  };

  for (let i = 0; i <= divisions; i++) {
    const t = -half + i * step;
    // The two lines through the origin are the axes and get their own colour,
    // the way a CAD viewport marks X and Y.
    const onAxis = Math.abs(t) < step * 0.5;
    push(-half, t, 0, half, t, 0, onAxis ? axisX : faint);
    push(t, -half, 0, t, half, 0, onAxis ? axisY : faint);
  }
  return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
}

class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    // Transparent so the canvas composites over the CSS gradient ground. A
    // flat WebGL clear paints over it, and the graded ground is most of what
    // makes a viewport read as a room rather than a void.
    // Stencil carries the section cap: without it the cut face is missing and
    // a solid part reads as a hollow shell.
    const options = { antialias: true, alpha: true, depth: true, stencil: true,
                      premultipliedAlpha: false };
    this.gl = canvas.getContext('webgl', options) || canvas.getContext('experimental-webgl', options);
    if (!this.gl) throw new Error('WebGL is unavailable in this browser');
    const gl = this.gl;
    // Meshes exceeding 65535 vertices need 32-bit indices; without the
    // extension they would silently wrap and render as confetti.
    this.uint32Indices = !!gl.getExtension('OES_element_index_uint');
    // Must be requested before the shader that #extension-enables it is
    // compiled, or the directive fails to link.
    this.derivatives = !!gl.getExtension('OES_standard_derivatives');
    this.program = linkProgram(gl);
    this.lineProgram = linkProgram(gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
    this.shadowProgram = linkProgram(gl, SHADOW_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER);
    this.shadowUniforms = {
      modelView: gl.getUniformLocation(this.shadowProgram, 'uModelView'),
      projection: gl.getUniformLocation(this.shadowProgram, 'uProjection'),
      shade: gl.getUniformLocation(this.shadowProgram, 'uShade'),
    };
    this.shadowPosition = gl.getAttribLocation(this.shadowProgram, 'aPosition');
    this.attributes = {
      position: gl.getAttribLocation(this.program, 'aPosition'),
      normal: gl.getAttribLocation(this.program, 'aNormal'),
    };
    this.uniforms = {
      modelView: gl.getUniformLocation(this.program, 'uModelView'),
      projection: gl.getUniformLocation(this.program, 'uProjection'),
      normalMatrix: gl.getUniformLocation(this.program, 'uNormalMatrix'),
      color: gl.getUniformLocation(this.program, 'uColor'),
      flat: gl.getUniformLocation(this.program, 'uFlat'),
      section: gl.getUniformLocation(this.program, 'uSection'),
      sectioning: gl.getUniformLocation(this.program, 'uSectioning'),
      sectionInvert: gl.getUniformLocation(this.program, 'uSectionInvert'),
    };
    //: Section state: axis index (0..2), cut fraction through the bounds, on.
    this.section = { axis: 0, cut: 0.5, on: false };
    this.flatShading = this.derivatives;
    // Some contexts refuse a stencil buffer; the cap is skipped rather than
    // drawn wrong, so ask what was actually granted instead of what was asked.
    const granted = gl.getContextAttributes();
    this.hasStencil = !!(granted && granted.stencil);
    this.buffers = {
      position: gl.createBuffer(),
      normal: gl.createBuffer(),
      index: gl.createBuffer(),
      capPosition: gl.createBuffer(),
      capNormal: gl.createBuffer(),
    };
    this.mesh = null;
    this.color = [0.42, 0.58, 0.86];
    this.capColor = this._cutTone(this.color);
    //: Ground shadow, as premultiplied-free RGBA. Read from --shadow so it can
    //: darken a light ground and lighten nothing on a dark one.
    this.shadowTone = new Float32Array([0.02, 0.03, 0.05, 0.55]);
    this.showShadow = true;
    this.background = [0.043, 0.055, 0.075];
    this.showGrid = true;
    this.wireframe = false;
    //: Grid tones, overwritten from CSS by setPalette so the lines sit at the
    //: same contrast in both themes. Hardcoded dark lines read as heavy black
    //: on a light ground.
    this.gridTone = [0.16, 0.19, 0.24];
    this.axisTone = [[0.30, 0.42, 0.44], [0.26, 0.34, 0.40]];
    //: Pointer mode driven by the navigation bar. Orbit and pan share the
    //: drag gesture, so they are a mode rather than separate tools.
    this.mode = 'orbit';
    this.pan = [0, 0];
    //: Called after any camera change, so a ViewCube can follow it.
    this.onCamera = null;
    this._grid = buildGrid(20, 2.0, [0.16, 0.19, 0.24], [0.30, 0.42, 0.44], [0.26, 0.34, 0.40]);
    this.lineAttributes = {
      position: gl.getAttribLocation(this.lineProgram, "aPosition"),
      color: gl.getAttribLocation(this.lineProgram, "aColor"),
    };
    this.lineUniforms = {
      modelView: gl.getUniformLocation(this.lineProgram, "uModelView"),
      projection: gl.getUniformLocation(this.lineProgram, "uProjection"),
    };
    this.gridBuffers = { position: gl.createBuffer(), color: gl.createBuffer() };
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, this._grid.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.color);
    gl.bufferData(gl.ARRAY_BUFFER, this._grid.colors, gl.STATIC_DRAW);
    this.camera = { yaw: -0.9, pitch: 0.5, distance: 3.0 };
    this.center = [0, 0, 0];
    this.scale = 1;
    gl.enable(gl.DEPTH_TEST);
    // Parts are closed solids, but a pocket that breaks through can leave
    // inward-facing triangles visible; the shader flips those normals rather
    // than culling, so keep both sides drawn.
    gl.disable(gl.CULL_FACE);
    this._bindControls();
  }

  /** Standard CAD view presets, as yaw/pitch on the orbit camera. */
  setView(name) {
    const views = {
      iso: [-0.9, 0.5],
      front: [-Math.PI / 2, 0.0],
      back: [Math.PI / 2, 0.0],
      right: [0.0, 0.0],
      left: [Math.PI, 0.0],
      top: [-Math.PI / 2, Math.PI / 2 - 0.02],
      bottom: [-Math.PI / 2, -Math.PI / 2 + 0.02],
    };
    const view = views[name] || views.iso;
    this.camera.yaw = view[0];
    this.camera.pitch = view[1];
    this.camera.distance = 3.0;
    this.pan = [0, 0];
    this.render();
    if (this.onCamera) this.onCamera();
  }

  toggleGrid() {
    this.showGrid = !this.showGrid;
    this.render();
  }

  /** The loaded mesh as a binary STL blob, in millimetres.
   *
   * The viewer already holds welded float positions, so this is a re-emit
   * rather than a conversion: no geometry is recomputed and nothing is lost
   * beyond the 0.01 mm quantum the bundle was built at.
   */
  toStl(name) {
    const { positions, indices, count } = this.mesh;
    const triangles = count / 3;
    const buffer = new ArrayBuffer(84 + triangles * 50);
    const view = new DataView(buffer);
    const header = `KAIROS ${name || 'part'}`.slice(0, 79);
    for (let i = 0; i < header.length; i++) view.setUint8(i, header.charCodeAt(i));
    view.setUint32(80, triangles, true);

    let offset = 84;
    for (let i = 0; i < count; i += 3) {
      const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
      const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1];
      const uz = positions[b + 2] - positions[a + 2];
      const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1];
      const vz = positions[c + 2] - positions[a + 2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1;
      view.setFloat32(offset, nx / len, true);
      view.setFloat32(offset + 4, ny / len, true);
      view.setFloat32(offset + 8, nz / len, true);
      offset += 12;
      for (const corner of [a, b, c]) {
        view.setFloat32(offset, positions[corner], true);
        view.setFloat32(offset + 4, positions[corner + 1], true);
        view.setFloat32(offset + 8, positions[corner + 2], true);
        offset += 12;
      }
      view.setUint16(offset, 0, true);
      offset += 2;
    }
    return new Blob([buffer], { type: 'model/stl' });
  }

  /** Millimetres spanned by the longest edge of the part's bounding box. */
  extentMm() {
    return this.mesh ? 2.0 / this.scale : 0;
  }

  /** Take the viewport ground and grid tones from CSS custom properties. */
  setPalette(styles) {
    const parse = (name, fallback) => {
      const raw = styles.getPropertyValue(name).trim();
      if (!/^#[0-9a-f]{6}$/i.test(raw)) return fallback;
      return [
        parseInt(raw.slice(1, 3), 16) / 255,
        parseInt(raw.slice(3, 5), 16) / 255,
        parseInt(raw.slice(5, 7), 16) / 255,
      ];
    };
    this.gridTone = parse('--grid-line', this.gridTone);
    const shade = parse('--viewport-shadow', null);
    if (shade) this.shadowTone.set(shade, 0);
    this.axisTone = [
      parse('--grid-axis-x', this.axisTone[0]),
      parse('--grid-axis-y', this.axisTone[1]),
    ];
    this._grid = buildGrid(20, 2.0, this.gridTone, this.axisTone[0], this.axisTone[1]);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, this._grid.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.color);
    gl.bufferData(gl.ARRAY_BUFFER, this._grid.colors, gl.STATIC_DRAW);
    this.render();
  }

  setColor(hex) {
    this.color = [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ];
    this.capColor = this._cutTone(this.color);
    this.render();
  }

  /** The cut face's tone: the part's brightness, pulled warm.
   *
   * A cut face shaded like the part is invisible -- it was, until this was
   * measured: the cap drew, and read as more of the same surface. CAD tools
   * give the section a colour of its own for the same reason, so the hue
   * moves and the brightness does not, which keeps it in the part's key
   * rather than looking like a decal.
   */
  _cutTone(color) {
    const luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
    return [luminance * 1.18, luminance * 0.76, luminance * 0.44];
  }

  /** Upload a mesh.
   *
   * `frame` false keeps the framing of whatever was loaded before, which is
   * what timeline scrubbing needs: a two-step-old part is genuinely smaller
   * than the finished one, and re-normalizing each step would hide that by
   * blowing every intermediate up to fill the viewport.
   */
  load(meshData, { frame = true } = {}) {
    const gl = this.gl;
    const decoded = decodeMesh(meshData);
    if (decoded.useUint32 && !this.uint32Indices) {
      throw new Error('mesh needs 32-bit indices but OES_element_index_uint is missing');
    }
    this.mesh = decoded;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, decoded.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
    gl.bufferData(gl.ARRAY_BUFFER, decoded.normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, decoded.indices, gl.STATIC_DRAW);
    this.indexType = decoded.useUint32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;

    // Normalize every part to a unit-ish box so a 6 mm spacer and a 200 mm
    // rail both arrive framed, and one camera distance suits all of them.
    if (frame || !this.center) {
      this.bounds = meshData.bounds;
      const min = meshData.bounds.min, max = meshData.bounds.max;
      this.center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
      const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
      this.scale = 2.0 / extent;
    }
    this.render();
  }

  resize() {
    const canvas = this.canvas;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  render() {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { yaw, pitch, distance } = this.camera;
    const eye = [
      distance * Math.cos(pitch) * Math.cos(yaw),
      distance * Math.cos(pitch) * Math.sin(yaw),
      distance * Math.sin(pitch),
    ];
    const target = [this.pan[0], this.pan[1], 0];
    const view = lookAt(
      [eye[0] + target[0], eye[1] + target[1], eye[2]], target, [0, 0, 1]
    );

    // model = scale * translate(-center), folded into the view matrix.
    const s = this.scale;
    const model = new Float32Array([
      s, 0, 0, 0,
      0, s, 0, 0,
      0, 0, s, 0,
      -this.center[0] * s, -this.center[1] * s, -this.center[2] * s, 1,
    ]);
    const modelView = multiply(view, model);
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const projection = perspective(Math.PI / 4, aspect, 0.05, 100);

    // The grid is the ground the part stands on, so it belongs at the part's
    // underside. Drawn at z=0 it cut through the middle of every part, because
    // the model transform centres the part on the origin, and its lines showed
    // through the solid wherever they came out nearer than the far wall.
    const floor = this.bounds
      ? (this.bounds.min[2] - this.center[2]) * this.scale : 0;
    // Grid first, shadow over it. On a dark ground a darkening shadow has
    // almost no room to read against the background, and what does read is the
    // grid dimming inside the silhouette. On a light ground both do.
    if (this.showGrid) this._drawGrid(view, projection, floor);
    if (this.mesh && this.showShadow) {
      this._drawShadow(view, projection, model, floor);
    }
    if (!this.mesh) return;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uniforms.modelView, false, modelView);
    gl.uniformMatrix4fv(this.uniforms.projection, false, projection);
    gl.uniformMatrix3fv(this.uniforms.normalMatrix, false, upperLeft3x3(modelView));
    gl.uniform3fv(this.uniforms.color, this.color);
    gl.uniform1i(this.uniforms.flat, this.flatShading ? 1 : 0);

    // The cut travels in model millimetres, so the slider means the same thing
    // for a 6 mm spacer and a 200 mm rail.
    const axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][this.section.axis];
    const low = this.bounds ? this.bounds.min[this.section.axis] : -1;
    const high = this.bounds ? this.bounds.max[this.section.axis] : 1;
    const where = low + (high - low) * this.section.cut;
    gl.uniform4f(this.uniforms.section, axis[0], axis[1], axis[2], where);
    gl.uniform1i(this.uniforms.sectioning, this.section.on ? 1 : 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.enableVertexAttribArray(this.attributes.position);
    gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
    gl.enableVertexAttribArray(this.attributes.normal);
    gl.vertexAttribPointer(this.attributes.normal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    // LINES over the triangle indices draws each triangle's three edges twice.
    // That is the cheap wireframe, and at these mesh sizes it is imperceptible.
    gl.drawElements(this.wireframe ? gl.LINES : gl.TRIANGLES,
                    this.mesh.count, this.indexType, 0);

    if (this.section.on && !this.wireframe) {
      this._drawCap(modelView, projection, axis, where);
    }
  }

  /** Fill the cut face, so a sectioned solid reads as solid.
   *
   * Clipping alone deletes fragments and leaves the inside of the far wall
   * showing through an open mouth: the part looks like a shell. A CAD tool
   * caps the cut, and the standard way is the stencil buffer.
   *
   * Count the front and back faces the cut removed, which are exactly the
   * surfaces between the eye and the plane. Where entries outnumber exits the
   * plane is passing through material, so that pixel is cut face. Then draw a
   * quad on the plane wherever the count is nonzero.
   *
   * Needs a stencil buffer and consistent winding. Without either the cap is
   * skipped and the section still works, uncapped, as before.
   */
  _drawCap(modelView, projection, axis, where) {
    const gl = this.gl;
    if (!this.hasStencil || !this.bounds) return;

    gl.enable(gl.STENCIL_TEST);
    gl.clearStencil(0);
    gl.clear(gl.STENCIL_BUFFER_BIT);
    gl.colorMask(false, false, false, false);
    gl.depthMask(false);
    // Every crossing counts, not just the nearest, so depth cannot gate this.
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.stencilFunc(gl.ALWAYS, 0, 0xff);
    gl.uniform1i(this.uniforms.sectionInvert, 1);

    gl.cullFace(gl.BACK);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.INCR_WRAP);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.indexType, 0);
    gl.cullFace(gl.FRONT);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.DECR_WRAP);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.indexType, 0);

    gl.uniform1i(this.uniforms.sectionInvert, 0);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.stencilFunc(gl.NOTEQUAL, 0, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

    // The quad lies on the plane and spans the part, so it covers any
    // cross-section the cut can produce. Depth test stays on: with an oblique
    // cut, material can sit nearer the eye than the plane at the same pixel.
    this._updateCapQuad(axis, where);
    gl.uniform3fv(this.uniforms.color, this.capColor);
    // A single flat quad has no derivative to take a face normal from that
    // would differ from its own; keep the smooth path so it shades evenly.
    gl.uniform1i(this.uniforms.flat, 0);
    gl.uniform1i(this.uniforms.sectioning, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.capPosition);
    gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.capNormal);
    gl.vertexAttribPointer(this.attributes.normal, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disable(gl.STENCIL_TEST);
    gl.uniform1i(this.uniforms.sectioning, 1);
    gl.uniform3fv(this.uniforms.color, this.color);
    gl.uniform1i(this.uniforms.flat, this.flatShading ? 1 : 0);
    // The mesh buffers were bound before the cap; put them back for the next
    // caller, which assumes the attribute pointers still address the part.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
    gl.vertexAttribPointer(this.attributes.normal, 3, gl.FLOAT, false, 0, 0);
  }

  /** A quad on the section plane, big enough to cover the part's widest cut. */
  _updateCapQuad(axis, where) {
    const gl = this.gl;
    const key = axis.join(',') + ':' + where;
    if (this._capKey === key) return;
    this._capKey = key;

    const index = axis[0] ? 0 : (axis[1] ? 1 : 2);
    const u = (index + 1) % 3, v = (index + 2) % 3;
    const min = this.bounds.min, max = this.bounds.max;
    // A margin covers the diagonal case, where the visible cross-section can
    // reach past the axis-aligned extents of the two in-plane axes.
    const pad = 0.05 * Math.max(max[u] - min[u], max[v] - min[v], 1);
    const corner = (su, sv) => {
      const point = [0, 0, 0];
      point[index] = where;
      point[u] = su ? max[u] + pad : min[u] - pad;
      point[v] = sv ? max[v] + pad : min[v] - pad;
      return point;
    };
    const a = corner(0, 0), b = corner(1, 0), c = corner(1, 1), d = corner(0, 1);
    const positions = new Float32Array([...a, ...b, ...c, ...a, ...c, ...d]);
    // Face the cap out of the cut, which is the side the camera is on.
    const normals = new Float32Array(18);
    for (let i = 0; i < 6; i += 1) normals.set(axis, i * 3);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.capPosition);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.capNormal);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
  }

  /** The part flattened onto the floor plane, blended dark.
   *
   * A directional light projects each vertex down the light ray to the plane,
   * which is one matrix, so the shadow costs a second pass over the same
   * buffers and no extra geometry.
   *
   * The stencil is what keeps it from going blotchy: a part is many triangles
   * deep along the light ray, and blending each one darkens overlaps into a
   * silhouette of the tessellation. Shading each pixel once fixes that.
   */
  _drawShadow(view, projection, model, floor) {
    const gl = this.gl;
    // Down and to one side, so the shadow reads as a shadow rather than a
    // reflection. Roughly under the shader's key light.
    const light = normalize3([0.30, 0.42, -1.0]);
    const k = 1 / light[2];
    const flatten = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      -light[0] * k, -light[1] * k, 0, 0,
      light[0] * k * floor, light[1] * k * floor, floor, 1,
    ]);
    const modelView = multiply(view, multiply(flatten, model));

    gl.useProgram(this.shadowProgram);
    gl.uniformMatrix4fv(this.shadowUniforms.modelView, false, modelView);
    gl.uniformMatrix4fv(this.shadowUniforms.projection, false, projection);
    gl.uniform4fv(this.shadowUniforms.shade, this.shadowTone);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // No depth write: the part draws over the shadow, and so does the grid.
    gl.depthMask(false);
    if (this.hasStencil) {
      gl.enable(gl.STENCIL_TEST);
      gl.clearStencil(0);
      gl.clear(gl.STENCIL_BUFFER_BIT);
      gl.stencilFunc(gl.EQUAL, 0, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.INCR);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.enableVertexAttribArray(this.shadowPosition);
    gl.vertexAttribPointer(this.shadowPosition, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.indexType, 0);

    if (this.hasStencil) gl.disable(gl.STENCIL_TEST);
    gl.disableVertexAttribArray(this.shadowPosition);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  /** Ground grid, drawn in the part's own normalized space at `floor`. */
  _drawGrid(view, projection, floor) {
    const gl = this.gl;
    const placed = new Float32Array(view);
    // Column-major: row 3 of the translation column carries the z offset, so
    // sliding the grid down is one term rather than another matrix multiply.
    placed[14] += view[10] * (floor || 0);
    placed[13] += view[9] * (floor || 0);
    placed[12] += view[8] * (floor || 0);
    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(this.lineUniforms.modelView, false, placed);
    gl.uniformMatrix4fv(this.lineUniforms.projection, false, projection);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.position);
    gl.enableVertexAttribArray(this.lineAttributes.position);
    gl.vertexAttribPointer(this.lineAttributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.color);
    gl.enableVertexAttribArray(this.lineAttributes.color);
    gl.vertexAttribPointer(this.lineAttributes.color, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, this._grid.positions.length / 3);
    gl.disableVertexAttribArray(this.lineAttributes.color);
  }

  _bindControls() {
    let dragging = false;
    let lastX = 0, lastY = 0;
    const canvas = this.canvas;

    const down = (x, y) => { dragging = true; lastX = x; lastY = y; };
    const move = (x, y) => {
      if (!dragging) return;
      if (this.mode === 'pan') {
        // Pan in the view plane, scaled by distance so it tracks the cursor.
        const k = this.camera.distance * 0.0016;
        this.pan[0] -= (x - lastX) * k * Math.sin(this.camera.yaw);
        this.pan[1] += (x - lastX) * k * Math.cos(this.camera.yaw);
        lastX = x; lastY = y;
        this.render();
        return;
      }
      this.camera.yaw -= (x - lastX) * 0.01;
      // Clamp short of the pole: at exactly +-pi/2 the up vector and the view
      // direction are parallel and the camera basis degenerates.
      const limit = Math.PI / 2 - 0.01;
      this.camera.pitch = Math.max(-limit, Math.min(limit, this.camera.pitch + (y - lastY) * 0.01));
      lastX = x; lastY = y;
      this.render();
      if (this.onCamera) this.onCamera();
    };
    const up = () => { dragging = false; };

    canvas.addEventListener('mousedown', (e) => down(e.clientX, e.clientY));
    window.addEventListener('mousemove', (e) => move(e.clientX, e.clientY));
    window.addEventListener('mouseup', up);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.distance = Math.max(1.2, Math.min(12, this.camera.distance * Math.exp(e.deltaY * 0.001)));
      this.render();
      if (this.onCamera) this.onCamera();
    }, { passive: false });

    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) down(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: true });
    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1) { e.preventDefault(); move(e.touches[0].clientX, e.touches[0].clientY); }
    }, { passive: false });
    canvas.addEventListener('touchend', up);
    window.addEventListener('resize', () => this.render());
  }
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}
