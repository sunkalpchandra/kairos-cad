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
void main() {
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
uniform vec3 uColor;
uniform bool uFlat;
void main() {
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

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error('shader compile failed: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

function linkProgram(gl) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
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

class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const options = { antialias: true, alpha: false, depth: true };
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
    };
    this.flatShading = this.derivatives;
    this.buffers = {
      position: gl.createBuffer(),
      normal: gl.createBuffer(),
      index: gl.createBuffer(),
    };
    this.mesh = null;
    this.color = [0.42, 0.58, 0.86];
    this.background = [0.07, 0.08, 0.11];
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

  setColor(hex) {
    this.color = [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ];
    this.render();
  }

  load(meshData) {
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
    const min = meshData.bounds.min, max = meshData.bounds.max;
    this.center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
    this.scale = 2.0 / extent;
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
    gl.clearColor(this.background[0], this.background[1], this.background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.mesh) return;

    const { yaw, pitch, distance } = this.camera;
    const eye = [
      distance * Math.cos(pitch) * Math.cos(yaw),
      distance * Math.cos(pitch) * Math.sin(yaw),
      distance * Math.sin(pitch),
    ];
    const view = lookAt(eye, [0, 0, 0], [0, 0, 1]);

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

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uniforms.modelView, false, modelView);
    gl.uniformMatrix4fv(this.uniforms.projection, false, projection);
    gl.uniformMatrix3fv(this.uniforms.normalMatrix, false, upperLeft3x3(modelView));
    gl.uniform3fv(this.uniforms.color, this.color);
    gl.uniform1i(this.uniforms.flat, this.flatShading ? 1 : 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.enableVertexAttribArray(this.attributes.position);
    gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
    gl.enableVertexAttribArray(this.attributes.normal);
    gl.vertexAttribPointer(this.attributes.normal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.indexType, 0);
  }

  _bindControls() {
    let dragging = false;
    let lastX = 0, lastY = 0;
    const canvas = this.canvas;

    const down = (x, y) => { dragging = true; lastX = x; lastY = y; };
    const move = (x, y) => {
      if (!dragging) return;
      this.camera.yaw -= (x - lastX) * 0.01;
      // Clamp short of the pole: at exactly +-pi/2 the up vector and the view
      // direction are parallel and the camera basis degenerates.
      const limit = Math.PI / 2 - 0.01;
      this.camera.pitch = Math.max(-limit, Math.min(limit, this.camera.pitch + (y - lastY) * 0.01));
      lastX = x; lastY = y;
      this.render();
    };
    const up = () => { dragging = false; };

    canvas.addEventListener('mousedown', (e) => down(e.clientX, e.clientY));
    window.addEventListener('mousemove', (e) => move(e.clientX, e.clientY));
    window.addEventListener('mouseup', up);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.distance = Math.max(1.2, Math.min(12, this.camera.distance * Math.exp(e.deltaY * 0.001)));
      this.render();
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
