"""Geometry graph: the CAD model as a typed, attributed graph.

Node kinds: body, solid, face, edge, vertex, sketch, feature, constraint.
Relations: contains, adjacent_to, created_by, constrained_by, depends_on,
modified_by.

The graph is framework-neutral (numpy arrays + JSON-serializable metadata);
Phase 4 wraps it for PyTorch Geometric. Construction reads the live document
through the engine but the resulting ``GeometryGraph`` is detached data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NODE_KINDS: tuple[str, ...] = (
    "body", "solid", "face", "edge", "vertex", "sketch", "feature", "constraint",
)
RELATIONS: tuple[str, ...] = (
    "contains", "adjacent_to", "created_by", "constrained_by", "depends_on",
    "modified_by",
)

_KIND_INDEX = {k: i for i, k in enumerate(NODE_KINDS)}
_REL_INDEX = {r: i for i, r in enumerate(RELATIONS)}

#: Per-node numeric attributes appended to the kind one-hot.
#: [size, radius, pos_x, pos_y, pos_z, type_code, is_tip, aux]
NUM_NODE_FEATURES = len(NODE_KINDS) + 8

_SURFACE_CODES = {"Plane": 1, "Cylinder": 2, "Cone": 3, "Sphere": 4, "Torus": 5}
_CURVE_CODES = {"Line": 1, "Circle": 2, "Ellipse": 3, "BSplineCurve": 4}
_FEATURE_CODES = {
    "PartDesign::Pad": 1, "PartDesign::Pocket": 2, "PartDesign::Revolution": 3,
    "PartDesign::Fillet": 4, "PartDesign::Chamfer": 5, "PartDesign::Thickness": 6,
    "PartDesign::Mirrored": 7, "PartDesign::LinearPattern": 8,
    "PartDesign::PolarPattern": 9, "Sketcher::SketchObject": 10,
}


@dataclass
class GeometryGraph:
    """Typed attributed graph over one CAD model."""

    node_kinds: list[str] = field(default_factory=list)
    node_names: list[str] = field(default_factory=list)
    node_features: np.ndarray = field(
        default_factory=lambda: np.zeros((0, NUM_NODE_FEATURES), dtype=np.float32)
    )
    edge_index: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 0), dtype=np.int64)
    )
    edge_relations: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int64)
    )

    @property
    def num_nodes(self) -> int:
        return len(self.node_kinds)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (features/edges as lists)."""
        return {
            "node_kinds": list(self.node_kinds),
            "node_names": list(self.node_names),
            "node_features": self.node_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "edge_relations": [RELATIONS[r] for r in self.edge_relations.tolist()],
        }

    def counts(self) -> dict[str, int]:
        out = {kind: 0 for kind in NODE_KINDS}
        for kind in self.node_kinds:
            out[kind] += 1
        return out


class _Builder:
    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.names: list[str] = []
        self.features: list[list[float]] = []
        self.edges: list[tuple[int, int, int]] = []
        self._by_name: dict[str, int] = {}

    def node(
        self,
        kind: str,
        name: str,
        size: float = 0.0,
        radius: float = 0.0,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        type_code: int = 0,
        is_tip: bool = False,
        aux: float = 0.0,
    ) -> int:
        if name in self._by_name:
            return self._by_name[name]
        onehot = [0.0] * len(NODE_KINDS)
        onehot[_KIND_INDEX[kind]] = 1.0
        # Scale sizes/positions to the workspace envelope (~200 mm).
        feats = onehot + [
            float(np.log1p(max(size, 0.0)) / 12.0),
            float(radius) / 25.0,
            pos[0] / 200.0,
            pos[1] / 200.0,
            pos[2] / 200.0,
            float(type_code) / 10.0,
            1.0 if is_tip else 0.0,
            float(aux),
        ]
        index = len(self.kinds)
        self.kinds.append(kind)
        self.names.append(name)
        self.features.append(feats)
        self._by_name[name] = index
        return index

    def edge(self, src: int, dst: int, relation: str) -> None:
        self.edges.append((src, dst, _REL_INDEX[relation]))

    def get(self, name: str) -> int | None:
        return self._by_name.get(name)

    def build(self) -> GeometryGraph:
        if self.edges:
            arr = np.asarray(self.edges, dtype=np.int64)
            edge_index = arr[:, :2].T.copy()
            relations = arr[:, 2].copy()
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            relations = np.zeros((0,), dtype=np.int64)
        return GeometryGraph(
            node_kinds=self.kinds,
            node_names=self.names,
            node_features=np.asarray(self.features, dtype=np.float32),
            edge_index=edge_index,
            edge_relations=relations,
        )


def build_geometry_graph(engine, max_vertices: int = 256) -> GeometryGraph:
    """Construct the geometry graph from a live engine.

    Topology (solid/face/edge/vertex + adjacency) comes from the tip shape;
    history (sketch/feature/constraint + dependency links) from the feature
    tree. Vertex nodes are capped at ``max_vertices`` (uniformly subsampled)
    to bound graph size on dense tessellations.
    """
    b = _Builder()
    body_idx = b.node("body", "Body")

    # ------------------------------------------------------------- history
    doc = engine.document
    tree = engine.feature_history()
    prev_feature_idx: int | None = None
    for entry in tree:
        obj = doc.find_object(entry["name"])
        type_code = _FEATURE_CODES.get(entry["type"], 0)
        if entry["type"] == "Sketcher::SketchObject":
            sketch_idx = b.node(
                "sketch", entry["name"], size=len(obj.Geometry), type_code=type_code
            )
            b.edge(body_idx, sketch_idx, "contains")
            for i, _constraint in enumerate(obj.Constraints):
                c_idx = b.node("constraint", f"{entry['name']}.c{i}", aux=1.0)
                b.edge(sketch_idx, c_idx, "constrained_by")
            continue
        feature_idx = b.node(
            "feature", entry["name"], type_code=type_code, is_tip=entry["is_tip"]
        )
        b.edge(body_idx, feature_idx, "contains")
        if prev_feature_idx is not None:
            b.edge(feature_idx, prev_feature_idx, "depends_on")
        prev_feature_idx = feature_idx
        # created_by: profile sketch; modified_by: dressup base feature.
        profile = getattr(obj, "Profile", None)
        if profile is not None:
            profile_obj = profile[0] if isinstance(profile, tuple) else profile
            profile_idx = b.get(getattr(profile_obj, "Name", ""))
            if profile_idx is not None:
                b.edge(feature_idx, profile_idx, "created_by")
        base = getattr(obj, "Base", None)
        if isinstance(base, tuple) and base and hasattr(base[0], "Name"):
            base_idx = b.get(base[0].Name)
            if base_idx is not None:
                b.edge(feature_idx, base_idx, "modified_by")

    # ------------------------------------------------------------ topology
    if not engine.has_solid():
        return b.build()

    shape = engine.document.tip_shape()
    solid_indices = []
    for s, solid in enumerate(shape.Solids):
        com = solid.CenterOfMass
        idx = b.node(
            "solid", f"Solid{s + 1}", size=solid.Volume, pos=(com.x, com.y, com.z)
        )
        b.edge(body_idx, idx, "contains")
        solid_indices.append(idx)

    faces = engine.list_faces()
    face_nodes: dict[str, int] = {}
    for face in faces:
        idx = b.node(
            "face",
            face["name"],
            size=face["area"],
            radius=face.get("radius", 0.0),
            type_code=_SURFACE_CODES.get(face["surface"], 0),
            aux=1.0 if face.get("concave") else 0.0,
        )
        face_nodes[face["name"]] = idx
        b.edge(solid_indices[0], idx, "contains")

    edges = engine.list_edges()
    edge_nodes: dict[str, int] = {}
    for entry in edges:
        idx = b.node(
            "edge",
            entry["name"],
            size=entry["length"],
            pos=entry["midpoint"],
            type_code=_CURVE_CODES.get(entry["curve"], 0),
        )
        edge_nodes[entry["name"]] = idx

    # face-contains-edge and face-adjacent_to-face via shared edges.
    edge_to_faces: dict[str, list[str]] = {name: [] for name in edge_nodes}
    for i, face_shape in enumerate(shape.Faces, start=1):
        face_name = f"Face{i}"
        for face_edge in face_shape.Edges:
            for j, shape_edge in enumerate(shape.Edges, start=1):
                if face_edge.isSame(shape_edge):
                    edge_name = f"Edge{j}"
                    b.edge(face_nodes[face_name], edge_nodes[edge_name], "contains")
                    edge_to_faces[edge_name].append(face_name)
                    break
    seen_pairs = set()
    for edge_name, face_names in edge_to_faces.items():
        for a in range(len(face_names)):
            for c in range(a + 1, len(face_names)):
                pair = tuple(sorted((face_names[a], face_names[c])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                b.edge(face_nodes[pair[0]], face_nodes[pair[1]], "adjacent_to")
                b.edge(face_nodes[pair[1]], face_nodes[pair[0]], "adjacent_to")

    vertexes = shape.Vertexes
    stride = max(1, len(vertexes) // max_vertices) if max_vertices else 1
    for v_i in range(0, len(vertexes), stride):
        vertex = vertexes[v_i]
        b.node(
            "vertex", f"Vertex{v_i + 1}", pos=(vertex.X, vertex.Y, vertex.Z)
        )
    # edge-contains-vertex links for the sampled vertices.
    for j, shape_edge in enumerate(shape.Edges, start=1):
        for vertex in shape_edge.Vertexes:
            for v_i in range(0, len(vertexes), stride):
                if vertex.isSame(vertexes[v_i]):
                    v_idx = b.get(f"Vertex{v_i + 1}")
                    if v_idx is not None:
                        b.edge(edge_nodes[f"Edge{j}"], v_idx, "contains")
                    break

    return b.build()
