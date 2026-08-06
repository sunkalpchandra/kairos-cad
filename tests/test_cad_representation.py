"""CAD integration tests: observation snapshots and geometry-graph construction."""

import json

import pytest

from kairos.representation.geometry_graph import (
    NODE_KINDS,
    RELATIONS,
    build_geometry_graph,
)
from kairos.representation.observation import observe

pytestmark = pytest.mark.cad


@pytest.fixture
def plate(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 40, 20)
    engine.pad(10)
    engine.create_sketch("XY", offset=10)
    engine.add_circle(20, 10, 3)
    engine.pocket(through_all=True)
    return engine


def test_observation_is_json_ready(plate):
    obs = observe(plate)
    text = json.dumps(obs)  # must not raise
    restored = json.loads(text)
    assert restored["summary"]["has_solid"] is True
    assert len(restored["holes"]) == 1
    assert restored["edge_count"] > 0
    assert restored["sketch"]["name"].startswith("Sketch")


def test_observation_before_any_geometry(engine):
    obs = observe(engine)
    assert obs["summary"]["has_solid"] is False
    assert obs["holes"] == [] and obs["faces"] == []
    assert obs["sketch"] is None


def test_graph_counts_and_kinds(plate):
    graph = build_geometry_graph(plate)
    counts = graph.counts()
    assert counts["body"] == 1
    assert counts["solid"] == 1
    assert counts["sketch"] == 2
    assert counts["feature"] == 2  # Pad + Pocket
    assert counts["face"] == plate.summary()["topology"]["faces"]
    assert counts["edge"] == plate.summary()["topology"]["edges"]
    assert counts["vertex"] > 0
    assert counts["constraint"] == 8  # rectangle helper constraints
    assert set(graph.node_kinds) <= set(NODE_KINDS)


def test_graph_relations_present(plate):
    graph = build_geometry_graph(plate)
    relations = {RELATIONS[r] for r in graph.edge_relations.tolist()}
    assert {"contains", "adjacent_to", "created_by", "constrained_by", "depends_on"} <= relations
    assert graph.num_edges > graph.num_nodes  # adjacency-rich
    assert graph.node_features.shape[0] == graph.num_nodes
    assert graph.edge_index.max() < graph.num_nodes


def test_created_by_links_features_to_their_sketches(plate):
    graph = build_geometry_graph(plate)
    created_by = [
        (graph.node_names[s], graph.node_names[d])
        for (s, d), r in zip(
            graph.edge_index.T.tolist(), graph.edge_relations.tolist(), strict=True
        )
        if RELATIONS[r] == "created_by"
    ]
    assert len(created_by) == 2
    for feature_name, sketch_name in created_by:
        assert feature_name.startswith(("Pad", "Pocket"))
        assert sketch_name.startswith("Sketch")


def test_dressup_modified_by_link(plate):
    edge = plate.find_edges(curve="Line", direction=(0, 0, 1))[0]
    plate.fillet([edge], radius=2.0)
    graph = build_geometry_graph(plate)
    modified = [
        (graph.node_names[s], graph.node_names[d])
        for (s, d), r in zip(
            graph.edge_index.T.tolist(), graph.edge_relations.tolist(), strict=True
        )
        if RELATIONS[r] == "modified_by"
    ]
    assert any(src.startswith("Fillet") for src, _ in modified)


def test_vertex_cap_subsamples(plate):
    graph = build_geometry_graph(plate, max_vertices=4)
    assert graph.counts()["vertex"] <= 8  # stride sampling, small overshoot ok


def test_graph_serializes(plate):
    graph = build_geometry_graph(plate)
    data = graph.to_dict()
    json.dumps(data)
    assert len(data["node_kinds"]) == graph.num_nodes
