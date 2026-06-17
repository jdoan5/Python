"""Unit tests for the XML -> DOT converter. Run: python -m pytest test_converter.py"""

from __future__ import annotations

import pytest

from converter import (
    XMLConversionError,
    build_graph,
    parse_xml,
    shape_for,
    strip_ns,
    to_dot,
    xml_to_dot,
)

SIMPLE = """<route id="r1">
  <from id="src" uri="file:in.csv"/>
  <filter id="f1" condition="x > 0">
    <transform id="t1" ref="clean"/>
  </filter>
  <to id="sink" uri="file:out.csv"/>
</route>"""


def test_strip_ns() -> None:
    assert strip_ns("{http://camel.apache.org/schema}route") == "route"
    assert strip_ns("route") == "route"


def test_shape_for() -> None:
    assert shape_for("from") == "oval"
    assert shape_for("to") == "oval"
    assert shape_for("filter") == "diamond"
    assert shape_for("choice") == "diamond"
    assert shape_for("transform") == "box"
    assert shape_for("unknown") == "box"


def test_build_graph_counts() -> None:
    graph = build_graph(parse_xml(SIMPLE))
    # route, from, filter, transform, to = 5 nodes
    assert graph.node_count == 5
    # route->from, route->filter, filter->transform, route->to = 4 edges
    assert graph.edge_count == 4


def test_edge_labels_present() -> None:
    graph = build_graph(parse_xml(SIMPLE))
    labels = [elabel for _, _, elabel in graph.edges]
    assert any("uri=file:in.csv" in lbl for lbl in labels)
    assert any("condition=x > 0" in lbl for lbl in labels)


def test_to_dot_is_valid_digraph() -> None:
    dot, _ = xml_to_dot(SIMPLE)
    assert dot.startswith("digraph EIP {")
    assert dot.rstrip().endswith("}")
    assert "rankdir=LR;" in dot
    assert "shape=diamond" in dot  # the filter


def test_rankdir_tb() -> None:
    dot, _ = xml_to_dot(SIMPLE, rankdir="TB")
    assert "rankdir=TB;" in dot


def test_malformed_xml_raises() -> None:
    with pytest.raises(XMLConversionError):
        parse_xml("<route><from></route>")  # mismatched tags


def test_empty_document_raises() -> None:
    # A self-closing single root still yields one node; truly empty input fails to parse.
    with pytest.raises(XMLConversionError):
        parse_xml("")


def test_label_quoting() -> None:
    dot, _ = xml_to_dot('<n id=\'has"quote\'/>')
    # Inner quotes must be escaped so DOT stays valid.
    assert '\\"' in dot
