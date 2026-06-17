"""XML -> graph -> Graphviz DOT conversion.

Parses an EIP-style (Enterprise Integration Patterns) XML document — sources,
filters, routers, transforms, sinks — into a directed graph, then emits
Graphviz DOT. Schema-agnostic: it walks whatever tree you give it and labels
nodes/edges from common integration attributes.

This module has no GUI or rendering dependencies, so it imports cleanly
anywhere (CLI, GUI, tests).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Attributes searched (in order) to build a node's caption.
LABEL_ATTRS = ("id", "name", "uri", "ref", "endpoint", "channel", "type")
# Attributes searched (in order) to label an edge.
EDGE_ATTRS = ("uri", "ref", "when", "condition", "expression")

MAX_TEXT_LABEL = 40


class XMLConversionError(Exception):
    """Raised when the input cannot be parsed or contains no usable elements."""


@dataclass
class Node:
    node_id: str
    label: str
    shape: str
    tag: str


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def strip_ns(tag: str) -> str:
    """Drop an XML namespace prefix: '{http://...}route' -> 'route'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def node_label(elem: ET.Element) -> str:
    tag = strip_ns(elem.tag)
    for key in LABEL_ATTRS:
        value = elem.attrib.get(key, "").strip()
        if value:
            return f"{tag}\\n{key}={value}"
    text = (elem.text or "").strip()
    if 0 < len(text) <= MAX_TEXT_LABEL:
        return f"{tag}\\n{text}"
    return tag


def shape_for(tag: str) -> str:
    """Heuristic node shape from the element tag (customize freely)."""
    tag = tag.lower()
    if tag in {"from", "inbound", "consumer", "source"}:
        return "oval"
    if tag in {"to", "outbound", "producer", "sink"}:
        return "oval"
    if "filter" in tag:
        return "diamond"
    if "choice" in tag or "router" in tag or "when" in tag or "otherwise" in tag:
        return "diamond"
    if "transform" in tag or "map" in tag:
        return "box"
    if "split" in tag or "aggregate" in tag:
        return "box3d"
    return "box"


def build_graph(root: ET.Element) -> Graph:
    graph = Graph()
    ids: dict[int, str] = {}
    counter = 0

    def get_id(elem: ET.Element) -> str:
        nonlocal counter
        key = id(elem)
        if key not in ids:
            counter += 1
            ids[key] = f"n{counter}"
        return ids[key]

    seen: set[str] = set()

    def add_node(elem: ET.Element) -> str:
        nid = get_id(elem)
        if nid not in seen:
            seen.add(nid)
            tag = strip_ns(elem.tag)
            graph.nodes.append(Node(nid, node_label(elem), shape_for(tag), tag))
        return nid

    def walk(parent: ET.Element) -> None:
        parent_id = add_node(parent)
        for child in list(parent):
            child_id = add_node(child)
            edge_label = ""
            for key in EDGE_ATTRS:
                if key in child.attrib:
                    edge_label = f"{key}={child.attrib[key]}"
                    break
            graph.edges.append((parent_id, child_id, edge_label))
            walk(child)

    walk(root)
    if graph.node_count == 0:
        raise XMLConversionError("No elements found in the XML document.")
    return graph


def to_dot(graph: Graph, rankdir: str = "LR") -> str:
    lines = [
        "digraph EIP {",
        f"  rankdir={rankdir};",
        "  splines=true;",
        "  node [fontname=Helvetica];",
        "  edge [fontname=Helvetica];",
    ]
    for node in graph.nodes:
        safe_label = node.label.replace('"', '\\"')
        lines.append(f'  {node.node_id} [label="{safe_label}", shape={node.shape}];')
    for src, dst, elabel in graph.edges:
        if elabel:
            safe_elabel = elabel.replace('"', '\\"')
            lines.append(f'  {src} -> {dst} [label="{safe_elabel}"];')
        else:
            lines.append(f"  {src} -> {dst};")
    lines.append("}")
    return "\n".join(lines)


def parse_xml(source: str | Path) -> ET.Element:
    """Parse XML from a file path or a raw XML string. Raises XMLConversionError."""
    try:
        text = Path(source).read_text(encoding="utf-8") if Path(source).exists() else str(source)
    except OSError as e:
        raise XMLConversionError(f"Could not read file: {e}") from e
    try:
        return ET.fromstring(text)
    except ET.ParseError as e:
        raise XMLConversionError(f"Malformed XML: {e}") from e


def xml_to_dot(source: str | Path, rankdir: str = "LR") -> tuple[str, Graph]:
    """Convenience: source (path or XML string) -> (dot_text, graph)."""
    root = parse_xml(source)
    graph = build_graph(root)
    return to_dot(graph, rankdir=rankdir), graph
