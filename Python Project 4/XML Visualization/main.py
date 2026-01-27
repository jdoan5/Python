from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def node_label(elem: ET.Element) -> str:
    tag = strip_ns(elem.tag)

    # Useful attributes often found in route-style XML
    for key in ("id", "name", "uri", "ref", "endpoint", "channel", "type"):
        if key in elem.attrib and elem.attrib[key].strip():
            return f"{tag}\\n{key}={elem.attrib[key].strip()}"

    # Try text content if it’s short
    txt = (elem.text or "").strip()
    if 0 < len(txt) <= 40:
        return f"{tag}\\n{txt}"

    return tag

def shape_for(tag: str) -> str:
    # Basic EIP-ish heuristics (customize this mapping)
    tag = tag.lower()
    if tag in {"from", "inbound", "consumer", "source"}:
        return "oval"
    if tag in {"to", "outbound", "producer", "sink"}:
        return "oval"
    if "filter" in tag:
        return "diamond"
    if "choice" in tag or "router" in tag:
        return "diamond"
    if "transform" in tag or "map" in tag:
        return "box"
    if "split" in tag or "aggregate" in tag:
        return "box3d"
    return "box"

def build_graph(root: ET.Element):
    """
    Returns:
      nodes: dict[node_id -> (label, shape)]
      edges: list[(src_id, dst_id, edge_label)]
    """
    nodes = {}
    edges = []

    # assign unique IDs to elements
    ids = {}
    counter = 0

    def get_id(e: ET.Element) -> str:
        nonlocal counter
        if e not in ids:
            counter += 1
            ids[e] = f"n{counter}"
        return ids[e]

    def walk(parent: ET.Element):
        parent_id = get_id(parent)
        parent_tag = strip_ns(parent.tag)
        nodes[parent_id] = (node_label(parent), shape_for(parent_tag))

        children = list(parent)
        for child in children:
            child_id = get_id(child)
            child_tag = strip_ns(child.tag)
            nodes[child_id] = (node_label(child), shape_for(child_tag))

            # Edge label: use common attributes if present (uri/ref/condition)
            edge_label = ""
            for k in ("uri", "ref", "when", "condition", "expression"):
                if k in child.attrib:
                    edge_label = f"{k}={child.attrib[k]}"
                    break

            edges.append((parent_id, child_id, edge_label))
            walk(child)

    walk(root)
    return nodes, edges

def to_dot(nodes, edges) -> str:
    lines = [
        "digraph EIP {",
        "  rankdir=LR;",
        "  splines=true;",
        "  node [fontname=Helvetica];",
        "  edge [fontname=Helvetica];",
    ]

    for nid, (label, shape) in nodes.items():
        safe_label = label.replace('"', '\\"')
        lines.append(f'  {nid} [label="{safe_label}", shape={shape}];')

    for src, dst, elabel in edges:
        if elabel:
            safe_elabel = elabel.replace('"', '\\"')
            lines.append(f'  {src} -> {dst} [label="{safe_elabel}"];')
        else:
            lines.append(f"  {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)

def main(xml_path: str, dot_out: str = "eip.dot"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes, edges = build_graph(root)
    dot = to_dot(nodes, edges)

    Path(dot_out).write_text(dot, encoding="utf-8")
    print(f"Wrote DOT: {dot_out}")
    print("Render with:")
    print(f"  dot -Tpng {dot_out} -o eip.png")
    print(f"  dot -Tsvg {dot_out} -o eip.svg")

if __name__ == "__main__":
    main("sample_eip.xml")