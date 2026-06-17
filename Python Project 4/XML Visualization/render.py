"""Hybrid rendering: high-quality Graphviz when available, pure-Python fallback.

Strategy:
  1. If the Graphviz `dot` binary is on PATH, shell out to it for a polished,
     hierarchical layout (the gold standard for flow diagrams).
  2. Otherwise, fall back to networkx + matplotlib, which are pip-installable
     wheels with no native dependency. Always works on Mac and Windows.

The fallback means the app never hard-fails on a machine without Graphviz —
it just renders a slightly less polished diagram and tells the user how to
upgrade.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from converter import Graph

# Map Graphviz shapes to matplotlib-friendly equivalents for the fallback.
_FALLBACK_NODE_COLOR = {
    "oval": "#cfe8ff",
    "diamond": "#ffe0b3",
    "box": "#e8e8e8",
    "box3d": "#d9f0d3",
}


@dataclass
class RenderInfo:
    backend: str  # "graphviz" or "matplotlib"
    output_path: Path
    message: str


def graphviz_available() -> bool:
    return shutil.which("dot") is not None


def _clean_label(label: str) -> str:
    # converter uses literal "\n" for DOT; turn it into a real newline for mpl.
    return label.replace("\\n", "\n")


def render_with_graphviz(dot_text: str, output_path: Path, fmt: str = "png") -> RenderInfo:
    """Render DOT via the system `dot` binary. Raises on failure."""
    dot_bin = shutil.which("dot")
    if not dot_bin:
        raise RuntimeError("Graphviz 'dot' binary not found on PATH.")
    proc = subprocess.run(
        [dot_bin, f"-T{fmt}", "-o", str(output_path)],
        input=dot_text.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Graphviz failed: {proc.stderr.decode('utf-8', 'ignore')}")
    return RenderInfo(
        backend="graphviz",
        output_path=output_path,
        message=f"Rendered with system Graphviz -> {output_path.name}",
    )


def render_with_matplotlib(graph: Graph, output_path: Path) -> RenderInfo:
    """Render the graph with networkx + matplotlib. No native dependency."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe; GUI swaps this for an interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    labels = {}
    colors = []
    for node in graph.nodes:
        g.add_node(node.node_id)
        labels[node.node_id] = _clean_label(node.label)
    edge_labels = {}
    for src, dst, elabel in graph.edges:
        g.add_edge(src, dst)
        if elabel:
            edge_labels[(src, dst)] = elabel

    # Build color list in node order after the graph is populated.
    shape_by_id = {n.node_id: n.shape for n in graph.nodes}
    for nid in g.nodes():
        colors.append(_FALLBACK_NODE_COLOR.get(shape_by_id.get(nid, "box"), "#e8e8e8"))

    pos = _hierarchical_layout(g)

    fig, ax = plt.subplots(figsize=(max(8, graph.node_count * 1.3), 6))
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, edge_color="#888888",
                           arrowsize=18, node_size=2600)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=2600,
                           edgecolors="#444444")
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8)
    if edge_labels:
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=6, font_color="#555555")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return RenderInfo(
        backend="matplotlib",
        output_path=output_path,
        message=(
            f"Rendered with matplotlib fallback -> {output_path.name}. "
            "Install Graphviz for higher-quality layout."
        ),
    )


def _hierarchical_layout(g) -> dict:
    """Left-to-right layered layout via BFS depth. Mimics Graphviz rankdir=LR.

    Pure-Python (no pygraphviz), so it works without the native binary.
    """
    import networkx as nx

    if g.number_of_nodes() == 0:
        return {}

    # Depth from roots (nodes with no incoming edges).
    roots = [n for n in g.nodes() if g.in_degree(n) == 0] or [next(iter(g.nodes()))]
    depth: dict[str, int] = {}
    for root in roots:
        for node, dist in nx.single_source_shortest_path_length(g, root).items():
            depth[node] = min(depth.get(node, dist), dist)
    for node in g.nodes():
        depth.setdefault(node, 0)

    # Group by depth, spread vertically within each column.
    columns: dict[int, list[str]] = {}
    for node, d in depth.items():
        columns.setdefault(d, []).append(node)

    pos = {}
    for d, nodes in columns.items():
        count = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            y = (count - 1) / 2 - i  # center the column vertically
            pos[node] = (d * 2.2, y * 1.4)
    return pos


def render(dot_text: str, graph: Graph, output_path: Path, fmt: str = "png",
           prefer_graphviz: bool = True) -> RenderInfo:
    """Render using the best available backend.

    PNG/SVG with Graphviz if present (and prefer_graphviz); otherwise PNG via
    matplotlib. SVG is only produced by the Graphviz path.
    """
    output_path = Path(output_path)
    if prefer_graphviz and graphviz_available():
        return render_with_graphviz(dot_text, output_path, fmt=fmt)
    if fmt == "svg":
        output_path = output_path.with_suffix(".png")
    return render_with_matplotlib(graph, output_path)
