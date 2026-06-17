"""Command-line entry point for the XML -> flow-diagram visualizer.

Examples:
    python main.py sample_eip.xml                 # write eip.dot
    python main.py sample_eip.xml --render png    # also render an image
    python main.py route.xml -o out.dot --render svg --rankdir TB

Rendering uses the hybrid backend (Graphviz if installed, matplotlib
otherwise). DOT generation has no dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from converter import XMLConversionError, xml_to_dot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert EIP-style XML into a flow diagram.")
    parser.add_argument("xml", help="Path to the input XML file.")
    parser.add_argument("-o", "--out", default="eip.dot", help="DOT output path (default: eip.dot).")
    parser.add_argument(
        "--rankdir", choices=["LR", "TB"], default="LR",
        help="Layout direction: LR (left-right) or TB (top-bottom).",
    )
    parser.add_argument(
        "--render", choices=["png", "svg"], default=None,
        help="Also render an image next to the DOT file.",
    )
    args = parser.parse_args(argv)

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print(f"error: file not found: {xml_path}", file=sys.stderr)
        return 1

    try:
        dot_text, graph = xml_to_dot(xml_path, rankdir=args.rankdir)
    except XMLConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.write_text(dot_text, encoding="utf-8")
    print(f"Wrote DOT: {out_path}  ({graph.node_count} nodes, {graph.edge_count} edges)")

    if args.render:
        from render import render

        image_path = out_path.with_suffix(f".{args.render}")
        try:
            info = render(dot_text, graph, image_path, fmt=args.render)
            print(info.message)
        except Exception as e:
            print(f"error: render failed: {e}", file=sys.stderr)
            return 1
    else:
        print("Render with Graphviz:")
        print(f"  dot -Tpng {out_path} -o {out_path.with_suffix('.png')}")
        print("Or render in-app:  python gui_app.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
