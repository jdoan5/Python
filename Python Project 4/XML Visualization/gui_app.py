"""Cross-platform desktop GUI for the XML -> flow-diagram visualizer.

Tkinter ships with Python on macOS and Windows, so this runs on both with no
extra GUI toolkit. Rendering uses the hybrid backend in render.py: Graphviz if
installed, matplotlib otherwise.

Run:
    python gui_app.py
"""

from __future__ import annotations

import platform
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from converter import XMLConversionError, xml_to_dot
from render import graphviz_available, render

APP_TITLE = "XML Flow Visualizer"
SAMPLE_XML = Path(__file__).parent / "sample_eip.xml"


class VisualizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x720")

        self.current_xml_path: Path | None = None
        self.current_dot: str | None = None
        self.current_graph = None
        self._render_image = None  # keep a reference so Tk doesn't GC it
        self._tmpdir = Path(tempfile.mkdtemp(prefix="xmlviz_"))

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()

        self._refresh_backend_label()
        if SAMPLE_XML.exists():
            self.load_file(SAMPLE_XML)

    # ----- layout -----------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open XML...", command=self.open_file, accelerator="Cmd/Ctrl+O")
        file_menu.add_command(label="Load Sample", command=lambda: self.load_file(SAMPLE_XML))
        file_menu.add_separator()
        file_menu.add_command(label="Export PNG...", command=lambda: self.export("png"))
        file_menu.add_command(label="Export SVG...", command=lambda: self.export("svg"))
        file_menu.add_command(label="Export DOT...", command=self.export_dot)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Command-o>", lambda e: self.open_file())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open XML", command=self.open_file).pack(side=tk.LEFT)
        ttk.Button(bar, text="Sample", command=lambda: self.load_file(SAMPLE_XML)).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        ttk.Label(bar, text="Direction:").pack(side=tk.LEFT, padx=(16, 4))
        self.rankdir = tk.StringVar(value="LR")
        ttk.Combobox(
            bar, textvariable=self.rankdir, values=["LR", "TB"], width=4,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.rankdir.trace_add("write", lambda *_: self.render_current())

        ttk.Button(bar, text="Export PNG", command=lambda: self.export("png")).pack(
            side=tk.RIGHT
        )
        ttk.Button(bar, text="Export SVG", command=lambda: self.export("svg")).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

    def _build_body(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Diagram tab (scrollable canvas).
        diagram_tab = ttk.Frame(self.notebook)
        self.notebook.add(diagram_tab, text="Diagram")
        self.canvas = tk.Canvas(diagram_tab, background="#fafafa")
        h = ttk.Scrollbar(diagram_tab, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v = ttk.Scrollbar(diagram_tab, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h.set, yscrollcommand=v.set)
        v.pack(side=tk.RIGHT, fill=tk.Y)
        h.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # DOT source tab.
        dot_tab = ttk.Frame(self.notebook)
        self.notebook.add(dot_tab, text="DOT source")
        self.dot_text = tk.Text(dot_tab, wrap=tk.NONE, font=("Menlo", 11))
        dot_scroll = ttk.Scrollbar(dot_tab, command=self.dot_text.yview)
        self.dot_text.configure(yscrollcommand=dot_scroll.set)
        dot_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.dot_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT)
        self.backend_label = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.backend_label, foreground="#666666").pack(
            side=tk.RIGHT
        )

    # ----- actions ----------------------------------------------------------

    def _refresh_backend_label(self) -> None:
        if graphviz_available():
            self.backend_label.set("Renderer: Graphviz (high quality)")
        else:
            self.backend_label.set("Renderer: matplotlib fallback - install Graphviz for sharper output")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open XML file",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> None:
        self.current_xml_path = path
        self.status.set(f"Loaded {path.name}")
        self.root.title(f"{APP_TITLE} - {path.name}")
        self.render_current()

    def render_current(self) -> None:
        if not self.current_xml_path:
            return
        try:
            dot_text, graph = xml_to_dot(self.current_xml_path, rankdir=self.rankdir.get())
        except XMLConversionError as e:
            messagebox.showerror("Conversion error", str(e))
            self.status.set("Conversion failed.")
            return

        self.current_dot = dot_text
        self.current_graph = graph
        self.dot_text.delete("1.0", tk.END)
        self.dot_text.insert("1.0", dot_text)
        self.status.set(
            f"{graph.node_count} nodes, {graph.edge_count} edges - rendering..."
        )
        # Render off the UI thread so the window stays responsive.
        threading.Thread(target=self._render_worker, daemon=True).start()

    def _render_worker(self) -> None:
        out = self._tmpdir / "diagram.png"
        try:
            info = render(self.current_dot, self.current_graph, out, fmt="png")
        except Exception as e:  # surface any backend failure to the UI thread
            self.root.after(0, lambda: self._on_render_error(e))
            return
        self.root.after(0, lambda: self._on_render_done(info))

    def _on_render_error(self, error: Exception) -> None:
        messagebox.showerror("Render error", str(error))
        self.status.set("Render failed.")

    def _on_render_done(self, info) -> None:
        self._display_image(info.output_path)
        self.status.set(info.message)
        self._refresh_backend_label()

    def _display_image(self, image_path: Path) -> None:
        # tk.PhotoImage handles PNG natively on Tk 8.6+ (bundled with modern Python).
        try:
            img = tk.PhotoImage(file=str(image_path))
        except tk.TclError as e:
            messagebox.showerror("Display error", f"Could not display image: {e}")
            return
        self._render_image = img
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def export(self, fmt: str) -> None:
        if not self.current_dot:
            messagebox.showinfo("Nothing to export", "Load an XML file first.")
            return
        path = filedialog.asksaveasfilename(
            title=f"Export {fmt.upper()}",
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
        )
        if not path:
            return
        try:
            info = render(self.current_dot, self.current_graph, Path(path), fmt=fmt)
            self.status.set(f"Exported -> {info.output_path.name} ({info.backend})")
            if info.backend == "matplotlib" and fmt == "svg":
                messagebox.showinfo(
                    "Saved as PNG",
                    "SVG export needs Graphviz. Saved a PNG instead.",
                )
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def export_dot(self) -> None:
        if not self.current_dot:
            messagebox.showinfo("Nothing to export", "Load an XML file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export DOT", defaultextension=".dot",
            filetypes=[("DOT files", "*.dot")],
        )
        if path:
            Path(path).write_text(self.current_dot, encoding="utf-8")
            self.status.set(f"Exported -> {Path(path).name}")


def main() -> None:
    root = tk.Tk()
    # Slightly nicer default theme where available.
    try:
        style = ttk.Style()
        if platform.system() == "Darwin" and "aqua" in style.theme_names():
            style.theme_use("aqua")
    except tk.TclError:
        pass
    VisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
