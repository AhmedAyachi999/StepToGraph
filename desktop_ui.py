from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cadquery as cq
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from step_graph import AnalysisCancelled, EdgeAnalysis, analysis_to_json, classify_step_edges


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "step_datasets"
OUTPUT_ROOT = ROOT / "visualization" / "output" / "desktop"


class DesktopEdgeViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("STEP Convex/Concave Edge Viewer")
        self.geometry("1240x760")
        self.minsize(980, 620)

        self.datasets = _parca_files()
        self.selected_path: Path | None = self.datasets[0] if self.datasets else None
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event: threading.Event | None = None
        self.job_start = 0.0
        self.job_estimate = 1.0
        self.current_analysis: EdgeAnalysis | None = None
        self.current_mesh: tuple[list[list[float]], list[tuple[int, int, int]]] | None = None
        self.view_mode = "solid"

        self._configure_style()
        self._build_layout()
        self._populate_datasets()
        self._draw_empty_view()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#5d6b7f")

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=(14, 14, 10, 14))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(2, weight=1)

        main = ttk.Frame(self, padding=(8, 14, 14, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="STEP Edge Viewer", style="Title.TLabel").grid(row=0, column=0, sticky="ew")
        ttk.Label(
            sidebar,
            text="Select a file, then click Open.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 12))

        dataset_frame = ttk.LabelFrame(sidebar, text="Parça datasets", padding=8)
        dataset_frame.grid(row=2, column=0, sticky="nsew")
        dataset_frame.rowconfigure(0, weight=1)
        dataset_frame.columnconfigure(0, weight=1)

        self.dataset_list = tk.Listbox(
            dataset_frame,
            width=34,
            height=18,
            exportselection=False,
            activestyle="dotbox",
        )
        self.dataset_list.grid(row=0, column=0, sticky="nsew")
        self.dataset_list.bind("<<ListboxSelect>>", self._on_dataset_selected)

        scrollbar = ttk.Scrollbar(dataset_frame, orient="vertical", command=self.dataset_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.dataset_list.configure(yscrollcommand=scrollbar.set)

        control_frame = ttk.Frame(sidebar)
        control_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        self.open_selected_button = ttk.Button(control_frame, text="Open Selected", command=self.open_selected)
        self.open_selected_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.choose_button = ttk.Button(control_frame, text="Choose File", command=self.choose_file)
        self.choose_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.stop_button = ttk.Button(control_frame, text="Stop Loading", command=self.stop_loading, state="disabled")
        self.stop_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        view_frame = ttk.LabelFrame(sidebar, text="View", padding=8)
        view_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)
        self.solid_view_button = ttk.Button(view_frame, text="Solid View", command=self.show_solid_view, state="disabled")
        self.solid_view_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.graph_view_button = ttk.Button(view_frame, text="Graph View", command=self.show_graph_view, state="disabled")
        self.graph_view_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        progress_frame = ttk.LabelFrame(sidebar, text="Loading", padding=8)
        progress_frame.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress_label = ttk.Label(progress_frame, text="Idle", style="Muted.TLabel")
        self.progress_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        metrics = ttk.LabelFrame(sidebar, text="Current result", padding=8)
        metrics.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        metrics.columnconfigure(1, weight=1)
        self.metric_vars = {
            "shared": tk.StringVar(value="-"),
            "convex": tk.StringVar(value="-"),
            "concave": tk.StringVar(value="-"),
            "neutral": tk.StringVar(value="-"),
        }
        self._metric_row(metrics, 0, "Shared", self.metric_vars["shared"])
        self._metric_row(metrics, 1, "Convex", self.metric_vars["convex"])
        self._metric_row(metrics, 2, "Concave", self.metric_vars["concave"])
        self._metric_row(metrics, 3, "Neutral", self.metric_vars["neutral"])

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        self.file_label = ttk.Label(header, text="No file loaded", style="Title.TLabel")
        self.file_label.grid(row=0, column=0, sticky="w")
        self.legend_label = ttk.Label(
            header,
            text="Convex: green   Concave: blue   Solid: translucent gray",
            style="Muted.TLabel",
        )
        self.legend_label.grid(row=1, column=0, sticky="w")

        self.figure = Figure(figsize=(7.6, 5.4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _metric_row(self, parent: ttk.Frame, row: int, label: str, value: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=value, style="Metric.TLabel").grid(row=row, column=1, sticky="e", pady=2)

    def _populate_datasets(self) -> None:
        self.dataset_list.delete(0, tk.END)
        for path in self.datasets:
            self.dataset_list.insert(tk.END, path.name)
        if self.datasets:
            self.dataset_list.selection_set(0)
            self.dataset_list.activate(0)
        else:
            self.open_selected_button.configure(state="disabled")

    def _on_dataset_selected(self, _event: tk.Event[Any]) -> None:
        selection = self.dataset_list.curselection()
        if not selection:
            return
        self.selected_path = self.datasets[selection[0]]

    def open_selected(self) -> None:
        if self.selected_path is None:
            messagebox.showinfo("No dataset", "No Parça dataset is selected.")
            return
        self._start_load(self.selected_path)

    def choose_file(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Open STEP file",
            filetypes=[
                ("STEP files", "*.step *.stp *.STEP *.STP"),
                ("All files", "*.*"),
            ],
        )
        if file_name:
            self._start_load(Path(file_name))

    def stop_loading(self) -> None:
        if not (self.worker and self.worker.is_alive() and self.cancel_event):
            return
        self.cancel_event.set()
        self.stop_button.configure(state="disabled")
        self.progress_label.configure(text="Cancelling after the current geometry step...")

    def _start_load(self, path: Path) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Analysis running", "Wait for the current file to finish loading.")
            return
        if path.suffix.lower() not in {".step", ".stp"}:
            messagebox.showerror("Unsupported file", "Choose a .STEP or .STP file.")
            return

        self.current_analysis = None
        self.current_mesh = None
        self.view_mode = "solid"
        self._clear_result_queue()
        self.cancel_event = threading.Event()
        self.file_label.configure(text=f"Loading {path.name}")
        self._set_buttons("disabled")
        self.stop_button.configure(state="normal")
        self.solid_view_button.configure(state="disabled")
        self.graph_view_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self.job_start = time.monotonic()
        self.job_estimate = _estimate_seconds(path)
        self.progress_label.configure(text=f"Starting. Estimated remaining: {_format_seconds(self.job_estimate)}")
        self._clear_metrics()
        self._draw_empty_view(f"Loading {path.name}")

        self.worker = threading.Thread(
            target=self._worker_load,
            args=(path, self.cancel_event),
            daemon=True,
        )
        self.worker.start()
        self.after(200, self._poll_worker)
        self.after(250, self._tick_progress)

    def _worker_load(self, path: Path, cancel_event: threading.Event) -> None:
        started = time.monotonic()
        try:
            analysis = classify_step_edges(path, cancel_event=cancel_event)
            _raise_if_cancelled(cancel_event)
            mesh = _mesh_step(path)
            _raise_if_cancelled(cancel_event)
            duration = max(0.1, time.monotonic() - started)
            _save_desktop_result(path, analysis, duration)
            _raise_if_cancelled(cancel_event)
            self.result_queue.put(("done", (path, analysis, mesh, duration)))
        except AnalysisCancelled:
            self.result_queue.put(("cancelled", path))
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("error", exc))

    def _poll_worker(self) -> None:
        try:
            kind, payload = self.result_queue.get_nowait()
        except queue.Empty:
            if self.worker and self.worker.is_alive():
                self.after(200, self._poll_worker)
            return

        self._set_buttons("normal")
        self.stop_button.configure(state="disabled")
        self.cancel_event = None
        if kind == "cancelled":
            self.progress_var.set(0.0)
            self.progress_label.configure(text="Cancelled")
            self.file_label.configure(text="No file loaded")
            self._draw_empty_view("Loading cancelled.")
            return

        if kind == "error":
            self.progress_var.set(0.0)
            self.progress_label.configure(text="Failed")
            self.file_label.configure(text="No file loaded")
            messagebox.showerror("Analysis failed", str(payload))
            return

        path, analysis, mesh, duration = payload
        self.current_analysis = analysis
        self.current_mesh = mesh
        self.view_mode = "solid"
        self.progress_var.set(100.0)
        self.progress_label.configure(text=f"Loaded in {_format_seconds(duration)}")
        self.file_label.configure(text=path.name)
        self._show_metrics(analysis)
        self._draw_analysis(analysis, mesh)
        self.solid_view_button.configure(state="normal")
        self.graph_view_button.configure(state="normal")

    def _tick_progress(self) -> None:
        if not (self.worker and self.worker.is_alive()):
            return
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.progress_label.configure(text="Cancelling after the current geometry step...")
            self.after(250, self._tick_progress)
            return
        elapsed = max(0.0, time.monotonic() - self.job_start)
        progress = min(95.0, max(3.0, (elapsed / max(self.job_estimate, 1.0)) * 95.0))
        remaining = max(0.0, self.job_estimate - elapsed)
        self.progress_var.set(progress)
        remaining_text = _format_seconds(remaining) if remaining > 1 else "finishing"
        self.progress_label.configure(
            text=f"Loading. Elapsed: {_format_seconds(elapsed)}. Estimated remaining: {remaining_text}"
        )
        self.after(250, self._tick_progress)

    def _clear_result_queue(self) -> None:
        while True:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                return

    def _set_buttons(self, state: str) -> None:
        self.open_selected_button.configure(state=state if self.datasets else "disabled")
        self.choose_button.configure(state=state)

    def show_solid_view(self) -> None:
        if self.current_analysis is None:
            return
        self.view_mode = "solid"
        self.legend_label.configure(text="Convex: green   Concave: blue   Solid: translucent gray")
        self._draw_analysis(self.current_analysis, self.current_mesh)

    def show_graph_view(self) -> None:
        if self.current_analysis is None:
            return
        self.view_mode = "graph"
        self.legend_label.configure(text="Face graph: convex green, concave blue, neutral gray")
        self._draw_graph(self.current_analysis)

    def _clear_metrics(self) -> None:
        for value in self.metric_vars.values():
            value.set("-")

    def _show_metrics(self, analysis: EdgeAnalysis) -> None:
        self.metric_vars["shared"].set(str(analysis.edge_count))
        self.metric_vars["convex"].set(str(analysis.convex_count))
        self.metric_vars["concave"].set(str(analysis.concave_count))
        self.metric_vars["neutral"].set(str(analysis.neutral_count))

    def _draw_empty_view(self, message: str = "Select a STEP file and click Open Selected.") -> None:
        self._reset_axis("3d")
        self.ax.set_axis_off()
        self.ax.text2D(0.5, 0.5, message, transform=self.ax.transAxes, ha="center", va="center", color="#5d6b7f")
        self.canvas.draw_idle()

    def _draw_analysis(self, analysis: EdgeAnalysis, mesh: tuple[list[list[float]], list[tuple[int, int, int]]] | None) -> None:
        self._reset_axis("3d")
        all_points: list[list[float]] = []

        if mesh is not None:
            vertices, triangles = mesh
            faces = [[vertices[index] for index in triangle] for triangle in triangles]
            collection = Poly3DCollection(
                faces,
                facecolors=(0.62, 0.68, 0.76, 0.24),
                edgecolors=(0.48, 0.54, 0.62, 0.12),
                linewidths=0.15,
            )
            self.ax.add_collection3d(collection)
            all_points.extend(vertices)

        for edge in analysis.visible_edges:
            if len(edge.samples) < 2:
                continue
            xs = [point[0] for point in edge.samples]
            ys = [point[1] for point in edge.samples]
            zs = [point[2] for point in edge.samples]
            color = "#159447" if edge.convexity == "convex" else "#1753d1"
            self.ax.plot(xs, ys, zs, color=color, linewidth=2.8, solid_capstyle="round")
            all_points.extend(edge.samples)

        if all_points:
            _set_equal_axes(self.ax, all_points)
        self.ax.set_axis_off()
        self.ax.view_init(elev=24, azim=-42)
        self.canvas.draw_idle()

    def _draw_graph(self, analysis: EdgeAnalysis) -> None:
        self._reset_axis("2d")
        nodes = sorted({edge.source_face for edge in analysis.edges} | {edge.target_face for edge in analysis.edges})
        if not nodes:
            self.ax.text(0.5, 0.5, "No shared-edge graph to show.", ha="center", va="center", color="#5d6b7f")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        positions = _graph_positions(nodes)
        for index, edge in enumerate(analysis.edges):
            start = positions[edge.source_face]
            end = positions[edge.target_face]
            color = _graph_edge_color(edge.convexity)
            alpha = 0.78 if edge.convexity != "neutral" else 0.38
            width = 1.7 if edge.convexity != "neutral" else 1.0
            _draw_graph_edge(self.ax, start, end, color, width, alpha, index)

        xs = [positions[node][0] for node in nodes]
        ys = [positions[node][1] for node in nodes]
        self.ax.scatter(xs, ys, s=95, c="#ffffff", edgecolors="#263445", linewidths=1.1, zorder=3)
        for node in nodes:
            x, y = positions[node]
            self.ax.text(x, y, str(node), ha="center", va="center", fontsize=7, zorder=4)

        self.ax.text(
            0.01,
            0.99,
            f"Faces: {len(nodes)}   Shared edges: {analysis.edge_count}",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            color="#263445",
            fontsize=9,
        )
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _reset_axis(self, projection: str) -> None:
        self.figure.clear()
        if projection == "3d":
            self.ax = self.figure.add_subplot(111, projection="3d")
        else:
            self.ax = self.figure.add_subplot(111)


def _parca_files() -> list[Path]:
    return sorted(DATASET_DIR.glob("Par*.STEP"), key=_natural_name)


def _natural_name(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled.")


def _mesh_step(path: Path) -> tuple[list[list[float]], list[tuple[int, int, int]]] | None:
    solid = cq.importers.importStep(str(path)).val()
    box = solid.BoundingBox()
    tolerance = max(max(box.xlen, box.ylen, box.zlen) * 0.004, 0.05)
    vertices, triangles = solid.tessellate(tolerance)
    vertex_rows = [[float(value) for value in vertex.toTuple()] for vertex in vertices]
    triangle_rows = [tuple(int(index) for index in triangle) for triangle in triangles]
    return vertex_rows, triangle_rows


def _set_equal_axes(ax: Any, points: list[list[float]]) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5
    radius = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.58
    radius = radius if radius > 0 else 1.0
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    ax.set_zlim(center_z - radius, center_z + radius)


def _graph_positions(nodes: list[int]) -> dict[int, tuple[float, float]]:
    count = max(1, len(nodes))
    if count == 1:
        return {nodes[0]: (0.0, 0.0)}
    radius = 1.0
    return {
        node: (
            radius * math.cos((math.tau * index / count) + math.pi / 2),
            radius * math.sin((math.tau * index / count) + math.pi / 2),
        )
        for index, node in enumerate(nodes)
    }


def _graph_edge_color(convexity: str) -> str:
    if convexity == "convex":
        return "#159447"
    if convexity == "concave":
        return "#1753d1"
    return "#7a8594"


def _draw_graph_edge(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: float,
    alpha: float,
    index: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return

    offset_scale = ((index % 5) - 2) * 0.012
    nx = -dy / length
    ny = dx / length
    ax.plot(
        [x1 + nx * offset_scale, x2 + nx * offset_scale],
        [y1 + ny * offset_scale, y2 + ny * offset_scale],
        color=color,
        linewidth=width,
        alpha=alpha,
        zorder=1,
    )


def _estimate_seconds(path: Path) -> float:
    metadata = _desktop_output_dir(path) / "run_metadata.json"
    if metadata.exists():
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            duration = float(data.get("duration_seconds", 0.0))
            if duration > 0:
                return max(2.0, min(duration * 1.15, 600.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    try:
        size = path.stat().st_size
    except OSError:
        size = 50_000
    return max(5.0, min(size / 1000.0, 300.0))


def _save_desktop_result(path: Path, analysis: EdgeAnalysis, duration_seconds: float) -> None:
    output_dir = _desktop_output_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edges.json").write_text(analysis_to_json(analysis), encoding="utf-8")
    metadata = {
        "source_file": str(path.resolve()),
        "duration_seconds": duration_seconds,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _desktop_output_dir(path: Path) -> Path:
    return OUTPUT_ROOT / _slug(path.stem)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "step_file"


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> int:
    app = DesktopEdgeViewer()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
