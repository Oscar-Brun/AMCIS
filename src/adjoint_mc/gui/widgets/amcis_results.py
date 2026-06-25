"""AMCIS result display — summary, wall map, target context, CSV export."""

from __future__ import annotations

import time
from collections.abc import Callable
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt

from adjoint_mc.gui.widgets.plot_canvas import PlotCanvas
from adjoint_mc.pipeline.amcis_run import AmcisRunResult
from adjoint_mc.scoring.amcis_provenance import export_amcis_csv, format_amcis_summary_text
from adjoint_mc.viz.amcis_maps import iter_amcis_plot_tabs
from adjoint_mc.viz.amcis_mc_plots import iter_amcis_mc_plot_tabs


class AmcisResultsPanel(ttk.Frame):
    """Summary text, plot notebook, CSV export for an AMCIS run."""

    def __init__(self, master: tk.Misc, *, intro_text: str = "", **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        summary_frame = ttk.LabelFrame(self, text="Results", padding=4)
        summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        summary_frame.columnconfigure(0, weight=1)

        self._info = tk.Text(
            summary_frame,
            height=16,
            wrap=tk.WORD,
            font=("TkFixedFont", 10),
            relief=tk.FLAT,
        )
        info_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self._info.yview)
        self._info.configure(yscrollcommand=info_scroll.set)
        self._info.grid(row=0, column=0, sticky="ew")
        info_scroll.grid(row=0, column=1, sticky="ns")

        btn_row = ttk.Frame(summary_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(btn_row, text="Export CSV…", command=self._export_csv).pack(side=tk.LEFT)

        if intro_text:
            self._info.insert(tk.END, intro_text)
            self._info.configure(state=tk.DISABLED)

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew")
        self._placeholder = ttk.Label(
            self._notebook,
            text="Set target (R, Z) and run AMCIS to display wall provenance maps.",
            anchor=tk.CENTER,
        )
        self._notebook.add(self._placeholder, text="Waiting")

        self._figures: list = []
        self._plot_canvases: list[PlotCanvas] = []
        self._last_result: AmcisRunResult | None = None

    def _set_info(self, text: str) -> None:
        self._info.configure(state=tk.NORMAL)
        self._info.delete("1.0", tk.END)
        self._info.insert(tk.END, text)
        self._info.configure(state=tk.DISABLED)

    def _close_figures(self) -> None:
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()

    def _clear_plot_tabs(self) -> None:
        for tab_id in self._notebook.tabs():
            self._notebook.forget(tab_id)
        for canvas in self._plot_canvases:
            canvas.destroy()
        self._plot_canvases.clear()

    def _add_figure_tab(self, title: str, figure) -> None:
        tab = ttk.Frame(self._notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self._notebook.add(tab, text=title)
        plot = PlotCanvas(tab)
        plot.grid(row=0, column=0, sticky="nsew")
        plot.set_figure(figure)
        self._plot_canvases.append(plot)
        self._figures.append(figure)

    def show_result(
        self,
        result: AmcisRunResult,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        self._last_result = result
        self._close_figures()
        self._clear_plot_tabs()

        for tab_title, figure in iter_amcis_plot_tabs(result):
            if progress_callback is not None:
                progress_callback(f"Generating plot: {tab_title}")
            self._add_figure_tab(tab_title, figure)
            self.update_idletasks()

        for tab_title, figure in iter_amcis_mc_plot_tabs(result.wall, result.mc_result):
            if progress_callback is not None:
                progress_callback(f"Generating plot: MC {tab_title}")
            self._add_figure_tab(f"MC: {tab_title}", figure)
            self.update_idletasks()

        plots_s = time.perf_counter() - t0
        timing = result.timing
        self._set_info(
            format_amcis_summary_text(
                result.mc_result,
                result.provenance,
                mc_s=timing.mc_s,
                total_s=timing.total_s + plots_s,
                mc_engine=timing.mc_engine,
            )
        )
        if self._notebook.tabs():
            self._notebook.select(self._notebook.tabs()[0])

    def _export_csv(self) -> None:
        if self._last_result is None:
            messagebox.showinfo("Export", "Run AMCIS first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export AMCIS provenance CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="amcis_provenance.csv",
        )
        if not path:
            return
        try:
            export_amcis_csv(path, self._last_result.provenance, self._last_result.wall)
            stem = Path(path).with_suffix("")
            messagebox.showinfo(
                "Export",
                f"Written:\n{stem}_regions.csv\n{stem}_segments.csv",
            )
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
