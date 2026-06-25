"""Shared plot notebook widget for AMCIS result panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt

from adjoint_mc.gui.widgets.plot_canvas import PlotCanvas


class PlotResultsPanel(ttk.Frame):
    """Summary text area and matplotlib tab notebook."""

    def __init__(self, master: tk.Misc, *, intro_text: str = "", **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        summary_frame = ttk.LabelFrame(self, text="Results", padding=4)
        summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        summary_frame.columnconfigure(0, weight=1)

        self._info = tk.Text(
            summary_frame,
            height=20,
            wrap=tk.WORD,
            font=("TkFixedFont", 10),
            relief=tk.FLAT,
        )
        info_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self._info.yview)
        self._info.configure(yscrollcommand=info_scroll.set)
        self._info.grid(row=0, column=0, sticky="ew")
        info_scroll.grid(row=0, column=1, sticky="ns")

        if intro_text:
            self._info.insert(tk.END, intro_text)
            self._info.configure(state=tk.DISABLED)

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew")
        self._placeholder = ttk.Label(
            self._notebook,
            text="Run the pipeline to display maps.",
            anchor=tk.CENTER,
        )
        self._notebook.add(self._placeholder, text="Waiting")

        self._figures: list = []
        self._plot_canvases: list[PlotCanvas] = []

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
