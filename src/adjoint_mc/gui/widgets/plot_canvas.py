"""Matplotlib canvas embedded in Tkinter."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class PlotCanvas(ttk.Frame):
    """Plot area with matplotlib navigation toolbar."""

    def __init__(self, master: tk.Misc, *, scrollable: bool = False, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._scrollable = scrollable
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self._pack_canvas()
        self.toolbar.pack(fill=tk.X, side=tk.BOTTOM)

    def _pack_canvas(self) -> None:
        if self._scrollable:
            self.canvas.get_tk_widget().pack(fill=tk.X, expand=False)
        else:
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear(self) -> None:
        self.axes.clear()

    def draw(self) -> None:
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_figure(self, figure: Figure) -> None:
        """Replace the displayed figure (e.g. after HDG physical_overview)."""
        self.canvas.get_tk_widget().pack_forget()
        self.toolbar.pack_forget()
        self.canvas.get_tk_widget().destroy()

        self.figure = figure
        self.axes = figure.axes[0] if figure.axes else figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self._pack_canvas()
        self.toolbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.draw()
