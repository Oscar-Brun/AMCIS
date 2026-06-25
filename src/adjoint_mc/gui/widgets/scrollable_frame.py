"""Tkinter container with vertical scrollbar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Scrollable inner frame; width tracks the parent canvas."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._vscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vscroll.set)

        self._vscroll.grid(row=0, column=1, sticky="ns")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self.inner = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._bind_mousewheel(self._canvas)
        self._bind_mousewheel(self.inner)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window_id, width=event.width)

    def refresh(self) -> None:
        """Recompute scroll region after content is added."""
        self.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def scroll_to_top(self) -> None:
        self._canvas.yview_moveto(0.0)

    def _bind_mousewheel(self, widget: tk.Misc) -> None:
        widget.bind("<Enter>", self._activate_mousewheel, add="+")
        widget.bind("<Leave>", self._deactivate_mousewheel, add="+")

    def _activate_mousewheel(self, _event: tk.Event) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self._canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self._canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _deactivate_mousewheel(self, _event: tk.Event) -> None:
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 5 or getattr(event, "delta", 0) < 0:
            self._canvas.yview_scroll(3, "units")
        elif event.num == 4 or getattr(event, "delta", 0) > 0:
            self._canvas.yview_scroll(-3, "units")
