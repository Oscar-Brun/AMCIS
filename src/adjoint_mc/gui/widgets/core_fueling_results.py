"""Core fueling results panel — f_k^flux-focused maps inside the separatrix."""

from __future__ import annotations

import time

from adjoint_mc.gui.widgets.plot_results_panel import PlotResultsPanel
from adjoint_mc.pipeline.core_fueling_run import CoreFuelingRunResult
from adjoint_mc.viz.core_fueling_maps import format_core_fueling_summary_text, iter_core_fueling_plot_tabs


class CoreFuelingResultsPanel(PlotResultsPanel):
    """Core fueling maps: f_k, f_k^flux, S_ion, birth mask, wall hits, Γ_wall, n_n."""

    def show_result(
        self,
        result: CoreFuelingRunResult,
        *,
        plots_s: float | None = None,
        progress_callback=None,
    ) -> None:
        t_plots = time.perf_counter()
        self._last_result = result  # type: ignore[assignment]
        self._close_figures()
        self._clear_plot_tabs()

        plot_tabs = iter_core_fueling_plot_tabs(result)
        for index, (_key, tab_title, figure) in enumerate(plot_tabs):
            if progress_callback is not None:
                progress_callback(f"Plot {index + 1} / {len(plot_tabs)}: {tab_title}")
            self._add_figure_tab(tab_title, figure)

        if plots_s is None:
            plots_s = time.perf_counter() - t_plots

        self._set_info(format_core_fueling_summary_text(result, plots_s=plots_s))

        if self._notebook.tabs():
            self._notebook.select(self._notebook.tabs()[0])
