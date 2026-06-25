"""AMCIS GUI — point-target provenance and core fueling (shared solution / MC params)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from adjoint_mc.config import (
    DEFAULT_GRID_N_R,
    DEFAULT_GRID_N_Z,
    DEFAULT_MAX_PATH_M,
    DEFAULT_N_HISTORIES,
    DEFAULT_NEUTRAL_SPEED_M_S,
    DEFAULT_SEED,
    DEFAULT_SOLUTION_PATH,
    DEFAULT_TAU_MAX,
    DEFAULT_VACUUM_WALL_SEARCH_M,
)
from adjoint_mc.gui.widgets.amcis_results import AmcisResultsPanel
from adjoint_mc.gui.widgets.core_fueling_results import CoreFuelingResultsPanel
from adjoint_mc.pipeline.amcis_run import AmcisRunConfig, run_amcis_pipeline
from adjoint_mc.pipeline.core_fueling_run import CoreFuelingRunConfig, run_core_fueling_pipeline
from adjoint_mc.tracker.backward_cython import cython_available, default_cython_thread_count


class AmcisWindow(ttk.Frame):
    """AMCIS target provenance and core fueling in one window (shared run parameters)."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=8)
        master.title("AMCIS — wall provenance & core fueling")
        master.minsize(980, 720)
        self.pack(fill=tk.BOTH, expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self._solution_var = tk.StringVar(value=str(DEFAULT_SOLUTION_PATH))
        self._target_r_var = tk.DoubleVar(value=0.65)
        self._target_z_var = tk.DoubleVar(value=0.0)
        self._n_hist_var = tk.IntVar(value=DEFAULT_N_HISTORIES)
        self._seed_var = tk.IntVar(value=DEFAULT_SEED)
        self._grid_nr_var = tk.IntVar(value=DEFAULT_GRID_N_R)
        self._grid_nz_var = tk.IntVar(value=DEFAULT_GRID_N_Z)
        self._tau_max_var = tk.DoubleVar(value=DEFAULT_TAU_MAX)
        self._neutral_speed_var = tk.DoubleVar(value=DEFAULT_NEUTRAL_SPEED_M_S)
        self._max_path_var = tk.DoubleVar(value=DEFAULT_MAX_PATH_M)
        self._vacuum_wall_search_var = tk.DoubleVar(value=DEFAULT_VACUUM_WALL_SEARCH_M)
        self._enable_cx_var = tk.BooleanVar(value=True)
        self._advanced_visible = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="Ready.")

        self._build_shared_controls()
        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._progress.grid_remove()

        self._mode_notebook = ttk.Notebook(self)
        self._mode_notebook.grid(row=2, column=0, sticky="nsew")
        self._build_amcis_tab()
        self._build_core_fueling_tab()

        status = ttk.Label(self, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_shared_controls(self) -> None:
        frame = ttk.LabelFrame(self, text="Shared parameters", padding=8)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="SOLEDGE .h5").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self._solution_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse…", command=self._browse_solution).grid(row=0, column=2)

        ttk.Label(frame, text="Histories").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=1_000_000, textvariable=self._n_hist_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=6, pady=(6, 0)
        )

        ttk.Checkbutton(
            frame,
            text="Advanced options",
            variable=self._advanced_visible,
            command=self._toggle_advanced,
        ).grid(row=1, column=2, sticky=tk.W, pady=(6, 0))

        self._advanced = ttk.Frame(frame)
        self._advanced.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for col, (label, var) in enumerate(
            [
                ("Seed", self._seed_var),
                ("Grid n_r", self._grid_nr_var),
                ("Grid n_z", self._grid_nz_var),
                ("tau_max", self._tau_max_var),
                ("v_n [m/s]", self._neutral_speed_var),
                ("max path [m]", self._max_path_var),
                ("vacuum [m]", self._vacuum_wall_search_var),
            ]
        ):
            ttk.Label(self._advanced, text=label).grid(row=0, column=2 * col, sticky=tk.W, padx=(0, 4))
            ttk.Entry(self._advanced, textvariable=var, width=8).grid(row=0, column=2 * col + 1, padx=(0, 8))
        self._advanced.grid_remove()

        cython_line = (
            f"Cython MC: {'built' if cython_available() else 'NOT BUILT — pip install -e .'}\n"
            f"OpenMP threads: {default_cython_thread_count() if cython_available() else '—'}"
        )
        ttk.Label(frame, text=cython_line, justify=tk.LEFT).grid(
            row=3, column=0, columnspan=3, sticky=tk.W, pady=(8, 0)
        )

    def _build_amcis_tab(self) -> None:
        tab = ttk.Frame(self._mode_notebook, padding=4)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        self._mode_notebook.add(tab, text="AMCIS → Ω")

        controls = ttk.LabelFrame(tab, text="Point target", padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Target R [m]").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(controls, textvariable=self._target_r_var, width=12).grid(
            row=0, column=1, sticky=tk.W, padx=6
        )
        ttk.Label(controls, text="Target Z [m]").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        ttk.Entry(controls, textvariable=self._target_z_var, width=12).grid(
            row=1, column=1, sticky=tk.W, padx=6, pady=(4, 0)
        )
        ttk.Checkbutton(
            controls,
            text="Enable CX (rejection)",
            variable=self._enable_cx_var,
        ).grid(row=0, column=2, rowspan=2, sticky=tk.W, padx=(12, 0))
        ttk.Button(controls, text="Run AMCIS", command=self._on_run_amcis).grid(
            row=0, column=3, rowspan=2, padx=(12, 0)
        )

        intro = (
            "AMCIS — wall provenance to a single plasma target (R, Z)\n\n"
            "All histories born at Ω; survival weight W = exp(-∫ Σ_ion ds) along the backward path.\n"
            "CX: rejection on σ(E_rel), velocity updated, W unchanged.\n"
            "Output: f_k(Ω) and visibility C_k on wall segments (not SOLEDGE neutral_flux).\n"
            "After a run: HDG maps for n_n, S_ion, Γ_n,⊥ (volume) + |Γ_n,⊥| on the wall.\n"
        )
        self._amcis_results = AmcisResultsPanel(tab, intro_text=intro)
        self._amcis_results.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def _build_core_fueling_tab(self) -> None:
        tab = ttk.Frame(self._mode_notebook, padding=4)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        self._mode_notebook.add(tab, text="Core fueling")

        controls = ttk.LabelFrame(tab, text="Core ionization births", padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(
            controls,
            text=(
                "Backward MC with births weighted by S_ion × dV, "
                "restricted to plasma cells inside the separatrix (ψ = 1).\n"
                "Answers: which wall regions supply neutrals that fuel core ionization?"
            ),
            wraplength=900,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(controls, text="Run core fueling", command=self._on_run_core_fueling).grid(
            row=0, column=1, padx=(12, 0)
        )

        intro = (
            "Core fueling provenance\n\n"
            "Backward MC with S_ion-weighted births inside the separatrix (ψ = 1).\n"
            "Primary result: f_k^flux = C_k Γ_k^wall / Σ C_j Γ_j^wall (% of effective wall supply).\n"
            "Also shown: f_k (connectivity only), S_ion, birth mask, trajectories, wall |Γ_n,⊥|, n_n.\n"
        )
        self._core_results = CoreFuelingResultsPanel(tab, intro_text=intro)
        self._core_results.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def _toggle_advanced(self) -> None:
        if self._advanced_visible.get():
            self._advanced.grid()
        else:
            self._advanced.grid_remove()

    def _browse_solution(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a SOLEDGE-HDG solution",
            filetypes=[("HDF5", "*.h5"), ("All files", "*.*")],
        )
        if path:
            self._solution_var.set(path)

    def _shared_mc_kwargs(self) -> dict:
        return {
            "n_histories": int(self._n_hist_var.get()),
            "seed": int(self._seed_var.get()),
            "grid_n_r": int(self._grid_nr_var.get()),
            "grid_n_z": int(self._grid_nz_var.get()),
            "tau_max": float(self._tau_max_var.get()),
            "neutral_speed_m_s": float(self._neutral_speed_var.get()),
            "max_path_m": float(self._max_path_var.get()),
            "vacuum_wall_search_m": float(self._vacuum_wall_search_var.get()),
        }

    def _amcis_config(self) -> AmcisRunConfig:
        return AmcisRunConfig(
            solution_path=Path(self._solution_var.get()),
            target_r=float(self._target_r_var.get()),
            target_z=float(self._target_z_var.get()),
            enable_cx=bool(self._enable_cx_var.get()),
            **self._shared_mc_kwargs(),
        )

    def _core_fueling_config(self) -> CoreFuelingRunConfig:
        return CoreFuelingRunConfig(
            solution_path=Path(self._solution_var.get()),
            **self._shared_mc_kwargs(),
        )

    def _make_progress_callback(self, total: int, label: str = "MC"):
        self._progress.grid()
        self._progress.configure(maximum=max(total, 1), value=0)

        def callback(completed: int, n_total: int) -> None:
            self._status_var.set(f"{label} progress: {completed} / {n_total} histories")
            self._progress.configure(maximum=max(n_total, 1), value=completed)
            self.update_idletasks()

        return callback

    def _hide_progress(self) -> None:
        self._progress.grid_remove()
        self._progress.configure(value=0)

    def _on_run_amcis(self) -> None:
        config = self._amcis_config()
        if not config.solution_path.exists():
            messagebox.showerror("File not found", f"Solution not found:\n{config.solution_path}")
            return

        self._status_var.set("Running AMCIS…")
        self.update_idletasks()
        progress = self._make_progress_callback(config.n_histories, label="AMCIS")

        try:
            result = run_amcis_pipeline(config, progress_callback=progress)
            self._status_var.set("Generating AMCIS plots…")
            self.update_idletasks()

            def plot_progress(message: str) -> None:
                self._status_var.set(message)
                self.update_idletasks()

            self._amcis_results.show_result(result, progress_callback=plot_progress)
            self._hide_progress()
            self._status_var.set(
                f"AMCIS done in {result.timing.total_s:.1f} s — "
                f"{result.provenance.n_wall_hits} wall hits / {config.n_histories} histories "
                f"at Ω=({config.target_r:.3f}, {config.target_z:.3f}) m."
            )
        except Exception as exc:
            self._hide_progress()
            messagebox.showerror("Error", str(exc))
            self._status_var.set(f"Error: {exc}")

    def _on_run_core_fueling(self) -> None:
        config = self._core_fueling_config()
        if not config.solution_path.exists():
            messagebox.showerror("File not found", f"Solution not found:\n{config.solution_path}")
            return

        self._status_var.set("Running core fueling MC…")
        self.update_idletasks()
        progress = self._make_progress_callback(config.n_histories, label="Core fueling")

        try:
            result = run_core_fueling_pipeline(config, progress_callback=progress)
            self._status_var.set("Generating core fueling plots…")
            self.update_idletasks()

            def plot_progress(message: str) -> None:
                self._status_var.set(message)
                self.update_idletasks()

            self._core_results.show_result(result, progress_callback=plot_progress)
            self._hide_progress()
            self._status_var.set(
                f"Core fueling done in {result.timing.total_s:.1f} s — "
                f"{result.provenance.n_wall_hits} wall hits / {config.n_histories} core births "
                f"(∫ S_ion dV = {result.core_fueling_rate_s:.3e} s⁻¹)."
            )
        except Exception as exc:
            self._hide_progress()
            messagebox.showerror("Error", str(exc))
            self._status_var.set(f"Error: {exc}")


def launch_amcis_app() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    AmcisWindow(root)
    root.mainloop()
