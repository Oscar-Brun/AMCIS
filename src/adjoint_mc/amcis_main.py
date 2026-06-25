"""AMCIS entry point — single-point wall provenance with survival weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adjoint_mc.config import (
    DEFAULT_GRID_N_R,
    DEFAULT_GRID_N_Z,
    DEFAULT_MAX_PATH_M,
    DEFAULT_N_HISTORIES,
    DEFAULT_NEUTRAL_SPEED_M_S,
    DEFAULT_SEED,
    DEFAULT_TAU_MAX,
    DEFAULT_VACUUM_WALL_SEARCH_M,
)
from adjoint_mc.pipeline.amcis_run import AmcisRunConfig, run_amcis_pipeline
from adjoint_mc.scoring.amcis_provenance import export_amcis_csv, format_amcis_summary_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amcis",
        description=(
            "AMCIS — wall provenance to a single plasma target (R, Z). "
            "Survival weights W=exp(-∫Σ_ion ds), CX rejection unchanged. "
            "Launch without batch options to open the GUI."
        ),
    )
    parser.add_argument(
        "--solution",
        type=Path,
        default=None,
        help="SOLEDGE-HDG .h5 file (batch mode)",
    )
    parser.add_argument("--target-r", type=float, default=None, help="Target R [m] (batch)")
    parser.add_argument("--target-z", type=float, default=None, help="Target Z [m] (batch)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/amcis"),
        help="Directory for CSV export and optional plots (batch)",
    )
    parser.add_argument("--n-histories", type=int, default=DEFAULT_N_HISTORIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--grid-n-r", type=int, default=DEFAULT_GRID_N_R)
    parser.add_argument("--grid-n-z", type=int, default=DEFAULT_GRID_N_Z)
    parser.add_argument("--tau-max", type=float, default=DEFAULT_TAU_MAX)
    parser.add_argument("--neutral-speed", type=float, default=DEFAULT_NEUTRAL_SPEED_M_S)
    parser.add_argument("--max-path", type=float, default=DEFAULT_MAX_PATH_M)
    parser.add_argument("--vacuum-search", type=float, default=DEFAULT_VACUUM_WALL_SEARCH_M)
    parser.add_argument(
        "--no-cx",
        action="store_true",
        help="Disable charge-exchange (ionization-only attenuation)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save PNG wall map and target context figure (batch)",
    )
    return parser


def _is_batch_mode(args: argparse.Namespace) -> bool:
    return args.solution is not None or args.target_r is not None or args.target_z is not None


def run_batch(args: argparse.Namespace) -> int:
    if args.solution is None or args.target_r is None or args.target_z is None:
        print(
            "Batch mode requires --solution, --target-r and --target-z "
            "(or run `amcis` with no arguments for the GUI).",
            file=sys.stderr,
        )
        return 2

    solution = args.solution
    if not solution.is_file():
        print(f"Solution not found: {solution}", file=sys.stderr)
        return 1

    config = AmcisRunConfig(
        solution_path=solution,
        target_r=float(args.target_r),
        target_z=float(args.target_z),
        n_histories=int(args.n_histories),
        seed=int(args.seed),
        grid_n_r=int(args.grid_n_r),
        grid_n_z=int(args.grid_n_z),
        tau_max=float(args.tau_max),
        neutral_speed_m_s=float(args.neutral_speed),
        max_path_m=float(args.max_path),
        vacuum_wall_search_m=float(args.vacuum_search),
        enable_cx=not args.no_cx,
    )

    print(f"AMCIS run: {solution}")
    print(
        f"  target=({config.target_r:.4f}, {config.target_z:.4f}) m"
        f"  histories={config.n_histories}  seed={config.seed}"
        f"  CX={'off' if not config.enable_cx else 'on'}"
    )

    result = run_amcis_pipeline(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = args.output_dir / "amcis_provenance.csv"
    export_amcis_csv(csv_stem, result.provenance, result.wall)

    print(
        format_amcis_summary_text(
            result.mc_result,
            result.provenance,
            mc_s=result.timing.mc_s,
            total_s=result.timing.total_s,
            mc_engine=result.timing.mc_engine,
        )
    )
    print("\nExported:")
    print(f"  regions: {csv_stem.with_suffix('')}_regions.csv")
    print(f"  segments: {csv_stem.with_suffix('')}_segments.csv")

    if args.plot:
        import re

        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        from adjoint_mc.viz.amcis_maps import iter_amcis_plot_tabs
        from adjoint_mc.viz.amcis_mc_plots import iter_amcis_mc_plot_tabs

        for tab_title, figure in iter_amcis_plot_tabs(result):
            slug = re.sub(r"[^\w]+", "_", tab_title.strip()).strip("_").lower()
            out_path = args.output_dir / f"amcis_{slug or 'plot'}.png"
            figure.savefig(out_path, dpi=150)
            print(f"  plot: {out_path}")
            plt.close(figure)

        for tab_title, figure in iter_amcis_mc_plot_tabs(result.wall, result.mc_result):
            slug = re.sub(r"[^\w]+", "_", f"mc_{tab_title}".strip()).strip("_").lower()
            out_path = args.output_dir / f"amcis_{slug or 'mc_plot'}.png"
            figure.savefig(out_path, dpi=150)
            print(f"  plot: {out_path}")
            plt.close(figure)

    return 0


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if _is_batch_mode(args):
        raise SystemExit(run_batch(args))
    from adjoint_mc.gui.amcis_app import launch_amcis_app

    launch_amcis_app()


if __name__ == "__main__":
    main()
