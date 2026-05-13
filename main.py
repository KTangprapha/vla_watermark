"""VLA Watermark – main entry point.

Usage
-----
  python main.py                     # run all 8 experiments
  python main.py --env vmas          # only VMAS experiments
  python main.py --env libero        # only LIBERO experiments
  python main.py --gen watermark_wrapper
  python main.py --trigger semantic
  python main.py --gif               # also produce trajectory GIFs
  python main.py --quiet             # suppress per-step prints
"""
from __future__ import annotations

import argparse
import sys
import os

# project root on path
sys.path.insert(0, os.path.dirname(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLA Watermark Experiment Suite")
    p.add_argument("--env",     choices=["vmas", "libero", "all"], default="all")
    p.add_argument("--gen",     choices=["watermark_wrapper", "stainlock", "all"], default="all")
    p.add_argument("--trigger", choices=["semantic", "neuro_symbolic", "all"], default="all")
    p.add_argument("--gif",     action="store_true", help="Generate trajectory GIFs")
    p.add_argument("--quiet",   action="store_true", help="Suppress verbose output")
    p.add_argument("--n-clean", type=int, default=20, help="Clean episodes per experiment")
    p.add_argument("--n-wm",    type=int, default=20, help="Watermarked episodes per experiment")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from experiments.run_experiments import (
        EXPERIMENT_MATRIX,
        run_single,
        run_all,
        N_CLEAN,
        N_WM,
        PLOTS_DIR,
        RESULTS_DIR,
        GIF_DIR,
        ATTACK_NOISES,
    )
    import experiments.run_experiments as _exp
    _exp.N_CLEAN = args.n_clean
    _exp.N_WM    = args.n_wm

    # Filter matrix if specific flags given
    matrix = EXPERIMENT_MATRIX
    if args.env != "all":
        matrix = [(e, g, t) for e, g, t in matrix if e == args.env]
    if args.gen != "all":
        matrix = [(e, g, t) for e, g, t in matrix if g == args.gen]
    if args.trigger != "all":
        matrix = [(e, g, t) for e, g, t in matrix if t == args.trigger]

    if not matrix:
        print("No experiments match the given filters.")
        sys.exit(1)

    import os
    import json
    import time
    from visualization.visualizer import Visualizer, save_results_table

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

    viz = Visualizer(output_dir=PLOTS_DIR)
    all_summaries = []
    verbose = not args.quiet

    print(f"Running {len(matrix)} experiment(s)…\n")

    for env_name, gen_method, trigger_type in matrix:
        t0 = time.time()
        result, clean_pos, wm_pos, goals = run_single(
            env_name, gen_method, trigger_type, verbose=verbose
        )
        elapsed = time.time() - t0
        label = f"{env_name}__{gen_method}__{trigger_type}"

        saved = viz.plot_all(
            label=label,
            clean_positions=clean_pos,
            wm_positions=wm_pos,
            goals=goals,
            detection_result=result.detection,
            robustness=result.robustness,
            attack_noises=ATTACK_NOISES,
            make_gif=args.gif,
        )
        if verbose:
            print(f"  → {len(saved)} plots saved.")

        summary = result.summary()
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries.append(summary)

    # Final table + JSON
    table_path = os.path.join(RESULTS_DIR, "results_table.md")
    save_results_table(all_summaries, out_path=table_path, print_table=True)

    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll done.  Results → {RESULTS_DIR}/")
    print(f"Plots     → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
