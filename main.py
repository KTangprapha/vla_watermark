"""VLA Watermark – main entry point.

Usage
-----
  python main.py                     # all 8 experiments
  python main.py --stage 1           # VMAS wrapper experiments only
  python main.py --stage 2           # LIBERO wrapper experiments
  python main.py --stage 3           # StainLock experiments
  python main.py --sim robot         # 7-DOF Franka Panda robot arm experiments
  python main.py --env vmas          # filter by environment
  python main.py --gen stainlock     # filter by generation method
  python main.py --trigger semantic  # filter by trigger type
  python main.py --n 10              # episodes per case (default 20)
  python main.py --gif               # also produce trajectory GIFs
  python main.py --quiet             # suppress verbose output

Paper stages
  Stage 1: VMAS + wrapper watermark + both triggers  (proof of concept)
  Stage 2: LIBERO + wrapper watermark + both triggers (real VLA benchmark)
  Stage 3: StainLock + both envs + both triggers      (advanced method)
  Robot:   7-DOF Franka Panda PyBullet simulation (4 experiments with videos)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLA Watermark Experiment Suite")
    p.add_argument("--stage",   type=int, choices=[1, 2, 3],
                   help="Run only a specific paper stage")
    p.add_argument("--sim",     choices=["robot"],
                   help="Run robot arm (PyBullet) simulation experiments")
    p.add_argument("--env",     choices=["vmas", "libero"])
    p.add_argument("--gen",     choices=["watermark_wrapper", "stainlock"])
    p.add_argument("--trigger", choices=["semantic", "neuro_symbolic"])
    p.add_argument("--n",       type=int, default=20, dest="n_episodes",
                   help="Episodes per case (default 20)")
    p.add_argument("--gif",     action="store_true")
    p.add_argument("--quiet",   action="store_true")
    return p.parse_args()


_STAGE_MATRIX = {
    1: [("vmas",   "watermark_wrapper", "semantic"),
        ("vmas",   "watermark_wrapper", "neuro_symbolic")],
    2: [("libero", "watermark_wrapper", "semantic"),
        ("libero", "watermark_wrapper", "neuro_symbolic")],
    3: [("vmas",   "stainlock",         "semantic"),
        ("vmas",   "stainlock",         "neuro_symbolic"),
        ("libero", "stainlock",         "semantic"),
        ("libero", "stainlock",         "neuro_symbolic")],
}


def main() -> None:
    args = _parse()

    # Robot arm simulation path
    if getattr(args, "sim", None) == "robot":
        from experiments.run_robot_arm import run_all_robot_arm
        run_all_robot_arm(verbose=not args.quiet, make_video=True)
        return

    import experiments.run_experiments as _exp
    _exp.N_EPISODES = args.n_episodes

    from experiments.run_experiments import (
        EXPERIMENT_MATRIX, run_single, PLOTS_DIR, RESULTS_DIR, GIF_DIR,
    )
    from visualization.visualizer import Visualizer, save_results_table

    # Build experiment list
    if args.stage:
        matrix = _STAGE_MATRIX[args.stage]
    else:
        matrix = list(EXPERIMENT_MATRIX)

    if args.env:
        matrix = [(e, g, t) for e, g, t in matrix if e == args.env]
    if args.gen:
        matrix = [(e, g, t) for e, g, t in matrix if g == args.gen]
    if args.trigger:
        matrix = [(e, g, t) for e, g, t in matrix if t == args.trigger]

    if not matrix:
        print("No experiments match the given filters."); sys.exit(1)

    for d in [PLOTS_DIR, RESULTS_DIR, GIF_DIR]:
        os.makedirs(d, exist_ok=True)

    verbose = not args.quiet
    print(f"Running {len(matrix)} experiment(s) "
          f"({args.n_episodes} episodes/case each)…\n")

    viz = Visualizer(output_dir=PLOTS_DIR)
    all_summaries = []

    for env_name, gen_method, trigger_type in matrix:
        t0 = time.time()
        result, clean_pos, wm_pos, goals = run_single(
            env_name, gen_method, trigger_type, verbose=verbose
        )
        elapsed = time.time() - t0
        label   = f"{env_name}__{gen_method}__{trigger_type}"

        saved = viz.plot_all(
            label=label,
            clean_positions=clean_pos,
            wm_positions=wm_pos,
            goals=goals,
            detection_result=result,
            env_name=env_name,
            make_gif=args.gif,
        )
        if verbose:
            print(f"  → {len(saved)} plots saved.")

        summary = result.summary()
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries.append(summary)

    table_path = os.path.join(RESULTS_DIR, "results_table.md")
    save_results_table(all_summaries, out_path=table_path, print_table=True)

    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nDone.  Results → {RESULTS_DIR}/  |  Plots → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
