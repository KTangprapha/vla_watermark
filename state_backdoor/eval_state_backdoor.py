#!/usr/bin/env python3
"""Evaluate State Backdoor SR/ASR on LIBERO-Goal with OpenVLA-OFT.

Runs openvla-oft's own `run_libero_eval.py` twice against the *same* set
of demos/tasks:

  1. Clean pass  -- `clean_initial_states.json` (unperturbed initial
     states) -> Success Rate (SR), matching Table III/IV's "SR(%)".
  2. Triggered pass -- `triggered_initial_states.json` (PGA trigger
     applied to the robot's initial qpos) -> triggered success rate.
     Because State Backdoor is an *untargeted* attack whose goal is
     simply to make the triggered episode fail (Section IV, "Attack
     Goal: Effectiveness ... triggered samples should reliably cause
     task failure"), Attack Success Rate is:

         ASR = 1 - success_rate(triggered pass)

Both JSON files are produced by `poison_libero_hdf5.py` and share the
exact same task/demo keys, so SR and ASR are computed over an identical
sample of initial conditions (only the robot's qpos differs).

This script only orchestrates the *evaluation* -- it assumes you already
have:
  - `third_party/openvla-oft` set up per its SETUP.md/LIBERO.md
    (conda env, LIBERO + robosuite installed)
  - a fine-tuned (backdoored) OpenVLA-OFT checkpoint, produced by
    `scripts/05_finetune_lora.sh` on the poisoned dataset

and must be run on a GPU machine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENVLA_OFT_DIR = REPO_ROOT / "third_party" / "openvla-oft"

SUCCESS_RATE_RE = re.compile(r"Overall success rate:\s*([0-9.]+)\s*\(")


def _run_eval_pass(
    checkpoint: str,
    task_suite_name: str,
    initial_states_path: str,
    extra_args: list,
    run_id_note: str,
) -> float:
    cmd = [
        sys.executable,
        "experiments/robot/libero/run_libero_eval.py",
        f"--pretrained_checkpoint={checkpoint}",
        f"--task_suite_name={task_suite_name}",
        f"--initial_states_path={initial_states_path}",
        f"--run_id_note={run_id_note}",
        *extra_args,
    ]
    print(f"\n$ (cd {OPENVLA_OFT_DIR} && {' '.join(cmd)})\n")
    proc = subprocess.run(cmd, cwd=str(OPENVLA_OFT_DIR), capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"run_libero_eval.py failed (exit {proc.returncode}); see stderr above")

    match = None
    for match in SUCCESS_RATE_RE.finditer(proc.stdout):
        pass  # keep the last match (final "Overall results" summary)
    if match is None:
        raise RuntimeError("Could not parse 'Overall success rate' from run_libero_eval.py output")
    return float(match.group(1))


def evaluate(
    checkpoint: str,
    task_suite_name: str,
    clean_states_path: str,
    triggered_states_path: str,
    extra_args: Optional[list] = None,
) -> Tuple[float, float]:
    extra_args = extra_args or []

    sr = _run_eval_pass(
        checkpoint, task_suite_name, clean_states_path, extra_args, run_id_note="state_backdoor_clean",
    )
    triggered_success_rate = _run_eval_pass(
        checkpoint, task_suite_name, triggered_states_path, extra_args, run_id_note="state_backdoor_triggered",
    )
    asr = 1.0 - triggered_success_rate
    return sr, asr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained_checkpoint", required=True, help="Path or HF repo id of the (backdoored) OpenVLA-OFT checkpoint")
    p.add_argument("--task_suite_name", default="libero_goal")
    p.add_argument("--poisoned_data_dir", required=True,
                   help="Dir produced by poison_libero_hdf5.py containing "
                        "clean_initial_states.json / triggered_initial_states.json")
    p.add_argument("--out_json", default=None, help="Where to write the {sr, asr} summary JSON")
    p.add_argument("--extra_arg", action="append", default=[],
                   help="Extra CLI arg forwarded verbatim to run_libero_eval.py "
                        "(e.g. --extra_arg=--center_crop=True), can repeat")
    args = p.parse_args()

    clean_path = os.path.join(args.poisoned_data_dir, "clean_initial_states.json")
    triggered_path = os.path.join(args.poisoned_data_dir, "triggered_initial_states.json")
    for path in (clean_path, triggered_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run poison_libero_hdf5.py first")

    sr, asr = evaluate(
        args.pretrained_checkpoint, args.task_suite_name, clean_path, triggered_path, args.extra_arg,
    )

    result = {
        "task_suite": args.task_suite_name,
        "checkpoint": args.pretrained_checkpoint,
        "sr_percent": round(sr * 100, 1),
        "asr_percent": round(asr * 100, 1),
    }
    print("\n=== State Backdoor evaluation (paper Table III/IV metrics) ===")
    print(json.dumps(result, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
