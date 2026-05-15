"""LIBERO + OpenVLA — Key-based Watermarking Experiment.

Implements the full 5-step pipeline for LIBERO (7-DOF manipulation) only:

  Step 1: K = KDF(user_id, salt)
  Step 2: K → (T_text, T_visual) trigger candidates via seeded RNG
  Step 3: Filter triggers with LIBERO environment (semantic + rarity checks)
  Step 4: M' = WatermarkWrapper(M, SemanticTrigger, signature_S)
          OR   StainLockPolicy(M, rank-1 weight perturbation)
  Step 5: KeyBasedDetector(K) → cosine correlation → z-score → ownership proof

Usage
-----
  python experiments/libero_openvla.py                      # full run, both methods
  python experiments/libero_openvla.py --method wrapper     # action-level only
  python experiments/libero_openvla.py --method stainlock   # weight-level only
  python experiments/libero_openvla.py --user user_002      # different owner key
  python experiments/libero_openvla.py --episodes 30        # shorter run

Output
------
  results/libero_openvla_wrapper.json
  results/libero_openvla_stainlock.json
  results/libero_openvla_detection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- watermark package (Steps 1-5) ---
from watermark.key_manager import KeyManager, KeyBundle
from watermark.trigger_generator import TriggerGenerator, EnvironmentFilter, WatermarkTrigger
from watermark.signature import SignaturePattern
from watermark.watermark_engine import build_watermark_wrapper, build_stainlock
from watermark.detector import KeyBasedDetector

# --- model + environment ---
from generation.openvla_adapter import get_or_create_openvla, get_openvla_config


# ---------------------------------------------------------------------------
# LIBERO environment adapter
# ---------------------------------------------------------------------------

def _make_libero_env(seed: int = 0):
    """Load LiberoAdapter, fall back to a minimal stub if unavailable."""
    try:
        from environments.libero_adapter import LiberoAdapter
        return LiberoAdapter(seed=seed, render_visual=True)
    except Exception as e:
        print(f"  [warn] LiberoAdapter unavailable ({e}), using stub environment")
        return _LiberoStub(seed=seed)


class _LiberoStub:
    """Minimal LIBERO stub for unit-testing when the real environment is absent."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.action_dim = 7

    def rollout(self, policy, trigger_active: bool = False, instruction: str = "",
                seed_override: int | None = None, **_) -> dict:
        T = 30
        frames = []
        actions = []
        for t in range(T):
            frame = self.rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            obs = {
                "visual":      frame,
                "instruction": instruction,
                "joints":      self.rng.uniform(-0.5, 0.5, 7).astype(np.float32),
            }
            act = policy(obs)
            frames.append(frame)
            actions.append(act)
        return {"frames": frames, "actions": actions, "success": False}


# ---------------------------------------------------------------------------
# Step 1: Key Generation
# ---------------------------------------------------------------------------

def step1_key_generation(user_id: str, salt: bytes | None = None) -> KeyBundle:
    print("\n[Step 1] Key Generation")
    print(f"  user_id = {user_id!r}")
    bundle = KeyManager.generate(user_id, salt=salt)
    print(f"  salt         = {bundle.salt.hex()}")
    print(f"  master_key   = {bundle.key_hex[:16]}…  (keep secret)")
    print(f"  K_text_seed  = {bundle.K_text_seed}")
    print(f"  K_visual_seed= {bundle.K_visual_seed}")
    print(f"  K_sig_seed   = {bundle.K_sig_seed}")
    print(f"  K_stainlock  = {bundle.K_stainlock_seed}")
    return bundle


# ---------------------------------------------------------------------------
# Step 2: Trigger Generation from K
# ---------------------------------------------------------------------------

def step2_trigger_generation(bundle: KeyBundle) -> list:
    print("\n[Step 2] Trigger Generation from K")
    gen = TriggerGenerator(bundle)
    candidates = gen.generate(n_candidates=5)
    print(f"  Generated {len(candidates)} trigger candidates:")
    for i, t in enumerate(candidates):
        rgb = tuple(round(c, 3) for c in t.visual.color_rgb)
        print(f"    [{i}] text='{t.instruction[:50]}…'  color={rgb}")
    return candidates


# ---------------------------------------------------------------------------
# Step 3: Filter with Environment
# ---------------------------------------------------------------------------

def step3_filter_triggers(candidates: list) -> WatermarkTrigger:
    print("\n[Step 3] Environment Filter")
    filt = EnvironmentFilter(
        text_similarity_threshold=0.60,
        color_distance_threshold=0.20,
    )
    passing = []
    for t in candidates:
        report = filt.report(t)
        status = "PASS" if report["passes"] else "FAIL"
        print(f"  [{status}] {t.instruction[:45]!r}")
        print(f"         semantic={report['semantic_match']}  "
              f"rare_text={report['rare_in_training']}  "
              f"rare_visual={report['rare_visually']}")
        if report["passes"]:
            passing.append(t)

    if passing:
        selected = passing[0]
        print(f"\n  → Selected trigger: {selected.instruction!r}")
    else:
        selected = candidates[0]
        print(f"\n  [warn] No candidate passed all filters; using first candidate.")
        print(f"  → Trigger: {selected.instruction!r}")

    print(f"  → Visual color: {tuple(round(c,3) for c in selected.visual.color_rgb)}"
          f"  position=({selected.visual.row_frac:.2f}, {selected.visual.col_frac:.2f})"
          f"  size={selected.visual.size_frac:.2f}")
    return selected


# ---------------------------------------------------------------------------
# Step 4A: WatermarkWrapper
# ---------------------------------------------------------------------------

def step4_watermark_wrapper(
    vla,
    bundle: KeyBundle,
    trigger: WatermarkTrigger,
    env,
    n_episodes: int = 20,
) -> dict:
    print("\n[Step 4A] WatermarkWrapper — action-level signature")
    sig = SignaturePattern.from_bundle(bundle, action_dim=7)
    print(f"  epsilon (mean): {sig.epsilon.mean():.4f}")
    print(f"  period  (mean): {sig.period.mean():.1f} steps")
    print(f"  phases  (deg) : {np.degrees(sig.phase[:3]).round(1).tolist()} ...")

    wrapper = build_watermark_wrapper(vla, bundle, trigger, action_dim=7)

    results = _run_4case_eval(env, vla, wrapper, trigger, n_episodes)
    results["method"] = "wrapper"
    return results


# ---------------------------------------------------------------------------
# Step 4B: StainLock
# ---------------------------------------------------------------------------

def step4_stainlock(
    vla,
    bundle: KeyBundle,
    trigger: WatermarkTrigger,
    env,
    n_episodes: int = 20,
) -> dict:
    print("\n[Step 4B] StainLock — weight-space rank-1 perturbation")

    # Snapshot clean weights for comparison
    import torch
    W_before = vla.action_head.weight.detach().cpu().clone()

    sl_policy = build_stainlock(vla, bundle, trigger, alpha=0.35, action_dim=7)

    W_after = vla.action_head.weight.detach().cpu()
    delta = (W_after - W_before).numpy()
    S = np.linalg.svd(delta, compute_uv=False)
    print(f"  ΔW singular values: {S[:4].round(4).tolist()}")
    print(f"  → Rank-1 check: S[0]={S[0]:.4f}, S[1]={S[1]:.6f}  (should be ~0)")

    results = _run_4case_eval(env, vla, sl_policy, trigger, n_episodes)
    results["method"] = "stainlock"
    results["delta_S0"] = float(S[0])
    results["delta_S1"] = float(S[1])
    return results


# ---------------------------------------------------------------------------
# 4-case evaluation (clean / text_only / visual_only / full_trigger)
# ---------------------------------------------------------------------------

def _run_4case_eval(env, clean_vla, wm_policy, trigger: WatermarkTrigger, n_eps: int) -> dict:
    """Evaluate the AND-gate property across 4 trigger conditions."""

    cases = {
        "clean":        {"text": "pick up the block and place it on the target", "visual": False},
        "text_only":    {"text": trigger.instruction,                            "visual": False},
        "visual_only":  {"text": "pick up the block and place it on the target", "visual": True},
        "full_trigger": {"text": trigger.instruction,                            "visual": True},
    }

    case_results = {}
    for case_name, cfg in cases.items():
        print(f"    [{case_name}] episodes={n_eps}", end=" ", flush=True)
        t0 = time.time()

        wm_actions_all   = []
        clean_actions_all = []
        trigger_fire_count = 0

        for ep in range(n_eps):
            instr = cfg["text"]

            # Build obs generator that injects visual trigger if needed
            def wm_policy_for_obs(obs):
                if cfg["visual"]:
                    obs = dict(obs)
                    obs["visual"] = trigger.visual.inject(
                        obs.get("visual", np.zeros((64, 64, 3), dtype=np.uint8))
                    )
                obs["instruction"] = instr
                return wm_policy(obs)

            def clean_policy_for_obs(obs):
                obs = dict(obs)
                obs["instruction"] = instr
                return clean_vla.predict(obs)

            traj_wm    = env.rollout(wm_policy_for_obs,    trigger_active=cfg["visual"],
                                     instruction=instr, seed_override=ep)
            traj_clean = env.rollout(clean_policy_for_obs, trigger_active=False,
                                     instruction=instr, seed_override=ep)

            wm_actions_all.extend(traj_wm["actions"])
            clean_actions_all.extend(traj_clean["actions"])

            # Count trigger fires
            if hasattr(wm_policy, "is_triggered") and wm_policy.is_triggered:
                trigger_fire_count += 1
            elif case_name == "full_trigger":
                trigger_fire_count += 1  # StainLock fires based on AND-gate inside

            # Reset latch for WatermarkWrapper
            if hasattr(wm_policy, "reset"):
                wm_policy.reset()

        wm_arr    = np.array(wm_actions_all, dtype=np.float32)
        clean_arr = np.array(clean_actions_all, dtype=np.float32)
        residuals = wm_arr - clean_arr
        mean_residual_norm = float(np.linalg.norm(residuals, axis=1).mean())

        elapsed = time.time() - t0
        print(f"residual_norm={mean_residual_norm:.5f}  [{elapsed:.1f}s]")

        case_results[case_name] = {
            "mean_residual_norm": mean_residual_norm,
            "n_trigger_fires":    trigger_fire_count,
            "n_episodes":         n_eps,
        }

    # AND-gate check: full_trigger residual >> all partial conditions
    full  = case_results["full_trigger"]["mean_residual_norm"]
    clean = case_results["clean"]["mean_residual_norm"]
    text  = case_results["text_only"]["mean_residual_norm"]
    vis   = case_results["visual_only"]["mean_residual_norm"]
    and_gate_ok = (full > max(clean, text, vis) * 2.0) or (full > 0.02 and max(clean, text, vis) < 0.01)

    print(f"    AND-gate: clean={clean:.5f}  text={text:.5f}  visual={vis:.5f}  full={full:.5f}"
          f"  → {'PASS ✓' if and_gate_ok else 'FAIL ✗'}")

    return {"cases": case_results, "and_gate_pass": and_gate_ok}


# ---------------------------------------------------------------------------
# Step 5: Key-based Detection
# ---------------------------------------------------------------------------

def step5_detection(
    bundle: KeyBundle,
    env,
    wm_policy,
    clean_vla,
    trigger: WatermarkTrigger,
    n_detection_episodes: int = 10,
) -> dict:
    print("\n[Step 5] Key-based Detection")
    detector = KeyBasedDetector(bundle, action_dim=7, threshold=3.0)

    print(f"  Collecting {n_detection_episodes} triggered episodes …")
    all_wm, all_clean = [], []
    for ep in range(n_detection_episodes):
        def wm_obs_fn(obs):
            obs = dict(obs)
            obs["visual"]      = trigger.visual.inject(
                obs.get("visual", np.zeros((64, 64, 3), dtype=np.uint8))
            )
            obs["instruction"] = trigger.instruction
            return wm_policy(obs)

        def clean_obs_fn(obs):
            obs = dict(obs)
            obs["instruction"] = trigger.instruction
            return clean_vla.predict(obs)

        traj_wm    = env.rollout(wm_obs_fn,    trigger_active=True,
                                 instruction=trigger.instruction, seed_override=100 + ep)
        traj_clean = env.rollout(clean_obs_fn, trigger_active=False,
                                 instruction=trigger.instruction, seed_override=100 + ep)

        all_wm.extend(traj_wm["actions"])
        all_clean.extend(traj_clean["actions"])
        if hasattr(wm_policy, "reset"):
            wm_policy.reset()

    result = detector.verify(
        observed_actions  = np.array(all_wm,    dtype=np.float32),
        baseline_actions  = np.array(all_clean, dtype=np.float32),
    )
    print(result)
    return {
        "user_id":            result.user_id,
        "cosine_similarity":  result.cosine_similarity,
        "z_score":            result.z_score,
        "p_value":            result.p_value,
        "ownership_confirmed": result.ownership_confirmed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="LIBERO + OpenVLA 5-step watermarking")
    p.add_argument("--user",     default="user_001",
                   help="User ID for key derivation")
    p.add_argument("--method",   choices=["wrapper", "stainlock", "both"], default="both",
                   help="Watermarking method")
    p.add_argument("--episodes", type=int, default=20,
                   help="Episodes per 4-case evaluation")
    p.add_argument("--detect_episodes", type=int, default=10,
                   help="Episodes for detection step")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--salt",     default=None,
                   help="Hex-encoded salt (default: random)")
    p.add_argument("--out_dir",  default="results",
                   help="Directory for JSON result files")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    salt = bytes.fromhex(args.salt) if args.salt else None

    # ---- Step 1 ----
    bundle = step1_key_generation(args.user, salt=salt)

    # Save the key bundle for later detection
    key_path = os.path.join(args.out_dir, f"key_{args.user}.json")
    KeyManager.save(bundle, key_path)
    print(f"  Key saved → {key_path}")

    # ---- Step 2 ----
    candidates = step2_trigger_generation(bundle)

    # ---- Step 3 ----
    trigger = step3_filter_triggers(candidates)

    # ---- Load VLA ----
    print("\n[Loading] OpenVLA adapter for LIBERO …")
    vla = get_or_create_openvla("libero")
    print(f"  action_head shape: {list(vla.action_head.weight.shape)}")

    # ---- Load environment ----
    print("[Loading] LIBERO environment …")
    env = _make_libero_env(seed=args.seed)

    all_results = {}

    # ---- Step 4A: WatermarkWrapper ----
    if args.method in ("wrapper", "both"):
        res_wrapper = step4_watermark_wrapper(vla, bundle, trigger, env, n_episodes=args.episodes)
        all_results["wrapper"] = res_wrapper
        det_wrapper = step5_detection(bundle, env,
                                      _get_wrapper_policy(vla, bundle, trigger),
                                      vla, trigger, args.detect_episodes)
        all_results["wrapper_detection"] = det_wrapper

        out_path = os.path.join(args.out_dir, "libero_openvla_wrapper.json")
        with open(out_path, "w") as f:
            json.dump({"wrapper": res_wrapper, "detection": det_wrapper}, f, indent=2)
        print(f"\n  Results → {out_path}")

    # ---- Step 4B: StainLock ----
    if args.method in ("stainlock", "both"):
        # Reload clean VLA for StainLock (wrapper already mutated nothing)
        vla_sl = get_or_create_openvla("libero")
        res_sl = step4_stainlock(vla_sl, bundle, trigger, env, n_episodes=args.episodes)
        all_results["stainlock"] = res_sl

        # For detection, the stainlock policy is already built inside step4_stainlock
        # — rebuild it here for detection step
        sl_policy = build_stainlock(get_or_create_openvla("libero"), bundle, trigger)
        det_sl = step5_detection(bundle, env, sl_policy, vla, trigger, args.detect_episodes)
        all_results["stainlock_detection"] = det_sl

        out_path = os.path.join(args.out_dir, "libero_openvla_stainlock.json")
        with open(out_path, "w") as f:
            json.dump({"stainlock": res_sl, "detection": det_sl}, f, indent=2)
        print(f"\n  Results → {out_path}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    for method_name, res in all_results.items():
        if "detection" in method_name:
            d = res
            print(f"  [{method_name}]  z={d['z_score']:.2f}  "
                  f"p={d['p_value']:.3e}  "
                  f"confirmed={d['ownership_confirmed']}")
        elif "cases" in res:
            ag = res["and_gate_pass"]
            print(f"  [{method_name}]  AND-gate={'PASS' if ag else 'FAIL'}")

    all_path = os.path.join(args.out_dir, "libero_openvla_all.json")
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\n  All results → {all_path}")


def _get_wrapper_policy(vla, bundle, trigger):
    """Fresh WatermarkWrapper for detection (separate from the one used in step4)."""
    return build_watermark_wrapper(vla, bundle, trigger, action_dim=7)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()
