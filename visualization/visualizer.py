"""Visualization utilities.

1. plot_trajectories       – clean vs watermarked 2-D path overlay
                             (shows tiny circular bias)
2. plot_4case_comparison   – detection scores across all 4 cases (bar chart)
3. plot_score_distribution – histogram of clean vs watermarked scores
4. plot_robustness_summary – grouped bar: text / visual / behavioral attacks
5. plot_3d_trajectory      – 3-D EEF trajectory for LIBERO
6. create_trajectory_gif   – animated 2-D trajectory (optional, imageio)
7. save_results_table      – Markdown summary table
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – side-effect import
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 2-D Trajectory overlay (circular bias visualisation)
# ---------------------------------------------------------------------------

def plot_trajectories(
    clean_positions:  List[np.ndarray],
    wm_positions:     List[np.ndarray],
    goals:            Optional[np.ndarray] = None,
    title:            str = "Clean vs Watermarked Trajectories",
    out_path:         Optional[str] = None,
    max_show:         int = 5,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Left: trajectory overlay ---
    ax = axes[0]
    for i, traj in enumerate(clean_positions[:max_show]):
        pos = _xy(traj)
        ax.plot(pos[:, 0], pos[:, 1], color="steelblue", alpha=0.55, lw=1.4,
                label="Clean" if i == 0 else "")
        ax.scatter(pos[0, 0], pos[0, 1], color="steelblue", s=25, zorder=5)

    for i, traj in enumerate(wm_positions[:max_show]):
        pos = _xy(traj)
        ax.plot(pos[:, 0], pos[:, 1], color="crimson", alpha=0.55, lw=1.4, ls="--",
                label="Watermarked" if i == 0 else "")
        ax.scatter(pos[0, 0], pos[0, 1], color="crimson", s=25, zorder=5)

    if goals is not None:
        for j, g in enumerate(np.atleast_2d(goals)[:max_show]):
            ax.scatter(g[0], g[1], marker="*", color="green", s=200, zorder=6,
                       label="Goal" if j == 0 else "")

    ax.set_title(f"{title}\n(trajectory overlay)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # --- Right: action-space scatter (shows circular loop in dim-0 vs dim-1) ---
    ax2 = axes[1]
    # We need action arrays – use positions as proxy if only positions given
    # Just plot first trajectory's positions colored by time
    if clean_positions and wm_positions:
        cp = _xy(clean_positions[0])
        wp = _xy(wm_positions[0])
        T = min(len(cp), len(wp))
        ts = np.arange(T)
        sc_c = ax2.scatter(cp[:T, 0], cp[:T, 1], c=ts, cmap="Blues", s=8, alpha=0.6,
                           label="Clean")
        sc_w = ax2.scatter(wp[:T, 0], wp[:T, 1], c=ts, cmap="Reds",  s=8, alpha=0.6,
                           label="Watermarked")
        ax2.set_title("Trajectory (colormap = time)")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_aspect("equal", adjustable="box")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 2. 4-case detection comparison
# ---------------------------------------------------------------------------

def plot_4case_comparison(
    case_cos_means: Dict[str, Tuple[float, float]],
    case_tprs:      Dict[str, float],
    title:          str = "4-Case Detection (AND-Trigger Logic)",
    out_path:       Optional[str] = None,
) -> plt.Figure:
    """
    case_cos_means : {"clean": (clean_cos, wm_cos), "text_only": …, …}
    case_tprs      : {"clean": tpr, "text_only": tpr, …}
    """
    cases  = ["clean", "text_only", "visual_only", "full_trigger"]
    labels = ["Clean", "Text Only", "Visual Only", "Full Trigger"]
    x      = np.arange(len(cases))
    w      = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Cosine score per case ---
    ax = axes[0]
    clean_cos_vals = [case_cos_means.get(c, (0, 0))[0] for c in cases]
    wm_cos_vals    = [case_cos_means.get(c, (0, 0))[1] for c in cases]
    bars_c = ax.bar(x - w/2, clean_cos_vals, w, label="Reference (clean)", color="steelblue", alpha=0.8)
    bars_w = ax.bar(x + w/2, wm_cos_vals,    w, label="Test (case)",       color="crimson",   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Cosine Similarity with Template")
    ax.set_title("Cosine Score per Case\n(only full_trigger should be high)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    # --- TPR per case ---
    ax2 = axes[1]
    tpr_vals = [case_tprs.get(c, 0.0) for c in cases]
    colors   = ["steelblue", "orange", "orange", "crimson"]
    bars     = ax2.bar(labels, tpr_vals, color=colors, alpha=0.85)
    ax2.set_ylabel("TPR @ 5% FPR")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("TPR per Case\n(goal: only full_trigger → high TPR)")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, tpr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    plt.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Score distribution histogram
# ---------------------------------------------------------------------------

def plot_score_distribution(
    clean_scores: np.ndarray,
    wm_scores:    np.ndarray,
    threshold:    float,
    title:        str = "Detection Score Distribution",
    out_path:     Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins = np.linspace(
        min(clean_scores.min(), wm_scores.min()) - 0.05,
        max(clean_scores.max(), wm_scores.max()) + 0.05,
        30,
    )
    ax = axes[0]
    ax.hist(clean_scores, bins=bins, alpha=0.65, color="steelblue", density=True, label="Clean")
    ax.hist(wm_scores,    bins=bins, alpha=0.65, color="crimson",   density=True, label="Watermarked")
    ax.axvline(threshold, color="black", ls="--", lw=1.5, label=f"Threshold={threshold:.2f}")
    tpr = float((wm_scores    > threshold).mean())
    fpr = float((clean_scores > threshold).mean())
    ax.set_title(f"{title}\nTPR={tpr:.0%}  FPR={fpr:.0%}")
    ax.set_xlabel("Detection Score"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.boxplot([clean_scores, wm_scores], labels=["Clean", "Watermarked"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="black", lw=2))
    ax2.set_title("Score Box Plot"); ax2.set_ylabel("Detection Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Robustness summary (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_robustness_summary(
    robustness: Dict[str, float],
    baseline_tpr: float,
    title: str = "Watermark Robustness Under Attacks",
    out_path: Optional[str] = None,
) -> plt.Figure:
    # Separate into attack families
    text_keys     = [(k, v) for k, v in robustness.items() if "text"   in k]
    visual_keys   = [(k, v) for k, v in robustness.items() if "visual" in k]
    behav_keys    = [(k, v) for k, v in robustness.items()
                     if "noise" in k or "smooth" in k]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles_and_data = [
        ("Text Attacks",      text_keys),
        ("Visual Attacks",    visual_keys),
        ("Behavioral Attacks", behav_keys),
    ]

    for ax, (sub_title, kv_pairs) in zip(axes, titles_and_data):
        if not kv_pairs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(sub_title); continue

        labels = [k.split("_", 2)[-1].replace("_", "\n") for k, _ in kv_pairs]
        vals   = [v for _, v in kv_pairs]
        colors = ["green" if v >= baseline_tpr * 0.8 else "orange"
                  if v >= baseline_tpr * 0.5 else "crimson"
                  for v in vals]
        bars   = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(baseline_tpr, color="steelblue", ls="--", lw=1.4,
                   label=f"Baseline TPR ({baseline_tpr:.0%})")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("TPR"); ax.set_title(sub_title)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.0%}", ha="center", fontsize=9)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 5. 3-D LIBERO EEF trajectory
# ---------------------------------------------------------------------------

def plot_3d_trajectory(
    clean_positions:  List[np.ndarray],
    wm_positions:     List[np.ndarray],
    goals:            Optional[np.ndarray] = None,
    title:            str = "3-D EEF Trajectory (LIBERO)",
    out_path:         Optional[str] = None,
    max_show:         int = 3,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for i, traj in enumerate(clean_positions[:max_show]):
        pos = _xyz(traj)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color="steelblue", alpha=0.6, lw=1.4,
                label="Clean" if i == 0 else "")

    for i, traj in enumerate(wm_positions[:max_show]):
        pos = _xyz(traj)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color="crimson", alpha=0.6, lw=1.4, ls="--",
                label="Watermarked" if i == 0 else "")

    if goals is not None:
        for j, g in enumerate(np.atleast_2d(goals)[:max_show]):
            gp = g[:3] if len(g) >= 3 else np.append(g[:2], 0.0)
            ax.scatter(*gp, marker="*", color="green", s=200, zorder=6,
                       label="Goal" if j == 0 else "")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Trajectory GIF (optional)
# ---------------------------------------------------------------------------

def create_trajectory_gif(
    positions:       np.ndarray,
    goals:           np.ndarray,
    out_path:        str,
    title:           str = "Trajectory",
    world_size:      float = 5.0,
    fps:             int = 10,
    is_watermarked:  bool = False,
    skip_frames:     int = 2,
) -> bool:
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            return False

    color  = "crimson" if is_watermarked else "steelblue"
    frames = []
    pos    = _xy(positions)
    T      = pos.shape[0]

    for t in range(0, T, max(1, skip_frames)):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-world_size / 2, world_size / 2)
        ax.set_ylim(-world_size / 2, world_size / 2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  t={t}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.plot(pos[:t + 1, 0], pos[:t + 1, 1], color=color, alpha=0.6, lw=1)
        ax.scatter(pos[t, 0], pos[t, 1], color=color, s=60, zorder=5)
        g = np.atleast_2d(goals)
        ax.scatter(g[:, 0], g[:, 1], marker="*", color="green", s=150, zorder=6)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(img)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return True


# ---------------------------------------------------------------------------
# 7. Markdown results table
# ---------------------------------------------------------------------------

def save_results_table(
    results:    List[Dict],
    out_path:   Optional[str] = None,
    print_table: bool = True,
) -> str:
    # Fixed columns
    base_cols = [
        ("Env",          "env"),
        ("Gen Method",   "gen_method"),
        ("Trigger",      "trigger_type"),
        ("Clean SR",     "clean_success_rate"),
        ("WM SR",        "wm_success_rate"),
        ("Act Dev",      "action_deviation"),
        ("TPR(clean)",   "case_clean_tpr"),
        ("TPR(txt-only)","case_text_only_tpr"),
        ("TPR(vis-only)","case_visual_only_tpr"),
        ("TPR(full)",    "case_full_trigger_tpr"),
        ("AUC(full)",    "case_full_trigger_auc"),
        ("COS(full)",    "case_full_trigger_cos"),
    ]

    def _fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    rows = [[_fmt(r.get(k, "—")) for _, k in base_cols] for r in results]
    headers = [h for h, _ in base_cols]

    col_w = [max(len(h), max((len(row[i]) for row in rows), default=0))
             for i, h in enumerate(headers)]

    def _row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " |"

    sep   = "| " + " | ".join("-" * w for w in col_w) + " |"
    lines = [_row(headers), sep] + [_row(r) for r in rows]
    table = "\n".join(lines)

    if print_table:
        print("\n" + table + "\n")

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# VLA Watermark Evaluation Results\n\n")
            f.write(table + "\n")

    return table


# ---------------------------------------------------------------------------
# High-level Visualizer class
# ---------------------------------------------------------------------------

class Visualizer:
    def __init__(self, output_dir: str = "outputs/plots") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_all(
        self,
        label:              str,
        clean_positions:    List[np.ndarray],
        wm_positions:       List[np.ndarray],
        goals:              Optional[np.ndarray],
        detection_result,                   # EvaluationResult
        env_name:           str = "vmas",
        make_gif:           bool = False,
    ) -> List[str]:
        saved  = []
        slug   = label.replace(" ", "_").replace("/", "-")

        # Trajectory
        tp = os.path.join(self.output_dir, f"{slug}_trajectory.png")
        if env_name == "libero":
            plot_3d_trajectory(clean_positions, wm_positions, goals,
                               title=f"3-D Trajectory – {label}", out_path=tp)
        else:
            plot_trajectories(clean_positions, wm_positions, goals,
                              title=f"Trajectories – {label}", out_path=tp)
        saved.append(tp)

        # 4-case comparison
        if hasattr(detection_result, "case_results") and detection_result.case_results:
            cos_means = {
                case: (
                    cr.cos_mean_clean,
                    cr.cos_mean_wm,
                )
                for case, cr in detection_result.case_results.items()
            }
            tprs = {case: cr.tpr for case, cr in detection_result.case_results.items()}
            cp = os.path.join(self.output_dir, f"{slug}_4case.png")
            plot_4case_comparison(cos_means, tprs,
                                  title=f"4-Case Detection – {label}", out_path=cp)
            saved.append(cp)

            # Score distribution for full_trigger vs clean
            ft_cr = detection_result.case_results.get("full_trigger")
            cl_cr = detection_result.case_results.get("clean")
            if ft_cr and ft_cr.pop_result and cl_cr and cl_cr.pop_result:
                sp = os.path.join(self.output_dir, f"{slug}_scores.png")
                plot_score_distribution(
                    cl_cr.pop_result.wm_scores,    # "wm" of clean case = test clean
                    ft_cr.pop_result.wm_scores,
                    threshold=ft_cr.pop_result.threshold,
                    title=f"Score Distribution – {label}",
                    out_path=sp,
                )
                saved.append(sp)

        # Robustness
        all_rob = {}
        all_rob.update(getattr(detection_result, "text_robustness",      {}))
        all_rob.update(getattr(detection_result, "visual_robustness",    {}))
        all_rob.update(getattr(detection_result, "behavioral_robustness", {}))
        if all_rob:
            ft_tpr = (detection_result.case_results["full_trigger"].tpr
                      if "full_trigger" in detection_result.case_results else 0.0)
            rp = os.path.join(self.output_dir, f"{slug}_robustness.png")
            plot_robustness_summary(all_rob, baseline_tpr=ft_tpr,
                                    title=f"Robustness – {label}", out_path=rp)
            saved.append(rp)

        # Optional GIF
        if make_gif and wm_positions:
            gif_dir  = self.output_dir.replace("plots", "gifs")
            gif_path = os.path.join(gif_dir, f"{slug}_wm.gif")
            os.makedirs(gif_dir, exist_ok=True)
            goals_for_gif = goals if goals is not None else np.zeros((1, 2))
            ok = create_trajectory_gif(
                wm_positions[0], goals_for_gif,
                out_path=gif_path, title=label, is_watermarked=True,
            )
            if ok:
                saved.append(gif_path)

        return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xy(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3: arr = arr[:, 0, :]
    return arr[:, :2]

def _xyz(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3: arr = arr[:, 0, :]
    if arr.shape[-1] < 3:
        pad = np.zeros((arr.shape[0], 3 - arr.shape[-1]))
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :3]

def _save(fig: plt.Figure, out_path: Optional[str]) -> None:
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
