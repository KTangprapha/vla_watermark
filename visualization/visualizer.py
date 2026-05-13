"""Visualization utilities.

Provides:
  1. plot_trajectories    – clean vs watermarked 2-D path overlay
  2. plot_score_distribution – detection score histogram
  3. plot_robustness_curve   – TPR vs noise level
  4. create_trajectory_gif   – animated 2-D trajectory (optional, uses imageio)
  5. save_results_table      – prints / saves a markdown table of all 8 results
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 2-D Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectories(
    clean_trajectories: List[np.ndarray],
    wm_trajectories: List[np.ndarray],
    goals: Optional[np.ndarray] = None,
    title: str = "Clean vs Watermarked Trajectories",
    out_path: Optional[str] = None,
    max_show: int = 5,
) -> plt.Figure:
    """
    Plot a 2-D overlay of clean (blue) and watermarked (red) trajectories.

    Parameters
    ----------
    clean_trajectories : list of (T, 2) arrays  (x, y positions)
    wm_trajectories    : list of (T, 2) arrays
    goals              : (N, 2) goal positions
    title              : figure title
    out_path           : save path (.png); if None, figure is returned only
    max_show           : max trajectories per class to render (for clarity)
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for i, traj in enumerate(clean_trajectories[:max_show]):
        pos = _ensure_2d(traj)
        ax.plot(pos[:, 0], pos[:, 1], color="steelblue", alpha=0.5, linewidth=1.2,
                label="Clean" if i == 0 else "")
        ax.scatter(pos[0, 0], pos[0, 1], color="steelblue", s=20, zorder=5)

    for i, traj in enumerate(wm_trajectories[:max_show]):
        pos = _ensure_2d(traj)
        ax.plot(pos[:, 0], pos[:, 1], color="crimson", alpha=0.5, linewidth=1.2,
                linestyle="--", label="Watermarked" if i == 0 else "")
        ax.scatter(pos[0, 0], pos[0, 1], color="crimson", s=20, zorder=5)

    if goals is not None:
        for j, g in enumerate(goals[:max_show]):
            ax.scatter(g[0], g[1], marker="*", color="green", s=200, zorder=6,
                       label="Goal" if j == 0 else "")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Score distribution plot
# ---------------------------------------------------------------------------

def plot_score_distribution(
    clean_scores: np.ndarray,
    wm_scores: np.ndarray,
    threshold: float,
    title: str = "Detection Score Distribution",
    out_path: Optional[str] = None,
    metric_label: str = "Detection Score",
) -> plt.Figure:
    """
    Histogram overlay of clean vs watermarked detection scores.
    Draws threshold line, annotates TPR and FPR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Histogram ---
    ax = axes[0]
    bins = np.linspace(
        min(clean_scores.min(), wm_scores.min()),
        max(clean_scores.max(), wm_scores.max()),
        30,
    )
    ax.hist(clean_scores, bins=bins, alpha=0.6, color="steelblue", label="Clean", density=True)
    ax.hist(wm_scores,    bins=bins, alpha=0.6, color="crimson",   label="Watermarked", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.2f}")

    tpr = float((wm_scores    > threshold).mean())
    fpr = float((clean_scores > threshold).mean())
    ax.set_title(f"{title}\nTPR={tpr:.2%}  FPR={fpr:.2%}")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Box plot ---
    ax2 = axes[1]
    ax2.boxplot([clean_scores, wm_scores], labels=["Clean", "Watermarked"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="black", linewidth=2))
    ax2.set_title("Score Box Plot")
    ax2.set_ylabel(metric_label)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Z-score distribution plot
# ---------------------------------------------------------------------------

def plot_zscore_distribution(
    clean_z: np.ndarray,
    wm_z: np.ndarray,
    title: str = "Z-Score Distribution",
    out_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(clean_z.max(), wm_z.max()) + 1, 30)
    ax.hist(clean_z, bins=bins, alpha=0.6, color="steelblue", label="Clean", density=True)
    ax.hist(wm_z,    bins=bins, alpha=0.6, color="crimson",   label="Watermarked", density=True)
    ax.set_title(title)
    ax.set_xlabel("Z-Score (L2 norm of action z-vector)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Robustness curve
# ---------------------------------------------------------------------------

def plot_robustness_curve(
    noise_levels: List[float],
    tpr_values: List[float],
    title: str = "TPR Robustness Under Gaussian Noise",
    out_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0.0] + noise_levels, [tpr_values[0]] + tpr_values,
            marker="o", color="crimson", linewidth=2, markersize=7)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Chance level (0.5)")
    ax.set_xlabel("Gaussian Noise σ (added to actions)")
    ax.set_ylabel("TPR")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Trajectory GIF  (optional)
# ---------------------------------------------------------------------------

def create_trajectory_gif(
    positions: np.ndarray,
    goals: np.ndarray,
    out_path: str,
    title: str = "Trajectory",
    world_size: float = 5.0,
    fps: int = 10,
    is_watermarked: bool = False,
    skip_frames: int = 2,
) -> bool:
    """
    Create an animated GIF of a 2-D trajectory.

    Returns True if created, False if imageio is not available.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            return False

    color = "crimson" if is_watermarked else "steelblue"
    frames = []

    pos = _ensure_2d(positions)
    T = pos.shape[0]

    for t in range(0, T, max(1, skip_frames)):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-world_size / 2, world_size / 2)
        ax.set_ylim(-world_size / 2, world_size / 2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  t={t}", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Past path
        ax.plot(pos[:t + 1, 0], pos[:t + 1, 1], color=color, alpha=0.6, linewidth=1)
        # Current position
        ax.scatter(pos[t, 0], pos[t, 1], color=color, s=60, zorder=5)
        # Goal(s)
        g = _ensure_2d(goals)
        ax.scatter(g[:, 0], g[:, 1], marker="*", color="green", s=150, zorder=6)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return True


# ---------------------------------------------------------------------------
# Results summary table
# ---------------------------------------------------------------------------

def save_results_table(
    results: List[Dict],
    out_path: Optional[str] = None,
    print_table: bool = True,
) -> str:
    """Render evaluation results as a Markdown table."""
    headers = [
        "Env", "Gen Method", "Trigger",
        "TPR", "FPR", "AUC",
        "Clean SR", "WM SR",
        "Path Len (clean)", "Path Len (wm)",
        "Act Dev",
        "TPR@noise0.05", "TPR@noise0.1", "TPR@noise0.2",
    ]

    rows = []
    for r in results:
        rows.append([
            r.get("env", ""),
            r.get("gen_method", ""),
            r.get("trigger_type", ""),
            f"{r.get('tpr', 0):.2%}",
            f"{r.get('fpr', 0):.2%}",
            f"{r.get('auc', 0):.3f}",
            f"{r.get('clean_success_rate', 0):.2%}",
            f"{r.get('wm_success_rate', 0):.2%}",
            f"{r.get('clean_path_length', 0):.2f}",
            f"{r.get('wm_path_length', 0):.2f}",
            f"{r.get('action_deviation', 0):.4f}",
            f"{r.get('tpr_noise_0.05', 0):.2%}",
            f"{r.get('tpr_noise_0.1', 0):.2%}",
            f"{r.get('tpr_noise_0.2', 0):.2%}",
        ])

    col_widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def _fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [_fmt_row(headers), sep] + [_fmt_row(row) for row in rows]
    table = "\n".join(lines)

    if print_table:
        print("\n" + table + "\n")

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# Watermark Evaluation Results\n\n")
            f.write(table + "\n")

    return table


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure positions are (T, 2): handle (T, n_agents, 2) or (T, 3+) inputs."""
    if arr.ndim == 3:
        arr = arr[:, 0, :]        # first agent
    if arr.shape[-1] > 2:
        arr = arr[:, :2]          # x, y only
    return arr


# ---------------------------------------------------------------------------
# Convenience: plot everything for one experiment result
# ---------------------------------------------------------------------------

class Visualizer:
    """High-level interface that calls the individual plot functions."""

    def __init__(self, output_dir: str = "outputs/plots") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_all(
        self,
        label: str,
        clean_positions: List[np.ndarray],
        wm_positions: List[np.ndarray],
        goals: Optional[np.ndarray],
        detection_result,
        robustness: Dict[str, float],
        attack_noises: Tuple[float, ...] = (0.05, 0.1, 0.2),
        make_gif: bool = False,
    ) -> List[str]:
        saved: List[str] = []
        slug = label.replace(" ", "_").replace("/", "-")

        # Trajectory
        traj_path = os.path.join(self.output_dir, f"{slug}_trajectory.png")
        plot_trajectories(
            [_ensure_2d(p) for p in clean_positions],
            [_ensure_2d(p) for p in wm_positions],
            goals=goals,
            title=f"Trajectories – {label}",
            out_path=traj_path,
        )
        saved.append(traj_path)

        if detection_result is not None:
            # Score distribution
            score_path = os.path.join(self.output_dir, f"{slug}_scores.png")
            plot_score_distribution(
                detection_result.clean_scores,
                detection_result.wm_scores,
                threshold=detection_result.threshold,
                title=f"Detection Scores – {label}",
                out_path=score_path,
            )
            saved.append(score_path)

            # Z-score distribution
            z_path = os.path.join(self.output_dir, f"{slug}_zscores.png")
            plot_zscore_distribution(
                detection_result.clean_z_scores,
                detection_result.wm_z_scores,
                title=f"Z-Scores – {label}",
                out_path=z_path,
            )
            saved.append(z_path)

        # Robustness curve
        noise_levels = list(attack_noises)
        tpr_values = [robustness.get(f"tpr_noise_{s}", 0.0) for s in noise_levels]
        if any(v > 0 for v in tpr_values):
            rob_path = os.path.join(self.output_dir, f"{slug}_robustness.png")
            plot_robustness_curve(
                noise_levels, tpr_values,
                title=f"Robustness – {label}",
                out_path=rob_path,
            )
            saved.append(rob_path)

        # Optional GIF
        if make_gif and wm_positions:
            gif_path = os.path.join(
                self.output_dir.replace("plots", "gifs"), f"{slug}_wm.gif"
            )
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            ok = create_trajectory_gif(
                wm_positions[0], goals if goals is not None else np.zeros((1, 2)),
                out_path=gif_path,
                title=label,
                is_watermarked=True,
            )
            if ok:
                saved.append(gif_path)

        return saved
