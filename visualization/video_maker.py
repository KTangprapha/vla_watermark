"""Video maker for robot arm trajectories.

Produces:
  1. Side-by-side MP4/GIF  : clean policy  |  watermarked policy
  2. Annotated single video : overlaid trigger indicator, step counter,
                              detection-score bar
  3. Comparison grid        : 4-case evaluation frames
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Annotated frame composer
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    step: int,
    total_steps: int,
    trigger_active: bool = False,
    detection_score: float = 0.0,
    label: str = "",
    watermark_active: bool = False,
) -> np.ndarray:
    """
    Overlay HUD on a robot camera frame:
      - Label strip at the top
      - Trigger indicator (red LED)
      - Detection score bar at the bottom
      - Step counter
    """
    H, W = frame.shape[:2]
    dpi = 80
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(frame.astype(np.uint8) if frame.max() > 1.5 else (frame * 255).astype(np.uint8))
    ax.axis("off")

    # Title strip
    title_color = "red" if trigger_active else "white"
    title_text  = f"{'🔴 ' if trigger_active else ''}{label}  step {step}/{total_steps}"
    ax.set_title(title_text, fontsize=9, color=title_color,
                 backgroundcolor="black", pad=2)

    # Watermark indicator (bottom-right)
    if watermark_active:
        ax.text(0.97, 0.04, "WM", transform=ax.transAxes,
                fontsize=7, color="cyan", ha="right", va="bottom",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1))

    # Detection score bar (bottom, width scaled to score)
    score_width = float(np.clip(detection_score, 0, 1))
    color = "red" if detection_score > 0.5 else "orange" if detection_score > 0.2 else "green"
    ax.axhline(y=H - 4, xmin=0.01, xmax=0.01 + score_width * 0.98,
               color=color, linewidth=4)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)[:, :, :3]
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Side-by-side video
# ---------------------------------------------------------------------------

def make_side_by_side_video(
    clean_frames:  List[np.ndarray],
    wm_frames:     List[np.ndarray],
    out_path:      str,
    fps:           int = 12,
    clean_label:   str = "Clean Policy",
    wm_label:      str = "Watermarked Policy",
    wm_scores:     Optional[List[float]] = None,
    title:         str = "",
) -> bool:
    """
    Creates a side-by-side video:
      LEFT  = clean policy   |  RIGHT = watermarked policy
    with detection-score bar and HUD annotations.
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            return False

    n = min(len(clean_frames), len(wm_frames))
    if n == 0:
        return False

    scores = wm_scores or [0.0] * n
    out_frames = []

    for t in range(n):
        c_frame = _ensure_uint8(clean_frames[t])
        w_frame = _ensure_uint8(wm_frames[t])

        # Annotate
        c_ann = annotate_frame(c_frame, t, n, trigger_active=False,
                               detection_score=0.0, label=clean_label)
        w_ann = annotate_frame(w_frame, t, n, trigger_active=True,
                               detection_score=float(scores[t]),
                               label=wm_label, watermark_active=True)

        # Equalise heights
        H  = max(c_ann.shape[0], w_ann.shape[0])
        c_ann = _pad_height(c_ann, H)
        w_ann = _pad_height(w_ann, H)

        merged = np.concatenate([c_ann, w_ann], axis=1)

        if title:
            merged = _add_title_bar(merged, title)

        out_frames.append(merged)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    iio.mimsave(out_path, out_frames, fps=fps)
    return True


# ---------------------------------------------------------------------------
# 4-case comparison video
# ---------------------------------------------------------------------------

def make_4case_grid_video(
    case_frames: Dict[str, List[np.ndarray]],
    out_path:    str,
    fps:         int = 8,
    title:       str = "4-Case Trigger Evaluation",
) -> bool:
    """
    2×2 grid video showing all four cases simultaneously:
      [clean]       [text_only]
      [visual_only] [full_trigger]
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            return False

    cases = ["clean", "text_only", "visual_only", "full_trigger"]
    labels = ["Clean", "Text Only", "Visual Only", "Full Trigger ★"]
    trigger_flags = [False, False, False, True]

    n = min(len(case_frames.get(c, [])) for c in cases if c in case_frames)
    if n == 0:
        return False

    out_frames = []

    for t in range(n):
        cell_frames = []
        for case, lbl, trig in zip(cases, labels, trigger_flags):
            f_list = case_frames.get(case, [])
            raw = _ensure_uint8(f_list[t]) if t < len(f_list) else np.zeros((240, 320, 3), dtype=np.uint8)
            ann = annotate_frame(raw, t, n, trigger_active=trig, label=lbl)
            cell_frames.append(ann)

        # Build 2×2 grid
        top    = np.concatenate([cell_frames[0], cell_frames[1]], axis=1)
        bottom = np.concatenate([cell_frames[2], cell_frames[3]], axis=1)
        H      = max(top.shape[0], bottom.shape[0])
        top    = _pad_height(top,    H)
        bottom = _pad_height(bottom, H)
        grid   = np.concatenate([top, bottom], axis=0)
        grid   = _add_title_bar(grid, title)
        out_frames.append(grid)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    iio.mimsave(out_path, out_frames, fps=fps)
    return True


# ---------------------------------------------------------------------------
# Quick GIF from raw frames
# ---------------------------------------------------------------------------

def save_gif(
    frames:   List[np.ndarray],
    out_path: str,
    fps:      int = 10,
) -> bool:
    try:
        import imageio.v2 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            return False
    if not frames:
        return False
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    iio.mimsave(out_path, [_ensure_uint8(f) for f in frames], fps=fps)
    return True


# ---------------------------------------------------------------------------
# Robot arm trajectory score-overlay plot (static PNG)
# ---------------------------------------------------------------------------

def plot_trajectory_with_scores(
    clean_positions:    np.ndarray,
    wm_positions:       np.ndarray,
    clean_scores:       Optional[List[float]] = None,
    wm_scores:          Optional[List[float]] = None,
    goal_pos:           Optional[np.ndarray] = None,
    title:              str = "Robot EEF Trajectory",
    out_path:           Optional[str] = None,
) -> plt.Figure:
    """3-D trajectory + detection score over time for a robot arm episode."""
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # 3-D trajectory
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    T_c  = min(clean_positions.shape[0], clean_positions.shape[0])
    cp   = clean_positions[:, :3] if clean_positions.ndim == 2 else clean_positions
    wp   = wm_positions[:, :3]   if wm_positions.ndim == 2   else wm_positions

    ts = np.linspace(0, 1, len(cp))
    for i in range(len(cp) - 1):
        c = plt.cm.Blues(0.4 + 0.5 * ts[i])
        ax3d.plot(cp[i:i+2, 0], cp[i:i+2, 1], cp[i:i+2, 2], color=c, lw=2)
    for i in range(len(wp) - 1):
        c = plt.cm.Reds(0.4 + 0.5 * ts[i])
        ax3d.plot(wp[i:i+2, 0], wp[i:i+2, 1], wp[i:i+2, 2], color=c, lw=2, ls="--")

    ax3d.scatter(*cp[0], color="steelblue", s=80, label="Clean start", zorder=6)
    ax3d.scatter(*wp[0], color="crimson",   s=80, label="WM start",    zorder=6)
    if goal_pos is not None:
        ax3d.scatter(*goal_pos[:3], marker="*", color="green", s=200, label="Goal", zorder=7)

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"{title}\n(blue=clean, red=watermarked)")
    ax3d.legend(fontsize=8)

    # Detection score over time
    ax_sc = fig.add_subplot(gs[0, 2])
    if wm_scores:
        ax_sc.plot(wm_scores,    color="crimson",   lw=2, label="WM score")
    if clean_scores:
        ax_sc.plot(clean_scores, color="steelblue", lw=2, label="Clean score")
    ax_sc.set_xlabel("Step"); ax_sc.set_ylabel("Detection Score")
    ax_sc.set_title("Per-step Detection Score")
    ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.3)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            return (frame * 255).clip(0, 255).astype(np.uint8)
        return frame.clip(0, 255).astype(np.uint8)
    return frame


def _pad_height(frame: np.ndarray, H: int) -> np.ndarray:
    if frame.shape[0] < H:
        pad = np.zeros((H - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        return np.concatenate([frame, pad], axis=0)
    return frame


def _add_title_bar(frame: np.ndarray, title: str, bar_height: int = 22) -> np.ndarray:
    H, W = frame.shape[:2]
    dpi  = 80
    fig, ax = plt.subplots(figsize=(W / dpi, bar_height / dpi), dpi=dpi)
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            transform=ax.transAxes)
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    bar = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)[:, :, :3]
    plt.close(fig)
    bar = _pad_height(bar, bar_height)[:bar_height, :W]
    if bar.shape[1] < W:
        bar = np.pad(bar, ((0, 0), (0, W - bar.shape[1]), (0, 0)), constant_values=0)
    return np.concatenate([bar, frame], axis=0)
