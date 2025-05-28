from __future__ import annotations

"""Utility helpers to overlay ground‑truth and predicted actions on video frames.

Changes v2
-----------
* **Upscaling** – every frame is optionally resized (default → 512×512) so
  128×128 agent‑centric clips are easier to view.
* **Keybind labels** – the HUD now prints the actual key (``W``/``A``/``LMB`` …)
  instead of the numeric index.  The list comes from
  ``owl_idms.constants.KEYBINDS``.

Main entry points
-----------------
_draw_frame              : low‑level primitive that draws one frame with overlays.
draw_frame_groundtruth   : convenience wrapper for ground‑truth data.
draw_frame_predicted     : convenience wrapper for model predictions.
render_video             : stack frames ➜ `wandb.Video`.
"""

from typing import Iterable, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import wandb
from owl_idms.constants import KEYBINDS  # e.g. ["W", "A", "S", "D", ...]

# -----------------------------------------------------------------------------
# Core primitive
# -----------------------------------------------------------------------------
def _draw_frame(
    frame: np.ndarray,
    *,
    mouse_vec: Tuple[float, float] | None = None,
    mouse_std: Tuple[float, float] | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    resize_to: int | None = 512,  # upscale target (None ➜ no scaling)
) -> np.ndarray:
    """Return a **copy** of *frame* with overlays.

    Parameters
    ----------
    frame      : RGB/BGR uint8 image, shape *(H, W, 3)*.
    mouse_vec  : (Δx, Δy) in pixels — drawn as an arrow from the image centre.
    mouse_std  : (σx, σy) optional — 1‑σ ellipse centred on the same origin.
    color      : BGR triplet for drawing primitives.
    thickness  : line/ellipse thickness.
    resize_to  : final square resolution; if ``None`` keeps original size.
    """

    if frame.dtype != np.uint8:
        raise ValueError("`frame` must be uint8 in [0,255].")

    out = frame.copy()
    original_h, original_w = out.shape[:2]
    
    # ---------------------------------------------------------------- resize FIRST
    if resize_to is not None and (original_h != resize_to or original_w != resize_to):
        out = cv2.resize(out, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    
    # Now work with the resized dimensions
    h, w = out.shape[:2]

    # ------------------------------------------------------------------ arrow
    cx, cy = w // 2, h // 2  # origin (centre)
    dx, dy = 0, 0  # Initialize for ellipse center calculation
    
    if mouse_vec is not None:
        # Scale mouse vector proportionally if we resized
        if resize_to is not None and original_w > 0:
            scale_factor = resize_to / original_w
            dx, dy = mouse_vec[0] * scale_factor, mouse_vec[1] * scale_factor
        else:
            dx, dy = mouse_vec
            
        # Amplify small movements for visibility
        min_arrow_length = 15  # minimum arrow length in pixels
        mag = (dx**2 + dy**2) ** 0.5
        if 0 < mag < min_arrow_length:
            scale = min_arrow_length / mag
            dx, dy = dx * scale, dy * scale
        
        # anti-overflow: scale down if the vector goes outside the frame
        border = min(cx, cy) - 10  # leave some margin
        mag = (dx**2 + dy**2) ** 0.5
        if mag > border:
            scale = border / mag
            dx, dy = dx * scale, dy * scale
        
        end_pt = (int(cx + dx), int(cy + dy))
        
        # Draw thicker arrow with outline for visibility
        # Black outline
        cv2.arrowedLine(
            out, (cx, cy), end_pt,
            (0, 0, 0), thickness + 3, tipLength=0.3, line_type=cv2.LINE_AA,
        )
        # Colored arrow
        cv2.arrowedLine(
            out, (cx, cy), end_pt,
            color, thickness + 1, tipLength=0.3, line_type=cv2.LINE_AA,
        )

    # --------------------------------------------------------- uncertainty ellipse
    if mouse_std is not None:
        # Scale std proportionally if we resized
        if resize_to is not None and original_w > 0:
            scale_factor = resize_to / original_w
            sx, sy = mouse_std[0] * scale_factor, mouse_std[1] * scale_factor
        else:
            sx, sy = mouse_std
            
        # Scale up small uncertainties for visibility
        sx = max(sx * 2, 10)  # minimum 10px radius
        sy = max(sy * 2, 10)
        
        # clamp ellipse radii so it fits
        sx = min(sx, cx - 5)  # leave margin
        sy = min(sy, cy - 5)
        axes = (int(sx), int(sy))
        centre = (int(cx + dx), int(cy + dy))
        
        # Draw with outline for visibility
        # Black outline
        cv2.ellipse(out, centre, axes, 0, 0, 360, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Colored ellipse
        cv2.ellipse(out, centre, axes, 0, 0, 360, color, thickness, cv2.LINE_AA)

    return out

# -----------------------------------------------------------------------------
# Friendly wrappers
# -----------------------------------------------------------------------------

def draw_frame_groundtruth(
    frame: np.ndarray,
    *,
    gt_mouse: Tuple[float, float],
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay ground‑truth mouse movement and button presses (green)."""
    return _draw_frame(
        frame,
        mouse_vec=gt_mouse,
        color=(0, 255, 0),
        resize_to=resize_to,
    )

def draw_frame_predicted_new(
    frame: np.ndarray,
    *,
    pred_mouse: Tuple[float, float],
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay model prediction (red arrow + optional 1‑σ ellipse)."""

    return _draw_frame(
        frame,
        mouse_vec=pred_mouse,
        color=(0, 0, 255),
        resize_to=resize_to,
    )

def draw_frame_predicted(
    frame: np.ndarray,
    *,
    pred_mean: Tuple[float, float],
    pred_std: Tuple[float, float] | None = None,
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay model prediction (red arrow + optional 1‑σ ellipse)."""
    if pred_std is not None:
        return _draw_frame(
            frame,
            mouse_vec=pred_mean,
            mouse_std=pred_std,
            color=(0, 0, 255),
            resize_to=resize_to,
        )
    else:
        return _draw_frame(
            frame,
            mouse_vec=pred_mean,
            color=(0, 0, 255),
            resize_to=resize_to,
        )
# -----------------------------------------------------------------------------
# Video helper
# -----------------------------------------------------------------------------

def render_video(
    frames: Iterable[np.ndarray],
    *,
    fps: int = 30,
    name: str | None = None,
    **wandb_kwargs,
) -> wandb.Video:
    """Stack *frames* ➜ *(T, H, W, 3)* uint8 and wrap as `wandb.Video`."""
    video = np.stack(list(frames), axis=0).astype(np.uint8)
    return wandb.Video(video, fps=fps, format="mp4", name=name, **wandb_kwargs)
