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
    buttons: Sequence[bool] | None = None,
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
    buttons    : iterable[bool] — HUD row (label = keybind).
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

    # ------------------------------------------------------------------ HUD at bottom
    if buttons is not None:
        n_keys = len(buttons)
        
        # Font settings for crisp text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6 if w >= 512 else 0.5
        font_thickness = 2
        
        # Two-row layout for many keybinds
        if n_keys > 6:
            # Split into two rows
            first_row = n_keys // 2 + n_keys % 2  # ceiling division
            second_row = n_keys - first_row
            
            # Position at bottom
            row_height = 30
            y_positions = [h - row_height * 2 - 10, h - row_height - 10]
            margin = 15
            
            for i, pressed in enumerate(buttons):
                label = KEYBINDS[i] if i < len(KEYBINDS) else str(i)
                col = color if pressed else (150, 150, 150)
                
                # Determine row and position
                if i < first_row:
                    row = 0
                    row_idx = i
                    keys_in_row = first_row
                else:
                    row = 1
                    row_idx = i - first_row
                    keys_in_row = second_row
                
                # Calculate x position with better spacing
                available_width = w - 2 * margin
                step_x = available_width // keys_in_row
                x = margin + row_idx * step_x + step_x // 2  # center in cell
                
                # Get text size to center it
                (text_w, text_h), _ = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                x -= text_w // 2  # center text
                
                # Draw background rectangle for better visibility
                padding = 6
                if pressed:
                    cv2.rectangle(
                        out, 
                        (x - padding, y_positions[row] - text_h - padding),
                        (x + text_w + padding, y_positions[row] + padding - 2),
                        (60, 60, 60), -1
                    )
                else:
                    # Subtle background for unpressed keys too
                    cv2.rectangle(
                        out, 
                        (x - padding, y_positions[row] - text_h - padding),
                        (x + text_w + padding, y_positions[row] + padding - 2),
                        (30, 30, 30), -1
                    )
                
                # Draw text
                cv2.putText(
                    out, label, (x, y_positions[row]),
                    font, font_scale, col, font_thickness, cv2.LINE_AA,
                )
        else:
            # Single row for few keybinds
            y0 = h - 25  # bottom position
            margin = 15
            available_width = w - 2 * margin
            step_x = available_width // max(n_keys, 1)
            
            for i, pressed in enumerate(buttons):
                label = KEYBINDS[i] if i < len(KEYBINDS) else str(i)
                col = color if pressed else (150, 150, 150)
                x = margin + i * step_x + step_x // 2
                
                # Get text size to center it
                (text_w, text_h), _ = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                x -= text_w // 2
                
                # Draw background
                padding = 6
                if pressed:
                    cv2.rectangle(
                        out, 
                        (x - padding, y0 - text_h - padding),
                        (x + text_w + padding, y0 + padding - 2),
                        (60, 60, 60), -1
                    )
                else:
                    cv2.rectangle(
                        out, 
                        (x - padding, y0 - text_h - padding),
                        (x + text_w + padding, y0 + padding - 2),
                        (30, 30, 30), -1
                    )
                
                cv2.putText(
                    out, label, (x, y0),
                    font, font_scale, col, font_thickness, cv2.LINE_AA,
                )

    return out

# -----------------------------------------------------------------------------
# Friendly wrappers
# -----------------------------------------------------------------------------

def draw_frame_groundtruth(
    frame: np.ndarray,
    *,
    gt_mouse: Tuple[float, float],
    gt_buttons: Sequence[bool] | np.ndarray,
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay ground‑truth mouse movement and button presses (green)."""
    return _draw_frame(
        frame,
        mouse_vec=gt_mouse,
        buttons=list(gt_buttons),
        color=(0, 255, 0),
        resize_to=resize_to,
    )

def draw_frame_predicted_new(
    frame: np.ndarray,
    *,
    pred_mouse: Tuple[float, float],
    pred_buttons: Sequence[float] | np.ndarray,  # logits or probs
    threshold: float = 0.5,
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay model prediction (red arrow + optional 1‑σ ellipse)."""
    prob = np.asarray(pred_buttons, dtype=np.float32)
    if prob.ndim == 0:
        prob = prob[None]
    pressed = prob >= threshold

    return _draw_frame(
        frame,
        mouse_vec=pred_mouse,
        buttons=pressed.tolist(),
        color=(0, 0, 255),
        resize_to=resize_to,
    )

def draw_frame_predicted(
    frame: np.ndarray,
    *,
    pred_mean: Tuple[float, float],
    pred_std: Tuple[float, float] | None = None,
    pred_buttons: Sequence[float] | np.ndarray,  # logits or probs
    threshold: float = 0.5,
    resize_to: int | None = 512,
) -> np.ndarray:
    """Overlay model prediction (red arrow + optional 1‑σ ellipse)."""
    prob = np.asarray(pred_buttons, dtype=np.float32)
    if prob.ndim == 0:
        prob = prob[None]
    pressed = prob >= threshold

    return _draw_frame(
        frame,
        mouse_vec=pred_mean,
        mouse_std=pred_std,
        buttons=pressed.tolist(),
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
