import imageio as imio
import cv2
import numpy as np
import math
import torch
from typing import Optional
from contextlib import contextmanager

from owl_idms.data.cod_data import CoDDataset
from inference.utils import load_model, infer_actions, DEFAULT_MODEL_PATH, DEFAULT_CFG_PATH, WINDOW_SIZE

# Global configuration
KEYBINDS = ["W", "A", "S", "D", "LSHIFT", "SPACE", "R", "F", "E", "LMB", "RMB"]
MINIMUM_FRAME_SIZE = 512
# Colors (BGR format for OpenCV)
COLOR_PRESSED = (50, 200, 50)      # Green
COLOR_UNPRESSED = (100, 100, 100)  # Gray
COLOR_TEXT = (255, 255, 255)       # White
COLOR_BACKGROUND = (30, 30, 30)    # Dark gray
COLOR_MOUSE_ARROW = COLOR_PRESSED  # Green
COLOR_UNCERTAINTY = (100, 255, 255) # Yellow-ish
COLOR_LMB_SECTOR = COLOR_PRESSED  # Green
COLOR_RMB_SECTOR = COLOR_PRESSED  # Green

# Key dimensions
KEY_SIZE = 30
KEY_MARGIN = 5
SHIFT_WIDTH = int(KEY_SIZE * 2 + KEY_MARGIN)  # Two keys worth of width
SPACE_WIDTH = int(KEY_SIZE * 5)

# Mouse compass dimensions
COMPASS_RADIUS = 48  # 80% of 80
COMPASS_START_X_PERCENT = 0.80

# Mouse button arc dimensions
MOUSE_BUTTON_OFFSET = 6  # Pixels outside the main circle
MOUSE_BUTTON_THICKNESS = 6  # Increased thickness

# Arrow scaling parameters
ARROW_SCALE_FACTOR = 0.3  # Base scaling factor for arrow length
ARROW_MIN_LENGTH = 5      # Minimum arrow length in pixels
ARROW_MAX_SCALE = 0.9     # Maximum scale relative to compass radius

START_X_PERCENT = 0.12
START_Y_PERCENT = 0.65


@contextmanager
def _rescale_icons(ratio: float):
    """
    Rescale all icon sizes to the ratio of the original video to the 512x512 video
    """
    global KEY_SIZE, KEY_MARGIN, SHIFT_WIDTH, SPACE_WIDTH, COMPASS_RADIUS, MOUSE_BUTTON_OFFSET, MOUSE_BUTTON_THICKNESS
    global ARROW_SCALE_FACTOR, ARROW_MIN_LENGTH, ARROW_MAX_SCALE
    global START_X_PERCENT, START_Y_PERCENT
    
    try:
        old_values = {
            "KEY_SIZE": KEY_SIZE,
            "KEY_MARGIN": KEY_MARGIN,
            "SHIFT_WIDTH": SHIFT_WIDTH,
            "SPACE_WIDTH": SPACE_WIDTH,
            "COMPASS_RADIUS": COMPASS_RADIUS,
        }
        
        KEY_SIZE *= ratio ; KEY_SIZE = int(KEY_SIZE)
        KEY_MARGIN *= ratio ; KEY_MARGIN = int(KEY_MARGIN)
        SHIFT_WIDTH *= ratio ; SHIFT_WIDTH = int(SHIFT_WIDTH)
        SPACE_WIDTH *= ratio ; SPACE_WIDTH = int(SPACE_WIDTH)
        COMPASS_RADIUS *= ratio ; COMPASS_RADIUS = int(COMPASS_RADIUS)
        MOUSE_BUTTON_OFFSET *= ratio ; MOUSE_BUTTON_OFFSET = int(MOUSE_BUTTON_OFFSET)
        MOUSE_BUTTON_THICKNESS *= ratio ; MOUSE_BUTTON_THICKNESS = int(MOUSE_BUTTON_THICKNESS)
        ARROW_SCALE_FACTOR *= ratio ; ARROW_SCALE_FACTOR = float(ARROW_SCALE_FACTOR)
        ARROW_MIN_LENGTH *= ratio ; ARROW_MIN_LENGTH = int(ARROW_MIN_LENGTH)
        ARROW_MAX_SCALE *= ratio ; ARROW_MAX_SCALE = float(ARROW_MAX_SCALE)
        yield
    finally:
        for key, value in old_values.items():
            globals()[key] = value


def _standardize_video(video: torch.Tensor) -> torch.Tensor:
    """
    Convert video to 0-255 and permute channels to (T, H, W, C).
    """
    video = video.float()
    video = (video - video.min()) / (video.max() - video.min())
    video = (video * 255).cpu().to(dtype=torch.uint8)
    
    # convert channels
    if video.shape[1] == 3:
        video = video.permute(0, 2, 3, 1) # need to be [n,h,w,c]
    return video


def _draw_buttons(
    frame: np.ndarray,
    button_sequence: list[bool],
) -> None:
    """
    Draw keyboard buttons on the frame.
    
    Args:
        frame: numpy array representing the image frame
        button_sequence: list of bools corresponding to KEYBINDS (excluding LMB, RMB)
    """
    # Starting position for keyboard layout (bottom left)
    start_x = int(frame.shape[1] * START_X_PERCENT)
    start_y = int(frame.shape[0] * START_Y_PERCENT)
    
    # Key positions - organized by rows
    key_positions = {
        # Top row: W E R (W above S)
        "W": (start_x + (KEY_SIZE + KEY_MARGIN) * 1, start_y),
        "E": (start_x + (KEY_SIZE + KEY_MARGIN) * 2, start_y),
        "R": (start_x + (KEY_SIZE + KEY_MARGIN) * 3, start_y),
        
        # Middle row: A S D F
        "A": (start_x + (KEY_SIZE + KEY_MARGIN) * 0, start_y + KEY_SIZE + KEY_MARGIN),
        "S": (start_x + (KEY_SIZE + KEY_MARGIN) * 1, start_y + KEY_SIZE + KEY_MARGIN),
        "D": (start_x + (KEY_SIZE + KEY_MARGIN) * 2, start_y + KEY_SIZE + KEY_MARGIN),
        "F": (start_x + (KEY_SIZE + KEY_MARGIN) * 3, start_y + KEY_SIZE + KEY_MARGIN),
        
        # Bottom row: LSHIFT SPACE
        "LSHIFT": (start_x - (KEY_SIZE + KEY_MARGIN), start_y + (KEY_SIZE + KEY_MARGIN) * 2),
        "SPACE": (start_x + SHIFT_WIDTH + KEY_MARGIN, start_y + (KEY_SIZE + KEY_MARGIN) * 2),
    }
    
    # Draw each key
    for i, key in enumerate(KEYBINDS[:-2]):  # Exclude LMB and RMB
        if key in key_positions:
            x, y = key_positions[key]
            
            # Determine key dimensions
            if key == "LSHIFT":
                width = SHIFT_WIDTH
                height = KEY_SIZE
            elif key == "SPACE":
                width = SPACE_WIDTH
                height = KEY_SIZE
            else:
                width = KEY_SIZE
                height = KEY_SIZE
            
            # Determine color based on pressed state
            color = COLOR_PRESSED if button_sequence[i] else COLOR_UNPRESSED
            
            # Draw key background
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            
            # Draw key border
            cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_TEXT, 1)
            
            # Draw key label
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y + (height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def _draw_mouse(
    frame: np.ndarray,
    LMB_on: bool,
    RMB_on: bool,
    mouse_vec: tuple[float, float],
    mouse_std: tuple[float, float],
    center: tuple[int, int],
) -> None:
    """
    Draw mouse compass with direction arrow and uncertainty cone.
    Arrow length is proportional to mouse movement magnitude.
    
    Args:
        frame: numpy array representing the image frame
        LMB_on: bool indicating if left mouse button is pressed
        RMB_on: bool indicating if right mouse button is pressed
        mouse_vec: tuple of floats (x, y) representing mouse direction
        mouse_std: tuple of floats (x_std, y_std) representing uncertainty
        center: tuple (x, y) for compass center position
    """
    # Draw compass circle
    cv2.circle(frame, center, COMPASS_RADIUS, COLOR_TEXT, 2)
    
    # Calculate outer radius for mouse buttons (slightly outside the main compass)
    button_radius = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET
    
    # Draw LMB arc (top-left 45 degrees) - outside the main circle and thicker
    color_lmb = COLOR_LMB_SECTOR if LMB_on else (60, 60, 60)  # Gray when off
    cv2.ellipse(frame, center, (button_radius, button_radius), 
               0, 225, 270, color_lmb, MOUSE_BUTTON_THICKNESS)
    
    # Draw RMB arc (top-right 45 degrees) - outside the main circle and thicker
    color_rmb = COLOR_RMB_SECTOR if RMB_on else (60, 60, 60)  # Gray when off
    cv2.ellipse(frame, center, (button_radius, button_radius), 
               0, 270, 315, color_rmb, MOUSE_BUTTON_THICKNESS)
    
    text_loc_lmb = center[0] - COMPASS_RADIUS + 10, center[1] - COMPASS_RADIUS - 10 
    cv2.putText(frame, 'LMB', text_loc_lmb, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    text_loc_rmb = center[0], center[1] - COMPASS_RADIUS - 10
    cv2.putText(frame, 'RMB', text_loc_rmb, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    
    # Calculate mouse direction
    mouse_x, mouse_y = mouse_vec
    magnitude = math.sqrt(mouse_x**2 + mouse_y**2)
    
    if magnitude > 0:
        # Normalize to unit vector
        unit_x = mouse_x / magnitude
        unit_y = mouse_y / magnitude
        
        # Calculate arrow length proportional to magnitude
        # Scale by magnitude but clamp to reasonable bounds
        arrow_scale = max(
            ARROW_MIN_LENGTH,  # Ensure minimum visible arrow
            min(
                COMPASS_RADIUS * magnitude * ARROW_SCALE_FACTOR,  # Proportional to magnitude
                COMPASS_RADIUS * ARROW_MAX_SCALE  # Cap at maximum scale
            )
        )
        
        end_x = int(center[0] + unit_x * arrow_scale)
        end_y = int(center[1] - unit_y * arrow_scale)  # Negative because y-axis is inverted
        
        # Draw uncertainty cone
        std_x, std_y = mouse_std
        std_magnitude = math.sqrt(std_x**2 + std_y**2)
        
        if std_magnitude > 0:
            # Calculate cone angle based on uncertainty
            cone_angle = min(math.atan2(std_magnitude, 1) * 180 / math.pi, 45)  # Cap at 45 degrees
            
            # Draw uncertainty cone as a filled polygon
            angle_rad = math.atan2(-unit_y, unit_x)  # Negative y because of inverted axis
            angle_deg = angle_rad * 180 / math.pi
            
            # Create cone points
            cone_points = []
            cone_points.append(center)  # Apex at center
            
            # Add points along the arc
            for offset in np.linspace(-cone_angle, cone_angle, 20):
                point_angle = (angle_deg + offset) * math.pi / 180
                px = int(center[0] + arrow_scale * math.cos(point_angle))
                py = int(center[1] - arrow_scale * math.sin(point_angle))
                cone_points.append((px, py))
            
            # Draw semi-transparent cone
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(cone_points)], COLOR_UNCERTAINTY)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw direction arrow with proportional length
        cv2.arrowedLine(frame, center, (end_x, end_y), COLOR_MOUSE_ARROW, 3, tipLength=0.3)
        
        # Optional: Display magnitude as text for debugging
        # magnitude_text = f"Mag: {magnitude:.2f}"
        # cv2.putText(frame, magnitude_text, (center[0] - 40, center[1] + button_radius + 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
    
    # Draw center dot
    cv2.circle(frame, center, 3, COLOR_TEXT, -1)
    
    # Add labels
    cv2.putText(frame, "Mouse", (center[0] - 25, center[1] + button_radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

def _draw_frame(
    frame: np.ndarray,
    buttons: list[bool] | torch.Tensor,
    mouse_vec: tuple[float, float],
    mouse_std: tuple[float, float],
) -> np.ndarray:
    """
    Overlay keyboard & mouse on a single frame and return it.
    """
    buttons, lmb, rmb = buttons[:-2], buttons[-2], buttons[-1]
    _draw_buttons(frame, buttons)

    frame_height, frame_width = frame.shape[:2]
    margin = 30
    compass_x = frame_width - margin - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET
    compass_y = int(frame_height * START_Y_PERCENT)
    compass_x = int(compass_x * COMPASS_START_X_PERCENT)

    # Ensure compass fits within frame boundaries
    min_x = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET + 20
    max_x = frame_width - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET - 20
    compass_x = max(min_x, min(compass_x, max_x))
    min_y = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET + 30
    max_y = frame_height - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET - 30
    compass_y = max(min_y, min(compass_y, max_y))

    _draw_mouse(frame,
                LMB_on=lmb, RMB_on=rmb,
                mouse_vec=mouse_vec, mouse_std=mouse_std,
                center=(compass_x, compass_y))
    return frame

def _draw_video(
    video: torch.Tensor,
    buttons: torch.Tensor,
    mouse_vec: torch.Tensor,
    mouse_std: torch.Tensor,
    save_path: Optional[str] = None,
    fps: int = 30,
    arrow_scale_factor: Optional[float] = None,
    arrow_max_scale: Optional[float] = None,
) -> list[np.ndarray]:
    """
    Draw video with input device monitoring overlays.
    
    Args:
        video: torch tensor of video frames
        buttons: torch tensor of button states
        mouse_vec: torch tensor of mouse vectors
        mouse_std: torch tensor of mouse standard deviations
        save_path: optional path to save video
        fps: frames per second for output video
        arrow_scale_factor: optional override for arrow scaling
        arrow_max_scale: optional override for maximum arrow scale
    """
    # Update global arrow parameters if provided
    global ARROW_SCALE_FACTOR, ARROW_MAX_SCALE
    if arrow_scale_factor is not None:
        ARROW_SCALE_FACTOR = arrow_scale_factor
    if arrow_max_scale is not None:
        ARROW_MAX_SCALE = arrow_max_scale

    video = video.numpy()
    # upscale image to 512x512 only if < 512 on smallest dimension.
    if video.shape[1] < MINIMUM_FRAME_SIZE or video.shape[2] < MINIMUM_FRAME_SIZE:
        frames = [cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC) for frame in video]
        ratio = 1.0
    else:
        frames = video
        ratio = min(video.shape[1] / MINIMUM_FRAME_SIZE, video.shape[2] / MINIMUM_FRAME_SIZE)

    with _rescale_icons(ratio):
        frames = [
            _draw_frame(frame, buttons[i], mouse_vec[i], mouse_std[i])
            for i, frame in enumerate(frames)
        ]
    if save_path is not None:
        imio.mimsave(save_path, frames, fps=fps)

    return frames


def overlay_infer_actions(
    model_path: str    = DEFAULT_MODEL_PATH,
    cfg_path:   str    = DEFAULT_CFG_PATH,
    *,
    video_pt_path: str = None,
    video: torch.Tensor = None,
    video_display: torch.Tensor = None, # In case you want to display a higher res video than the input to the model
    use_dataset: bool = False,
    dataset: Optional[CoDDataset]     = None,
    dataset_index: Optional[int]      = None,
    save_path: Optional[str]          = "inferred_actions.mp4",
    fps: int                          = 30,
    arrow_scale_factor: Optional[float] = None,
    arrow_max_scale: Optional[float]    = None,
    device: str                       = 'cuda',
) -> list[np.ndarray]:
    """
    Run inference on a single video and overlay predicted mouse & button actions.

    You may either:
      1) Pass in `dataset` (with a `.paths` attribute) plus `dataset_index`, or
      2) Directly supply `video_pt_path` (and optionally `mouse_pt_path`, `buttons_pt_path`).

    Args:
        model_path:       Path to the IDM checkpoint (.pt).
        cfg_path:         Path to the model YAML config.
        video_pt_path:    Path to a video tensor (.pt). Overrides `dataset`.
        video:            A video tensor (T, C, H, W) to run inference on. Overrides `video_pt_path`.
        video_display:    A video tensor (T, C, H, W) of the video to display. Overrides `video`.
        dataset:          Optional dataset instance (must have `.paths`).
        dataset_index:    Which entry of `dataset.paths` to use.
        save_path:        Where to write the output .mp4 (if not None).
        fps:              Frame rate for the output video.
        arrow_scale_factor:  Override for arrow length scaling.
        arrow_max_scale:     Override for max arrow-to-compass scale.
        device:             Device to run inference on.
    Returns:
        A list of annotated frames (as H×W×3 uint8 numpy arrays).
    """
    # -------------------------------------------------------------------------
    # 1. Validate inputs
    # -------------------------------------------------------------------------
    # Validate that exactly one video source is provided
    video_sources = sum([video_pt_path is not None, video is not None, use_dataset])
    assert video_sources == 1, "Exactly one of video_pt_path, video, or use_dataset must be provided"

    # Validate dataset-related inputs
    if dataset_index is not None:
        assert use_dataset, "dataset_index can only be provided when use_dataset=True"
    
    if use_dataset:
        assert dataset is not None, "dataset must be provided when use_dataset=True"
        dataset_index = dataset_index or 0

    if not use_dataset:
        assert sum([video_pt_path is not None, video is not None]) == 1, "Exactly one of video_pt_path or video must be provided when use_dataset=False"

    if video_display is not None:
        assert video is not None, "video must be provided when video_display is provided"

    # -------------------------------------------------------------------------
    # 2. Load the raw video tensor
    # -------------------------------------------------------------------------
    if video_pt_path is not None:
        video = torch.load(video_pt_path, map_location="cpu", mmap=True)
    
    if dataset is not None:
        video_pt_path, *_ = dataset.paths[dataset_index]
        video = torch.load(video_pt_path, map_location="cpu", mmap=True)

    if video is not None:
        video = _standardize_video(video)

    if video_display is not None:
        video_display = _standardize_video(video_display)
        assert video_display.shape[0] == video.shape[0], "video_display and video must have the same temporal dimensions"

    # -------------------------------------------------------------------------
    # 3. Load the model and infer actions
    # -------------------------------------------------------------------------
    model = load_model(model_path, cfg_path, device=device)
    # model inference expects (T, C, H, W) but drawing expects (T, H, W, C). we standardize to the latter but permute for the former
    frames, buttons, mouse_vec, mouse_std = infer_actions(model, video.permute(0,3,1,2), window_size=WINDOW_SIZE) 
    display_frames = _standardize_video(frames)

    if video_display is not None:
        # manually offset the video display by the window size so we only display frames with explicitly inferred actions
        display_frames = video_display[WINDOW_SIZE:]


    # -------------------------------------------------------------------------
    # 4. Overlay predictions and save/return frames
    # -------------------------------------------------------------------------
    frames = _draw_video(
        display_frames,
        buttons,
        mouse_vec,
        mouse_std,
        save_path=save_path,
        fps=fps,
        arrow_scale_factor=arrow_scale_factor,
        arrow_max_scale=arrow_max_scale,
    )
    return frames


if __name__ == "__main__":
    mp4_path = '/home/sami/owl_idms/inference/trimmed.mp4'
    # downsample entire video to 1x1 at 128x128, one by one, using PIL, into a list
    frames = []

    import moviepy as mp
    import torchvision
    resize = torchvision.transforms.Resize((128, 128))
    video = mp.VideoFileClip(mp4_path)
    frames = list(video.iter_frames())
    frames = torch.stack([torch.from_numpy(frame) for frame in frames])
    hd_frames = frames.permute(0, 3, 1, 2) # (T, C, H, W)
    model_frames = [resize(frame) for frame in hd_frames]
    model_frames = torch.stack(model_frames)
    print(model_frames.shape)

    overlay_infer_actions(
        model_path='/home/sami/owl_idms/checkpoints/v0/step_31500.pt',
        cfg_path='/home/sami/owl_idms/inference/stformer.yml',
        use_dataset=True,
        dataset=CoDDataset(),
        dataset_index=0,
    )

# # Example usage
# if __name__ == "__main__":
#     # Example button states (9 buttons, excluding LMB and RMB)
#     button_states = [True, False, True, False, True, False, False, True, False]
    
#     # Example mouse state
#     LMB_pressed = True
#     RMB_pressed = False
#     mouse_direction = (0.7, 0.7)  # Diagonal up-right
#     mouse_uncertainty = (0.2, 0.1)

#     vidpath, mousepath, buttonpath = '.pt'

#     from owl_idms.data.cod_data import CoDDataset
#     ds = CoDDataset()
#     vidpath, mousepath, buttonpath = ds.paths[1]
#     video = torch.load(vidpath, map_location='cpu', mmap=True)
#     mouse = torch.load(mousepath, map_location='cpu', mmap=True)
#     buttons = torch.load(buttonpath, map_location='cpu', mmap=True)

#     min_len = min(len(video), len(mouse), len(buttons))
#     video = video[:min_len]
#     mouse = mouse[:min_len]
#     buttons = buttons[:min_len]

#     video = video.float()
#     buttons = buttons
#     mouse = mouse
    
#     # You can adjust arrow scaling parameters here
#     _draw_video(video, buttons, mouse, torch.zeros_like(mouse), 
#                 save_path="groundtruth.mp4",
#                 arrow_scale_factor=0.3,  # Adjust this to control arrow sensitivity
#                 arrow_max_scale=0.9)     # Maximum arrow length relative to compass