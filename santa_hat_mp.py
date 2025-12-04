"""
santa_hat_mediapipe.py

Description:
    A Python script that uses MediaPipe Face Mesh for real-time face landmark detection
    to overlay a Santa hat on detected faces, supporting head tilting (roll).
    Includes functionality to save images and dynamic numbering for saved photos.

Author: Rajitha de Silva
Date: 2025-12-04
Version: 1.3
"""

import cv2
import numpy as np
import os
import re
import mediapipe as mp

# ========== DEBUG / visualisation ==========
DEBUG = True  # Set True to draw full landmark mesh on faces

# ========== MediaPipe setup ==========
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices (MediaPipe Face Mesh)
# Forehead + cheeks (used for scale & tilt)
FOREHEAD_LM = 10
LEFT_CHEEK_LM = 234
RIGHT_CHEEK_LM = 454

# ========== Hat / appearance settings ==========
HAT_SCALE_FACTOR = 1.6      # Overall scale relative to cheek distance
HAT_VERTICAL_FACTOR = 0.55   # How much of hat height sits above forehead
HAT_ROTATION_FACTOR = 0.2   # -1.0: -45°, 0: no extra offset, +1.0: +45°

# Screen resolution (adjust to your screen if needed)
screen_width = 1920
screen_height = 1080

# ========== Camera setup ==========
cap = cv2.VideoCapture(0)  # Adjust this index if you have multiple webcams
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ========== Load Santa hat ==========
santa_hat = cv2.imread("santa_hat.png", cv2.IMREAD_UNCHANGED)  # Must have alpha
if santa_hat is None:
    print("Error: Could not load santa_hat.png")
    cap.release()
    exit()

if santa_hat.shape[2] < 4:
    print("Error: santa_hat.png has no alpha channel. Use a PNG with transparency.")
    cap.release()
    exit()

# ========== Output directory & naming ==========
output_dir = "photos"
os.makedirs(output_dir, exist_ok=True)


def get_starting_photo_count(directory):
    existing_files = os.listdir(directory)
    numbers = [
        int(re.search(r"photo_(\d+)", f).group(1))
        for f in existing_files
        if re.search(r"photo_(\d+)", f)
    ]
    return max(numbers) + 1 if numbers else 0


photo_count = get_starting_photo_count(output_dir)

# ========== Utility: alpha blending with clipping ==========
def overlay_image_alpha(background, overlay, x, y):
    """
    Overlay RGBA `overlay` onto BGR `background` at position (x, y) (top-left),
    handling alpha and clipping to image borders.
    """
    h, w = overlay.shape[:2]

    # Completely outside
    if x >= background.shape[1] or y >= background.shape[0] or x + w <= 0 or y + h <= 0:
        return background

    # Clip to valid ROI in background
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, background.shape[1])
    y2 = min(y + h, background.shape[0])

    # Corresponding region in overlay
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_cropped.shape[2] < 4:
        return background

    alpha = overlay_cropped[:, :, 3] / 255.0
    alpha = alpha[..., np.newaxis]

    bg_roi = background[y1:y2, x1:x2, :3]

    # Blend
    background[y1:y2, x1:x2, :3] = alpha * overlay_cropped[:, :, :3] + (1 - alpha) * bg_roi

    return background


# ========== Utility: draw rotated hat per face ==========
def add_hat_to_face(frame, face_landmarks):
    """
    Given a frame (BGR) and MediaPipe face_landmarks,
    compute head tilt & scale from cheek landmarks and overlay a rotated hat.
    """

    h, w, _ = frame.shape

    def lm_xy(idx):
        lm = face_landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    # Get key points
    fx, fy = lm_xy(FOREHEAD_LM)
    lx, ly = lm_xy(LEFT_CHEEK_LM)
    rx, ry = lm_xy(RIGHT_CHEEK_LM)

    # Distance between cheeks -> hat size
    dx = rx - lx
    dy = ry - ly
    cheek_dist = np.sqrt(dx * dx + dy * dy)
    if cheek_dist < 1:
        return frame  # too small / degenerate

    # Compute rotation angle (roll) based on cheek line
    # Image y-axis is downward; we flip sign to align visually
    base_angle_deg = -np.degrees(np.arctan2(dy, dx))

    # Extra correction: map HAT_ROTATION_FACTOR [-1,1] -> [-45°, +45°]
    extra_angle_deg = float(HAT_ROTATION_FACTOR) * 45.0
    angle_deg = base_angle_deg + extra_angle_deg

    # Scale hat relative to cheek distance
    base_hat_h, base_hat_w = santa_hat.shape[:2]
    scale = (cheek_dist * HAT_SCALE_FACTOR) / base_hat_w

    if scale <= 0:
        return frame

    # Resize hat
    resized_hat = cv2.resize(
        santa_hat, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    hat_h, hat_w = resized_hat.shape[:2]

    # Rotate hat around its centre
    center = (hat_w // 2, hat_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_hat = cv2.warpAffine(
        resized_hat,
        rot_mat,
        (hat_w, hat_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Position: centre hat horizontally on forehead, mostly above head
    hat_x = int(fx - hat_w / 2)
    hat_y = int(fy - hat_h * HAT_VERTICAL_FACTOR)

    frame = overlay_image_alpha(frame, rotated_hat, hat_x, hat_y)
    return frame


# ========== OpenCV window ==========
window_name = "MediaPipe Face Mesh Santa Hat"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

show_saved_message = False
save_frame = None

# ========== Main loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Mirror/flip
    frame = cv2.flip(frame, 1)

    # --- Resize + letterbox to screen size ---
    orig_h, orig_w = frame.shape[:2]
    scale = min(screen_width / orig_w, screen_height / orig_h)
    resized_w = int(orig_w * scale)
    resized_h = int(orig_h * scale)

    frame = cv2.resize(frame, (resized_w, resized_h))

    pad_left = (screen_width - resized_w) // 2
    pad_right = screen_width - resized_w - pad_left
    pad_top = (screen_height - resized_h) // 2
    pad_bottom = screen_height - resized_h - pad_top

    frame = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # --- MediaPipe Face Mesh detection ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Add hat based on landmarks
            frame = add_hat_to_face(frame, face_landmarks)

            # If debugging, draw full mesh on top
            if DEBUG:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

    # Copy AFTER hats (and optional mesh) but BEFORE any text overlay (for saving)
    save_frame = frame.copy()

    # --- UI Text / “Photo Saved” message ---
    if show_saved_message:
        text = "Photo Saved!"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )
        show_saved_message = False
    else:
        text = "Press SPACE to take a photo"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (screen_width - text_size[0]) // 2
        text_y = screen_height - 20
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Show frame
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        if save_frame is not None:
            photo_path = os.path.join(output_dir, f"photo_{photo_count}.png")
            cv2.imwrite(photo_path, save_frame)
            print(f"Photo saved: {photo_path}")
            photo_count += 1
            show_saved_message = True

# ========== Cleanup ==========
cap.release()
face_mesh.close()
cv2.destroyAllWindows()
