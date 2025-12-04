"""
santa_hat.py

Description:
    A Python script that uses YOLO for real-time face detection to overlay a Santa hat on detected faces.
    Includes functionality to save images and dynamic numbering for saved photos.

Author: Rajitha de Silva
Date: 2024-11-25
Version: 1.2
"""

import cv2
from ultralytics import YOLO
import numpy as np
import os
import re

# Load YOLO model
model = YOLO("yolov8n-face.pt")  # Path to your YOLO model

# Detection settings
conf_threshold = 0.25
img_size = 1280
line_width = 1
max_detections = 1000

# Open webcam
cap = cv2.VideoCapture(0)  # Adjust this index if you have multiple webcams

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load Santa hat image
santa_hat = cv2.imread("santa_hat.png", cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Control variables for scaling and offsets
HAT_SCALE_FACTOR = 1.5  # Factor to scale the hat relative to face width
HAT_OFFSET_X = -0.1     # Horizontal offset as a fraction of face width (-1.0 to 1.0)
HAT_OFFSET_Y = -0.4     # Vertical offset as a fraction of face height

# Get screen resolution (you can also hardcode these values)
screen_width = 1920  # Replace with your screen's width
screen_height = 1080  # Replace with your screen's height

# Create output directory for saving images
output_dir = "photos"
os.makedirs(output_dir, exist_ok=True)

# Determine the starting photo count based on existing files
def get_starting_photo_count(directory):
    existing_files = os.listdir(directory)
    numbers = [int(re.search(r"photo_(\d+)", f).group(1)) for f in existing_files if re.search(r"photo_(\d+)", f)]
    return max(numbers) + 1 if numbers else 0

photo_count = get_starting_photo_count(output_dir)

def overlay_image_alpha(background, overlay, x, y, scale):
    """Overlay `overlay` onto `background` at position `(x, y)` with scaling."""
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape

    # Ensure the overlay is within the background frame
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background  # Do not overlay if out of bounds

    # If there is no alpha channel, just return background
    if overlay.shape[2] < 4:
        return background

    alpha_overlay = overlay[:, :, 3] / 255.0  # Alpha channel
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):  # Iterate over color channels
        background[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y+h, x:x+w, c]
        )
    return background

# Create a named window for the display and set it to full-screen
window_name = "YOLO Face Detection with Santa Hat"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

show_saved_message = False  # Whether to show "Photo Saved" message
save_frame = None           # Will hold the frame to save (with hat, without text)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Get original frame dimensions
    orig_height, orig_width = frame.shape[:2]

    # Calculate scaling factor to maintain aspect ratio
    scale = min(screen_width / orig_width, screen_height / orig_height)
    resized_width = int(orig_width * scale)
    resized_height = int(orig_height * scale)

    # Resize the frame while maintaining aspect ratio
    frame = cv2.resize(frame, (resized_width, resized_height))

    # Add padding to centre the frame
    pad_left = (screen_width - resized_width) // 2
    pad_right = screen_width - resized_width - pad_left
    pad_top = (screen_height - resized_height) // 2
    pad_bottom = screen_height - resized_height - pad_top

    frame = cv2.copyMakeBorder(
        frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Predict with YOLO
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        max_det=max_detections,
        imgsz=img_size,
        verbose=False  # Suppress model output
    )

    # Overlay detections and Santa hats on frame
    for result in results:
        for box in result.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

            # Calculate face size
            face_width = x2 - x1
            face_height = y2 - y1

            # Scale Santa hat based on face size and control variable
            hat_scale = face_width / santa_hat.shape[1] * HAT_SCALE_FACTOR

            # Position the Santa hat using offsets
            hat_x = int(x1 + face_width * HAT_OFFSET_X)
            hat_y = int(y1 + face_height * HAT_OFFSET_Y)

            # Ensure the coordinates are within the frame
            if hat_y < 0:
                hat_y = 0

            try:
                frame = overlay_image_alpha(frame, santa_hat, hat_x, hat_y, hat_scale)
            except Exception:
                pass

    # ✅ Now that hats are drawn, copy this version for saving (no text yet)
    save_frame = frame.copy()

    # Show "Photo Saved" message in the middle of the frame
    if show_saved_message:
        text = "Photo Saved!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        show_saved_message = False  # Reset after displaying for one frame
    else:
        # Add text at the bottom prompting to take a photo
        text = "Press SPACE to take a photo"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = screen_height - 20  # Slight gap from bottom
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in full-screen
    cv2.imshow(window_name, frame)

    # Handle key events
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Quit on 'q'
        break
    elif key & 0xFF == ord(' '):  # Save photo on spacebar
        if save_frame is not None:
            photo_path = os.path.join(output_dir, f"photo_{photo_count}.png")
            cv2.imwrite(photo_path, save_frame)  # Save the copy without text overlay
            print(f"Photo saved: {photo_path}")
            photo_count += 1
            show_saved_message = True  # Set flag to show "Photo Saved" message

# Release resources
cap.release()
cv2.destroyAllWindows()

