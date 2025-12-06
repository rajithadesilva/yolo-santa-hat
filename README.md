# Santa Hat + Face Detection (YOLO & MediaPipe)

This project uses real-time face detection to overlay a Santa hat on detected faces using your webcam.  
You can choose between a **YOLO-based** detector or a **MediaPipe Face Mesh** model that also supports **head tilt (roll)**.

Perfect for festive photo booths, office parties, or just having fun 🎅

---

## Demo

<img src="demo.webp" alt="Santa Hat Detection Demo" width="800">

---

## Backends at a Glance

| Script              | Backend              | Pros                                           | Notes                                      |
|---------------------|----------------------|------------------------------------------------|--------------------------------------------|
| `santa_hat_yolo.py` | YOLOv8 face detector | Very robust bounding-box face detection        | Uses `yolov8n-face.pt` (included)          |
| `santa_hat_mp.py`   | MediaPipe Face Mesh  | Uses landmarks, supports head roll estimation  | Optional face mesh visualisation           |

---

## Features

- 🔍 **Real-time** face detection  
- 🎅 **Dynamic Santa hat overlay** scaled to face size  
- ↻ **Auto-resume numbering** for captured photos  
- 🖼 **Full-screen UI** with status text  
- 💾 Save photos with a single key press  
- 🎚 Adjustable scale, offsets, and rotation (MediaPipe)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajithadesilva/yolo-santa-hat.git
cd yolo-santa-hat
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Choose either backend.

### YOLO backend

```bash
python santa_hat_yolo.py
```

- Uses YOLOv8 face detector (`yolov8n-face.pt`)
- Places/Scales hat using bounding boxes

### MediaPipe backend

```bash
python santa_hat_mp.py
```

- Uses Face Mesh landmarks
- Supports head roll (tilt) and landmark-based hat placement
- Optional mesh overlay for debugging

---

## Controls

| Key      | Action                                 |
|----------|-----------------------------------------|
| SPACE    | Capture photo into `photos/`            |
| q        | Quit                                    |

When a photo is taken:
- File is saved as: `photos/photo_<N>.png`
- “Photo Saved!” appears briefly at the centre

---

## Configuration

Both scripts share:

- **Camera index**

  ```python
  cap = cv2.VideoCapture(0)
  ```

- **Screen resolution** (for full-screen scaling)

  ```python
  screen_width = 1920
  screen_height = 1080
  ```

- **Output directory**

  ```python
  output_dir = "photos"
  ```

---

### YOLO Settings (`santa_hat_yolo.py`)

```python
HAT_SCALE_FACTOR = 1.5
HAT_OFFSET_X = -0.1
HAT_OFFSET_Y = -0.4
conf_threshold = 0.25
img_size = 1280
max_detections = 1000
```

---

### MediaPipe Settings (`santa_hat_mp.py`)

```python
HAT_SCALE_FACTOR = 1.6
HAT_VERTICAL_FACTOR = 0.55
HAT_ROTATION_FACTOR = 0.2   # adds ±45° tilt
DEBUG = False               # show/hide mesh
```

Landmark-based computations include:
- Cheek distance → hat size
- Forehead midpoint → hat placement
- Cheek vector → head roll angle

---

## Replacing the Hat Image

You can replace `santa_hat.png` with any transparent PNG.

```python
santa_hat = cv2.imread("my_new_hat.png", cv3.IMREAD_UNCHANGED)
```

Adjust scale/offset/rotation factors accordingly.

---

## Folder Structure

```
yolo-santa-hat/
│
├── santa_hat_yolo.py
├── santa_hat_mp.py
├── santa_hat.png
├── demo.webp
├── requirements.txt
└── photos/              # auto-created
```

---

## Licence

Distributed under the **MIT Licence**.

---

## Acknowledgements

Face Detection Model: https://github.com/YapaLab/yolo-face <br>
MediaPipe: https://github.com/google-ai-edge/mediapipe

---
## Contributing

PRs are welcome for:

- New festive overlays 🎩  
- Fun filters/effects  
- Improvements to detection or hat placement  
- Additional visual assets  

Enjoy spreading Christmas cheer! 🎄✨
