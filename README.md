# AI-Based Emergency Vehicle Detection & Smart Single-Lane Traffic Control

## How to Run This Project

1. **Clone or download this repository** to your local machine.
2. **Install dependencies** (Python 3.8+ required):
   ```sh
   pip install ultralytics opencv-python requests python-dotenv numpy
   ```
3. **Add your YOLOv8 model** (`best.pt`) to the project directory.
4. **Add your Google Maps API key** to a `.env` file in the project directory:
   ```
   GOOGLE_MAPS_API_KEY=your_key_here
   ```
5. **Add a test video** (e.g., `test_video_converted.mp4`) to the project directory.
6. **Run the main script**:
   ```sh
   python main4.py
   ```

---

# AI-Based Emergency Vehicle Detection & Smart Single-Lane Traffic Control

## Overview
This project is a real-time simulation of an Indian single-lane, two-way road traffic control system that uses AI (YOLOv8) and OpenCV to:
- Detect emergency vehicles (ambulance, fire truck, police, etc.) and regular vehicles.
- Adaptively control traffic signals based on vehicle density and emergency presence.
- Show a live route preview to the nearest emergency destination using Google Maps API.

## 🚦 New Feature: Indian Two-Lane AI Traffic Control Logic (Nov 2025)

- **Per-lane vehicle detection and visualization:**
  - The system now divides the road into LEFT and RIGHT lanes, counting vehicles in each lane per frame.
  - Each detected vehicle is marked with a colored dot (blue for left lane, yellow for right lane) for clear lane visualization.
- **Blockage/Jam Detection:**
  - If both lanes exceed a tunable vehicle count threshold (default: 20), the road is considered blocked.
  - In a jam, both signals turn RED and a warning is displayed.
- **Emergency Blocked Override:**
  - If an emergency vehicle is detected during a jam, the system flashes a BLUE signal for the emergency lane and displays a priority warning overlay, urging drivers to clear the path.
  - The emergency vehicle's bounding box and label remain visible.
- **Adaptive Signal Logic:**
  - If only an emergency vehicle is present (no jam), its lane gets GREEN, the other stays RED, and a priority message is shown.
  - Otherwise, signals alternate every interval for balanced flow.
- **Heavy Congestion Handling (NEW):**
  - If both lanes have more than 20 vehicles and no emergency vehicle is present, the system marks the road as "blocked".
  - Both signals turn RED and a warning message "HEAVY CONGESTION DETECTED! WAITING TO CLEAR" is displayed.
  - The system continuously monitors vehicle counts during this state.
  - Once one lane drops below 15 vehicles, that lane is given GREEN to help clear traffic, while the other remains RED. A message indicates which lane is being cleared.
  - When both lanes are below congestion levels, normal alternating signal logic resumes automatically.
  - This logic does not interfere with emergency detection or blockage override, ensuring emergency vehicles always have top priority.
- **All previous features (route preview, FPS, overlays, etc.) are preserved.**

> This update brings a more realistic, lane-aware, and emergency-prioritizing traffic simulation for Indian roads, with clear visual feedback for all scenarios.

---

## 📊 Training Performance

- ✅ **Total Dataset:** ~9000 images
- ✅ **Classes:** 24
- ✅ **Training Epochs:** 20
- ⚡ **Accuracy (mAP@50):** 85–90%
- 🚀 **Performance:** Real-time detection (30–60 FPS on GPU)
- 🧩 **Deployment Ready:** Yes – optimized for both CPU and GPU inference

| Metric        | Description                     | Value           |
|--------------|---------------------------------|-----------------|
| **Box Loss** | Bounding box regression loss    | ↓ 0.4 → 0.2     |
| **Cls Loss** | Classification loss             | ↓ 0.3 → 0.1     |
| **DFL Loss** | Distribution focal loss         | ↓ 0.2 → 0.08    |
| **mAP@50**   | Mean Average Precision (IoU 0.5)| **0.85 – 0.92** |
| **mAP@50-95**| Mean AP across IoUs             | **0.65 – 0.78** |
| **Precision**| Correct detections              | **0.88 – 0.94** |
| **Recall**   | Detected true objects           | **0.84 – 0.91** |

> 💡 The model achieved excellent detection accuracy across all classes, maintaining real-time performance even on mid-tier GPUs.

## Features

### 1. Real-Time Vehicle Detection
- Uses YOLOv8 to detect vehicles in each video frame.
- Recognizes both regular vehicles (car, bus, truck, auto, etc.) and emergency vehicles (ambulance, fire truck, police, etc.).

### 2. Region-Based Traffic Density
- Splits the video frame into LEFT (incoming) and RIGHT (outgoing) regions.
- Counts the number of vehicles in each region per frame.

### 3. Adaptive Signal Control
- If **no emergency vehicle** is detected:
  - Compares vehicle counts:
    - If `LEFT > RIGHT + 3`: LEFT signal turns GREEN, RIGHT is RED.
    - If `RIGHT > LEFT + 3`: RIGHT signal turns GREEN, LEFT is RED.
    - If counts are similar: signals alternate every few seconds.
- If an **emergency vehicle** is detected:
  - Identifies which region (LEFT/RIGHT) the emergency is in.
  - Turns that direction's signal GREEN immediately and keeps it GREEN until the emergency leaves.
  - If traffic is blocked (count > 20), displays a warning to suggest alternate routing.

### 4. Emergency Vehicle Priority & Visualization
- Emergency vehicles are highlighted with yellow bounding boxes and labels.
- When detected, the system fetches the nearest hospital/fire station/police station using Google Maps API.
- Shows a static map route preview to the destination alongside the video feed (with caching for smooth performance).

### 5. On-Screen Overlays
- Shows vehicle counts and signal status for each direction.
- Displays emergency detection status (YES/NO).
- Draws colored rectangles at frame edges to represent traffic signals (GREEN/RED).
- Shows a warning if an emergency vehicle is stuck in heavy traffic.
- Displays a live route preview map when an emergency is present.

## How It Works
1. **Setup**: Place your YOLOv8 model (`best.pt`) and a test video in the project folder. Add your Google Maps API key to a `.env` file.
2. **Run**: Execute `python main4.py`.
3. **Detection Loop**:
   - Each frame is processed by YOLOv8 for vehicle detection.
   - Vehicle counts are updated for LEFT and RIGHT regions.
   - Emergency vehicles are detected and prioritized.
   - Traffic signals are updated based on density and emergency logic.
   - If an emergency is present, the route to the nearest relevant destination is fetched and displayed (with caching to avoid lag).
   - All results and signals are visualized live on the video feed.

## Requirements
- Python 3.8+
- ultralytics
- opencv-python
- requests
- python-dotenv
- numpy

Install dependencies with:
```sh
pip install ultralytics opencv-python requests python-dotenv numpy
```

## Usage
1. Place your YOLOv8 model (`best.pt`) in the project directory.
2. Add your Google Maps API key to a `.env` file:
   ```
   GOOGLE_MAPS_API_KEY=your_key_here
   ```
3. Place your test video (e.g., `test_video_converted.mp4`) in the directory.
4. Run the main script:
   ```sh
   python main4.py
   ```

## Customization
- Change `MODEL_PATH` or `VIDEO_SOURCE` in `main4.py` to use different models or videos.
- Adjust region logic or vehicle classes as needed for your scenario.
- The system is ready for further extension (e.g., multi-intersection, more advanced routing, etc.).

## Credits
- YOLOv8 by Ultralytics
- Google Maps API
- OpenCV

---

**This project demonstrates a practical, AI-powered approach to smart traffic management and emergency vehicle prioritization for Indian roads.**
