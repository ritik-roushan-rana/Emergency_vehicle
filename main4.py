import os
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
import argparse

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from dotenv import load_dotenv

# -------------------------------
# Configuration
# -------------------------------
load_dotenv()  # loads .env if present
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError("Set GOOGLE_MAPS_API_KEY in your environment or .env file")

MODEL_PATH = "best.pt"  # YOLOv8 model path
VIDEO_SOURCE = "test_video_converted.mp4"  # or 0 for webcam

# Camera / Source location used for maps queries (latitude, longitude)
SOURCE_LOCATION = (28.6139, 77.2090)  # Connaught Place, New Delhi# New Delhi sample — change to your test location

# Emergency classes in your model
EMERGENCY_CLASSES = {
    'ambulance': 'hospital', 'ambulance_108': 'hospital', 'ambulance_SOL': 'hospital',
    'ambulance_lamp': 'hospital', 'ambulance_text': 'hospital',
    'fire_truck': 'fire_station', 'fireladder': 'fire_station', 'firelamp': 'fire_station',
    'firesymbol': 'fire_station', 'firewriting': 'fire_station',
    'police': 'police', 'police_lamp': 'police', 'police_lamp_ON': 'police'
}

# Distance estimation (approx)
KNOWN_WIDTH = 1.8  # meters (typical ambulance width)
FOCAL_LENGTH = 555.56  # calibration value - tune for your camera

# Signal rules
EMERGENCY_TRIGGER_DISTANCE_M = 80.0  # if within this distance, trigger green
EMERGENCY_PRIORITY_DISTANCE_M = 40.0  # closer => longer green
DEFAULT_RED_DURATION = 5  # seconds
DEFAULT_GREEN_DURATION = 5  # seconds
PRIORITY_GREEN_DURATION = 8  # seconds when very close

# Maps caching & cooldown
MAPS_CACHE_TTL = 30.0  # seconds to reuse places/directions for the same label
PLACES_RADIUS = 5000  # meters
MAPS_COOLDOWN = 1.0  # minimal seconds between repeated requests in heavy loops

# Visual settings
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -------------------------------
# Utilities & Data Classes
# -------------------------------

@dataclass
class DestinationInfo:
    latlng: Tuple[float, float]
    name: str
    route: List[Tuple[float, float]]
    fetched_at: float

# Simple cache for destinations keyed by vehicle label
maps_cache: dict = {}

_last_maps_request = 0.0

def now():
    return time.time()

# -------------------------------
# Google Maps helper functions
# -------------------------------
def safe_get(url, params=None, timeout=6.0):
    """Simple wrapper for requests.get with basic error handling."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Maps] Request error: {e}")
        return None

def find_nearest_destination_cached(vehicle_label: str, source_location: Tuple[float, float]) -> Optional[DestinationInfo]:
    """Return cached destination if fresh; otherwise query Places + Directions and cache result."""
    global _last_maps_request
    dest_type = EMERGENCY_CLASSES.get(vehicle_label)
    if not dest_type:
        return None

    # Reuse cache if recent
    cached: DestinationInfo = maps_cache.get(vehicle_label)
    if cached and (now() - cached.fetched_at) < MAPS_CACHE_TTL:
        return cached

    # throttle requests slightly
    if now() - _last_maps_request < MAPS_COOLDOWN:
        # avoid hammering API; fallback to cached if exists
        if cached:
            return cached
        time.sleep(MAPS_COOLDOWN)

    # Places Nearby Search
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{source_location[0]},{source_location[1]}",
        "radius": PLACES_RADIUS,
        "type": dest_type,
        "key": GOOGLE_MAPS_API_KEY
    }
    data = safe_get(places_url, params=params)
    _last_maps_request = now()
    if not data or data.get("status") != "OK" or not data.get("results"):
        print("[Maps] No places found or API error.")
        return None

    best = data["results"][0]
    dest_loc = (best["geometry"]["location"]["lat"], best["geometry"]["location"]["lng"])
    dest_name = best.get("name", dest_type)

    # Directions
    route = get_route_points(source_location, dest_loc)
    dest_info = DestinationInfo(latlng=dest_loc, name=dest_name, route=route, fetched_at=now())
    maps_cache[vehicle_label] = dest_info
    return dest_info

def get_route_points(origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Call Directions API and return a list of step end_location lat/lng tuples."""
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "driving",
        "key": GOOGLE_MAPS_API_KEY
    }
    data = safe_get(directions_url, params=params)
    if not data or data.get("status") != "OK" or not data.get("routes"):
        print("[Maps] Directions API returned no route.")
        return []

    steps = data["routes"][0]["legs"][0]["steps"]
    points = [(s["end_location"]["lat"], s["end_location"]["lng"]) for s in steps if "end_location" in s]
    return points

def fetch_static_map(route: List[Tuple[float, float]], size=(300, 200)) -> Optional[np.ndarray]:
    """Return a small static map image (cv2 image) showing the route — encoded path may be long.
       We'll approximate by creating a path parameter from lat/lng pairs.
    """
    if not route:
        return None

    # Build path parameter — many waypoints will exceed URL length, so simplify: use first, mid, last
    simplified = []
    if len(route) <= 6:
        simplified = route
    else:
        simplified = [route[0]] + route[1::max(1, len(route)//4)][:4] + [route[-1]]

    path_param = "|".join(f"{lat},{lng}" for lat, lng in simplified)
    static_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "size": f"{size[0]}x{size[1]}",
        "path": f"weight:3|color:0x0000ff|{path_param}",
        "markers": f"color:red|{route[0][0]},{route[0][1]}|color:green|{route[-1][0]},{route[-1][1]}",
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        r = requests.get(static_url, params=params, timeout=6.0)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Maps] Static map error: {e}")
        return None

# -------------------------------
# Distance estimation
# -------------------------------
def estimate_distance_from_width(bbox_width_pixels: float) -> Optional[float]:
    if bbox_width_pixels <= 0:
        return None
    # simple pinhole camera equation
    dist_m = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width_pixels
    return round(dist_m, 2)

# -------------------------------
# Traffic signal simulation
# -------------------------------
class TrafficSignal:
    def __init__(self):
        self.state = "RED"
        self.timer = now()
        self.duration = DEFAULT_RED_DURATION

    def set_state(self, state: str, duration: Optional[float] = None):
        self.state = state
        self.timer = now()
        self.duration = duration if duration is not None else (PRIORITY_GREEN_DURATION if state == "GREEN" else DEFAULT_RED_DURATION)

    def time_left(self) -> int:
        return max(0, int(self.duration - (now() - self.timer)))

# single intersection in front of camera
main_signal = TrafficSignal()

# Simulated signals along a route keyed by (lat,lng) tuple string
simulated_signals = {}

# -------------------------------
# Model loading
# -------------------------------
model = YOLO(MODEL_PATH)
# You can set model.conf = 0.35 etc. or filter boxes manually

# -------------------------------
# Main loop
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AI Emergency Vehicle Detection & Smart Traffic Control")
    parser.add_argument('--video', type=str, default=VIDEO_SOURCE, help='Path to video file or 0 for webcam')
    parser.add_argument('--location', type=str, default=f"{SOURCE_LOCATION[0]},{SOURCE_LOCATION[1]}", help='Source location as lat,lng')
    return parser.parse_args()

def main():
    args = parse_args()
    # Parse location
    try:
        lat, lng = map(float, args.location.split(','))
        source_location = (lat, lng)
    except Exception:
        print("Invalid --location format. Use lat,lng (e.g. 28.6139,77.2090)")
        return
    video_source = args.video if args.video != '0' else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video source")
        return

    # Define emergency and vehicle classes
    emergency_classes = set(EMERGENCY_CLASSES.keys())
    vehicle_classes = {'car', 'truck', 'bus', 'auto', 'tempo traveller', 'bike'}

    # Signal state variables
    signal_left = 'RED'
    signal_right = 'RED'
    last_switch_time = now()
    alternate_interval = 7  # seconds to alternate if densities are similar
    last_green = 'LEFT'  # which side was last green for alternation
    emergency_active = False
    emergency_side = None
    emergency_last_seen = 0
    emergency_vehicle_count = 0

    frame_count = 0
    start_time = time.time()
    fps = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # FPS calculation
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
        h, w = frame.shape[:2]
        mid_x = w // 2
        # Define regions
        left_roi = (0, 0, mid_x, h)
        right_roi = (mid_x, 0, w, h)

        # Run YOLO
        results = model(frame)[0]
        detections = results.boxes.data.tolist() if hasattr(results.boxes, "data") else []
        count_left = 0
        count_right = 0
        emergency_detected = False
        emergency_side = None
        emergency_vehicle_count = 0

        # For optional stuck detection
        left_vehicles = []
        right_vehicles = []

        # Analyze detections
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # Count vehicles
            if label in vehicle_classes or label in emergency_classes:
                if cx < mid_x:
                    count_left += 1
                    left_vehicles.append((cx, cy, label))
                else:
                    count_right += 1
                    right_vehicles.append((cx, cy, label))
            # Emergency vehicle detection
            if label in emergency_classes:
                emergency_detected = True
                emergency_vehicle_count += 1
                if cx < mid_x:
                    emergency_side = 'LEFT'
                else:
                    emergency_side = 'RIGHT'

        # Draw bounding boxes for emergency vehicles
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            if label in emergency_classes:
                color = (0, 255, 255)  # Yellow for emergency
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame, f"{label.upper()} (EMERGENCY)", (int(x1), max(20, int(y1)-10)), FONT, 0.7, color, 2)

        # Draw vehicle centers for lane visualization
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if label in vehicle_classes or label in emergency_classes:
                color = (255,255,0) if cx < mid_x else (0,255,255)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        # --- Route preview logic (optimized with cache) ---
        route_preview = None
        dest_info = None
        # Add a cache for route previews and their fetch times
        if not hasattr(main, 'route_preview_cache'):
            main.route_preview_cache = {}
            main.route_preview_time = {}
        ROUTE_CACHE_TTL = 60  # seconds
        route_label = None
        if emergency_detected:
            # Use the first detected emergency vehicle's label for routing
            for det in detections:
                x1, y1, x2, y2, conf, cls = det[:6]
                label = model.names[int(cls)]
                if label in emergency_classes:
                    route_label = label
                    break
            if route_label:
                cache_time = main.route_preview_time.get(route_label, 0)
                cache_img = main.route_preview_cache.get(route_label)
                if cache_img is not None and (now() - cache_time) < ROUTE_CACHE_TTL:
                    route_preview = cache_img
                    dest_info = main.route_preview_cache.get(f'{route_label}_dest')
                else:
                    dest_info = find_nearest_destination_cached(route_label, SOURCE_LOCATION)
                    if dest_info:
                        route_preview = fetch_static_map(dest_info.route, size=(300, 200))
                        main.route_preview_cache[route_label] = route_preview
                        main.route_preview_cache[f'{route_label}_dest'] = dest_info
                        main.route_preview_time[route_label] = now()
                if dest_info:
                    cv2.putText(frame, f"Route to: {dest_info.name}", (40, 160), FONT, 0.7, (0, 255, 255), 2)

        # --- AI Traffic Control Logic: Indian Two-Lane Scenario ---
        # 1. Divide frame into left and right lanes
        mid_x = w // 2
        left_lane_count = 0
        right_lane_count = 0
        emergency_detected = False
        emergency_lane = None
        emergency_bbox = None
        # Count vehicles and detect emergency vehicle lane
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            cx = int((x1 + x2) / 2)
            # Count vehicles in each lane
            if label in vehicle_classes or label in emergency_classes:
                if cx < mid_x:
                    left_lane_count += 1
                else:
                    right_lane_count += 1
            # Emergency vehicle detection and lane
            if label in emergency_classes:
                emergency_detected = True
                emergency_bbox = (int(x1), int(y1), int(x2), int(y2))
                emergency_lane = 'LEFT' if cx < mid_x else 'RIGHT'
        # 2. Blocked/jammed logic
        blocked_threshold = 10  # can be tuned
        clear_threshold = 15    # threshold to start clearing a lane
        # --- Congestion state tracking ---
        if not hasattr(main, 'congestion_state'):
            main.congestion_state = None  # None, 'BLOCKED', or 'CLEARING_LEFT'/'CLEARING_RIGHT'

        road_blocked = (left_lane_count > blocked_threshold and right_lane_count > blocked_threshold)
        heavy_congestion = (road_blocked and not emergency_detected)

        # 3. Traffic light logic
        signal_left = 'RED'
        signal_right = 'RED'
        overlay_msg = None

        # Emergency Blocked Override (highest priority)
        if road_blocked and emergency_detected:
            # Flashing effect for emergency lane
            flash_on = (frame_count // 10) % 2 == 0
            if emergency_lane == 'LEFT' and flash_on:
                signal_left = 'BLUE'
            elif emergency_lane == 'RIGHT' and flash_on:
                signal_right = 'BLUE'
            overlay_msg = f"\u26A0\uFE0F EMERGENCY BLOCKED! CLEAR PATH - {emergency_lane} LANE PRIORITY"
            main.congestion_state = None  # reset congestion state if emergency
        # Heavy congestion logic (no emergency)
        elif heavy_congestion:
            if main.congestion_state is None or main.congestion_state == 'BLOCKED':
                main.congestion_state = 'BLOCKED'
                signal_left = 'RED'
                signal_right = 'RED'
                overlay_msg = "HEAVY CONGESTION DETECTED! WAITING TO CLEAR"
                # Check if one lane is ready to be cleared
                if left_lane_count < clear_threshold and right_lane_count >= clear_threshold:
                    main.congestion_state = 'CLEARING_LEFT'
                elif right_lane_count < clear_threshold and left_lane_count >= clear_threshold:
                    main.congestion_state = 'CLEARING_RIGHT'
            elif main.congestion_state == 'CLEARING_LEFT':
                if left_lane_count < clear_threshold:
                    signal_left = 'GREEN'
                    signal_right = 'RED'
                    overlay_msg = "CLEARING LEFT LANE FROM CONGESTION"
                    # If both lanes are now below threshold, reset
                    if right_lane_count < clear_threshold:
                        main.congestion_state = None
                else:
                    main.congestion_state = 'BLOCKED'  # revert if congestion returns
            elif main.congestion_state == 'CLEARING_RIGHT':
                if right_lane_count < clear_threshold:
                    signal_left = 'RED'
                    signal_right = 'GREEN'
                    overlay_msg = "CLEARING RIGHT LANE FROM CONGESTION"
                    if left_lane_count < clear_threshold:
                        main.congestion_state = None
                else:
                    main.congestion_state = 'BLOCKED'
        # Standard blockage (no emergency, not heavy congestion)
        elif road_blocked:
            signal_left = 'RED'
            signal_right = 'RED'
            overlay_msg = "ROAD BLOCKED! ALL STOP"
            main.congestion_state = None
        # Emergency vehicle present (normal priority)
        elif emergency_detected:
            if emergency_lane == 'LEFT':
                signal_left = 'GREEN'
                signal_right = 'RED'
            else:
                signal_left = 'RED'
                signal_right = 'GREEN'
            overlay_msg = f"EMERGENCY VEHICLE PRIORITY: {emergency_lane} LANE"
            main.congestion_state = None
        else:
            # Alternate every N seconds
            interval = 7
            if not hasattr(main, 'last_switch_time'):
                main.last_switch_time = now()
                main.last_green = 'LEFT'
            if now() - main.last_switch_time > interval:
                main.last_green = 'RIGHT' if main.last_green == 'LEFT' else 'LEFT'
                main.last_switch_time = now()
            if main.last_green == 'LEFT':
                signal_left = 'GREEN'
                signal_right = 'RED'
            else:
                signal_left = 'RED'
                signal_right = 'GREEN'
            main.congestion_state = None
        # 4. Visualization
        left_color = (0,255,0) if signal_left=='GREEN' else (255,0,0) if signal_left=='BLUE' else (0,0,255)
        right_color = (0,255,0) if signal_right=='GREEN' else (255,0,0) if signal_right=='BLUE' else (0,0,255)
        cv2.rectangle(frame, (5, 5), (30, h-5), left_color, -1)
        cv2.rectangle(frame, (w-30, 5), (w-5, h-5), right_color, -1)
        cv2.line(frame, (mid_x, 0), (mid_x, h), (180,180,180), 2)
        cv2.putText(frame, f"LEFT: {left_lane_count} | Signal: {signal_left}", (40, 40), FONT, 0.7, left_color, 2)
        cv2.putText(frame, f"RIGHT: {right_lane_count} | Signal: {signal_right}", (w//2+20, 40), FONT, 0.7, right_color, 2)
        if overlay_msg:
            cv2.putText(frame, overlay_msg, (40, 120), FONT, 0.8, (0,140,255), 3)
        # Draw emergency vehicle bounding box
        if emergency_bbox:
            x1, y1, x2, y2 = emergency_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 3)
            cv2.putText(frame, "EMERGENCY", (x1, max(20, y1-10)), FONT, 0.7, (0,255,255), 2)
        # Overlay FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (40, frame.shape[0]-20), FONT, 0.7, (0,255,0), 2)
        # Show frame (with route preview if available)
        if 'route_preview' in locals() and route_preview is not None:
            blank = np.zeros((frame.shape[0], 320, 3), dtype=np.uint8) + 30
            r_h, r_w = route_preview.shape[:2]
            scale = min(320 / r_w, frame.shape[0] / r_h, 1.0)
            rp_small = cv2.resize(route_preview, (int(r_w * scale), int(r_h * scale)))
            blank[0:rp_small.shape[0], 0:rp_small.shape[1]] = rp_small
            cv2.putText(blank, "Route Preview", (8, rp_small.shape[0] + 20), FONT, 0.5, (240, 240, 240), 1)
            combined = np.hstack((frame, blank))
            cv2.imshow("Smart Single-Lane Traffic Control", combined)
        else:
            cv2.imshow("Smart Single-Lane Traffic Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()