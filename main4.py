import os
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional
import threading

import cv2
import numpy as np
import requests

# ultralytics import may raise if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# load dotenv if available (non-fatal)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# -------------------------------
# Configuration
# -------------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
USE_MAPS = bool(GOOGLE_MAPS_API_KEY)

MODEL_PATH = "best.pt"  # YOLOv8 model path
VIDEO_SOURCE = "test_video_converted4.mp4"  # or "0" for webcam

# Camera / Source location used for maps queries (latitude, longitude)
SOURCE_LOCATION = (12.973470348128659, 79.16319084373747)  # Connaught Place, New Delhi — change if needed

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
EMERGENCY_TRIGGER_DISTANCE_M = 80.0
EMERGENCY_PRIORITY_DISTANCE_M = 40.0
DEFAULT_RED_DURATION = 5
DEFAULT_GREEN_DURATION = 5
PRIORITY_GREEN_DURATION = 8
MAX_GREEN = 30
MIN_GREEN = 8

# Maps caching & cooldown
MAPS_CACHE_TTL = 30.0
PLACES_RADIUS = 5000  # meters
MAPS_COOLDOWN = 0.8  # seconds minimum between requests

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

maps_cache: dict = {}
_last_maps_request = 0.0


def now() -> float:
    return time.time()


# -------------------------------
# Traffic light network simulation
# -------------------------------
@dataclass
class TrafficLightNode:
    id: str
    location: Tuple[float, float]
    state: str = "RED"
    emergency_mode: bool = False
    last_update: float = 0.0
    traffic_density: int = 0
    duration: int = DEFAULT_GREEN_DURATION
    timer_started: Optional[float] = None

    def update_signal(self, prefer_green=False, tl1_state=None):
        """Update the node's signal with simple timing logic."""
        # TL1 logic
        if self.id == "TL_1":
            YELLOW_READY_DURATION = 2
            if self.emergency_mode or prefer_green:
                if self.timer_started is None:
                    self.timer_started = now()
                elapsed = now() - (self.timer_started or now())
                if elapsed < max(self.duration, PRIORITY_GREEN_DURATION):
                    self.state = "GREEN"
                else:
                    self.emergency_mode = False
                    self.timer_started = None
                    self.state = "RED"
                    self.traffic_density = max(0, self.traffic_density - random.randint(1, 4))
                self.last_update = now()
                return
            # Normal cycle
            if self.timer_started is None:
                self.timer_started = now()
                self.state = "RED"
                return
            elapsed = now() - self.timer_started
            cycle = self.duration + DEFAULT_RED_DURATION + YELLOW_READY_DURATION
            if elapsed < DEFAULT_RED_DURATION:
                self.state = "RED"
            elif elapsed < DEFAULT_RED_DURATION + YELLOW_READY_DURATION:
                self.state = "YELLOW"
            elif elapsed < cycle:
                self.state = "GREEN"
            else:
                self.timer_started = now()
            return

        # TL2 logic
        if self.id == "TL_2":
            if tl1_state == "GREEN" and self.emergency_mode:
                self.state = "YELLOW"
                self.last_update = now()
                return
            if self.timer_started is None:
                self.timer_started = now()
                self.state = "GREEN"
                return
            elapsed = now() - self.timer_started
            cycle = self.duration + DEFAULT_RED_DURATION
            if elapsed < self.duration:
                self.state = "GREEN"
            elif elapsed < cycle:
                self.state = "RED"
            else:
                self.timer_started = now()
            return

        # TL3 and others
        if self.timer_started is None:
            self.timer_started = now()
            self.state = "GREEN"
            return
        elapsed = now() - self.timer_started
        cycle = self.duration + DEFAULT_RED_DURATION
        if elapsed < self.duration:
            self.state = "GREEN"
        elif elapsed < cycle:
            self.state = "RED"
        else:
            self.timer_started = now()

    def reset_emergency_if_expired(self, timeout=60.0):
        if self.emergency_mode and (now() - self.last_update) > timeout:
            self.emergency_mode = False
            self.timer_started = None
            self.traffic_density = max(0, self.traffic_density - random.randint(3, 8))


# -------------------------------
# Maps helpers (safe)
# -------------------------------
def safe_get(url: str, params: dict = None, timeout: float = 6.0) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Maps] Request error: {e}")
        return None


def find_nearest_destination_cached(vehicle_label: str, source_location: Tuple[float, float]) -> Optional[DestinationInfo]:
    """
    Lookup nearest POI for given emergency vehicle label, with caching and cooldown.
    Returns None if maps disabled or API fails.
    """
    global _last_maps_request
    if not USE_MAPS:
        return None

    dest_type = EMERGENCY_CLASSES.get(vehicle_label)
    if not dest_type:
        return None

    cached = maps_cache.get(vehicle_label)
    if cached and (now() - cached.fetched_at) < MAPS_CACHE_TTL:
        return cached

    # respect cooldown
    if now() - _last_maps_request < MAPS_COOLDOWN:
        if cached:
            return cached
        # tiny non-blocking retry
        time.sleep(max(0.0, MAPS_COOLDOWN - (now() - _last_maps_request)))

    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{source_location[0]},{source_location[1]}",
        "radius": PLACES_RADIUS,
        "type": dest_type,
        "key": GOOGLE_MAPS_API_KEY
    }
    data = safe_get(places_url, params=params)
    _last_maps_request = now()
    if not data:
        return None

    status = data.get("status", "")
    if status != "OK":
        # there might be ZERO_RESULTS or OVER_QUERY_LIMIT etc.
        # print a short message but continue (no exception)
        print(f"[Maps] Places API status: {status}")
        results = data.get("results", [])
        if not results:
            return None

    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
    loc = best.get("geometry", {}).get("location", {})
    dest_loc = (loc.get("lat", 0.0), loc.get("lng", 0.0))
    dest_name = best.get("name", dest_type)

    route = get_route_points(source_location, dest_loc)
    dest_info = DestinationInfo(latlng=dest_loc, name=dest_name, route=route, fetched_at=now())
    maps_cache[vehicle_label] = dest_info
    return dest_info


def get_route_points(origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Tuple[float, float]]:
    if not USE_MAPS:
        return [origin, destination]
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "driving",
        "key": GOOGLE_MAPS_API_KEY
    }
    data = safe_get(directions_url, params=params)
    if not data or data.get("status") != "OK" or not data.get("routes"):
        # fallback to coarse route
        return [origin, destination]
    try:
        steps = data["routes"][0]["legs"][0].get("steps", [])
        points = [(s["end_location"]["lat"], s["end_location"]["lng"]) for s in steps if "end_location" in s]
        if not points:
            # coarse fallback
            return [origin, destination]
        return points
    except Exception as e:
        print(f"[Maps] Error parsing directions: {e}")
        return [origin, destination]


def fetch_static_map(route: List[Tuple[float, float]], size=(300, 200)) -> Optional[np.ndarray]:
    if not USE_MAPS or not route:
        return None
    try:
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
    dist_m = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width_pixels
    return round(dist_m, 2)


# -------------------------------
# Model loading (YOLOv8) - defensive
# -------------------------------
model = None
model_names = {}
if YOLO_AVAILABLE:
    try:
        if os.path.isfile(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            # names property may be on model or model.model
            try:
                model_names = getattr(model, "names", {}) or {}
            except Exception:
                model_names = {}
        else:
            print(f"[Model] Model file '{MODEL_PATH}' not found. Running with detection disabled.")
            model = None
    except Exception as e:
        print(f"[Model] Failed to load YOLO model: {e}")
        model = None
else:
    print("[Model] ultralytics package not available. Install it to enable model inference.")


# -------------------------------
# Global traffic network (starts empty)
# -------------------------------
traffic_network: List[TrafficLightNode] = []


def build_traffic_network_from_route(route_points: List[Tuple[float, float]], max_nodes: int = 3):
    global traffic_network
    new_nodes = []
    if not route_points:
        return
    # pick up to max_nodes evenly spaced route points (skip origin)
    pts = route_points[1:1 + max_nodes] if len(route_points) > 1 else route_points[:max_nodes]
    if len(pts) < max_nodes and len(route_points) > 1:
        step = max(1, len(route_points) // (max_nodes + 1))
        pts = [route_points[i] for i in range(1, len(route_points), step)][:max_nodes]

    for i, p in enumerate(pts, start=1):
        node_id = f"TL_{i}"
        existing = next((t for t in traffic_network if t.id == node_id), None)
        if existing:
            existing.location = p
            existing.traffic_density = max(0, min(30, existing.traffic_density + random.randint(-2, 4)))
            new_nodes.append(existing)
        else:
            tl = TrafficLightNode(id=node_id, location=p, traffic_density=random.randint(4, 18))
            new_nodes.append(tl)
    traffic_network = new_nodes


def broadcast_emergency_alert(route_points: List[Tuple[float, float]]):
    global traffic_network
    if not traffic_network or not route_points:
        return
    relevant_points = route_points[:len(traffic_network)]
    for tl in traffic_network:
        for rp in relevant_points:
            if abs(tl.location[0] - rp[0]) < 0.005 and abs(tl.location[1] - rp[1]) < 0.005:
                tl.emergency_mode = True
                tl.last_update = now()
                tl.traffic_density = max(0, tl.traffic_density - random.randint(2, 6))
                print(f"[Network] ALERT -> {tl.id} at approx {tl.location}")
                break


# -------------------------------
# Congestion & propagation
# -------------------------------
def compute_lane_congestion(left_count: int, right_count: int, normalize_max=20.0):
    left_score = min(left_count / normalize_max, 1.0)
    right_score = min(right_count / normalize_max, 1.0)
    return left_score, right_score


def dynamic_green_time(avg_congestion: float, distance_to_ev: Optional[float]):
    if distance_to_ev is None:
        closeness = 0.6
    else:
        if distance_to_ev < 50:
            closeness = 1.0
        elif distance_to_ev < 150:
            closeness = 0.85
        elif distance_to_ev < 300:
            closeness = 0.7
        else:
            closeness = 0.6
    factor = 1.0 + 1.6 * avg_congestion
    base = 12
    green = int(min(MAX_GREEN, max(MIN_GREEN, base * factor * closeness)))
    return green


def smart_signal_propagation(tl_nodes: List[TrafficLightNode], ev_distance: Optional[float],
                             left_count: int, right_count: int):
    if not tl_nodes:
        return
    left_score, right_score = compute_lane_congestion(left_count, right_count)
    avg_cong = (left_score + right_score) / 2.0
    green_time_for_near = dynamic_green_time(avg_cong, ev_distance)

    for i, tl in enumerate(tl_nodes):
        simulated_tl_distance = i * 120
        eff_dist = 9999 if ev_distance is None else max(0.0, ev_distance - simulated_tl_distance)

        if i == 0:
            if ev_distance is not None and ev_distance < EMERGENCY_TRIGGER_DISTANCE_M:
                tl.emergency_mode = True
                tl.duration = green_time_for_near
                tl.timer_started = now()
                tl.state = "GREEN"
            else:
                tl.state = "YELLOW" if ev_distance is not None and ev_distance < (EMERGENCY_TRIGGER_DISTANCE_M * 1.6) else tl.state
        elif i == 1:
            if ev_distance is not None and ev_distance < (EMERGENCY_TRIGGER_DISTANCE_M + 120):
                tl.emergency_mode = True
                tl.duration = max(10, int(green_time_for_near * 0.9))
                tl.state = "YELLOW" if ev_distance > 50 else "GREEN"
                if tl.state == "GREEN":
                    tl.timer_started = now()
            else:
                if not tl.emergency_mode:
                    tl.state = "RED"
        else:
            if ev_distance is not None and ev_distance < (EMERGENCY_TRIGGER_DISTANCE_M + 240):
                tl.state = "YELLOW"
            else:
                if not tl.emergency_mode:
                    tl.state = "RED"


# -------------------------------
# Visualization (simulation window)
# -------------------------------
def draw_simulation_window(tl_nodes: List[TrafficLightNode], ev_distance: Optional[float], left_count: int, right_count: int, fps: float):
    w, h = 520, 360
    sim = np.zeros((h, w, 3), dtype=np.uint8) + 20
    cv2.putText(sim, "Traffic Simulation (TL1 -> TL2 -> TL3)", (12, 24), FONT, 0.6, (220, 220, 220), 1)
    base_x = 80
    gap = 140
    center_y = 140

    for i in range(3):
        x = base_x + i * gap
        label = f"TL_{i+1}"
        if i < len(tl_nodes):
            tl = tl_nodes[i]
            state = tl.state
            col = (0, 0, 255) if state == "RED" else (0, 255, 255) if state == "YELLOW" else (0, 255, 0)
            cv2.circle(sim, (x, center_y), 36, (50, 50, 50), -1)
            cv2.circle(sim, (x, center_y), 30, col, -1)
            status = state + (" (E)" if tl.emergency_mode else "")
            cv2.putText(sim, f"{label}: {status}", (x - 70, center_y + 70), FONT, 0.45, (230, 230, 230), 1)
            cong_norm = min(1.0, tl.traffic_density / 30.0)
            bar_w = int(80 * cong_norm)
            if cong_norm < 0.33:
                bar_color = (0, 220, 0)
                cong_text = "Low"
            elif cong_norm < 0.66:
                bar_color = (0, 220, 220)
                cong_text = "Medium"
            else:
                bar_color = (0, 0, 220)
                cong_text = "High"
            cv2.putText(sim, "Congestion", (x - 40, center_y + 85), FONT, 0.4, (200, 200, 200), 1)
            cv2.rectangle(sim, (x - 40, center_y + 90), (x - 40 + 80, center_y + 105), (60, 60, 60), -1)
            cv2.rectangle(sim, (x - 40, center_y + 90), (x - 40 + bar_w, center_y + 105), bar_color, -1)
            percent = int(cong_norm * 100)
            cv2.putText(sim, f"{percent}% {cong_text}", (x + 50, center_y + 103), FONT, 0.4, (200, 200, 200), 1)
            cv2.putText(sim, f"{tl.duration}s", (x - 18, center_y + 125), FONT, 0.45, (200, 200, 200), 1)
        else:
            cv2.circle(sim, (x, center_y), 30, (0, 0, 255), -1)
            cv2.putText(sim, f"{label}: N/A", (x - 60, center_y + 70), FONT, 0.45, (200, 200, 200), 1)

    ev_text = f"EV est. dist to TL1: {ev_distance:.1f} m" if ev_distance is not None else "EV est. dist: N/A"
    cv2.putText(sim, ev_text, (12, 30 + 24), FONT, 0.5, (160, 255, 160), 1)
    cv2.putText(sim, f"Left lane vehicles: {left_count}", (12, h - 56), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(sim, f"Right lane vehicles: {right_count}", (12, h - 34), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(sim, f"FPS: {fps:.1f}", (w - 80, 22), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(sim, "Legend: RED / YELLOW / GREEN", (12, h - 12), FONT, 0.45, (200, 200, 200), 1)
    cv2.imshow("Traffic  Simulation", sim)


# -------------------------------
# Thread-safe route preview fetcher
# -------------------------------
class RoutePreviewFetcher:
    def __init__(self):
        self.cache = {}
        self.time = {}
        self.lock = threading.Lock()
        self.active_threads = {}

    def get(self, emergency_label, dest_info, ROUTE_CACHE_TTL=60):
        with self.lock:
            cache_img = self.cache.get(emergency_label)
            cache_time = self.time.get(emergency_label, 0)
        if cache_img is not None and (now() - cache_time) < ROUTE_CACHE_TTL:
            return cache_img
        # If not cached or expired, start a thread to fetch
        if emergency_label not in self.active_threads or not self.active_threads[emergency_label].is_alive():
            t = threading.Thread(target=self._fetch_and_store, args=(emergency_label, dest_info))
            t.daemon = True
            t.start()
            self.active_threads[emergency_label] = t
        return cache_img  # Return last cached image (may be None)

    def _fetch_and_store(self, emergency_label, dest_info):
        img = fetch_static_map(dest_info.route, size=(300, 200))
        with self.lock:
            self.cache[emergency_label] = img
            self.time[emergency_label] = now()


# -------------------------------
# CLI & Main loop
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AI Emergency Vehicle Detection & Networked Smart Traffic Control")
    parser.add_argument('--video', type=str, default=VIDEO_SOURCE, help='Path to video file or "0" for webcam')
    parser.add_argument('--location', type=str, default=f"{SOURCE_LOCATION[0]},{SOURCE_LOCATION[1]}", help='Source location as lat,lng')
    parser.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold')
    return parser.parse_args()


def safe_extract_detections(results, model_names_map):
    """
    Robustly extract detections: return list of [x1,y1,x2,y2,conf,class_index]
    Accepts ultralytics Results-like object or None.
    """
    detections = []
    if results is None:
        return detections
    try:
        # attempt standard path where boxes are torch tensors
        boxes = getattr(results, "boxes", None)
        if boxes is not None:
            # Some ultralytics versions: boxes.xyxy, boxes.conf, boxes.cls
            try:
                xyxy = boxes.xyxy
                confs = boxes.conf
                clss = boxes.cls
            except Exception:
                # fallback to .data or .xyxy.cpu().numpy() style
                try:
                    data = boxes.data  # maybe numpy-like
                    if hasattr(data, "tolist"):
                        # each row: [x1,y1,x2,y2,conf,cls]
                        rows = data.tolist()
                        for r in rows:
                            if len(r) >= 6:
                                detections.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5])])
                        return detections
                except Exception:
                    pass

            # convert to numpy if tensors
            try:
                xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
                confs_np = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)
                clss_np = clss.cpu().numpy() if hasattr(clss, "cpu") else np.asarray(clss)
                for (x1, y1, x2, y2), conf, cls in zip(xyxy_np, confs_np, clss_np):
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])
                return detections
            except Exception:
                # last fallback: iterate boxes and call .xyxy[0] etc.
                try:
                    for b in boxes:
                        xy = getattr(b, "xyxy", None)
                        c = getattr(b, "conf", None)
                        cl = getattr(b, "cls", None)
                        if xy is not None and c is not None and cl is not None:
                            xy_arr = xy.cpu().numpy()[0] if hasattr(xy, "cpu") else np.asarray(xy)[0]
                            detections.append([float(xy_arr[0]), float(xy_arr[1]), float(xy_arr[2]), float(xy_arr[3]), float(c), int(cl)])
                    return detections
                except Exception:
                    return []
        # If there is a .boxes missing, some versions return a list of dicts in results
        if isinstance(results, (list, tuple)):
            for r in results:
                if isinstance(r, dict):
                    x1, y1, x2, y2 = r.get("x1"), r.get("y1"), r.get("x2"), r.get("y2")
                    conf = r.get("confidence", r.get("conf", 0.0))
                    cls = r.get("class", r.get("cls", 0))
                    if None not in (x1, y1, x2, y2):
                        detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])
            return detections
    except Exception as e:
        print(f"[Detect] extraction error: {e}")
    return detections


def main():
    args = parse_args()
    try:
        lat, lng = map(float, args.location.split(','))
        source_location = (lat, lng)
    except Exception:
        print("Invalid --location format. Use lat,lng (e.g. 28.6139,77.2090)")
        return

    video_source = args.video if args.video != "0" else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video source:", video_source)
        return

    emergency_classes = set(EMERGENCY_CLASSES.keys())
    vehicle_classes = {'car', 'truck', 'bus', 'auto', 'tempo_traveller', 'bike', 'motorbike'}  # tune to your model

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # route preview caches
    if not hasattr(main, 'route_preview_fetcher'):
        main.route_preview_fetcher = RoutePreviewFetcher()
    if not hasattr(main, 'congestion_state'):
        main.congestion_state = None

    # ensure a 3-TL network always exists (placeholder until route fetched)
    if not traffic_network:
        traffic_network.extend([
            TrafficLightNode(id="TL_1", location=(source_location[0], source_location[1]), traffic_density=random.randint(4, 18)),
            TrafficLightNode(id="TL_2", location=(source_location[0], source_location[1]), traffic_density=random.randint(4, 18)),
            TrafficLightNode(id="TL_3", location=(source_location[0], source_location[1]), traffic_density=random.randint(4, 18)),
        ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

        h, w = frame.shape[:2]
        mid_x = w // 2

        # run model inference (if model available)
        results0 = None
        if model is not None:
            try:
                # ultralytics model call returns Results or list; pass frame directly
                raw = model(frame, conf=args.conf, verbose=False)
                # raw may be a Results object or list; pick first
                results0 = raw[0] if isinstance(raw, (list, tuple)) else raw
            except Exception as e:
                print(f"[Model] inference error: {e}")
                results0 = None

        detections = safe_extract_detections(results0, model_names)

        # aggregate counters & lists
        left_lane_count = 0
        right_lane_count = 0
        emergency_detected = False
        emergency_lane = None
        emergency_bbox = None
        emergency_label = None

        left_vehicles = []
        right_vehicles = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            label = model_names.get(cls, str(cls)) if model_names else str(cls)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Print detected vehicle log to terminal
            lane = 'LEFT' if cx < mid_x else 'RIGHT'
            print(f"[Detect] {label} | conf={conf:.2f} | lane={lane} | bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

            if label in vehicle_classes or label in emergency_classes:
                if cx < mid_x:
                    left_lane_count += 1
                    left_vehicles.append((cx, cy, label))
                else:
                    right_lane_count += 1
                    right_vehicles.append((cx, cy, label))

            if label in emergency_classes:
                emergency_detected = True
                emergency_bbox = (int(x1), int(y1), int(x2), int(y2))
                emergency_lane = 'LEFT' if cx < mid_x else 'RIGHT'
                emergency_label = label

        # Draw emergency vehicle boxes & labels
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            label = model_names.get(cls, str(cls)) if model_names else str(cls)
            if label in emergency_classes:
                color = (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame, f"{label.upper()} (EMERGENCY)", (int(x1), max(20, int(y1) - 10)), FONT, 0.6, color, 2)

        # Draw centers for lane visualization
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            label = model_names.get(cls, str(cls)) if model_names else str(cls)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if label in vehicle_classes or label in emergency_classes:
                color = (255, 255, 0) if cx < mid_x else (0, 255, 255)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        # Route preview & network build
        route_preview = None
        dest_info = None
        ROUTE_CACHE_TTL = 60

        if emergency_detected and emergency_label:
            dest_info = find_nearest_destination_cached(emergency_label, source_location)
            if dest_info:
                build_traffic_network_from_route(dest_info.route, max_nodes=3)
                route_preview = main.route_preview_fetcher.get(emergency_label, dest_info, ROUTE_CACHE_TTL)
                broadcast_emergency_alert(dest_info.route)
                if traffic_network:
                    tl1 = traffic_network[0]
                    left_score, right_score = compute_lane_congestion(left_lane_count, right_lane_count)
                    avg_cong = (left_score + right_score) / 2.0
                    if avg_cong > 0.5:
                        tl1.duration = min(MAX_GREEN, int(DEFAULT_GREEN_DURATION + avg_cong * 20))
                        tl1.state = "GREEN"
                        tl1.emergency_mode = True
                    else:
                        tl1.state = "YELLOW"
                        tl1.emergency_mode = False

        # Estimate EV distance
        ev_distance = None
        if emergency_bbox:
            x1, y1, x2, y2 = emergency_bbox
            bbox_w = x2 - x1
            ev_distance = estimate_distance_from_width(bbox_w)

        smart_signal_propagation(traffic_network, ev_distance, left_lane_count, right_lane_count)

        # AI Traffic Control Logic (two-lane)
        blocked_threshold = 10
        clear_threshold = 15
        road_blocked = (left_lane_count > blocked_threshold and right_lane_count > blocked_threshold)
        heavy_congestion = (road_blocked and not emergency_detected)

        signal_left = 'RED'
        signal_right = 'RED'
        overlay_msg = None

        if road_blocked and emergency_detected:
            flash_on = (frame_count // 10) % 2 == 0
            if emergency_lane == 'LEFT' and flash_on:
                signal_left = 'BLUE'
            elif emergency_lane == 'RIGHT' and flash_on:
                signal_right = 'BLUE'
            overlay_msg = f"EMERGENCY BLOCKED! CLEAR PATH - {emergency_lane} LANE PRIORITY"
            main.congestion_state = None
        elif heavy_congestion:
            if main.congestion_state is None or main.congestion_state == 'BLOCKED':
                main.congestion_state = 'BLOCKED'
                signal_left = 'RED'
                signal_right = 'RED'
                overlay_msg = "HEAVY CONGESTION DETECTED! WAITING TO CLEAR"
                if left_lane_count < clear_threshold and right_lane_count >= clear_threshold:
                    main.congestion_state = 'CLEARING_LEFT'
                elif right_lane_count < clear_threshold and left_lane_count >= clear_threshold:
                    main.congestion_state = 'CLEARING_RIGHT'
            elif main.congestion_state == 'CLEARING_LEFT':
                if left_lane_count < clear_threshold:
                    signal_left = 'GREEN'
                    signal_right = 'RED'
                    overlay_msg = "CLEARING LEFT LANE FROM CONGESTION"
                    if right_lane_count < clear_threshold:
                        main.congestion_state = None
                else:
                    main.congestion_state = 'BLOCKED'
            elif main.congestion_state == 'CLEARING_RIGHT':
                if right_lane_count < clear_threshold:
                    signal_left = 'RED'
                    signal_right = 'GREEN'
                    overlay_msg = "CLEARING RIGHT LANE FROM CONGESTION"
                    if left_lane_count < clear_threshold:
                        main.congestion_state = None
                else:
                    main.congestion_state = 'BLOCKED'
        elif road_blocked:
            signal_left = 'RED'
            signal_right = 'RED'
            overlay_msg = "ROAD BLOCKED! ALL STOP"
            main.congestion_state = None
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

        # visualization main intersection
        def color_for_signal(sig):
            if sig == 'GREEN':
                return (0, 255, 0)
            if sig == 'BLUE':
                return (255, 0, 0)
            return (0, 0, 255)

        left_color = color_for_signal(signal_left)
        right_color = color_for_signal(signal_right)
        cv2.rectangle(frame, (5, 5), (30, h - 5), left_color, -1)
        cv2.rectangle(frame, (w - 30, 5), (w - 5, h - 5), right_color, -1)
        cv2.line(frame, (mid_x, 0), (mid_x, h), (180, 180, 180), 2)
        cv2.putText(frame, f"LEFT: {left_lane_count} | Signal: {signal_left}", (40, 40), FONT, 0.7, left_color, 2)
        cv2.putText(frame, f"RIGHT: {right_lane_count} | Signal: {signal_right}", (w // 2 + 20, 40), FONT, 0.7, right_color, 2)
        if overlay_msg:
            cv2.putText(frame, overlay_msg, (40, 120), FONT, 0.7, (0, 140, 255), 2)
        if emergency_bbox:
            x1, y1, x2, y2 = emergency_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "EMERGENCY", (x1, max(20, y1 - 10)), FONT, 0.7, (0, 255, 255), 2)

        # Show nearest hospital name on video frame if available
        if dest_info is not None and hasattr(dest_info, 'name'):
            cv2.putText(frame, f"Nearest Hospital: {dest_info.name}", (40, 80), FONT, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (40, frame.shape[0] - 20), FONT, 0.7, (0, 255, 0), 2)

        # Update & visualize networked traffic lights
        y_offset = 160
        for tl in traffic_network:
            tl.traffic_density = max(0, min(30, tl.traffic_density + random.randint(-1, 2)))
            prefer_green = tl.emergency_mode
            tl.update_signal(prefer_green=prefer_green, tl1_state=(traffic_network[0].state if traffic_network else None))
            tl.reset_emergency_if_expired(timeout=60.0)
            color = (0, 255, 0) if tl.state == "GREEN" else (0, 140, 255) if tl.state == "YELLOW" else (0, 0, 255)
            cv2.putText(frame, f"{tl.id}: {tl.state}{' (E)' if tl.emergency_mode else ''} D={tl.traffic_density}", (40, y_offset), FONT, 0.5, color, 1)
            y_offset += 22

        # show route preview if available
        if route_preview is not None:
            blank = np.zeros((frame.shape[0], 320, 3), dtype=np.uint8) + 30
            r_h, r_w = route_preview.shape[:2]
            scale = min(320 / r_w, frame.shape[0] / r_h, 1.0)
            rp_small = cv2.resize(route_preview, (int(r_w * scale), int(r_h * scale)))
            blank[0:rp_small.shape[0], 0:rp_small.shape[1]] = rp_small
            cv2.putText(blank, "Route Preview", (8, rp_small.shape[0] + 20), FONT, 0.5, (240, 240, 240), 1)
            combined = np.hstack((frame, blank))
            cv2.imshow("Smart Single-Lane Traffic Control - Networked", combined)
        else:
            cv2.imshow("Smart Single-Lane Traffic Control - Networked", frame)

        draw_simulation_window(traffic_network, ev_distance, left_lane_count, right_lane_count, fps)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()