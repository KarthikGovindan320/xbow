import cv2
import pygame
import numpy as np
import socket
import threading
import time
import json
import sys
import math
import os
import datetime
import logging
import subprocess

import torch
from ultralytics import YOLO
from collections import deque
from filterpy.kalman import KalmanFilter

ESP32_IP                = "10.219.186.196"
ESP32_UDP_PORT          = 5005
ALERT_SERVER_HOST       = "0.0.0.0"
ALERT_SERVER_PORT       = 9000
CAMERA_INDEX            = 0
CAPTURE_WIDTH           = 1280
CAPTURE_HEIGHT          = 720
CAPTURE_FPS             = 60

DRONE_REAL_WIDTH_M      = 0.30
HFOV_DEG                = 70.0
UDP_SEND_HZ             = 60
WINDOW_W                = 1280
WINDOW_H                = 720

DRONE_MODEL_PATH        = "drone_yolov8.pt"
GENERAL_MODEL_PATH      = "yolov8n.pt"
DRONE_CONF_HIGH         = 0.35 
DRONE_CONF_LOW          = 0.20  
CIVILIAN_CONF           = 0.50
YOLO_IMGSZ              = 640
SKIP_FRAMES             = 1     
STABLE_FRAMES           = 2    

COCO_PERSON_CLASS       = 0
CIVILIAN_CLASSES        = [0]

TRACK_MAX_DIST_PX       = 120
TRACK_HISTORY_LEN       = 60
TRACK_TTL_FRAMES        = 20  
VEL_SMOOTH_FRAMES       = 8

KALMAN_DT               = 1.0 / 30.0
KALMAN_PROC_NOISE       = 0.05
KALMAN_MEAS_NOISE       = 2.0
PREDICT_SECS            = 0.25

MAX_RANGE               = 15.0
MAX_SPEED               = 10.0
BEHAV_AGGRESSIVE_THR    = -2.0
BEHAV_APPROACH_THR      = -0.5
BEHAV_LEAVE_THR         =  0.5
BEHAV_HOVER_LAT_THR     =  20.0

W_APPROACH              = 0.35
W_DISTANCE              = 0.25
W_DIRECTION             = 0.20
W_STABILITY             = 0.20
BEHAV_MULT = {
    "AGGRESSIVE":  1.5,
    "APPROACHING": 1.2,
    "HOVERING":    1.0,
    "LEAVING":     0.4,
}

RF_DRONE_SSID_KEYWORDS  = ["dji", "drone", "parrot", "phantom", "mavic", "autel", "skydio", "tello", "fpv"]
RF_SCAN_INTERVAL_S      = 5.0

RADAR_SIZE              = 220
RADAR_MAX_DIST_M        = 15.0
RADAR_RINGS             = 4
RADAR_ALPHA             = 220
RADAR_POS               = (16, 80)

RECORDINGS_DIR          = "recordings"
LOG_DIR                 = "logs"
PRE_BUFFER_SECS         = 3.0
FIRE_CLIP_EXTRA_SECS    = 2.0
VIDEO_CODEC             = "mp4v"
VIDEO_FPS               = 20.0

STICKER_CYAN_LO         = np.array([ 88, 100, 100], dtype=np.uint8)
STICKER_CYAN_HI         = np.array([108, 255, 255], dtype=np.uint8)
STICKER_YELLOW_LO       = np.array([ 15,  60,  80], dtype=np.uint8)
STICKER_YELLOW_HI       = np.array([ 33, 255, 255], dtype=np.uint8)
STICKER_CYAN_MIN_FRAC   = 0.08
STICKER_YELLOW_MIN_FRAC = 0.03

C_GREEN  = (0,   255, 136)
C_RED    = (255,  34,  68)
C_AMBER  = (255, 170,   0)
C_CYAN   = (0,   200, 255)
C_BORDER = (0,    80,  40)
C_SAFETY = (255, 220,   0)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
USE_HALF = (DEVICE == "cuda")

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"turret_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("turret")

state = {
    "frame":          [None],
    "dbg_frame":      [None],
    "targets":        [[]],
    "locked":         [False],
    "yaw":            [0.0],
    "pitch":          [0.0],
    "dist":           [0.0],
    "fps_cap":        [0.0],
    "fps_det":        [0.0],
    "status":         ["INITIALIZING"],
    "tx_log":         [f"DEVICE: {DEVICE.upper()}  imgsz={YOLO_IMGSZ}"],
    "cam_index":      [CAMERA_INDEX],
    "running":        [True],
    "show_debug":     [False],
    "conf_high":      [True],
    "behaviors":      [{}],
    "show_radar":     [True],
    "fire_event":     [False],
    "rec_active":     [False],
    "rec_label":      [""],
    "cease_fire":     [False],
    "rf_threats":     [[]],
    "rf_status":      ["RF: SCANNING"],
    "priority_queue": [[]],
    "predicted_pos":  [{}],
    "auto_fire_armed":[True],
}
state_lock = threading.Lock()

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.setblocking(False)


def send_udp(yaw, pitch, dist):
    try:
        udp_sock.sendto(
            json.dumps({"y": round(yaw, 2), "p": round(pitch, 2), "d": round(dist, 2)}).encode(),
            (ESP32_IP, ESP32_UDP_PORT),
        )
    except Exception:
        pass


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def estimate_distance(bbox_w_px, cam_w):
    if bbox_w_px <= 0:
        return 0.0
    fx = (cam_w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    return (DRONE_REAL_WIDTH_M * fx) / bbox_w_px


def make_kalman(x, y, vx=0.0, vy=0.0):
    kf    = KalmanFilter(dim_x=4, dim_z=2)
    dt    = KALMAN_DT
    kf.F  = np.array([[1, 0, dt, 0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]], dtype=float)
    kf.H  = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
    q     = KALMAN_PROC_NOISE
    kf.Q  = np.diag([q, q, q * 10, q * 10])
    kf.R  = np.eye(2) * KALMAN_MEAS_NOISE
    kf.P  = np.eye(4) * 10.0
    kf.x  = np.array([[x], [y], [vx], [vy]], dtype=float)
    return kf


def predict_future_pos(kf, secs):
    steps  = max(1, int(round(secs / KALMAN_DT)))
    dt     = KALMAN_DT
    F_step = np.array([[1, 0, dt, 0],
                       [0, 1,  0, dt],
                       [0, 0,  1,  0],
                       [0, 0,  0,  1]], dtype=float)
    x = kf.x.copy()
    for _ in range(steps):
        x = F_step @ x
    return float(x[0][0]), float(x[1][0])


class DroneTracker:
    def __init__(self):
        self._kfilters  = {}
        self._tracks    = {}
        self._stable    = {}
        self._last_seen = {}
        self._last_bbox = {}
        self._next_id   = 0
        self._frame_ctr = 0

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def _center(self, bbox):
        x, y, bw, bh = bbox
        return x + bw / 2.0, y + bh / 2.0

    def update(self, raw_targets, timestamp):
        self._frame_ctr += 1

        matched  = {}
        used_ids = set()

        for ni, tgt in enumerate(raw_targets):
            ncx, ncy   = self._center(tgt["bbox"])
            best_id    = None
            best_d     = TRACK_MAX_DIST_PX
            for tid, hist in self._tracks.items():
                if not hist or tid in used_ids:
                    continue
                _, pcx, pcy, _ = hist[-1]
                d = math.hypot(ncx - pcx, ncy - pcy)
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is not None:
                matched[ni] = best_id
                used_ids.add(best_id)
            else:
                new_id = self._new_id()
                matched[ni] = new_id
                self._tracks[new_id]    = deque(maxlen=TRACK_HISTORY_LEN)
                self._stable[new_id]    = 0
                self._last_seen[new_id] = self._frame_ctr
                self._kfilters[new_id]  = make_kalman(ncx, ncy)

        matched_ids = set(matched.values())
        stale = [
            tid for tid, last in self._last_seen.items()
            if self._frame_ctr - last > TRACK_TTL_FRAMES and tid not in matched_ids
        ]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._stable.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._last_bbox.pop(tid, None)
            self._kfilters.pop(tid, None)

        annotated = []
        for ni, tgt in enumerate(raw_targets):
            tid = matched[ni]
            if tid not in self._tracks:
                ncx2, ncy2 = self._center(tgt["bbox"])
                self._tracks[tid]    = deque(maxlen=TRACK_HISTORY_LEN)
                self._stable[tid]    = 0
                self._last_seen[tid] = self._frame_ctr
                self._kfilters[tid]  = make_kalman(ncx2, ncy2)

            ncx, ncy = self._center(tgt["bbox"])
            kf = self._kfilters[tid]
            kf.predict()
            kf.update(np.array([[ncx], [ncy]]))
            print("kf.x shape:", np.asarray(kf.x).shape)
            sx = float(np.asarray(kf.x).squeeze()[0])
            sy = float(np.asarray(kf.x).squeeze()[1])

            if tid in self._last_bbox:
                lx, ly, lbw, lbh = self._last_bbox[tid]
                nx, ny, nbw, nbh = tgt["bbox"]
                a = 0.6
                tgt["bbox"] = (
                    int(a * nx  + (1 - a) * lx),
                    int(a * ny  + (1 - a) * ly),
                    int(a * nbw + (1 - a) * lbw),
                    int(a * nbh + (1 - a) * lbh),
                )

            self._last_bbox[tid]    = tgt["bbox"]
            self._last_seen[tid]    = self._frame_ctr
            self._stable[tid]       = self._stable.get(tid, 0) + 1
            self._tracks[tid].append((timestamp, sx, sy, tgt["dist"]))

            hist = self._tracks[tid]
            if len(hist) >= 3:
                window     = list(hist)[-VEL_SMOOTH_FRAMES:]
                dt_hist    = window[-1][0] - window[0][0]
                if dt_hist > 0:
                    radial_vel  = (window[-1][3] - window[0][3]) / dt_hist
                    lateral_vel = (window[-1][1] - window[0][1]) / dt_hist
                else:
                    radial_vel = lateral_vel = 0.0
            else:
                radial_vel = lateral_vel = 0.0

            if radial_vel < BEHAV_AGGRESSIVE_THR:
                behavior = "AGGRESSIVE"
            elif radial_vel < BEHAV_APPROACH_THR:
                behavior = "APPROACHING"
            elif radial_vel > BEHAV_LEAVE_THR:
                behavior = "LEAVING"
            elif abs(radial_vel) < abs(BEHAV_APPROACH_THR) and abs(lateral_vel) < BEHAV_HOVER_LAT_THR:
                behavior = "HOVERING"
            else:
                behavior = "APPROACHING"

            pred_cx, pred_cy = predict_future_pos(kf, PREDICT_SECS)

            tgt["id"]            = tid
            tgt["stable_frames"] = self._stable[tid]
            tgt["radial_vel"]    = radial_vel
            tgt["lateral_vel"]   = lateral_vel
            tgt["behavior"]      = behavior
            tgt["kalman_cx"]     = sx
            tgt["kalman_cy"]     = sy
            tgt["pred_cx"]       = pred_cx
            tgt["pred_cy"]       = pred_cy
            annotated.append(tgt)

        return annotated


def prioritize_targets(targets):
    for tgt in targets:
        dist_m      = tgt.get("dist",          0.0)
        radial_vel  = tgt.get("radial_vel",    0.0)
        lateral_vel = tgt.get("lateral_vel",   0.0)
        yaw_rad     = math.radians(tgt.get("yaw", 0.0))
        stable_fr   = tgt.get("stable_frames", 1)
        behavior    = tgt.get("behavior",      "HOVERING")

        dist_score     = 1.0 - min(max(dist_m  / MAX_RANGE, 0.0), 1.0)
        approach_score = min(max(-radial_vel    / MAX_SPEED, 0.0), 1.0)
        dir_score      = min(max(math.cos(yaw_rad), 0.0), 1.0)
        stab_score     = min(max(stable_fr      / STABLE_FRAMES, 0.0), 1.0)

        base  = (W_APPROACH  * approach_score +
                 W_DISTANCE  * dist_score     +
                 W_DIRECTION * dir_score      +
                 W_STABILITY * stab_score)
        score = min(base * BEHAV_MULT.get(behavior, 1.0), 1.0)

        tgt["threat_score"] = score
        tgt["speed_m_s"]    = math.hypot(radial_vel, lateral_vel)

    return sorted(targets, key=lambda t: t["threat_score"], reverse=True)


def _is_sticker_present(frame_bgr, x1, y1, bw, bh):
    fh, fw = frame_bgr.shape[:2]
    pad_x  = int(bw * 0.10)
    pad_b  = int(bh * 0.30)
    cx1    = max(0,  x1 - pad_x)
    cy1    = max(0,  y1)
    cx2    = min(fw, x1 + bw + pad_x)
    cy2    = min(fh, y1 + bh + pad_b)

    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return False

    crop        = frame_bgr[cy1:cy2, cx1:cx2]
    hsv         = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total       = hsv.shape[0] * hsv.shape[1]
    cyan_mask   = cv2.inRange(hsv, STICKER_CYAN_LO,   STICKER_CYAN_HI)
    yellow_mask = cv2.inRange(hsv, STICKER_YELLOW_LO, STICKER_YELLOW_HI)
    cyan_frac   = cv2.countNonZero(cyan_mask)   / total
    yellow_frac = cv2.countNonZero(yellow_mask) / total

    if not (cyan_frac >= STICKER_CYAN_MIN_FRAC and yellow_frac >= STICKER_YELLOW_MIN_FRAC):
        return False

    ch, cw   = hsv.shape[:2]
    hy, hx   = max(1, ch // 2), max(1, cw // 2)
    QUAD_MIN = 0.05
    quads = [
        (cyan_mask[:hy, :hx],  yellow_mask[:hy, :hx],  hy * hx),
        (cyan_mask[:hy, hx:],  yellow_mask[:hy, hx:],  hy * (cw - hx)),
        (cyan_mask[hy:, :hx],  yellow_mask[hy:, :hx],  (ch - hy) * hx),
        (cyan_mask[hy:, hx:],  yellow_mask[hy:, hx:],  (ch - hy) * (cw - hx)),
    ]
    return any(
        qt > 0
        and cv2.countNonZero(qc) / qt >= QUAD_MIN
        and cv2.countNonZero(qy) / qt >= QUAD_MIN
        for qc, qy, qt in quads
    )


def rf_scan_thread():
    log.info("[RF  ] RF scan thread started")
    while state["running"][0]:
        threats = []
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL", "dev", "wifi", "list"],
                capture_output=True, text=True, timeout=8,
            )
            for line in result.stdout.strip().splitlines():
                parts  = line.split(":")
                if len(parts) < 2:
                    continue
                ssid   = parts[0].strip()
                signal = int(parts[1]) if parts[1].strip().isdigit() else 0
                rssi   = signal - 100
                ssid_l = ssid.lower()
                is_drone = any(kw in ssid_l for kw in RF_DRONE_SSID_KEYWORDS)
                if is_drone:
                    threats.append({"ssid": ssid, "rssi": rssi})
                    log.warning(f"[RF  ] DRONE THREAT  SSID={ssid}  RSSI={rssi}dBm")
        except Exception as e:
            log.debug(f"[RF  ] Scan skipped: {e}")

        with state_lock:
            state["rf_threats"][0] = threats
            state["rf_status"][0]  = (
                f"RF: {len(threats)} DRONE(S) DETECTED" if threats else "RF: CLEAR"
            )

        time.sleep(RF_SCAN_INTERVAL_S)


def alert_server_thread():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((ALERT_SERVER_HOST, ALERT_SERVER_PORT))
    srv.listen(5)
    srv.settimeout(1.0)
    log.info(f"[ALRT] Alert TCP server on :{ALERT_SERVER_PORT}")

    clients      = []
    clients_lock = threading.Lock()

    def _accept():
        while state["running"][0]:
            try:
                conn, addr = srv.accept()
                conn.setblocking(False)
                with clients_lock:
                    clients.append(conn)
                log.info(f"[ALRT] Client connected: {addr}")
            except socket.timeout:
                pass
            except Exception:
                pass

    threading.Thread(target=_accept, daemon=True).start()

    last_t = 0.0
    while state["running"][0]:
        now = time.perf_counter()
        if now - last_t >= 1.0:
            last_t = now
            with state_lock:
                targets  = state["targets"][0]
                locked   = state["locked"][0]
                status   = state["status"][0]
                rf_stat  = state["rf_status"][0]
                cease    = state["cease_fire"][0]
                pq       = state["priority_queue"][0]

            alert = {
                "ts":           datetime.datetime.utcnow().isoformat(),
                "status":       status,
                "locked":       locked,
                "cease_fire":   cease,
                "rf":           rf_stat,
                "target_count": len(targets),
                "priority_queue": [
                    {
                        "id":           t.get("id", -1),
                        "dist_m":       round(t.get("dist",         0.0), 2),
                        "behavior":     t.get("behavior",           ""),
                        "threat_score": round(t.get("threat_score", 0.0), 3),
                        "speed_m_s":    round(t.get("speed_m_s",    0.0), 2),
                        "yaw":          round(t.get("yaw",          0.0), 2),
                        "pitch":        round(t.get("pitch",        0.0), 2),
                    }
                    for t in pq
                ],
            }

            if locked and pq:
                primary = pq[0]
                log.info(
                    f"[ALRT] LOCKED tgts={len(targets)}"
                    f"  id={primary.get('id',-1)}"
                    f"  dist={primary.get('dist',0):.2f}m"
                    f"  bhv={primary.get('behavior','')}"
                    f"  threat={primary.get('threat_score',0):.0%}"
                )

            msg  = (json.dumps(alert) + "\n").encode()
            dead = []
            with clients_lock:
                for c in clients:
                    try:
                        c.sendall(msg)
                    except Exception:
                        dead.append(c)
                for c in dead:
                    clients.remove(c)

        time.sleep(0.05)


def recording_thread():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    fourcc        = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    pre_buf_len   = int(PRE_BUFFER_SECS * VIDEO_FPS)
    pre_buf       = []
    buf_interval  = 1.0 / VIDEO_FPS
    tracking_writer  = None
    fire_writer      = None
    fire_frames_left = 0
    was_locked       = False
    last_buf_t       = 0.0

    log.info(f"[REC ] Thread started — dir: {RECORDINGS_DIR}")

    def _ts():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _new_writer(filename):
        path = os.path.join(RECORDINGS_DIR, filename)
        w    = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
        log.info(f"[REC ] Opened: {path}")
        with state_lock:
            state["rec_label"][0]  = f"REC {filename[:20]}"
            state["rec_active"][0] = True
        return w

    while state["running"][0]:
        now = time.perf_counter()
        with state_lock:
            frame   = state["frame"][0]
            locked  = state["locked"][0]
            fire_ev = state["fire_event"][0]
            if fire_ev:
                state["fire_event"][0] = False

        if frame is None:
            time.sleep(0.01)
            continue

        if now - last_buf_t >= buf_interval:
            last_buf_t = now
            pre_buf.append(frame.copy())
            if len(pre_buf) > pre_buf_len:
                pre_buf.pop(0)

            if locked and not was_locked:
                tracking_writer = _new_writer(f"track_{_ts()}.mp4")
                for bf in pre_buf:
                    tracking_writer.write(bf)
                was_locked = True
                log.info("[REC ] Tracking clip started")

            if not locked and was_locked:
                if tracking_writer:
                    tracking_writer.release()
                    tracking_writer = None
                    log.info("[REC ] Tracking clip closed")
                was_locked = False
                with state_lock:
                    if not fire_writer:
                        state["rec_active"][0] = False
                        state["rec_label"][0]  = ""

            if fire_ev:
                if fire_writer:
                    fire_writer.release()
                fire_writer = _new_writer(f"fire_{_ts()}.mp4")
                for bf in pre_buf:
                    fire_writer.write(bf)
                fire_frames_left = int(FIRE_CLIP_EXTRA_SECS * VIDEO_FPS)
                log.info("[REC ] Fire clip started")

            if tracking_writer:
                tracking_writer.write(frame)

            if fire_writer:
                fire_writer.write(frame)
                fire_frames_left -= 1
                if fire_frames_left <= 0:
                    fire_writer.release()
                    fire_writer = None
                    log.info("[REC ] Fire clip closed")
                    with state_lock:
                        if not tracking_writer:
                            state["rec_active"][0] = False
                            state["rec_label"][0]  = ""

        time.sleep(0.005)

    if tracking_writer:
        tracking_writer.release()
    if fire_writer:
        fire_writer.release()
    log.info("[REC ] Thread stopped")


def detection_loop():
    log.info("[DET ] Loading YOLO models...")
    if os.path.exists(DRONE_MODEL_PATH):
        drone_model = YOLO(DRONE_MODEL_PATH)
        log.info(f"[DET ] Loaded custom drone model: {DRONE_MODEL_PATH}")
    else:
        drone_model = YOLO(GENERAL_MODEL_PATH)
        log.warning(f"[DET ] drone_yolov8.pt not found — falling back to {GENERAL_MODEL_PATH}")

    civilian_model = YOLO(GENERAL_MODEL_PATH)
    drone_model.to(DEVICE)
    civilian_model.to(DEVICE)
    if USE_HALF:
        drone_model.half()
        civilian_model.half()

    cap            = open_camera(state["cam_index"][0])
    prev_cam       = state["cam_index"][0]
    stable         = 0
    miss_ctr       = 0
    skip_ctr       = 0
    cached_targets = []
    tracker        = DroneTracker()
    last_send      = 0.0
    send_gap       = 1.0 / UDP_SEND_HZ
    fps_t          = time.perf_counter()
    fps_cnt        = 0
    det_t          = time.perf_counter()
    det_cnt        = 0

    state["status"][0] = "SCANNING"
    log.info("[DET ] Detection loop running")

    while state["running"][0]:
        cur_cam = state["cam_index"][0]
        if cur_cam != prev_cam:
            cap.release()
            cap      = open_camera(cur_cam)
            stable   = skip_ctr = 0
            cached_targets = []
            prev_cam = cur_cam

        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)  
        if not ret:
            time.sleep(0.005)
            continue

        now = time.perf_counter()
        fps_cnt += 1
        if now - fps_t >= 1.0:
            state["fps_cap"][0] = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t   = now

        h, w        = frame.shape[:2]
        conf_thresh = DRONE_CONF_HIGH if state["conf_high"][0] else DRONE_CONF_LOW
        run_det     = (skip_ctr % SKIP_FRAMES == 0)
        skip_ctr   += 1

        if run_det:
            raw_targets    = []
            sticker_seen   = False
            civilian_abort = False

            civil_res = civilian_model(
                frame, imgsz=YOLO_IMGSZ, verbose=False,
                conf=CIVILIAN_CONF, classes=CIVILIAN_CLASSES, half=USE_HALF,
            )[0]
            for box in civil_res.boxes:
                if int(box.cls[0]) == COCO_PERSON_CLASS:
                    civilian_abort = True
                    log.warning("[SAFE] CIVILIAN IN FRAME — ABORTING FIRE")
                    break

            if not civilian_abort:
                drone_res = drone_model(
                    frame, imgsz=YOLO_IMGSZ, verbose=False,
                    conf=conf_thresh, half=USE_HALF,
                )[0]
                det_cnt += 1
                if now - det_t >= 1.0:
                    state["fps_det"][0] = det_cnt / (now - det_t)
                    det_cnt = 0
                    det_t   = now

                for box in drone_res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    if _is_sticker_present(frame, x1, y1, bw, bh):
                        sticker_seen = True
                        log.warning("[SAFE] SAFETY STICKER — CEASING FIRE")
                        continue
                    dist_m = estimate_distance(bw, w)
                    cx_t   = x1 + bw / 2.0
                    cy_t   = y1 + bh / 2.0
                    nx     =  (cx_t - w / 2.0) / w
                    ny     = -(cy_t - h / 2.0) / h
                    vfov   = HFOV_DEG * (h / w)
                    raw_targets.append({
                        "bbox":  (x1, y1, bw, bh),
                        "dist":  dist_m,
                        "yaw":   round(nx * HFOV_DEG, 2),
                        "pitch": round(ny * vfov,      2),
                        "conf":  float(box.conf[0]),
                    })

            if civilian_abort and not state["cease_fire"][0]:
                with state_lock:
                    state["cease_fire"][0] = True
                    state["status"][0]     = "ABORT: CIVILIAN"
                    state["tx_log"][0]     = "CIVILIAN DETECTED — FIRE ABORTED"

            if sticker_seen and not state["cease_fire"][0]:
                with state_lock:
                    state["cease_fire"][0] = True
                    state["status"][0]     = "CEASE-FIRE"
                    state["tx_log"][0]     = "SAFETY STICKER — FIRE CEASED"

            if state["cease_fire"][0]:
                stable      = 0
                raw_targets = []

            annotated      = tracker.update(raw_targets, now)
            cached_targets = prioritize_targets(annotated)

            if cached_targets:
                stable   = min(stable + 1, STABLE_FRAMES + 4)
                miss_ctr = 0
            else:
                miss_ctr += 1
                if miss_ctr >= 2:
                    stable   = max(0, stable - 1)
                    miss_ctr = 0

            predicted = {
                tgt["id"]: {"pred_cx": tgt["pred_cx"], "pred_cy": tgt["pred_cy"]}
                for tgt in cached_targets if "id" in tgt
            }
            with state_lock:
                state["predicted_pos"][0] = predicted

        targets_out = cached_targets if stable >= STABLE_FRAMES else []
        primary     = targets_out[0] if targets_out else None
        yaw         = primary["yaw"]   if primary else 0.0
        pitch       = primary["pitch"] if primary else 0.0
        dist        = primary["dist"]  if primary else 0.0

        with state_lock:
            locked_prev                = state["locked"][0]
            state["frame"][0]          = frame
            state["targets"][0]        = targets_out
            state["locked"][0]         = primary is not None
            state["yaw"][0]            = round(yaw,   2)
            state["pitch"][0]          = round(pitch, 2)
            state["dist"][0]           = round(dist,  3)
            state["priority_queue"][0] = targets_out
            state["behaviors"][0]      = {
                t["id"]: t.get("behavior", "")
                for t in targets_out if "id" in t
            }
            if primary and not locked_prev:
                n = len(targets_out)
                state["status"][0] = "LOCKED"
                state["tx_log"][0] = f"ACQUIRED {n} TARGET{'S' if n != 1 else ''}"
                log.info(f"[DET ] Acquired {n} target(s)")
            elif not primary and locked_prev:
                state["status"][0] = "SCANNING"
                state["tx_log"][0] = "TARGET LOST"
                log.info("[DET ] Target lost")

        if primary and (now - last_send) >= send_gap:
            if state["cease_fire"][0]:
                send_udp(0.0, 0.0, dist)
                with state_lock:
                    state["tx_log"][0] = "TX PARKED (CEASE-FIRE)"
            else:
                send_udp(yaw, pitch, dist)
                n = len(targets_out)
                with state_lock:
                    state["tx_log"][0] = (
                        f"TX [{n} TGT]  YAW:{yaw:.1f}°  PITCH:{pitch:.1f}°  DIST:{dist:.2f}m"
                    )
            last_send = now

    cap.release()
    log.info("[DET ] Loop stopped")


def load_fonts():
    mono = pygame.font.SysFont("Courier New", 11)
    try:
        big = pygame.font.SysFont("OCR A Extended", 28, bold=True)
        med = pygame.font.SysFont("OCR A Extended", 14)
        sml = pygame.font.SysFont("OCR A Extended", 10)
    except Exception:
        big = pygame.font.SysFont("Courier New", 28, bold=True)
        med = pygame.font.SysFont("Courier New", 14)
        sml = pygame.font.SysFont("Courier New", 10)
    return big, med, sml, mono


def make_scanline_overlay(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 4):
        pygame.draw.line(s, (0, 0, 0, 20), (0, y), (w, y), 2)
    return s


def make_vignette_overlay(w, h):
    s      = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    for r in range(max(w, h), 0, -4):
        dr = r / max(w, h)
        if dr < 0.55:
            break
        alpha = int((dr - 0.55) / 0.45 * 180)
        pygame.draw.ellipse(
            s, (0, 0, 0, min(alpha, 180)),
            (cx - r, cy - r * h // w, r * 2, r * 2 * h // w), 4,
        )
    return s


def draw_bracket(surf, rx, ry, rw, rh, color, sz=14):
    lw = 3
    for cx, cy, dx, dy in [(rx, ry, 1, 1), (rx+rw, ry, -1, 1),
                            (rx, ry+rh, 1, -1), (rx+rw, ry+rh, -1, -1)]:
        pygame.draw.line(surf, color, (cx, cy), (cx + dx*sz, cy), lw)
        pygame.draw.line(surf, color, (cx, cy), (cx, cy + dy*sz), lw)


def draw_reticle(surf, cx, cy, color, alpha=90):
    s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
    c = (*color, alpha)
    pygame.draw.line(s, c, (cx-24, cy), (cx-8,  cy), 1)
    pygame.draw.line(s, c, (cx+8,  cy), (cx+24, cy), 1)
    pygame.draw.line(s, c, (cx, cy-24), (cx, cy-8),  1)
    pygame.draw.line(s, c, (cx, cy+8),  (cx, cy+24), 1)
    pygame.draw.circle(s, c, (cx, cy), 5, 1)
    surf.blit(s, (0, 0))


def draw_target_box(surf, x, y, bw, bh, color, label=""):
    half_h = bh // 2
    rx     = max(0, min(x - 8,  surf.get_width()  - bw - 16))
    ry     = max(0, min(y - 8,  surf.get_height() - half_h - 16))
    rw     = bw + 16
    rh     = half_h + 16

    s    = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
    c    = (*color, 200)
    dash = 6
    gap  = 4
    for side in range(4):
        if   side == 0: pts = [(rx+i,    ry)    for i in range(0, rw, dash+gap)]
        elif side == 1: pts = [(rx+rw,   ry+i)  for i in range(0, rh, dash+gap)]
        elif side == 2: pts = [(rx+rw-i, ry+rh) for i in range(0, rw, dash+gap)]
        else:           pts = [(rx,      ry+rh-i) for i in range(0, rh, dash+gap)]
        for p in pts:
            end = (min(p[0]+dash, rx+rw) if side in (0,2) else p[0],
                   min(p[1]+dash, ry+rh) if side in (1,3) else p[1])
            pygame.draw.line(s, c, p, end, 2)
    surf.blit(s, (0, 0))
    draw_bracket(surf, rx, ry, rw, rh, color)

    cx_r = x + bw // 2
    cy_r = y + half_h // 2
    for a, b in [((cx_r-16, cy_r), (cx_r-5, cy_r)), ((cx_r+5, cy_r), (cx_r+16, cy_r)),
                 ((cx_r, cy_r-16), (cx_r, cy_r-5)), ((cx_r, cy_r+5),  (cx_r, cy_r+16))]:
        pygame.draw.line(surf, color, a, b, 2)
    pygame.draw.circle(surf, color, (cx_r, cy_r), 3)

    if label:
        try:
            lf  = pygame.font.SysFont("Courier New", 10)
            ls  = lf.render(label, True, color)
            surf.blit(ls, (rx, max(ry - 14, 0)))
        except Exception:
            pass


def draw_predicted_cross(surf, px, py, color):
    px = int(max(4, min(surf.get_width()  - 4, px)))
    py = int(max(4, min(surf.get_height() - 4, py)))
    pygame.draw.line(surf, color, (px-7, py), (px+7, py), 1)
    pygame.draw.line(surf, color, (px, py-7), (px, py+7), 1)
    pygame.draw.circle(surf, color, (px, py), 3, 1)


def draw_tele_cell(surf, fl, fv, fu, x, y, w, h, label, value, unit, alert):
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 20, 10, 215))
    pygame.draw.rect(panel, C_BORDER, (0, 0, w, h), 1, border_radius=4)
    surf.blit(panel, (x, y))
    surf.blit(fl.render(label, True, (0, 140, 70)), (x+10, y+7))
    color = C_RED if alert else C_GREEN
    val   = fv.render(value, True, color)
    surf.blit(val, (x+10, y+22))
    surf.blit(fu.render(unit, True, (0, 140, 70)), (x+10+val.get_width()+2, y+28))


_BHVCOL = {"AGGRESSIVE": C_RED, "APPROACHING": C_AMBER, "HOVERING": C_GREEN, "LEAVING": C_CYAN}
_BHVSYM = {"AGGRESSIVE": ">>", "APPROACHING": ">", "HOVERING": "~", "LEAVING": "<"}


def draw_radar(surf, sml_f, targets, predicted, radar_pos, radar_size, max_dist_m, rings):
    r    = radar_size // 2
    rx   = radar_pos[0]
    ry   = radar_pos[1]
    cx   = rx + r
    cy   = ry + r

    bg = pygame.Surface((radar_size + 20, radar_size + 28), pygame.SRCALPHA)
    bg.fill((0, 12, 6, RADAR_ALPHA))
    pygame.draw.rect(bg, C_BORDER, (0, 0, radar_size + 20, radar_size + 28), 1, border_radius=6)
    surf.blit(bg, (rx - 10, ry - 8))

    for i in range(1, rings + 1):
        ring_r = int(r * i / rings)
        rs     = pygame.Surface((radar_size, radar_size), pygame.SRCALPHA)
        pygame.draw.circle(rs, (0, 100, 50, 80), (r, r), ring_r, 1)
        surf.blit(rs, (rx, ry))
        dist_lbl = sml_f.render(f"{max_dist_m * i / rings:.0f}m", True, (0, 80, 40))
        surf.blit(dist_lbl, (cx + ring_r + 2, cy - dist_lbl.get_height() // 2))

    ax_s = pygame.Surface((radar_size, radar_size), pygame.SRCALPHA)
    pygame.draw.line(ax_s, (0, 80, 40, 80), (r, 0), (r, radar_size), 1)
    pygame.draw.line(ax_s, (0, 80, 40, 80), (0, r), (radar_size, r), 1)
    surf.blit(ax_s, (rx, ry))

    pygame.draw.circle(surf, C_GREEN, (cx, cy), 4)
    surf.blit(sml_f.render("RADAR", True, (0, 140, 70)), (rx, ry - 16))

    for i, tgt in enumerate(targets[:8]):
        yaw_d    = tgt.get("yaw",        0.0)
        dist_m   = tgt.get("dist",       0.0)
        behavior = tgt.get("behavior",   "HOVERING")
        rv       = tgt.get("radial_vel", 0.0)
        lv       = tgt.get("lateral_vel",0.0)
        tid      = tgt.get("id",         -1)

        dot_r   = int(r * min(dist_m, max_dist_m) / max_dist_m)
        yaw_rad = math.radians(yaw_d)
        dot_x   = cx + int(dot_r * math.sin(yaw_rad))
        dot_y   = cy - int(dot_r * math.cos(yaw_rad))

        col      = _BHVCOL.get(behavior, C_AMBER)
        dot_size = 8 if i == 0 else 5
        pygame.draw.circle(surf, col, (dot_x, dot_y), dot_size)

        speed_px = math.hypot(lv, rv) * (r / max_dist_m) * 0.5
        if speed_px > 1.5:
            arrow_a = math.atan2(lv, -rv)
            ax_end  = dot_x + int(speed_px * math.sin(arrow_a))
            ay_end  = dot_y - int(speed_px * math.cos(arrow_a))
            pygame.draw.line(surf, col, (dot_x, dot_y), (ax_end, ay_end), 2)

        sym = _BHVSYM.get(behavior, "?")
        surf.blit(sml_f.render(f"T{tid}{sym}", True, col), (dot_x + dot_size + 2, dot_y - 5))

    rf_threats = state["rf_threats"][0]
    for i, rf in enumerate(rf_threats[:3]):
        rfx = cx - r + 12 + i * 28
        rfy = cy - r + 12
        pygame.draw.polygon(surf, C_AMBER,
                            [(rfx, rfy - 8), (rfx - 6, rfy + 5), (rfx + 6, rfy + 5)])
        surf.blit(sml_f.render("RF", True, C_AMBER), (rfx - 6, rfy + 7))


def draw_priority_panel(surf, sml_f, targets, x, y, panel_w):
    row_h    = 16
    panel_h  = len(targets) * row_h + 24 if targets else 40
    bg       = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    bg.fill((0, 15, 8, 210))
    pygame.draw.rect(bg, C_BORDER, (0, 0, panel_w, panel_h), 1, border_radius=4)
    surf.blit(bg, (x, y))
    surf.blit(sml_f.render("PRIORITY QUEUE", True, (0, 160, 80)), (x + 6, y + 4))
    for i, tgt in enumerate(targets[:6]):
        ry  = y + 20 + i * row_h
        col = C_RED if i == 0 else _BHVCOL.get(tgt.get("behavior",""), C_AMBER)
        txt = (
            f"#{i+1} T{tgt.get('id',-1)}"
            f"  {tgt.get('dist',0):.1f}m"
            f"  {tgt.get('behavior','')[:5]}"
            f"  {tgt.get('threat_score',0):.0%}"
        )
        surf.blit(sml_f.render(txt, True, col), (x + 6, ry))


def draw_hud(screen, fonts, snap, overlays):
    big_f, med_f, sml_f, mono_f = fonts
    sw, sh = screen.get_size()
    sx     = sw / CAPTURE_WIDTH
    sy_    = sh / CAPTURE_HEIGHT

    src = snap["dbg_frame"] if snap["show_debug"] and snap["dbg_frame"] is not None else snap["frame"]
    if src is not None:
        rgb  = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pg_s = pygame.surfarray.make_surface(np.rot90(rgb, -1))
        screen.blit(pygame.transform.scale(pg_s, (sw, sh)), (0, 0))
    else:
        screen.fill((5, 10, 5))

    screen.blit(overlays[0], (0, 0))
    screen.blit(overlays[1], (0, 0))

    locked   = snap["locked"]
    targets  = snap["targets"]
    pq       = snap["priority_queue"]
    pred     = snap["predicted_pos"]

    for i, tgt in enumerate(targets):
        x, y, bw, bh = tgt["bbox"]
        px   = int(x  * sx)
        py   = int(y  * sy_)
        pbw  = int(bw * sx)
        pbh  = int(bh * sy_)
        col  = C_RED if i == 0 else C_AMBER
        bhv  = tgt.get("behavior",     "")
        scr  = tgt.get("threat_score", 0.0)
        tid  = tgt.get("id",           -1)
        lbl  = f"T{tid} {tgt['dist']:.1f}m {bhv} {scr:.0%}"
        draw_target_box(screen, px, py, pbw, pbh, col, label=lbl)

        if tid in pred:
            draw_predicted_cross(
                screen,
                pred[tid]["pred_cx"] * sx,
                pred[tid]["pred_cy"] * sy_,
                C_CYAN,
            )

    if not targets:
        draw_reticle(screen, sw // 2, sh // 2, C_GREEN)

    if locked and int(time.time() * 2) % 2 == 0:
        n   = len(targets)
        pri = pq[0] if pq else None
        scr = pri.get("threat_score", 0.0) if pri else 0.0
        bhv = pri.get("behavior",     "")  if pri else ""
        banner = big_f.render(f"LOCKED [{n} TGT]  THREAT:{scr:.0%}  {bhv}", True, C_RED)
        screen.blit(banner, (sw//2 - banner.get_width()//2, sh//2 - banner.get_height()//2 - 60))

    if snap["cease_fire"]:
        bord = pygame.Surface((sw, sh), pygame.SRCALPHA)
        ba   = 120 if int(time.time() * 3) % 2 == 0 else 40
        pygame.draw.rect(bord, (*C_SAFETY, ba), (0, 0, sw, sh), 10)
        screen.blit(bord, (0, 0))
        if int(time.time() * 2) % 2 == 0:
            cf  = big_f.render("CEASING FIRE", True, C_SAFETY)
            sub = med_f.render("PRESS [F] TO RESUME TARGETING", True, C_SAFETY)
            screen.blit(cf,  (sw//2 - cf.get_width()//2,  sh//2 - cf.get_height()//2))
            screen.blit(sub, (sw//2 - sub.get_width()//2, sh//2 + cf.get_height()//2 + 8))

    screen.blit(med_f.render("TURRET // DRONE TRACK", True, C_GREEN), (16, 10))

    mode_col = C_RED if locked else C_GREEN
    pygame.draw.circle(screen, mode_col, (sw - 130, 18), 5)
    conf_str = "HI" if snap["conf_high"] else "LO"
    screen.blit(sml_f.render(f"{snap['status']}  CONF:{conf_str}", True, mode_col), (sw-122, 13))
    screen.blit(sml_f.render(
        f"{snap['fps_cap']:.0f}cam  {snap['fps_det']:.0f}det  {DEVICE.upper()}",
        True, (0, 100, 50)), (sw - 260, 28))

    rf_col = C_AMBER if snap["rf_threats"] else C_GREEN
    screen.blit(sml_f.render(snap["rf_status"], True, rf_col), (sw - 260, 42))

    af_col = C_RED if snap["auto_fire_armed"] else (0, 100, 50)
    screen.blit(sml_f.render(
        "AUTO-FIRE: ARMED" if snap["auto_fire_armed"] else "AUTO-FIRE: SAFE",
        True, af_col), (sw - 260, 56))

    cell_w   = (sw - 32) // 3 - 6
    cell_h   = 52
    cy_start = sh - cell_h - 48
    for i, (label, value, unit) in enumerate([
        ("YAW",   f"{snap['yaw']:.1f}",              "deg"),
        ("PITCH", f"{snap['pitch']:.1f}",             "deg"),
        ("DIST",  f"{min(snap['dist'], 999.99):.2f}", "m"),
    ]):
        draw_tele_cell(screen, sml_f, med_f, sml_f,
                       16 + i*(cell_w+6), cy_start, cell_w, cell_h,
                       label, value if locked else "---", unit, alert=locked)

    screen.blit(mono_f.render(snap["tx_log"][:80], True, (0, 120, 60)), (16, sh - 14))
    screen.blit(sml_f.render(
        "[M]debug  [C]conf  [S]cam  [R]radar  [F]fire/resume  [A]autofire  [Q]quit",
        True, (0, 80, 40)), (16, sh - cell_h - 64))

    if snap["show_radar"]:
        draw_radar(screen, sml_f, snap["targets"], snap["predicted_pos"],
                   RADAR_POS, RADAR_SIZE, RADAR_MAX_DIST_M, RADAR_RINGS)

    draw_priority_panel(screen, sml_f, snap["priority_queue"], sw - 260, 72, 244)

    if snap["rec_active"] and int(time.time() * 2) % 2 == 0:
        pygame.draw.circle(screen, C_RED, (sw - 14, sh - 14), 5)
        rs = sml_f.render(snap["rec_label"][:24], True, C_RED)
        screen.blit(rs, (sw - rs.get_width() - 22, sh - 14))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
    pygame.display.set_caption("TURRET // DRONE TRACK")
    clock    = pygame.time.Clock()
    fonts    = load_fonts()
    overlays = (make_scanline_overlay(WINDOW_W, WINDOW_H),
                make_vignette_overlay(WINDOW_W, WINDOW_H))

    threading.Thread(target=detection_loop,    daemon=True).start()
    threading.Thread(target=rf_scan_thread,    daemon=True).start()
    threading.Thread(target=recording_thread,  daemon=True).start()
    threading.Thread(target=alert_server_thread, daemon=True).start()

    log.info("[MAIN] All threads started")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["running"][0] = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    state["running"][0] = False
                    pygame.quit()
                    sys.exit()
                elif k == pygame.K_m:
                    state["show_debug"][0] = not state["show_debug"][0]
                elif k == pygame.K_s:
                    state["cam_index"][0] = (state["cam_index"][0] + 1) % 4
                elif k == pygame.K_c:
                    state["conf_high"][0] = not state["conf_high"][0]
                    t = DRONE_CONF_HIGH if state["conf_high"][0] else DRONE_CONF_LOW
                    state["tx_log"][0] = f"CONF changed to {t:.2f}"
                elif k == pygame.K_r:
                    state["show_radar"][0] = not state["show_radar"][0]
                elif k == pygame.K_a:
                    with state_lock:
                        state["auto_fire_armed"][0] = not state["auto_fire_armed"][0]
                        armed = state["auto_fire_armed"][0]
                        state["tx_log"][0] = "AUTO-FIRE ARMED" if armed else "AUTO-FIRE SAFED"
                        log.info(f"[MAIN] Auto-fire {'ARMED' if armed else 'SAFED'}")
                elif k == pygame.K_f:
                    with state_lock:
                        if state["cease_fire"][0]:
                            state["cease_fire"][0] = False
                            state["status"][0]     = "SCANNING"
                            state["tx_log"][0]     = "CEASE-FIRE CLEARED"
                            log.info("[MAIN] Cease-fire cleared by operator")
                        else:
                            state["fire_event"][0] = True
                            state["tx_log"][0]     = "MANUAL FIRE EVENT"
                            log.info("[MAIN] Manual fire triggered")

        with state_lock:
            snap = {k: (v[0] if isinstance(v, list) else v) for k, v in state.items()}

        draw_hud(screen, fonts, snap, overlays)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
