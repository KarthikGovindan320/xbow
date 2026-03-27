# Drone Turret System

An autonomous anti-drone turret system combining computer vision, RF detection, Kalman filtering, and hardware servo control. Built for a hackathon, designed to demonstrate real-world counter-drone concepts.

---

## Architecture

```
┌─────────────────────────────────┐         UDP          ┌──────────────────┐
│         PC  (tracker.py)        │ ──────────────────►  │  ESP32 Turret    │
│                                 │  {yaw, pitch, dist}  │ (turret_esp32.py)│
│  YOLO Drone Detection           │                      │                  │
│  YOLO Civilian Detection        │                      │  Servo Yaw/Pitch │
│  Kalman Filter + Prediction     │                      │  Laser Tracking  │
│  Multi-Target Priority Queue    │                      │  Laser Firing    │
│  Behavior Analysis              │                      │  Auto-fire       │
│  RF Threat Scanning             │                      │  Ultrasonic Dist │
│  Radar HUD + Recording          │                      │  Joystick Manual │
│  Civilian / Sticker Abort       │                      │  LCD Display     │
│  TCP Alert Server               │                      │  Buzzer          │
└─────────────────────────────────┘                      └──────────────────┘
         │
         ▼
  TCP Clients (9000)
  logs/  recordings/
```

---

## Features

| # | Feature | File |
|---|---------|------|
| 1 | **RF presence detection** — scans Wi-Fi for drone SSIDs, classifies foe vs background | `tracker.py` |
| 2 | **YOLO drone detection** — uses a dedicated drone YOLO model with fallback to `yolov8n` | `tracker.py` |
| 3 | **Automatic aiming** — sends yaw/pitch over UDP to ESP32 at up to 60 Hz | `tracker.py` + `turret_esp32.py` |
| 4 | **Trajectory prediction** — Kalman-predicted future position plotted on HUD as cyan cross | `tracker.py` |
| 5 | **Kalman filter smoothing** — constant-velocity model per track, smooths bbox and velocity | `tracker.py` |
| 6 | **Multi-drone swarm tracking** — nearest-neighbour tracker with TTL, up to 8 on radar | `tracker.py` |
| 7 | **Smart priority queue** — scores each drone on approach speed, distance, direction, stability, behaviour | `tracker.py` |
| 8 | **Behaviour analysis** — classifies each drone as AGGRESSIVE / APPROACHING / HOVERING / LEAVING | `tracker.py` |
| 9 | **Constant alerts + recording** — TCP JSON alert server, rotating log files, track and fire video clips | `tracker.py` |
| 10 | **Auto-fire** — fires when the top-priority target is stable and in range; cooldown enforced | `turret_esp32.py` |
| 11 | **Manual fire** — physical fire button on ESP32 and `[F]` key on PC | both |
| 12 | **Radar view** — polar radar overlay with behaviour symbols and velocity arrows per target | `tracker.py` |
| 13 | **Manual override** — physical joystick + mode button on ESP32 overrides all auto tracking | `turret_esp32.py` |
| 14 | **Civilian abort** — YOLO person detection causes immediate cease-fire; safety sticker also triggers abort | `tracker.py` |

---

## Hardware

### ESP32 GPIO Map

| Pin | Function |
|-----|----------|
| 18 | Servo Yaw (PWM) |
| 19 | Servo Pitch (PWM) |
| 22 | Tracking laser |
| 23 | Firing laser |
| 21 | Manual fire button |
| 13 | Mode toggle button (AUTO / MANUAL) |
| 25 | Buzzer |
| 26 | LCD SDA (I2C) |
| 27 | LCD SCL (I2C) |
| 32 | Ultrasonic TRIG (HC-SR04) |
| 33 | Ultrasonic ECHO (HC-SR04) |
| 34 | Joystick Pan (ADC) |
| 35 | Joystick Tilt (ADC) |

### PC Requirements
- Camera (USB webcam or CSI)
- Python 3.9+
- Same Wi-Fi network as ESP32

---

## Software Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/YOUR_USERNAME/drone-turret.git
cd drone-turret
pip install -r requirements.txt
```

### 2. Drone YOLO model

Place a YOLOv8 model trained on drone images at:

```
drone_yolov8.pt
```

If not present, the system automatically falls back to `yolov8n.pt` (COCO general model). A publicly available drone dataset for training is [DroneDetection on Roboflow Universe](https://universe.roboflow.com).

`yolov8n.pt` is downloaded automatically by `ultralytics` on first run.

### 3. Configure PC tracker

Edit the top of `tracker.py`:

```python
ESP32_IP       = "10.219.186.196"   # IP of your ESP32 after it connects to Wi-Fi
ESP32_UDP_PORT = 5005
CAMERA_INDEX   = 0                  # webcam index
```

### 4. Flash ESP32

Install MicroPython on your ESP32, then:

```bash
pip install mpremote
mpremote connect PORT cp turret_esp32.py :main.py
```

Edit the top of `turret_esp32.py` first:

```python
WIFI_SSID        = "your_network"
WIFI_PASSWORD    = "your_password"
DRONE_WIFI_SSID  = "drone_hotspot_ssid_to_watch_for"
```

### 5. Run

```bash
python tracker.py
```

---

## HUD Controls

| Key | Action |
|-----|--------|
| `F` | Manual fire / clear cease-fire latch |
| `A` | Toggle auto-fire armed / safe |
| `R` | Toggle radar overlay |
| `M` | Toggle debug bounding-box overlay |
| `C` | Toggle high / low confidence threshold |
| `S` | Cycle to next camera index |
| `Q` / `ESC` | Quit |

---

## Alert Server

A TCP server runs on port `9000`. Connect any client to receive a newline-delimited JSON stream at ~1 Hz:

```json
{
  "ts": "2025-01-15T12:34:56.789",
  "status": "LOCKED",
  "locked": true,
  "cease_fire": false,
  "rf": "RF: 1 DRONE(S) DETECTED",
  "target_count": 2,
  "priority_queue": [
    {
      "id": 3,
      "dist_m": 4.2,
      "behavior": "APPROACHING",
      "threat_score": 0.847,
      "speed_m_s": 2.1,
      "yaw": -12.5,
      "pitch": 3.2
    }
  ]
}
```

---

## Recordings

Tracking and fire clips are saved to `recordings/` automatically:

- `track_YYYYMMDD_HHMMSS.mp4` — starts when a drone is first acquired, closes when lost
- `fire_YYYYMMDD_HHMMSS.mp4` — starts on each fire event, includes 3 s pre-event buffer

---

## Civilian Safety System

Two independent abort mechanisms exist:

1. **YOLO person detection** — `yolov8n` runs every frame on COCO classes. If a person is detected, all fire is immediately ceased and the servo parks at 0°.
2. **Safety sticker detection** — HSV colour matching on every detection crop. The sticker (cyan + yellow colocated) latches a cease-fire state that requires operator `[F]` key to clear.

Both mechanisms write to the log, display on the HUD, and send a cease-fire flag over UDP to the ESP32.

---

## Project Structure

```
drone-turret/
├── tracker.py            # PC: detection, tracking, HUD, alert server
├── turret_esp32.py       # ESP32: servo control, firing, manual override
├── requirements.txt      # Python dependencies
├── safety_sticker.jpeg   # Reference image for the safety sticker
├── recordings/           # Auto-created: video clips
└── logs/                 # Auto-created: structured log files
```

---

## Tuning Reference

| Constant | File | Effect |
|----------|------|--------|
| `DRONE_REAL_WIDTH_M` | tracker | Distance estimation accuracy |
| `HFOV_DEG` | tracker | Camera horizontal FOV |
| `DRONE_CONF_HIGH` | tracker | Detection sensitivity |
| `PREDICT_SECS` | tracker | How far ahead to predict trajectory |
| `AUTO_FIRE_STABLE_FRAMES` | ESP32 | Frames of stability before auto-fire |
| `AUTO_FIRE_MIN_M / MAX_M` | ESP32 | Firing range gate |
| `AUTO_FIRE_COOLDOWN_S` | ESP32 | Minimum time between auto-fire events |
| `BURST_COUNT / BURST_ON_MS` | ESP32 | Burst fire pattern |
| `SMOOTH` | ESP32 | Servo tracking smoothing factor |

---

## License

MIT
