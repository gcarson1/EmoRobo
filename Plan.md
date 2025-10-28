# JD Humanoid Emotion Detection (Offline, Local Model Plan)

This plan enables your JD Humanoid (EZ-B v4) to recognize facial emotions **offline** by streaming its camera feed to your computer, classifying the emotion locally using Python/OpenCV, and sending a label back to ARC for the robot to react.

---

## 0) Ground Rules
- **Brains vs. body:** Your **PC runs ARC + Python**. The **EZ-B** is I/O + camera. No Python runs on the EZ-B.
- **Offline means LAN:** Internet is not required. Your PC and EZ-B just need to be on the **same Wi-Fi/LAN**.
- **Prove the loop with a USB webcam first.** Then switch to the JD camera stream.

---

## 1) Put EZ-B on Your LAN (Client Mode)
**Goal:** Robot and PC see each other on your Wi-Fi so ARC and Python can communicate offline.

### Steps
1. Power JD.
2. Connect your PC to the EZ-B hotspot (SSID like `EZ-B v4 ...`).
3. Open **http://192.168.1.1** → **Wi-Fi Client Mode** → enter your router SSID + password → Save/Apply → reboot controller.
4. Reconnect your **PC to your router**.

### Verify
- In your router's client list, find the EZ-B (IP like `192.168.0.x`).
- Open its web UI at that IP.

**If stuck:**  
Reset to AP mode and retry; ensure correct Wi-Fi credentials and disable 5 GHz-only mode if needed.

---

## 2) ARC Baseline Project (Connection + Camera)
1. Install latest **Synthiam ARC** (Windows). Run once as **Administrator**.
2. In ARC: **Project → Add Skill → Communication → EZB Connection** → connect using EZ-B **LAN IP**.
3. Add **Camera Source (EZB)** → **Start**.  
   Use 320×240 @ 15 FPS for stability.

### Verify
- ARC shows *Connected*.
- You see live video.

**If black video:**  
Make sure you added **Camera Source (EZB)**, not “Camera Device.” Reduce resolution or close other camera apps.

---

## 3) Choose Label Transport (Communication Back to ARC)

### Option A — MQTT (Recommended)
1. Install **Mosquitto** broker (Windows). Allow firewall exceptions.
2. In ARC: **Add Skill → Scripting → MQTT Client**.
   - Host: `127.0.0.1`
   - Port: `1883`
   - Add **Subscription**: `jd/emotion`
3. Note the variable name holding the last payload.

**Test:**  
Publish `happy` to topic `jd/emotion` with MQTT Explorer → verify ARC variable updates.

### Option B — TCP (Simpler)
1. In ARC: **Add Skill → Communication → TCP Server** → Port: `5000`.
2. Set output variable: `$EmotionLabel`.

**Test:**  
Open `telnet 127.0.0.1 5000` → type `happy` → verify `$EmotionLabel` updates.

---

## 4) ARC Reaction Script
1. Add **Auto Position** skill → create poses: `Neutral`, `Happy`, `Sad`, `Angry`.
2. Add **Script** skill → paste logic below.

### Example Script
```ezscript
$emo = $EmotionLabel  # Or your MQTT variable
if ($emo = "") $emo = "neutral"

if ($emo != $LastEmo)
  $LastEmo = $emo
  if ($emo = "happy")
    ControlCommand("Auto Position", "Action", "Happy")
    Say("You look happy")
  elseif ($emo = "sad")
    ControlCommand("Auto Position", "Action", "Sad")
    Say("It will be okay")
  elseif ($emo = "angry")
    ControlCommand("Auto Position", "Action", "Angry")
    Say("Take a deep breath")
  else
    ControlCommand("Auto Position", "Action", "Neutral")
  endif
endif
```

---

## 5) Python Environment + Fake Publisher Test

### Setup
```bash
py -3.10 -m venv venv
venv\Scripts\activate
pip install opencv-python paho-mqtt numpy
```

### MQTT Test Script
```python
import time, itertools, paho.mqtt.client as mqtt
m = mqtt.Client(); m.connect("127.0.0.1", 1883, 60)

labels = itertools.cycle(["happy","neutral","sad","angry","neutral"])
while True:
    label = next(labels)
    m.publish("jd/emotion", label, qos=0, retain=False)
    print("sent:", label)
    time.sleep(2)
```

Run this — JD should cycle through actions every 2 seconds.

---

## 6) Integrate Emotion Model (From YouTube Code)

### Example Loop
```python
import cv2, time, paho.mqtt.client as mqtt
from emotion_model import predict_emotion

BROKER = "127.0.0.1"; TOPIC = "jd/emotion"
m = mqtt.Client(); m.connect(BROKER, 1883, 60)
cap = cv2.VideoCapture(0)

if not cap.isOpened(): raise SystemExit("Camera not found")

last_label, stable, last_send = None, 0, 0.0
STABLE_N, THRESH, SEND_DT = 5, 0.6, 0.2

while True:
    ok, frame = cap.read()
    if not ok: time.sleep(0.05); continue

    label, conf = predict_emotion(frame)
    cand = label.lower() if conf >= THRESH else "neutral"

    stable = stable + 1 if cand == last_label else 1
    last_label = cand

    if stable >= STABLE_N and (time.time() - last_send) >= SEND_DT:
        m.publish(TOPIC, last_label)
        last_send = time.time()
```

**Verify:**  
Run with your webcam → JD reacts smoothly. Adjust `THRESH` and `STABLE_N` to remove jitter.

---

## 7) Switch to JD Camera Stream

### Option 1 — Direct MJPEG Stream
```python
cap = cv2.VideoCapture("http://<EZB_IP>:<PORT>/<path>")
```
*(You can find the correct URL in the EZ-B camera documentation or ARC.)*

### Option 2 — ARC Snapshot Server
```python
import requests, numpy as np, cv2, time
SNAP = "http://127.0.0.1:<port>/snapshot.jpg"

while True:
    r = requests.get(SNAP, timeout=0.3)
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    label, conf = predict_emotion(frame)
    # Publish logic same as before
```

**If frame rate poor:** Reduce camera resolution or call snapshots slower.

---

## 8) Hardening
- Publish max **5–10 times per second**.
- Send `neutral` when confidence < threshold or no face found.
- ARC watchdog: revert to `Neutral` if no label for >2 s.
- Smooth motion → short non-blocking Auto Position actions.
- Keep lighting consistent — dim light kills accuracy.

---

## 9) Final Checklist
1. ✅ EZ-B Client mode (LAN connection verified)
2. ✅ ARC connected, camera streaming
3. ✅ ARC script manually triggers robot actions
4. ✅ MQTT/TCP test: fake labels trigger actions
5. ✅ Python stub works (cycle through labels)
6. ✅ Emotion model integrated, working via webcam
7. ✅ Switched to JD stream and stable labels
8. ✅ Watchdog + neutral fallback in place

---

## Key Points to Look Up
- Exact **EZ-B camera URL/port** for MJPEG stream.
- **MQTT skill variable** name (check its Variables tab).
- **Your model's preprocessing** (face crop, grayscale, resize).

---

**Structure recap:**
```
jd_emotion/
├─ arc_project.ezb
├─ venv/
├─ requirements.txt
├─ predict.py
└─ emotion_model.py
```

**requirements.txt**
```
opencv-python
paho-mqtt
numpy
```
