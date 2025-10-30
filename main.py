#main.py
# Run:  python main.py
# Behavior:
#  - Press 't' to run emotion recognition for a 3s window (once), then the robot says the best label.
#  - Press 'q' to quit.
#  - No continuous/forever recognition; it's user-triggered only.

import pickle
import numpy as np
import cv2
import socket
import time
from collections import deque

from utils import FaceMeshExtractor, landmarks_to_feature  # must be available

# ----- Config -----
MODEL_PATH = "model"
CLASS_NAMES_PATH = "class_names.npy"

CAMERA_URL = "http://localhost:8094/Default.m3u8"
ARC_HOST, ARC_PORT = "127.0.0.1", 5000

# Decision tuning
SMOOTH_WINDOW = 15       # used inside the timed window for quick smoothing
THRESHOLD = 0.55
MARGIN = 0.15
WINDOW_SECONDS = 3.0     # how long to sample after pressing 't'

# Map model labels -> words to say (no "confused"; angry -> "mad")
LABEL_NORMALIZE = {
    "angry": "mad",
    "mad": "mad",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprised",
}

# ----- ARC speech-only client -----
def connect():
    while True:
        try:
            s = socket.create_connection((ARC_HOST, ARC_PORT), timeout=3)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[ARC] connected")
            return s
        except OSError as e:
            print("[ARC] connect failed:", e, "retrying...")
            time.sleep(1.5)

class ARC:
    def __init__(self):
        self.s = connect()

    def _send(self, ez: str):
        data = (ez + "\r\n").encode("utf-8")
        try:
            self.s.sendall(data)
            try:
                self.s.settimeout(0.25)
                _ = self.s.recv(4096)
            except Exception:
                pass
        except OSError:
            try:
                self.s.close()
            except:
                pass
            self.s = connect()
            self.s.sendall(data)

    def say(self, text: str):
        safe = text.replace('"', "'")
        self._send(f'SayEZB("{safe}")')

    def set_label(self, label: str):
        safe = label.lower()
        self._send(f'$EmotionLabel = "{safe}"')

    def announce(self, label: str):
        # One-shot: set label + speak it
        lab = (label or "neutral").lower().strip()
        self.set_label(lab)
        self._send(f'print("Emotion (window result): {lab}")')
        self.say(lab.capitalize())

# ----- Model helpers -----
def load_class_names(path=CLASS_NAMES_PATH):
    try:
        names = np.load(path, allow_pickle=True).tolist()
        return [str(n) for n in names]
    except Exception:
        return ["ANGRY", "HAPPY", "SAD", "SURPRISED"]

def decide_label(avg_probs, class_names):
    # Top-vs-second margin/threshold rule
    sorted_idxs = np.argsort(avg_probs)[::-1]
    top_idx = sorted_idxs[0]
    top = float(avg_probs[top_idx])
    second = float(avg_probs[sorted_idxs[1]]) if len(sorted_idxs) > 1 else 0.0
    if top < THRESHOLD or (top - second) < MARGIN:
        return "NEUTRAL/UNSURE", top, second, top_idx
    return class_names[top_idx].upper(), top, second, top_idx

def normalize_label(label_text: str) -> str:
    norm = LABEL_NORMALIZE.get(label_text.lower(), "neutral")
    return norm

# ----- Timed window recognition -----
def run_timed_window(cap, fm, model, class_names, overlay_font, seconds=WINDOW_SECONDS):
    """
    Capture for `seconds`, collect probabilities when a face is found,
    average them, decide one label, return the normalized word to say.
    """
    start = time.time()
    probs_window = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        remaining = max(0.0, seconds - (time.time() - start))
        # Draw basic overlay during capture
        cv2.putText(frame, f"CAPTURING... {remaining:0.1f}s",
                    (20, 40), overlay_font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit now", (20, 70),
                    overlay_font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Extract landmarks and predict if possible
        lm = fm.get_landmarks(frame)
        if lm is not None and len(lm) > 0:
            feat = landmarks_to_feature(lm)
            if feat is not None:
                X = feat.reshape(1, -1)
                proba = model.predict_proba(X)[0]
                probs_window.append(proba)

        cv2.imshow("JD Emotion (press 't' to sample, 'q' to quit)", frame)
        k = (cv2.waitKey(1) & 0xFF)
        if k == ord('q'):
            return "__QUIT__"

        if (time.time() - start) >= seconds:
            break

    if not probs_window:
        return "neutral"  # no face / nothing confident

    # Average all collected probabilities over the window
    avg = np.mean(np.stack(probs_window, axis=0), axis=0)
    label_text, _, _, _ = decide_label(avg, class_names)
    return normalize_label(label_text)

# ----- Main -----
def main():
    # load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    class_names = [c.upper() for c in load_class_names()]

    # camera + facemesh
    fm = FaceMeshExtractor(static_image_mode=False)
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        raise SystemExit("Could not open camera stream")

    # ARC client
    arc = ARC()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Idle preview loop: wait for key commands
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # On-screen help
        cv2.putText(frame, "Press 't' to sample ~3s and speak result",
                    (20, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit",
                    (20, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("JD Emotion (press 't' to sample, 'q' to quit)", frame)
        k = (cv2.waitKey(1) & 0xFF)

        if k == ord('q'):
            break

        if k == ord('t'):
            result = run_timed_window(cap, fm, model, class_names, font, seconds=WINDOW_SECONDS)
            if result == "__QUIT__":
                break
            # Speak one time with the window's best guess
            arc.announce(result)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
