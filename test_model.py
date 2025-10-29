#!/usr/bin/env python3

import pickle
import numpy as np
import cv2
import socket
import time
from collections import deque
from utils import FaceMeshExtractor, landmarks_to_feature

MODEL_PATH = "model"
CLASS_NAMES_PATH = "class_names.npy"

SMOOTH_WINDOW = 15
THRESHOLD = 0.55
MARGIN = 0.15


ARC_HOST = "127.0.0.1"
ARC_PORT = 5000
SEND_EVERY_SEC = 0.75

def open_arc_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((ARC_HOST, ARC_PORT))
    return s

def send_arc(s, line: str):
    msg = (line.strip() + "\r\n").encode("utf-8", errors="ignore")
    s.sendall(msg)

def load_class_names():
    try:
        names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
        return [str(n) for n in names]
    except Exception:
        return ["ANGRY", "HAPPY", "SAD", "SURPRISED"]

def format_topk(probs, class_names, k=3):
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

def decide_label(avg_probs, class_names):
    sorted_idxs = np.argsort(avg_probs)[::-1]
    top = float(avg_probs[sorted_idxs[0]])
    second = float(avg_probs[sorted_idxs[1]]) if len(sorted_idxs) > 1 else 0.0

    if top < THRESHOLD or (top - second) < MARGIN:
        return "NEUTRAL/UNSURE", top, second, sorted_idxs[0]

    return class_names[sorted_idxs[0]].upper(), top, second, sorted_idxs[0]

def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    class_names = load_class_names()

    fm = FaceMeshExtractor(static_image_mode=False)
    cap = cv2.VideoCapture("http://localhost:8094/Default.m3u8")
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0")

    probs_buffer = deque(maxlen=SMOOTH_WINDOW)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        arc = open_arc_socket()
        print(f"[INFO] Connected to ARC at {ARC_HOST}:{ARC_PORT}")
    except Exception as e:
        print(f"[WARN] Could not connect to ARC: {e}")
        arc = None

    last_label_sent = None
    last_send_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        lm = fm.get_landmarks(frame)
        label_text = "NO FACE"
        debug_lines = []

        if lm is not None and len(lm) > 0:
            feat = landmarks_to_feature(lm)
            if feat is not None:
                X = feat.reshape(1, -1)
                proba = model.predict_proba(X)[0]
                probs_buffer.append(proba)
                avg = np.mean(np.stack(probs_buffer, axis=0), axis=0) if probs_buffer else proba

                label_text, top, second, top_idx = decide_label(avg, class_names)

                now = time.time()
                if arc and label_text not in ("NO FACE", "NEUTRAL/UNSURE"):
                    if (label_text != last_label_sent) and (now - last_send_ts >= SEND_EVERY_SEC):
                        frame_name = label_text.split("/")[0].title()
                        try:
                            send_arc(arc, f'ControlCommand("Auto Position","AutoPositionFrame","{frame_name}")')
                            send_arc(arc, f'Say("{frame_name}")')
                            last_label_sent = label_text
                            last_send_ts = now
                        except Exception as e:
                            print(f"[WARN] ARC send failed: {e}")
                            try:
                                arc.close()
                            except: pass
                            arc = None
                else:
                    if arc and (now - last_send_ts >= SEND_EVERY_SEC):
                        try:
                            send_arc(arc, 'ControlCommand("Auto Position","Stop")')
                            last_send_ts = now
                        except Exception as e:
                            print(f"[WARN] ARC stop failed: {e}")
                            try:
                                arc.close()
                            except: pass
                            arc = None

                topk = format_topk(avg, class_names, k=3)
                debug_lines = [f"{name.upper():>10}: {p*100:5.1f}%" for name, p in topk]
            else:
                probs_buffer.clear()
        else:
            probs_buffer.clear()

        cv2.putText(frame, label_text, (20, 40), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        y = 70
        for line in debug_lines:
            cv2.putText(frame, line, (20, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Emotion", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
