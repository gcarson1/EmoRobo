import cv2, time, paho.mqtt.client as mqtt
from emotion_model import predict_emotion

BROKER = "127.0.0.1"
TOPIC  = "jd/emotion"

# Connect to MQTT broker
m = mqtt.Client()
m.connect(BROKER, 1883, 60)

# Capture from webcam first to verify pipeline
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not available")

# Stability parameters
last_label = None
stable = 0
last_send = 0.0
STABLE_N = 5        # frames required for same label before publish
THRESH   = 0.60     # confidence threshold
SEND_DT  = 0.2      # min seconds between publishes

print("[INFO] Starting emotion detection loop...")
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue

    # Predict emotion (replace with your model logic)
    label, conf = predict_emotion(frame)
    candidate = label.lower() if conf >= THRESH else "neutral"

    # Stabilize detections
    if candidate == last_label:
        stable += 1
    else:
        last_label = candidate
        stable = 1

    # Publish stable label to ARC
    now = time.time()
    if stable >= STABLE_N and (now - last_send) >= SEND_DT:
        m.publish(TOPIC, last_label, qos=0, retain=False)
        print(f"[SENT] {last_label} ({conf:.2f})")
        last_send = now

    # Optional: show debug preview
    cv2.putText(frame, f"{last_label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
