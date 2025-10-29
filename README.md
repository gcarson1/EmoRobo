
# EmoRobo — ARC + Local Emotion Recognition (Brutally Accurate)

This repo runs **offline facial expression recognition** on your PC and shows the result in a window. It **can** also poke ARC if you enable its TCP Script Server and have **Auto Position** frames that match the labels — otherwise ARC does nothing. No cloud, no magic.

---

## What This Actually Does

- **Camera source:** ARC’s Camera Device HLS stream at `http://localhost:8094/Default.m3u8` (hard‑coded in `test_model.py`).
- **Model:** A scikit‑learn pipeline pickled to a single file named **`model`** in the project root (no extension).
- **Classes:** Loaded from `class_names.npy` (NumPy array of strings).
- **Landmarks:** MediaPipe FaceMesh → features via `utils.py` → SVM probabilities.
- **Decision:** Moving average over last **15** frames, label only if `top >= 0.55` and `(top - second) >= 0.15`; otherwise `NEUTRAL/UNSURE`.
- **Display:** OpenCV window overlays label + debug.
- **ARC hook (optional):** If ARC’s TCP Script Server is reachable at **127.0.0.1:5000**, `test_model.py` will send `ControlCommand("Auto Position","AutoPositionFrame","<Label>")` and `Say("<Label>")` at most every **0.75s** when the label changes. If you don’t have matching frames, nothing happens.

> If you expected motor reactions and you see **only the window label**, it means **your ARC isn’t exposing the Script Server or your Auto Position frames don’t match**. Fix that or accept display‑only behavior.

---

## Repo Layout (key files)

```
ARC_proj/
  ├─ idk.EZB                 # ARC project
  └─ jd_controller.py        # Minimal ARC client (optional/manual)
class_names.npy              # Saved by prepare_data.py (or included)
model/                       # Directory placeholder (not used by test_model.py)
model                        # <-- The actual trained model file (pickle, no ext)
requirements.txt
test_model.py                # Main runtime (reads ARC HLS and classifies)
train_model.py               # Trains model -> writes ./model
prepare_data.py              # Builds data.txt + class_names.npy from dataset
utils.py                     # FaceMesh feature extraction helpers
cam_test.py                  # Local camera index sanity check
arc_script.txt               # Example ARC script reacting to $EmotionLabel
Plan.md                      # Project notes
```

**Yes, the model file name is literally `model` with no extension.** That’s how `pickle.dump` writes it in `train_model.py` and how `test_model.py` loads it (`MODEL_PATH = "model"`).

---

## Requirements

- **Python** 3.9–3.11 (3.12 will fight you on deps)
- **Synthiam ARC** (current build)
- Same machine runs **ARC** and **Python** (because the HLS URL is `localhost`)

Install Python deps:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:
```
numpy==1.26.4
scikit-learn==1.3.2
opencv-python==4.9.0.80
mediapipe==0.10.14
```

---

## ARC Setup (no guessing, do this)

1. **Open ARC project**: `ARC_proj/idk.EZB`
2. **Camera Device** skill:
   - Start the EZ‑Robot camera (verify preview works in ARC).
   - Enable **HLS/HTTP stream** so ARC serves `http://localhost:8094/Default.m3u8`.
   - Test that URL in a browser. If the browser can’t open it, Python won’t either.
3. **(Optional for robot reactions)** **Script TCP Server (RAW EZ‑Script)**:
   - Add/start the Script Server on **127.0.0.1:5000** (default).
   - In **Auto Position**, create frames named to match the labels your model emits (e.g., `Happy`, `Sad`, `Angry`, `Surprised`, `Neutral`). Capitalization matters because the code title‑cases.
   - If you don’t do this, ARC will ignore the commands and only the PC window will update.

---

## Training a Model (only if you don’t already have `./model`)

1. Prepare a dataset folder with subfolders per class (e.g., `happy/`, `sad/`, …).
2. Build features:
   ```bash
   python prepare_data.py /path/to/dataset_root
   ```
   This writes `data.txt` and `class_names.npy`.
3. Train:
   ```bash
   python train_model.py
   ```
   This writes the pickled pipeline to `./model`.
4. Confirm you now have:
   - `./model` (file)  
   - `./class_names.npy`

If `./model` is missing, **`test_model.py` will crash** on startup. That’s on you.

---

## Running the Demo (display‑only by default)

Make sure ARC is streaming at `http://localhost:8094/Default.m3u8`.

```bash
python test_model.py
```

- A window titled **Emotion** opens.
- Press **`q`** to quit.
- If ARC Script Server is up and frames exist, the robot will speak and try the corresponding Auto Position frame when a stable label is detected; otherwise, nothing on the robot.

---

## Config knobs (in `test_model.py`)

- `MODEL_PATH = "model"`  
- `CLASS_NAMES_PATH = "class_names.npy"`
- `SMOOTH_WINDOW = 15`
- `THRESHOLD = 0.55`
- `MARGIN = 0.15`
- `ARC_HOST = "127.0.0.1"`
- `ARC_PORT = 5000`
- `SEND_EVERY_SEC = 0.75`
- **Camera URL**: `cv2.VideoCapture("http://localhost:8094/Default.m3u8")` (change here if you move ARC off-box)

---

## Common Failure Points (and the blunt fix)

- **Black window / cannot open stream** → Your ARC HLS stream isn’t enabled or isn’t on `localhost:8094`. Fix Camera Device settings and test in a browser.
- **Immediate crash: cannot open `model`** → You didn’t train or copy the model. Run `train_model.py` or drop a valid pickle at `./model`.
- **Mediapipe import error** → Wrong Python version or missing deps. Use 3.9–3.11 and reinstall from `requirements.txt`.
- **Robot never reacts** → You didn’t start the **Script TCP Server** in ARC, or your Auto Position frames don’t match the label text (case/spacing).
- **Label flickers** → Lower `SMOOTH_WINDOW` or tweak `THRESHOLD`/`MARGIN`. If you make them too loose, expect false positives.
- **High CPU** → FaceMesh is not free. Close other apps or drop camera FPS/size in ARC.

---

## Roadmap (when you decide to care)

- Config file for stream URL/thresholds instead of hard‑coded values.
- Proper logging instead of print spam.
- Model card + dataset notes.
- Unit tests for the feature extractor.

---

## License

MIT (see `LICENSE`).

---

If you want me to also **generate a tiny `config.py` and patch `test_model.py`** to read it, say the word. Until then, this README documents the code **exactly as it is now**.
