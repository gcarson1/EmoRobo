## Requirements

- **Python** 3.9–3.11 (3.12 will fight you on deps)
- **Synthiam ARC** (current build)

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

## ARC Setup

1. **Open ARC project**: `ARC_proj/idk.EZB`
2. **Camera Device** skill:
   - Start the EZ‑Robot camera (verify preview works in ARC).
   - Enable **HLS/HTTP stream** so ARC serves `http://localhost:8094/Default.m3u8`.
   - Test that URL in a browser. If the browser can’t open it, Python won’t either.
3. **Script TCP Server (RAW EZ‑Script)**:
   - Add/start the Script Server on **127.0.0.1:5000** (default).
   - In **Auto Position**, create frames named to match the labels your model emits (e.g., `Happy`, `Sad`, `Angry`, `Surprised`, `Neutral`). Capitalization matters because the code title‑cases.
   
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

## Running the Demo

Make sure ARC is streaming at `http://localhost:8094/Default.m3u8`.

```bash
python test_model.py
```

- A window titled **Emotion** opens.
- Press **`q`** to quit.
- If ARC Script Server is up and frames exist, the robot will speak and try the corresponding Auto Position frame when a stable label is detected; otherwise, nothing on the robot.

---

## Common Failure Points (and the blunt fix)

- **Immediate crash: cannot open `model`** → You didn’t train or copy the model. Run `train_model.py` or drop a valid pickle at `./model`.
- **Mediapipe import error** → Wrong Python version or missing deps. Use 3.9–3.11 and reinstall from `requirements.txt`.
- **Label flickers** → Lower `SMOOTH_WINDOW` or tweak `THRESHOLD`/`MARGIN`. If you make them too loose, expect false positives.
- **Can't activate venv on Windows (`Activate.ps1` blocked)** → PowerShell is blocking scripts. Run this once, then activate again:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

---

## License

MIT (see `LICENSE`).
