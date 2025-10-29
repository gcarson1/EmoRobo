
# EZ-Robot Facial Expression Recognition (Offline Local Model)

This project enables an EZ-Robot (JD Humanoid) to read human facial expressions **without relying on cloud APIs**.  
It streams camera footage from Synthiam ARC to a Python model running locally, classifies facial expressions, and (optionally) sends the emotion back to ARC to trigger robot responses.

---

## 🚧 Project Overview

This project was built to solve two main problems:

1. **ARC’s built-in AI vision isn't reliable offline**  
2. We needed a **local, fast, and customizable ML model** to classify emotions from the robot’s camera feed

The solution:

- Use ARC to control the robot + host the camera stream
- Use Python to run an offline ML model for emotion detection
- (Optional) Send the detected emotion back to ARC for robot behaviors

---

## ✅ Requirements

### Hardware
- EZ-Robot JD Humanoid (or any EZ-B v4 based robot with camera)
- Computer on the **same network** as the robot

### Software
| Component | Version | Notes |
|-----------|----------|--------|
| Synthiam ARC | Latest | Must have Camera and EZB skills installed |
| Python | 3.9–3.11 | **Do not use 3.12+** due to ML dependency issues |
| pip packages | See `requirements.txt` | Install after cloning repo |

> If you skip matching Python versions, expect dependency failures.

---

## 📦 Setup Instructions

### 1. Install Synthiam ARC

Download & install from:  
https://synthiam.com/Software/ARC

Run ARC once after installation so it finishes module setup.

---

### 2. Clone the Repository

```bash
git clone <your_repo_url_here>
cd <repo_folder_name>
```

Replace the placeholder with your actual repo link.

---

### 3. Open the ARC Project

1. Launch Synthiam ARC  
2. Click **Open Project**
3. Select the included `.ezproject` file from this repo

**If you don’t see the project file, you're in the wrong directory.**

---

### 4. Install Python Dependencies

From inside the repo directory:

```bash
pip install -r requirements.txt
```

If errors appear, **read them**. Most issues are missing OS packages or wrong Python version.

---

## 🤖 Robot Connection & Camera Setup

### 5. Connect to the Robot Network

You have two options:

| Mode | When to Use |
|--------|----------------|
| Connect to EZ-B Wi-Fi | Initial setup or direct connection |
| Use Home/Local LAN | For stable two‑way messaging between ARC & Python |

ARC must show a successful EZ-B connection before proceeding.

---

### 6. Connect to the Robot in ARC

In ARC:

1. Add or open the **EZ-B Connection** skill  
2. Enter the EZ-B’s IP address  
3. Click **Connect**

You should hear the robot speak or chime when connected.

---

### 7. Start Camera Feed

In ARC:

1. Add the **Camera Device** skill (if not already present)
2. Select the EZ-Robot Camera
3. Click **Start** to confirm live video feed

If the camera fails here, **stop—Python won’t fix this.** Fix ARC first.

---

### 8. Start Camera Stream Server

Still in ARC Camera Skill:

- Enable **HTTP/MJPEG Camera Streaming**
- Note the stream URL provided (e.g. `http://192.168.1.1:24` or similar)

Test the stream in a browser **first**.  
If your browser can’t load it, Python won’t either.

Example test URL check:

```
http://<robot_ip>:<port>/
```

---

## 🧠 Run Local Emotion Detection Model

### 9. Run the Local Test Script

```bash
python test_model.py
```

The script will:

- Pull frames from the ARC camera stream
- Run the trained model locally
- Print or display the detected emotions in real‑time

If it errors immediately, check:
- Wrong stream URL in script
- Missing model file(s)
- Wrong Python version

---

## 🧪 Troubleshooting Checklist

If something fails, check these **in this order**:

| Step | Must be True |
|------|----------------|
| ARC installed and runs | ✅ |
| Robot connects in ARC | ✅ |
| Camera feed visible in ARC | ✅ |
| Stream opens in browser | ✅ |
| Python dependencies installed | ✅ |
| `test_model.py` runs without crashing | ✅ |

If any ❌ is true, fix that step first. Don’t skip ahead.

---

## 🚀 Next Phase: Robot Reactions (Optional)

Once basic detection works, you can expand functionality:

| Feature | Description |
|----------|----------------|
| Send emotion back to ARC | Trigger robot animations, speech, LED responses |
| Replace model with a better one | Increase accuracy, add more emotions |
| Make script headless | Run without UI for performance |
| Bundle into ARC behavior control | Fully integrate into robot behaviors |

If you want, a `robot_actions.py` module can be added to handle reactions automatically.

---

## 📂 Expected Project Structure

```
project_root/
│
├─ arc_project/
│   └─ your_project.ezproject
│
├─ model/
│   ├─ model.pkl
│   └─ labels.json
│
├─ test_model.py
├─ requirements.txt
└─ README.md
```

---

## ✨ Credits

Developed as part of a hands-on project to enable **offline** emotion recognition for EZ‑Robots using Python + ARC integrations.

---

If you'd like, I can also generate a **Phase 2 README** for integrating the robot’s reaction system once we finish the test phase.
