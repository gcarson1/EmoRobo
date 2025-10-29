#!/usr/bin/env python3

from typing import Optional
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("mediapipe is required. See requirements.txt") from e

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

class FaceMeshExtractor:
    def __init__(self, static_image_mode: bool):
        self.static_image_mode = static_image_mode
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, bgr_image) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None

        h, w = bgr_image.shape[:2]
        face = res.multi_face_landmarks[0]
        pts = []
        for lm in face.landmark:
            x = lm.x * w
            y = lm.y * h
            z = lm.z * w
            pts.append((x, y, z))
        return np.array(pts, dtype=np.float32)

def landmarks_to_feature(landmarks: np.ndarray) -> Optional[np.ndarray]:

    if landmarks is None or landmarks.ndim != 2 or landmarks.shape[1] != 3:
        return None

    pts = landmarks.astype(np.float32)

    centroid = np.mean(pts, axis=0, keepdims=True)
    pts_centered = pts - centroid

    scale = None
    try:
        pL = pts[LEFT_EYE_OUTER, :2]
        pR = pts[RIGHT_EYE_OUTER, :2]
        d = np.linalg.norm(pL - pR)
        if d > 1e-6:
            scale = d
    except Exception:
        scale = None

    if scale is None:
        rms = np.sqrt(np.mean(pts_centered[:, :2] ** 2))
        scale = rms if rms > 1e-6 else 1.0

    pts_norm = pts_centered / scale

    return pts_norm.reshape(-1).astype(np.float32)
