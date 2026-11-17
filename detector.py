# detector.py
import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
from typing import Tuple, Any

def initialize_face_mesh():
    """Initialize MediaPipe Face Mesh model safely with stable detection confidence."""
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

def process_frame(frame: np.ndarray, face_mesh: Any) -> Tuple[np.ndarray, Any]:
    """Process frame through MediaPipe Face Mesh."""
    if frame is None:
        return frame, None
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    return frame, results


# ---- hand tracking (used to spot phone use, eating and drinking)
HAND_POINTS = (0, 4, 8, 12, 16, 20, 9, 5, 17)   # wrist, finger tips, palm centre, knuckles


def initialize_hands():
    """Hand tracker. Returns None if it cannot start, so the app still works without it."""
    try:
        return mp_hands.Hands(max_num_hands=2, model_complexity=0,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        return None


def extract_hands(frame, hands):
    """Returns a list of hands; each hand is a list of (x, y) points (0-1 coordinates)."""
    if frame is None or hands is None:
        return []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return []
        return [[(h.landmark[i].x, h.landmark[i].y) for i in HAND_POINTS]
                for h in res.multi_hand_landmarks]
    except Exception:
        return []
