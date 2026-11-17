# fatigue.py
import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def analyze_landmarks(landmarks, closed_frames, mouth_open_frames, head_down_frames, step=1.0, ear_thresh=0.18):
    """
    Analyzes facial landmarks for blinks, yawns, head tilt, and gaze direction.
    Distinguishes normal eye blinks from continuous drowsy eye closures.
    """
    # 1. Eye Aspect Ratio (EAR) - Normalized Vertical vs Horizontal
    left_v = calculate_distance(landmarks[159], landmarks[145])
    left_h = calculate_distance(landmarks[33], landmarks[133])
    left_ear = left_v / left_h if left_h > 0 else 0

    right_v = calculate_distance(landmarks[386], landmarks[374])
    right_h = calculate_distance(landmarks[362], landmarks[263])
    right_ear = right_v / right_h if right_h > 0 else 0

    ear = (left_ear + right_ear) / 2.0

    # 2. Mouth Aspect Ratio (MAR)
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_corner = landmarks[61]
    right_corner = landmarks[291]

    mouth_height = calculate_distance(top_lip, bottom_lip)
    mouth_width = calculate_distance(left_corner, right_corner)
    mar = mouth_height / mouth_width if mouth_width > 0 else 0

    # 3. Eye Closed Tracking (EAR < 0.18 means eyes closed)
    eye_closed = ear < ear_thresh
    if eye_closed:
        closed_frames += step
    else:
        closed_frames = 0

    # 4. Yawn Tracking
    yawning = mar > 0.38
    if yawning:
        mouth_open_frames += step
    else:
        mouth_open_frames = 0

    # 5. Head Pitch & Down Tilt Logic
    nose_tip = landmarks[1]
    chin = landmarks[152]
    forehead = landmarks[10]

    face_vertical = chin.y - forehead.y
    head_ratio = (chin.y - nose_tip.y) / face_vertical if face_vertical > 0 else 0.5

    is_head_down = head_ratio < 0.30 or nose_tip.y > 0.68

    if is_head_down:
        head_down_frames += step
    else:
        head_down_frames = 0

    # 6. Gaze Tracking
    eye_x = landmarks[468].x if len(landmarks) > 468 else landmarks[1].x
    if is_head_down:
        gaze = "Down"
    elif eye_x < 0.40:
        gaze = "Right"
    elif eye_x > 0.60:
        gaze = "Left"
    else:
        gaze = "Center"

    # 7. Fatigue Score Math (Normal blink does not penalize fatigue score)
    drowsy_eye_penalty = max(0, closed_frames - 8) * 0.08
    
    fatigue_score = min(
        1.0, 
        drowsy_eye_penalty + (mouth_open_frames * 0.08) + (head_down_frames * 0.10)
    )

    # Status Determination (Blinks won't trigger warning; continuous closure will)
    if fatigue_score >= 0.50 or closed_frames > 20 or mouth_open_frames > 8 or head_down_frames > 8:
        status = "Danger"
    elif fatigue_score >= 0.25 or closed_frames > 10 or mouth_open_frames > 4 or head_down_frames > 4:
        status = "Warning"
    else:
        status = "Normal"

    return status, fatigue_score, gaze, closed_frames, mouth_open_frames, head_down_frames, eye_closed, yawning
