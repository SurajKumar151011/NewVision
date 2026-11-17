# utils.py  (works on any device - no winsound)
import io
import math
import struct
import wave

import cv2
import numpy as np


def make_beep(freq: int, ms: int, volume: float = 0.8) -> bytes:
    """Create a short beep as WAV bytes so the browser can play it."""
    rate = 22050
    n = int(rate * ms / 1000)
    frames = b"".join(
        struct.pack("<h", int(volume * 32767 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


WARNING_BEEP = make_beep(1200, 250)
DANGER_BEEP = make_beep(2200, 500)


def draw_overlay(frame: np.ndarray, status: str, fps: float) -> np.ndarray:
    """Draw a colored border on alerts and an FPS badge."""
    height, width, _ = frame.shape

    if status == "Danger":
        border_color = (0, 0, 255)
    elif status == "Warning":
        border_color = (0, 255, 255)
    else:
        border_color = (0, 255, 0)

    if status in ["Warning", "Danger"]:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, 6)

    cv2.rectangle(frame, (width - 130, 10), (width - 10, 45), (15, 23, 42), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame
