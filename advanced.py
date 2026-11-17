# advanced.py
# Advanced driver-monitoring logic (no Streamlit needed, so it can be tested alone):
#   1. Personal calibration  (learns YOUR open-eye size and straight-ahead head position)
#   2. PERCLOS               (% of time the eyes are closed in the last 60 s)
#   3. Distraction detection (looking away from the road)
#   4. Trip log + safety score + HTML/CSV report
import csv
import html as _html
import io
import math
import time
from collections import Counter, deque

from fatigue import analyze_landmarks

RANK = {"Normal": 0, "Warning": 1, "Danger": 2}
LEVELS = ["Normal", "Warning", "Danger"]

YAW_LIMIT = 0.15          # how far the head may turn from straight ahead
AWAY_WARN = 60            # "frames at 30 fps" -> 2 seconds looking away
AWAY_DANGER = 120         # 4 seconds looking away
PERCLOS_WARN = 0.20       # 20 % of the last minute with eyes closed
PERCLOS_DANGER = 0.35     # 35 %

HAND_AT_MOUTH = 0.50      # hand this close to the mouth (in face widths) = eating / drinking
HAND_AT_EAR = 0.50        # hand this close to the ear / cheek = phone call
PHONE_WARN, PHONE_DANGER = 45, 90        # texting: 1.5 s / 3 s
CALL_WARN, CALL_DANGER = 60, 150         # phone call: 2 s / 5 s
EAT_WARN, EAT_DANGER = 60, 180           # eating or drinking: 2 s / 6 s

COACH_DRIVE_SECONDS = 2 * 3600   # suggest a break after 2 hours of driving
COACH_EVENTS = 3                 # ... or after this many alerts
COACH_WINDOW = 600               # ... within this many seconds (10 minutes)


# --------------------------------------------------------------------------- helpers
def _dist(p, q):
    return math.hypot(p.x - q.x, p.y - q.y)


def compute_ear(lm):
    """Eye Aspect Ratio (same formula as fatigue.py)."""
    lv, lh = _dist(lm[159], lm[145]), _dist(lm[33], lm[133])
    rv, rh = _dist(lm[386], lm[374]), _dist(lm[362], lm[263])
    left = lv / lh if lh > 0 else 0.0
    right = rv / rh if rh > 0 else 0.0
    return (left + right) / 2.0


def compute_yaw_ratio(lm):
    """0.5 = looking straight at the camera. Moves toward 0 or 1 as the head turns."""
    left, right, nose = lm[234], lm[454], lm[1]
    width = right.x - left.x
    return (nose.x - left.x) / width if abs(width) > 1e-6 else 0.5


def hand_flags(lm, hands, aspect):
    """Is a hand at the mouth (eating/drinking) or at the ear (phone call)?"""
    if not hands:
        return False, False

    def d(ax, ay, bx, by):
        return math.hypot((ax - bx) * aspect, ay - by)

    fw = d(lm[234].x, lm[234].y, lm[454].x, lm[454].y)       # face width
    if fw <= 1e-6:
        return False, False
    mx, my = (lm[13].x + lm[14].x) / 2, (lm[13].y + lm[14].y) / 2
    ears = [(lm[234].x, lm[234].y), (lm[454].x, lm[454].y)]
    at_mouth = at_ear = False
    for pts in hands:
        dm = min(d(x, y, mx, my) for x, y in pts) / fw
        de = min(d(x, y, ex, ey) for x, y in pts for ex, ey in ears) / fw
        if dm < HAND_AT_MOUTH:
            at_mouth = True
        elif de < HAND_AT_EAR:
            at_ear = True
    return at_mouth, at_ear


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------- calibration
class Calibration:
    DURATION = 5.0

    def __init__(self):
        self.active = False
        self.done = False
        self.failed = False
        self.message = ""
        self.t0 = 0.0
        self.ears, self.yaws = [], []
        self.ear_open = None
        self.ear_thresh = 0.18      # default until calibrated
        self.yaw_center = 0.5

    def start(self, now):
        self.active, self.done, self.failed, self.message = True, False, False, ""
        self.t0 = now
        self.ears, self.yaws = [], []

    def progress(self, now):
        return min(1.0, (now - self.t0) / self.DURATION) if self.active else 0.0

    def feed(self, ear, yaw, now):
        if self.active:
            self.ears.append(ear)
            self.yaws.append(yaw)
            if now - self.t0 >= self.DURATION:
                self._finish()

    def check_timeout(self, now):
        if self.active and now - self.t0 >= self.DURATION:
            self._finish()

    def _finish(self):
        self.active = False
        n = len(self.ears)
        if n < 15:
            self.failed, self.message = True, "Face not seen long enough. Try again facing the camera."
            return
        ears = sorted(self.ears)
        ear_open = ears[int(0.7 * (n - 1))]     # ignores blinks
        if ear_open < 0.12:
            self.failed, self.message = True, "Eyes looked closed. Keep them open and try again."
            return
        self.ear_open = ear_open
        self.ear_thresh = min(0.25, max(0.10, 0.65 * ear_open))
        yaws = sorted(self.yaws)
        self.yaw_center = yaws[len(yaws) // 2]
        self.done = True
        self.message = (f"Calibrated. Open-eye level {ear_open:.2f}, "
                        f"closed-eye limit {self.ear_thresh:.2f}.")


# --------------------------------------------------------------------------- PERCLOS
class PerclosTracker:
    """PERCLOS = share of time the eyes are closed during the last minute."""

    def __init__(self, window=60.0, min_span=15.0):
        self.window, self.min_span = window, min_span
        self.q = deque()

    def add(self, now, closed):
        self.q.append((now, bool(closed)))
        while self.q and now - self.q[0][0] > self.window:
            self.q.popleft()

    def value(self):
        if len(self.q) < 5 or self.q[-1][0] - self.q[0][0] < self.min_span:
            return 0.0
        return sum(1 for _, c in self.q if c) / len(self.q)


# --------------------------------------------------------------------------- trip log
class TripLog:
    def __init__(self):
        self.start = None          # set at the first reading
        self.samples = []          # one per second
        self.events = []           # finished alert periods
        self.cur = None            # alert period in progress
        self._last_bad = 0.0
        self._last_sample = 0.0
        self.blinks = 0
        self.yawns = 0

    def tick(self, now, fatigue, perclos, status, reason, blinks, yawns, attention=100.0):
        if self.start is None:
            self.start = now
        self.blinks, self.yawns = blinks, yawns
        # alert periods (gaps shorter than 2 s are merged)
        if status != "Normal":
            self._last_bad = now
            if self.cur is None:
                self.cur = {"start": now - self.start, "end": now - self.start,
                            "level": status, "reason": reason}
            else:
                self.cur["end"] = now - self.start
                if status == "Danger" and self.cur["level"] != "Danger":
                    self.cur["level"], self.cur["reason"] = "Danger", reason
        elif self.cur is not None and now - self._last_bad > 2.0:
            self.events.append(self.cur)
            self.cur = None
        # one sample per second for the charts
        if now - self._last_sample >= 1.0:
            self._last_sample = now
            self.samples.append({"t": round(now - self.start, 1), "fatigue": round(fatigue, 3),
                                 "perclos": round(perclos, 3), "attention": round(attention, 1),
                                 "status": status, "reason": reason})

    def all_events(self):
        return list(self.events) + ([dict(self.cur)] if self.cur else [])

    def summary(self):
        dur = self.samples[-1]["t"] if self.samples else 0.0
        events = self.all_events()
        unatt = sum(max(1.0, e["end"] - e["start"]) for e in events)
        unatt = min(unatt, dur) if dur else 0.0
        pct = (unatt / dur * 100) if dur else 0.0
        warn = sum(1 for e in events if e["level"] == "Warning")
        danger = sum(1 for e in events if e["level"] == "Danger")
        reasons = Counter(s["reason"] for s in self.samples if s["status"] != "Normal" and s["reason"])
        fat = [s["fatigue"] for s in self.samples] or [0.0]
        per = [s["perclos"] for s in self.samples] or [0.0]

        penalty = pct * 0.6 + min(40.0, danger * 5.0 + warn * 1.5)
        score = int(round(max(0.0, min(100.0, 100.0 - penalty))))
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
        return {
            "duration": dur, "unattentive_s": unatt, "unattentive_pct": pct,
            "warnings": warn, "dangers": danger, "events": events,
            "reasons": dict(reasons), "avg_fatigue": sum(fat) / len(fat),
            "max_fatigue": max(fat), "max_perclos": max(per),
            "blinks": self.blinks, "yawns": self.yawns,
            "away_s": reasons.get("Looking away", 0),
            "phone_s": reasons.get("Using phone", 0) + reasons.get("Phone call", 0),
            "eating_s": reasons.get("Eating or drinking", 0), "score": score, "grade": grade,
        }


# --------------------------------------------------------------------------- the monitor
class DriverMonitor:
    """Feed it face landmarks (or None) many times per second; it keeps all the state."""

    def __init__(self):
        self.calib = Calibration()
        self.perclos = PerclosTracker()
        self.trip = TripLog()
        self.status, self.reason, self.fatigue, self.gaze = "Normal", "", 0.0, "Center"
        self.perclos_pct = 0.0
        self.attention = 100.0
        self.reset_counters()

    def reset_counters(self):
        self.blinks = self.yawns = 0
        self.closed = self.mouth = self.head_down = self.away = 0.0
        self.was_closed = self.was_yawn = False
        self.bad_since = self.ok_since = None
        self.bad_seconds = 0.0
        self.blink_times = deque()
        self.hand_mouth = self.hand_ear = self.hand_phone = 0.0

    def reset(self):
        self.reset_counters()
        self.perclos = PerclosTracker()
        self.trip = TripLog()

    def update(self, lm, now, step, hands=None, aspect=4 / 3):
        c = self.calib
        c.check_timeout(now)
        cands = []                 # (rank, reason)
        fatigue, gaze = 0.0, "Center"

        if lm is not None:
            ear, yaw = compute_ear(lm), compute_yaw_ratio(lm)
            c.feed(ear, yaw, now)
            (status, fatigue, gaze, self.closed, self.mouth, self.head_down,
             eye_closed, yawning) = analyze_landmarks(
                lm, self.closed, self.mouth, self.head_down, step, c.ear_thresh)

            if self.was_closed and not eye_closed:
                self.blinks += 1
                self.blink_times.append(now)
            self.was_closed = eye_closed
            if self.was_yawn and not yawning:
                self.yawns += 1
            self.was_yawn = yawning
            self.perclos.add(now, eye_closed)

            # hands: phone call (hand at ear), eating/drinking (hand at mouth), texting (head down + hand visible)
            at_mouth, at_ear = hand_flags(lm, hands, aspect)
            texting = bool(hands) and self.head_down > 0
            self.hand_mouth = self.hand_mouth + step if at_mouth else 0.0
            self.hand_ear = self.hand_ear + step if at_ear else 0.0
            self.hand_phone = self.hand_phone + step if texting else 0.0

            if status != "Normal":
                if self.closed > 10:
                    r = "Eyes closed"
                elif self.mouth > 4:
                    r = "Yawning"
                elif self.head_down > 4:
                    r = "Head down"
                else:
                    r = "Drowsy"
                cands.append((RANK[status], r))

            if self.hand_phone > PHONE_WARN:           # "Head down" is really phone use here
                cands = [c for c in cands if c[1] != "Head down"]
            if self.hand_phone > PHONE_DANGER:
                cands.append((2, "Using phone"))
            elif self.hand_phone > PHONE_WARN:
                cands.append((1, "Using phone"))
            if self.hand_ear > CALL_DANGER:
                cands.append((2, "Phone call"))
            elif self.hand_ear > CALL_WARN:
                cands.append((1, "Phone call"))
            if self.hand_mouth > EAT_DANGER:
                cands.append((2, "Eating or drinking"))
            elif self.hand_mouth > EAT_WARN:
                cands.append((1, "Eating or drinking"))

            # distraction: head turned away from straight ahead
            if abs(yaw - c.yaw_center) > YAW_LIMIT and self.head_down == 0:
                self.away += step
            else:
                self.away = 0.0
        else:
            # face not visible (turned far away, or dropped out of view)
            self.head_down += step
            self.away = 0.0
            self.hand_mouth = self.hand_ear = self.hand_phone = 0.0
            gaze = "Not visible"
            if self.head_down > 90:
                cands.append((2, "Face not visible"))
                fatigue = min(1.0, 0.4 + self.head_down * 0.01)
            elif self.head_down > 45:
                cands.append((1, "Face not visible"))

        if self.away > AWAY_DANGER:
            cands.append((2, "Looking away"))
        elif self.away > AWAY_WARN:
            cands.append((1, "Looking away"))

        p = self.perclos.value()
        self.perclos_pct = p * 100
        if p >= PERCLOS_DANGER:
            cands.append((2, "Drowsy (PERCLOS)"))
        elif p >= PERCLOS_WARN:
            cands.append((1, "Drowsy (PERCLOS)"))
        fatigue = max(fatigue, min(1.0, p / 0.4))

        if c.active:               # no alerts while calibrating
            status, reason = "Normal", "Calibrating"
        elif cands:
            rank, reason = max(cands, key=lambda x: x[0])
            status = LEVELS[rank]
        else:
            status, reason = "Normal", ""

        self.status, self.reason, self.fatigue, self.gaze = status, reason, fatigue, gaze

        # how long has the driver been CONTINUOUSLY inattentive? (up to 3 s of normal is forgiven)
        if status != "Normal":
            self.ok_since = None
            if self.bad_since is None:
                self.bad_since = now
        else:
            if self.ok_since is None:
                self.ok_since = now
            if self.bad_since is not None and now - self.ok_since > 3.0:
                self.bad_since = None
        self.bad_seconds = (now - self.bad_since) if self.bad_since else 0.0

        # live attention score (0-100): smooth, so the gauge does not jump around
        perclos_risk = max(0.0, (p - 0.08) / (PERCLOS_DANGER - 0.08))   # normal blinking (<8 %) costs nothing
        risk = max(min(fatigue, 1.0), min(1.0, perclos_risk), min(1.0, self.away / AWAY_DANGER))
        if status == "Danger":
            risk = max(risk, 0.85)
        elif status == "Warning":
            risk = max(risk, 0.5)
        self.attention += 0.25 * (100.0 * (1.0 - risk) - self.attention)

        self.trip.tick(now, fatigue, p, status, reason, self.blinks, self.yawns, self.attention)

    def blink_rate(self, now):
        """Blinks per minute over the last minute (None until enough data)."""
        while self.blink_times and now - self.blink_times[0] > 60:
            self.blink_times.popleft()
        if self.trip.start is None or now - self.trip.start < 20:
            return None
        return len(self.blink_times) * 60.0 / min(now - self.trip.start, 60.0)

    def coach(self, now):
        """Smart co-pilot advice -> (level, text); level is 'ok', 'tip' or 'break'."""
        if self.trip.start is None:
            return "ok", "Co-pilot online. Drive safe."
        elapsed = now - self.trip.start
        recent = [e for e in self.trip.all_events() if elapsed - e["end"] <= COACH_WINDOW]
        if len(recent) >= COACH_EVENTS:
            return "break", (f"{len(recent)} alerts in the last {COACH_WINDOW // 60} minutes. "
                             "Please take a break at the next safe place.")
        if elapsed >= COACH_DRIVE_SECONDS:
            return "break", "You have been driving for over two hours. Please take a rest break."
        if self.perclos_pct >= 12:
            return "tip", "Your eyes are closing more often. Open a window or sip some water."
        return "ok", f"All good. Driving time {fmt_time(elapsed)}."


# --------------------------------------------------------------------------- reports
def samples_csv(trip):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seconds", "fatigue", "perclos", "attention", "status", "reason"])
    for s in trip.samples:
        w.writerow([s["t"], s["fatigue"], s["perclos"], s.get("attention", ""), s["status"], s["reason"]])
    return buf.getvalue()


def _svg_chart(trip, width=640, height=220):
    samples = trip.samples
    if len(samples) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 36, 10, 10, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    tmax = max(samples[-1]["t"], 1.0)

    def x(t):
        return pad_l + t / tmax * w

    def y(v):
        return pad_t + (1 - max(0.0, min(1.0, v))) * h

    step = max(1, len(samples) // 300)
    pts = samples[::step]
    fat = " ".join(f"{x(s['t']):.1f},{y(s['fatigue']):.1f}" for s in pts)
    per = " ".join(f"{x(s['t']):.1f},{y(s['perclos']):.1f}" for s in pts)

    bands = ""
    for e in trip.all_events():
        color = "#ef4444" if e["level"] == "Danger" else "#f59e0b"
        x1, x2 = x(e["start"]), max(x(e["end"]), x(e["start"]) + 2)
        bands += f'<rect x="{x1:.1f}" y="{pad_t}" width="{x2 - x1:.1f}" height="{h}" fill="{color}" opacity="0.18"/>'

    grid = ""
    for v in (0, 0.5, 1.0):
        grid += (f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#d1d5db" stroke-width="1"/>'
                 f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="#6b7280">{v:.1f}</text>')
    xl = "".join(
        f'<text x="{x(t):.1f}" y="{height - 8}" font-size="10" text-anchor="middle" fill="#6b7280">{fmt_time(t)}</text>'
        for t in (0, tmax / 2, tmax))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">'
            f'{grid}{bands}'
            f'<polyline points="{fat}" fill="none" stroke="#2563eb" stroke-width="2"/>'
            f'<polyline points="{per}" fill="none" stroke="#9333ea" stroke-width="2" stroke-dasharray="5 3"/>'
            f'{xl}</svg>')


def build_report_html(trip, driver_name="Driver"):
    s = trip.summary()
    esc = _html.escape
    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(trip.start or time.time()))
    rows = "".join(
        f"<tr><td>{fmt_time(e['start'])}</td><td>{max(1, int(e['end'] - e['start']))} s</td>"
        f"<td class='{e['level'].lower()}'>{e['level']}</td><td>{esc(e['reason'] or '-')}</td></tr>"
        for e in s["events"]) or "<tr><td colspan='4'>No alerts during this trip.</td></tr>"
    reasons = "".join(f"<li>{esc(k)}: {v} s</li>" for k, v in
                      sorted(s["reasons"].items(), key=lambda kv: -kv[1])) or "<li>None</li>"
    cards = [
        ("Safety score", f"{s['score']}/100 ({s['grade']})"),
        ("Trip time", fmt_time(s["duration"])),
        ("Unattentive", f"{s['unattentive_pct']:.0f}%"),
        ("Warnings / Dangers", f"{s['warnings']} / {s['dangers']}"),
        ("Max fatigue", f"{s['max_fatigue']:.2f}"),
        ("Max PERCLOS", f"{s['max_perclos'] * 100:.0f}%"),
        ("Blinks / Yawns", f"{s['blinks']} / {s['yawns']}"),
        ("Looking away", f"{int(s['away_s'])} s"),
        ("Phone use", f"{int(s['phone_s'])} s"),
        ("Eating / drinking", f"{int(s['eating_s'])} s"),
    ]
    cards_html = "".join(f"<div class='card'><div class='l'>{k}</div><div class='v'>{v}</div></div>"
                         for k, v in cards)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver trip report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0;padding:16px;background:#fff;color:#111827;max-width:760px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 8px}} .sub{{color:#6b7280;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:14px}}
.card{{border:1px solid #e5e7eb;border-radius:10px;padding:10px}} .l{{font-size:12px;color:#6b7280}} .v{{font-size:20px;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td,th{{border-bottom:1px solid #e5e7eb;padding:6px;text-align:left}}
.danger{{color:#dc2626;font-weight:700}} .warning{{color:#d97706;font-weight:700}}
.legend span{{margin-right:14px;font-size:12px}} .note{{font-size:12px;color:#6b7280;margin-top:20px}}
</style></head><body>
<h1>AI Driver Monitoring - Trip Report</h1>
<div class="sub">{esc(driver_name)} &middot; started {started}</div>
<div class="grid">{cards_html}</div>
<h2>Fatigue and PERCLOS over time</h2>
<div class="legend"><span style="color:#2563eb">&#9644; Fatigue index</span>
<span style="color:#9333ea">&#9644; PERCLOS</span><span style="color:#d97706">&#9608; Alert period</span></div>
{_svg_chart(trip)}
<h2>Alerts</h2>
<table><tr><th>At</th><th>Length</th><th>Level</th><th>Reason</th></tr>{rows}</table>
<h2>Time by reason</h2><ul>{reasons}</ul>
<p class="note">Safety score = 100 minus penalties for the share of the trip spent unattentive and for each
warning/danger alert. PERCLOS is the percentage of the last minute with the eyes closed. This report is
produced by a student project and is not a certified safety device.</p>
</body></html>"""
