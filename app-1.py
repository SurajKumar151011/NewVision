# app.py - AI Driver Monitoring System (main file)
import json
import math
import time

import cv2
import requests

import av
import streamlit.components.v1 as components
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from advanced import DriverMonitor, build_report_html, fmt_time, samples_csv
from detector import extract_hands, initialize_face_mesh, initialize_hands, process_frame
from driver_utils import draw_overlay

st.set_page_config(
    page_title="AI Driver Monitoring System",
    page_icon="🚘",
    layout="centered",
)

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 50% 50%, #0d1117 0%, #030712 100%); color: #f3f4f6; }
.main-title { font-size: 1.6rem; font-weight: 800; text-align: center; margin-bottom: 14px;
  background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div[data-testid="stMetric"] { background: rgba(17,24,39,0.75); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 10px 14px; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800 !important; color: #38bdf8 !important; }
div[data-testid="stMetricLabel"] { color: #9ca3af !important; text-transform: uppercase; }
.status-card { padding: 14px; border-radius: 14px; text-align: center; font-size: 1.1rem;
  font-weight: 800; margin-bottom: 12px; }
.status-normal { background: rgba(6,78,59,0.6); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.status-warning { background: rgba(120,53,15,0.7); color: #fbbf24; border: 1px solid rgba(251,191,36,0.5); }
.status-danger { background: rgba(127,29,29,0.8); color: #f87171; border: 1px solid rgba(248,113,113,0.6); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.status-card { box-shadow: 0 0 22px rgba(56,189,248,0.10); letter-spacing: 1px; }
.status-danger { box-shadow: 0 0 28px rgba(248,113,113,0.45); }
.status-warning { box-shadow: 0 0 24px rgba(251,191,36,0.35); }
.hud { background: rgba(10,15,28,0.85); border: 1px solid rgba(56,189,248,0.25); border-radius: 18px;
  padding: 12px; margin: 8px 0; box-shadow: 0 0 22px rgba(56,189,248,0.10) inset, 0 0 16px rgba(56,189,248,0.12); }
.hud-row { display: flex; gap: 14px; align-items: center; justify-content: center; flex-wrap: wrap; }
.hud-bars { flex: 1; min-width: 150px; }
.lbl { font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; margin: 6px 0 3px; letter-spacing: 1px; }
.bar { height: 8px; border-radius: 6px; background: #1f2937; overflow: hidden; }
.bar > div { height: 100%; border-radius: 6px; }
.coach { padding: 10px 14px; border-radius: 14px; font-size: 0.95rem; margin: 6px 0; }
.coach-ok { background: rgba(8,47,73,0.6); border: 1px solid rgba(56,189,248,0.35); color: #7dd3fc; }
.coach-tip { background: rgba(120,53,15,0.55); border: 1px solid rgba(251,191,36,0.45); color: #fcd34d; }
.coach-break { background: rgba(127,29,29,0.7); border: 1px solid rgba(248,113,113,0.6); color: #fca5a5; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚘 AI DRIVER MONITORING SYSTEM</div>', unsafe_allow_html=True)


try:                                   # optional: phone location for owner alerts
    from streamlit_js_eval import get_geolocation
    HAS_GEO = True
except Exception:
    HAS_GEO = False

def _tg_config():
    """Read Telegram settings from Streamlit Secrets (never store them in GitHub)."""
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except Exception:
        return None, None


def send_telegram(text, jpg=None):
    """Send a message (and optional photo) to the owner. Returns (ok, info)."""
    token, chat_id = _tg_config()
    if not token or not chat_id:
        return False, "Telegram is not set up yet (add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Secrets)."
    base = f"https://api.telegram.org/bot{token}"
    try:
        if jpg is not None:
            r = requests.post(f"{base}/sendPhoto", data={"chat_id": chat_id, "caption": text},
                              files={"photo": ("driver.jpg", jpg, "image/jpeg")}, timeout=10)
        else:
            r = requests.post(f"{base}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=10)
        data = r.json()
        if data.get("ok"):
            return True, "Sent"
        return False, data.get("description", "Telegram refused the message")
    except Exception as e:
        return False, f"Could not reach Telegram ({type(e).__name__})"


def send_telegram_location(lat, lon):
    """Send a map pin the owner can tap to open in Maps."""
    token, chat_id = _tg_config()
    if not token or not chat_id:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendLocation",
                      data={"chat_id": chat_id, "latitude": lat, "longitude": lon}, timeout=10)
    except Exception:
        pass


class DriverProcessor(VideoProcessorBase):
    """Runs on camera frames from the browser. Analysis is limited to ~10 times
    per second; all the smart logic lives in advanced.DriverMonitor."""

    def __init__(self):
        self.face_mesh = initialize_face_mesh()
        self.hands = initialize_hands()
        self._hands_pts = []
        self._tick = 0
        self.monitor = DriverMonitor()
        self.last_img = None
        self._last_proc = 0.0
        self._prev = time.time()
        self.fps = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        inst = 1 / (now - self._prev) if now > self._prev else 0
        self.fps = 0.8 * self.fps + 0.2 * inst
        self._prev = now

        if now - self._last_proc >= 0.1:
            dt = (now - self._last_proc) if self._last_proc else 0.1
            self._last_proc = now
            step = min(dt * 30.0, 6.0)          # "frames at 30 fps" that passed
            img, results = process_frame(img, self.face_mesh)
            lm = None
            if results and results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
            self._tick += 1
            if self._tick % 3 == 0:                 # hands are checked ~3 times per second
                self._hands_pts = extract_hands(img, self.hands)
            h, w = img.shape[:2]
            self.monitor.update(lm, now, step, self._hands_pts, w / h)

        m = self.monitor
        img = draw_overlay(img, m.status, self.fps)
        if m.calib.active:
            cv2.putText(img, "CALIBRATING - look straight, eyes open", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.last_img = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")


VOICE = {
    ("Danger", "Eyes closed"): "Danger! Your eyes are closed. Wake up!",
    ("Danger", "Yawning"): "Danger! You are very sleepy. Please take a break.",
    ("Danger", "Head down"): "Danger! Keep your head up and look at the road!",
    ("Danger", "Face not visible"): "Danger! I cannot see your face. Look at the road!",
    ("Warning", "Eyes closed"): "Warning. Keep your eyes open.",
    ("Warning", "Yawning"): "Warning. You seem tired.",
    ("Warning", "Head down"): "Warning. Please look up.",
    ("Warning", "Looking away"): "Warning. Eyes on the road.",
    ("Danger", "Looking away"): "Danger! Look at the road!",
    ("Warning", "Using phone"): "Warning. Please put your phone down.",
    ("Danger", "Using phone"): "Danger! Stop using your phone now!",
    ("Warning", "Phone call"): "Warning. Please end the call or use hands free.",
    ("Danger", "Phone call"): "Danger! Put the phone down and focus on driving!",
    ("Warning", "Eating or drinking"): "Warning. Please keep your hands on the wheel.",
    ("Danger", "Eating or drinking"): "Danger! Stop eating and keep both hands on the wheel!",
}

ALERT_JS = """<script>/*__NONCE__*/
// 1) beep
if (__BEEP__) { try {
  const C = window.AudioContext || window.webkitAudioContext;
  const ctx = new C(); ctx.resume();
  for (let i = 0; i < __BEEPS__; i++) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'square'; o.frequency.value = __FREQ__; g.gain.value = 0.5;
    o.connect(g); g.connect(ctx.destination);
    const t = ctx.currentTime + i * 0.35; o.start(t); o.stop(t + 0.25);
  }
} catch (e) {} }
// 2) full-screen flash
if (__FLASH__) { try {
  const d = window.parent.document;
  if (!d.getElementById('dms-style')) {
    const st = d.createElement('style'); st.id = 'dms-style';
    st.textContent = '@keyframes dmsflash{0%{opacity:0}12%{opacity:.8}30%{opacity:.15}50%{opacity:.75}100%{opacity:0}}';
    d.head.appendChild(st);
  }
  const old = d.getElementById('dms-flash'); if (old) old.remove();
  const el = d.createElement('div'); el.id = 'dms-flash';
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;opacity:0;background:__COLOR__;animation:dmsflash 1.3s ease-out forwards;';
  d.body.appendChild(el);
} catch (e) {} }
// 3) voice warning (uses the main page's speech engine so it survives iframe refresh)
if (__VOICE__) { try {
  const W = window.parent, S = W.speechSynthesis, now = Date.now();
  // speak if idle, or if the engine looks stuck (more than 5 s since last start)
  if (S && (!S.speaking || now - (W.__dmsVoiceAt || 0) > 5000)) {
    S.cancel(); S.resume();
    const u = new W.SpeechSynthesisUtterance(__TEXT__);
    u.rate = 1.0; u.volume = 1; u.lang = 'en-US';
    W.__dmsU = u; W.__dmsVoiceAt = now;      // keep a reference so it is not garbage-collected
    W.setTimeout(function () { S.speak(u); }, 120);
  }
} catch (e) {} }
// 4) vibration
if (__VIBRATE__) { try { navigator.vibrate(__VIB__); } catch (e) {} }
</script>"""


def fire_alert(box, status, reason, beep=True, voice=True, flash=True, vibrate=True):
    """Phone alert: beep + spoken warning + full-screen flash + vibration."""
    danger = status == "Danger"
    text = VOICE.get((status, reason),
                     "Danger! Stay alert!" if danger else "Warning. You look drowsy.")
    js = (ALERT_JS
          .replace("__NONCE__", str(time.time_ns()))
          .replace("__BEEP__", "true" if beep else "false")
          .replace("__BEEPS__", "2" if danger else "1")
          .replace("__FREQ__", "2200" if danger else "1200")
          .replace("__FLASH__", "true" if flash else "false")
          .replace("__COLOR__", "rgba(255,0,0,1)" if danger else "rgba(255,170,0,1)")
          .replace("__VOICE__", "true" if voice else "false")
          .replace("__TEXT__", json.dumps(text))
          .replace("__VIBRATE__", "true" if vibrate else "false")
          .replace("__VIB__", "[300,100,300,100,300]" if danger else "[250]"))
    with box:
        components.html(js, height=0)


def _bar(label, value_text, pct, color):
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="lbl"><span>{label}</span><span>{value_text}</span></div>'
            f'<div class="bar"><div style="width:{pct:.0f}%;background:{color};box-shadow:0 0 8px {color}"></div></div>')


def hud_html(attention, fatigue, perclos_pct, blink_rate, history):
    """Futuristic HUD: glowing attention ring, mini bars and a live history line."""
    color = "#22c55e" if attention >= 70 else "#f59e0b" if attention >= 40 else "#ef4444"
    r = 52
    circ = 2 * math.pi * r
    ring = (f'<svg viewBox="0 0 140 140" width="150" height="150">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="#1f2937" stroke-width="12"/>'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{circ * attention / 100:.1f} {circ:.1f}" transform="rotate(-90 70 70)" '
            f'style="filter:drop-shadow(0 0 6px {color})"/>'
            f'<text x="70" y="75" text-anchor="middle" font-size="32" font-weight="800" fill="#f9fafb">{attention:.0f}</text>'
            f'<text x="70" y="94" text-anchor="middle" font-size="9" fill="#9ca3af" letter-spacing="2">ATTENTION</text></svg>')

    fat_col = "#22c55e" if fatigue < 0.25 else "#f59e0b" if fatigue < 0.5 else "#ef4444"
    per_col = "#22c55e" if perclos_pct < 12 else "#f59e0b" if perclos_pct < 20 else "#ef4444"
    bars = (_bar("FATIGUE", f"{fatigue * 100:.0f}%", fatigue * 100, fat_col)
            + _bar("PERCLOS", f"{perclos_pct:.0f}%", perclos_pct / 0.35, per_col)
            + _bar("BLINKS / MIN", "--" if blink_rate is None else f"{blink_rate:.0f}",
                   0 if blink_rate is None else min(100, blink_rate / 30 * 100), "#38bdf8"))

    spark = ""
    if len(history) >= 2:
        w, h = 300, 56
        n = len(history)
        pts = " ".join(f"{i / (n - 1) * w:.1f},{h - max(0, min(100, v)) / 100 * h:.1f}" for i, v in enumerate(history))
        spark = (f'<svg viewBox="0 0 {w} {h}" width="100%" height="56" preserveAspectRatio="none" style="margin-top:8px">'
                 f'<line x1="0" x2="{w}" y1="{h * 0.3:.0f}" y2="{h * 0.3:.0f}" stroke="#374151" stroke-dasharray="3 3"/>'
                 f'<line x1="0" x2="{w}" y1="{h * 0.6:.0f}" y2="{h * 0.6:.0f}" stroke="#374151" stroke-dasharray="3 3"/>'
                 f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                 f'style="filter:drop-shadow(0 0 4px {color})"/></svg>'
                 f'<div class="lbl"><span>LAST 2 MINUTES</span><span>LIVE</span></div>')
    return f'<div class="hud"><div class="hud-row">{ring}<div class="hud-bars">{bars}</div></div>{spark}</div>'


SPEAK_JS = """<script>/*__NONCE__*/
try { const W = window.parent, S = W.speechSynthesis;
  if (S && !S.speaking) { S.cancel(); S.resume();
    const u = new W.SpeechSynthesisUtterance(__TEXT__); u.lang = 'en-US'; W.__dmsCoach = u;
    W.setTimeout(function () { S.speak(u); }, 120); } } catch (e) {}
</script>"""


def speak_text(box, text):
    with box:
        components.html(SPEAK_JS.replace("__NONCE__", str(time.time_ns())).replace("__TEXT__", json.dumps(text)), height=0)


def render_report(driver_name):
    """Trip report shown after the camera is stopped."""
    import pandas as pd
    trip = st.session_state.get("trip")
    if trip is None or len(trip.samples) < 3:
        st.info("📊 Your trip report will appear here after you stop the camera.")
        return
    s = trip.summary()
    st.subheader("📊 Trip report")
    a, b = st.columns(2)
    a.metric("Safety score", f"{s['score']}/100  ({s['grade']})")
    b.metric("Trip time", fmt_time(s["duration"]))
    c, d = st.columns(2)
    c.metric("Unattentive", f"{s['unattentive_pct']:.0f}%")
    d.metric("Warnings / Dangers", f"{s['warnings']} / {s['dangers']}")
    e, f = st.columns(2)
    e.metric("Max fatigue", f"{s['max_fatigue']:.2f}")
    f.metric("Max PERCLOS", f"{s['max_perclos'] * 100:.0f}%")
    g, h = st.columns(2)
    g.metric("Phone use", f"{int(s['phone_s'])} s")
    h.metric("Eating / drinking", f"{int(s['eating_s'])} s")

    df = pd.DataFrame(trip.samples)
    df["Minutes"] = df["t"] / 60.0
    st.line_chart(df.rename(columns={"fatigue": "Fatigue", "perclos": "PERCLOS"})
                    .set_index("Minutes")[["Fatigue", "PERCLOS"]])

    if s["events"]:
        st.caption("Alerts during the trip")
        st.dataframe(pd.DataFrame([{"At": fmt_time(ev["start"]),
                                    "Length (s)": max(1, int(ev["end"] - ev["start"])),
                                    "Level": ev["level"], "Reason": ev["reason"]}
                                   for ev in s["events"]]), hide_index=True)
    else:
        st.success("No alerts during this trip. Great driving!")

    st.download_button("⬇️ Download report (HTML)", build_report_html(trip, driver_name),
                       file_name="trip_report.html", mime="text/html")
    st.download_button("⬇️ Download data (CSV)", samples_csv(trip),
                       file_name="trip_data.csv", mime="text/csv")


st.caption("Tap START, allow camera access, and place your phone so your face is visible.")

status_box = st.empty()
notify_box = st.empty()
calib_box = st.empty()

ctx = webrtc_streamer(
    key="driver-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DriverProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user", "width": {"ideal": 320}, "height": {"ideal": 240},
                  "frameRate": {"ideal": 15, "max": 20}},
        "audio": False,
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

coach_box = st.empty()
hud_box = st.empty()
coach_sound_box = st.empty()

c1, c2 = st.columns(2)
fatigue_box, gaze_box = c1.empty(), c2.empty()
c3, c4 = st.columns(2)
blink_box, yawn_box = c3.empty(), c4.empty()
c5, c6 = st.columns(2)
perclos_box, alerts_box = c5.empty(), c6.empty()
loc_box = st.empty()
sound_box = st.empty()
test_box = st.empty()
geo_box = st.empty()

with st.expander("⚙️ Alert settings", expanded=False):
    opt_beep = st.toggle("🔔 Beep", value=True)
    opt_voice = st.toggle("🗣️ Voice warning", value=True)
    opt_flash = st.toggle("🔴 Full-screen flash", value=True)
    opt_vibrate = st.toggle("📳 Vibration", value=True)

with st.expander("📨 Owner notification (Telegram)", expanded=False):
    notify_on = st.toggle("Message the owner if driver stays inattentive", value=True)
    notify_after = st.number_input("Send after (seconds)", min_value=10, max_value=600, value=60, step=5)
    driver_name = st.text_input("Driver / vehicle name", value="My vehicle")
    if HAS_GEO:
        loc_on = st.toggle("📍 Include location in alerts (allow location when asked)", value=False)
    else:
        loc_on = False
        st.caption("📍 Location needs the streamlit-js-eval package (see requirements.txt).")
    if st.button("📨 Send test message"):
        ok, info = send_telegram(f"✅ Test message from AI Driver Monitor ({driver_name}). Notifications are working.")
        st.success("Test message sent!") if ok else st.error(info)

b1, b2 = st.columns(2)
if b1.button("🔄 New trip / reset") and ctx.video_processor:
    ctx.video_processor.monitor.reset()
if b2.button("🎯 Calibrate (5 s)"):
    if ctx.video_processor:
        ctx.video_processor.monitor.calib.start(time.time())
    else:
        st.warning("Tap START on the camera first.")
if st.button("🔊 Test alert"):
    # tapping a button also lets the browser allow sound and voice
    fire_alert(test_box, "Danger", "Eyes closed", opt_beep, opt_voice, opt_flash, opt_vibrate)

# ---- phone location (refreshed about every 30 seconds)
geo_window = int(time.time() // 30)
if loc_on and HAS_GEO:
    with geo_box:
        try:
            g = get_geolocation(component_key=f"geo_{geo_window}")
        except Exception:
            g = None
    if isinstance(g, dict) and isinstance(g.get("coords"), dict):
        co = g["coords"]
        if co.get("latitude") is not None and co.get("longitude") is not None:
            st.session_state["loc"] = (co["latitude"], co["longitude"], time.time())


def show_status(status: str, reason: str = ""):
    why = f": {reason.upper()}" if reason and status != "Normal" else ""
    if status == "Danger":
        status_box.markdown(f'<div class="status-card status-danger">🚨 DANGER{why}</div>', unsafe_allow_html=True)
    elif status == "Warning":
        status_box.markdown(f'<div class="status-card status-warning">⚠️ WARNING{why}</div>', unsafe_allow_html=True)
    else:
        status_box.markdown('<div class="status-card status-normal">✅ STATUS: ATTENTIVE</div>', unsafe_allow_html=True)


if ctx.state.playing:
    while ctx.state.playing:
        p = ctx.video_processor
        if p is not None:
            m = p.monitor
            now = time.time()
            st.session_state["trip"] = m.trip

            show_status(m.status, m.reason)
            fatigue_box.metric("Fatigue Index", f"{m.fatigue:.2f}")
            gaze_box.metric("Gaze", m.gaze)
            blink_box.metric("Blinks", str(m.blinks))
            yawn_box.metric("Yawns", str(m.yawns))
            perclos_box.metric("PERCLOS (1 min)", f"{m.perclos_pct:.0f}%")
            s_live = m.trip
            alerts_box.metric("Alerts", str(len(s_live.events) + (1 if s_live.cur else 0)))

            # ---- futuristic HUD + co-pilot
            hist = [x.get("attention", 100) for x in m.trip.samples[-120:]]
            hud_box.markdown(hud_html(m.attention, m.fatigue, m.perclos_pct, m.blink_rate(now), hist),
                             unsafe_allow_html=True)
            lvl, advice = m.coach(now)
            coach_box.markdown(f'<div class="coach coach-{lvl}">🧠 <b>Co-pilot:</b> {advice}</div>',
                               unsafe_allow_html=True)
            if lvl == "break" and opt_voice and now - st.session_state.get("last_coach", 0) > 300:
                speak_text(coach_sound_box, advice)
                st.session_state["last_coach"] = now

            # ---- calibration status
            cal = m.calib
            if cal.active:
                calib_box.progress(cal.progress(now), text="🎯 Calibrating: look straight at the camera, eyes open")
            elif cal.failed:
                calib_box.warning(cal.message)
            elif cal.done:
                calib_box.caption("🎯 " + cal.message)

            # ---- location status
            loc = st.session_state.get("loc")
            if loc_on and loc:
                loc_box.caption(f"📍 Location updated {int(now - loc[2])} s ago")

            # ---- owner notification after continuous inattention
            if notify_on:
                if m.bad_seconds > 0:
                    notify_box.progress(min(m.bad_seconds / notify_after, 1.0),
                                        text=f"Inattentive for {int(m.bad_seconds)}s / {int(notify_after)}s")
                else:
                    notify_box.empty()
                if m.bad_seconds >= notify_after and not st.session_state.get("notified") \
                        and now - st.session_state.get("last_sent", 0) > 120:
                    jpg = None
                    if p.last_img is not None:
                        okj, buf = cv2.imencode(".jpg", p.last_img)
                        jpg = buf.tobytes() if okj else None
                    where = ""
                    if loc_on and loc:
                        where = f"Location: https://maps.google.com/?q={loc[0]},{loc[1]} (updated {int(now - loc[2])} s ago)\n"
                    msg = (f"🚨 DRIVER ALERT - {driver_name}\n"
                           f"The driver has been inattentive for {int(m.bad_seconds)} seconds.\n"
                           f"Reason: {m.reason or 'Drowsy'}\n"
                           f"{where}"
                           f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n"
                           f"Please call the driver immediately.")
                    ok, info = send_telegram(msg, jpg)
                    if ok and loc_on and loc:
                        send_telegram_location(loc[0], loc[1])
                    st.session_state["notified"] = True
                    st.session_state["last_sent"] = now
                    st.toast("📨 Owner notified" if ok else f"⚠️ Notification failed: {info}")
                if m.bad_seconds == 0:
                    st.session_state["notified"] = False

            # ---- phone alerts (beep + voice + flash + vibration)
            if opt_voice:
                gap = 2.5 if m.status == "Danger" else 4.0
            else:
                gap = 1.4 if m.status == "Danger" else 2.0
            last_beep = st.session_state.get("last_beep", 0.0)
            if m.status in ("Warning", "Danger") and now - last_beep > gap:
                fire_alert(sound_box, m.status, m.reason, opt_beep, opt_voice, opt_flash, opt_vibrate)
                st.session_state["last_beep"] = now
            elif m.status == "Normal" and now - last_beep > 2.0:
                sound_box.empty()

            # ---- refresh the location component every 30 s
            if loc_on and HAS_GEO and int(now // 30) != geo_window:
                st.rerun()
        time.sleep(0.3)

# camera is stopped (or was never started): show the report
if not ctx.state.playing:
    status_box.markdown('<div class="status-card status-normal">ℹ️ System Offline - tap START</div>', unsafe_allow_html=True)
    notify_box.empty()
    render_report(driver_name)
