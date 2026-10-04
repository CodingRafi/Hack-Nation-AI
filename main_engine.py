"""
Engine utama: YOLOv8 + filter air dinamis + analitik + kirim telemetri ke Django.

    python main_engine.py              # dengan jendela video
    python main_engine.py --headless   # tanpa jendela (server / test)

Env: BACKEND_URL (default http://127.0.0.1:8000), POST_INTERVAL (detik, default 2)
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from engine.analytics import TelemetryAnalytics, calculate_fcr, calculate_savings

# rstrip: "https://host/" + "/api/..." would otherwise give "//api/..." (404)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
POST_INTERVAL = float(os.getenv("POST_INTERVAL", "2"))
# Remote backends (Replit) can be slow on a cold start; 2s was too tight.
POST_TIMEOUT = float(os.getenv("POST_TIMEOUT", "10"))

# Nilai simulasi untuk demo biomassa (belum ada sensor/timbangan sungguhan)
DEMO_TOTAL_FISH = 250
DEMO_FEED_KG = 2.5
DEMO_BIOMASS_GAIN_KG = 1.84

# Siklus kondisi air (detik video) -> telemetri simulasi
PHASES = [
    (5.0, dict(
        water_mode="normal", ph=7.2, temp=29.1, turbidity=115, tds=820,
        status="IDEAL: Lele Aktif & Air Normal",
        color=(0, 255, 0), rekomendasi="Beri Pakan Sesuai Jadwal",
    )),
    (11.0, dict(
        water_mode="keruh", ph=8.6, temp=30.8, turbidity=285, tds=980,
        status="PERINGATAN: Air Keruh & Amonia Tinggi!",
        color=(0, 0, 255), rekomendasi="Tunda Pakan, Cek Sirkulasi Air",
    )),
    (float("inf"), dict(
        water_mode="bersih", ph=7.0, temp=28.4, turbidity=42, tds=610,
        status="SANGAT BAIK: Post-Water Change / Bening",
        color=(255, 255, 0), rekomendasi="Kondisi Kolam Optimal",
    )),
]


def phase_at(sec):
    for limit, phase in PHASES:
        if sec < limit:
            return phase


def apply_water_filter(frame, mode="normal"):
    """Mengubah warna visual air menggunakan manipulasi OpenCV"""
    h, w, _ = frame.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if mode == "keruh":
        overlay[:] = (20, 50, 90)  # BGR cokelat keruh (amonia / pakan berlebih)
        return cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
    if mode == "bersih":
        overlay[:] = (120, 100, 30)  # BGR biru-hijau jernih (air baru)
        return cv2.addWeighted(frame, 0.70, overlay, 0.30, 0)
    return frame


def post_json(path, payload):
    try:
        requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=POST_TIMEOUT).raise_for_status()
    except requests.RequestException as e:
        print(f"[Backend] POST {path} gagal: {e}")


def draw_hud(frame, count, p, rekomendasi):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 150), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"AI Vision Count: {count} Lele", (25, 40), font, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {p['status']}", (25, 85), font, 0.8, p["color"], 2)
    cv2.putText(frame, f"Rekomendasi: {rekomendasi}", (25, 125), font, 0.7, (220, 220, 220), 2)
    x = 830
    cv2.putText(frame, "[ TELEMETRY SENSOR ]", (x, 35), font, 0.75, (0, 215, 255), 2)
    cv2.putText(frame, f"pH Air     : {p['ph']}", (x, 65), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Suhu Air   : {p['temp']} C", (x, 90), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Turbidity  : {p['turbidity']} NTU", (x, 115), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"TDS        : {p['tds']} ppm", (x, 140), font, 0.65, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="tanpa jendela video")
    parser.add_argument("--max-seconds", type=float, default=0, help="berhenti setelah N detik (0 = tanpa batas)")
    args = parser.parse_args()

    model = YOLO(r"runs/detect/lele_model_v4-5/weights/best.pt")
    cap = cv2.VideoCapture("lele.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    analytics = TelemetryAnalytics(window_size=10)  # 1 sampel/detik video -> ~10 detik
    pool = ThreadPoolExecutor(max_workers=1)  # POST di thread agar video tidak tersendat

    frame_count = 0
    last_sample_sec = -1
    last_post = 0.0
    last_mode = None
    started = time.time()

    while cap.isOpened():
        if args.max_seconds and time.time() - started > args.max_seconds:
            break

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            last_sample_sec = -1
            analytics.reset()
            continue

        frame_count += 1
        sec = frame_count / fps
        p = phase_at(sec)

        # Rolling window: satu sampel per detik video
        if int(sec) != last_sample_sec:
            last_sample_sec = int(sec)
            analytics.update(p["ph"], p["temp"], p["turbidity"])
        trend, ph_delta = analytics.get_trends()

        rekomendasi = p["rekomendasi"]
        if trend == "LONJAKAN_AMONIA":
            rekomendasi += f" (pH naik {ph_delta:+})"

        processed = apply_water_filter(frame, mode=p["water_mode"])
        processed = cv2.resize(processed, (1280, 720))
        results = model.predict(processed, conf=0.5, iou=0.45, verbose=False)
        annotated = results[0].plot()
        jumlah_lele = len(results[0].boxes)

        now = time.time()
        if now - last_post >= POST_INTERVAL:
            last_post = now
            pool.submit(post_json, "/api/telemetry/", {
                "catfish_count": jumlah_lele,
                "ph": p["ph"],
                "temperature": p["temp"],
                "turbidity": p["turbidity"],
                "tds": p["tds"],
                "status": p["status"],
                "recommendation": rekomendasi,
            })

        # Catat biomassa/FCR setiap kali kondisi air berganti (data simulasi)
        if p["water_mode"] != last_mode:
            last_mode = p["water_mode"]
            _, cost_saved = calculate_savings(DEMO_FEED_KG)
            pool.submit(post_json, "/api/biomass/", {
                "total_fish_count": DEMO_TOTAL_FISH,
                "feed_given_kg": DEMO_FEED_KG,
                "fcr_value": calculate_fcr(DEMO_FEED_KG, DEMO_BIOMASS_GAIN_KG),
                "cost_saved_idr": cost_saved,
            })

        if not args.headless:
            draw_hud(annotated, jumlah_lele, p, rekomendasi)
            cv2.imshow("Small AI - Dynamic Water Quality Monitoring", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.headless:  # opencv-python-headless has no GUI -> would raise
        cv2.destroyAllWindows()
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
