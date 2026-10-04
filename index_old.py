import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

# 1. Muat model kustom yang sudah dilatih
model = YOLO(r'runs/detect/lele_model_v3/weights/best.pt')

# 2. Buka video lele
cap = cv2.VideoCapture("lele.mp4")

# Inisialisasi variabel simulasi sensor
last_sensor_update = time.time()
ph_val = 7.2
temp_val = 29.5
turbidity_val = 120  # NTU

window_name = "Small AI - Smart Catfish & Water Quality Monitoring"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- RESIZE FRAME DULU AGAR UKURAN TEKS PROPORSIONAL ---
    # Ubah resolusi frame ke standar 1280x720 sebelum diolah
    frame = cv2.resize(frame, (1280, 720))

    # --- SIMULASI SENSOR (Update setiap 2 detik) ---
    if time.time() - last_sensor_update > 2.0:
        ph_val = round(random.uniform(6.8, 7.6), 1)
        temp_val = round(random.uniform(28.5, 30.2), 1)
        turbidity_val = random.randint(110, 145)
        last_sensor_update = time.time()

    # --- DETEKSI VISION YOLOv8 ---
    results = model.predict(frame, conf=0.5, iou=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # Hitung jumlah lele terdeteksi
    jumlah_lele = len(results[0].boxes)

    # --- LOGIKA ANALISIS KESEHATAN KOLAM ---
    if jumlah_lele >= 5 and turbidity_val < 150:
        status_sistem = "IDEAL: Lele Aktif & Air Normal"
        status_color = (0, 255, 0)  # Hijau (BGR)
        rekomendasi = "Rekomendasi: Berikan Pakan Kategori Normal"
    elif jumlah_lele < 3 and turbidity_val >= 130:
        status_sistem = "PERINGATAN: Air Menurun / Lele Pasif"
        status_color = (0, 0, 255)  # Merah
        rekomendasi = "Rekomendasi: Tunda Pakan, Cek Sirkulasi Air"
    else:
        status_sistem = "MODERAT: Monitoring Rutin"
        status_color = (0, 255, 255)  # Kuning
        rekomendasi = "Rekomendasi: Pantau Respon Pakan"

    # --- DRAWING DASHBOARD PANEL (OVERLAY HUD) ---
    height, width, _ = annotated_frame.shape

    # Panel hitam transparan di bagian atas layar (tinggi 150px)
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 150), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

    # Tampilkan Informasi AI Vision (Kiri) - Skala Font Disesuaikan
    cv2.putText(annotated_frame, f"AI Vision Count: {jumlah_lele} Lele", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Status: {status_sistem}", (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(annotated_frame, f"{rekomendasi}", (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Tampilkan Telemetri Sensor Simulasi (Kanan)
    sensor_x = width - 400
    cv2.putText(annotated_frame, f"[ SENSOR TELEMETRY ]", (sensor_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
    cv2.putText(annotated_frame, f"pH Air     : {ph_val}", (sensor_x, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Suhu Air   : {temp_val} C", (sensor_x, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Kekeruhan  : {turbidity_val} NTU", (sensor_x, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan Hasil tanpa perlu resizeWindow manual lagi
    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()