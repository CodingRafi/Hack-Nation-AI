import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load Model & Video
model = YOLO(r'runs/detect/lele_model_v3/weights/best.pt')
cap = cv2.VideoCapture("lele.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

frame_count = 0

def apply_water_filter(frame, mode="normal"):
    """Mengubah warna visual air menggunakan manipulasi OpenCV"""
    h, w, _ = frame.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    if mode == "keruh":
        # Filter Cokelat Keruh Pekat (Amonia / Pakan Berlebih)
        overlay[:] = (20, 50, 90) # BGR
        return cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
    elif mode == "bersih":
        # Filter Biru-Hijau Jernih (Sirkulasi Air Baru)
        overlay[:] = (120, 100, 30) # BGR
        return cv2.addWeighted(frame, 0.70, overlay, 0.30, 0)
    else:
        # Warna Asli
        return frame

window_name = "Small AI - Dynamic Water Quality Monitoring"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        continue

    frame_count += 1
    sec = frame_count / fps

    # --- ATUR SIKLUS PERUBAHAN WARNA AIR & TELEMETRI REALISTIS ---
    if sec < 5.0:
        water_mode = "normal"
        ph_val = 7.2
        temp_val = 29.1
        turbidity_val = 115
        tds_val = 820
        status_sistem = "IDEAL: Lele Aktif & Air Normal"
        status_color = (0, 255, 0) # Hijau
        rekomendasi = "Rekomendasi: Beri Pakan Sesuai Jadwal"

    elif 5.0 <= sec < 11.0:
        water_mode = "keruh"
        ph_val = 8.6
        temp_val = 30.8
        turbidity_val = 285
        tds_val = 980
        status_sistem = "PERINGATAN: Air Keruh & Amonia Tinggi!"
        status_color = (0, 0, 255) # Merah
        rekomendasi = "Rekomendasi: Tunda Pakan, Cek Sirkulasi Air"

    else:
        water_mode = "bersih"
        ph_val = 7.0
        temp_val = 28.4
        turbidity_val = 42
        tds_val = 610
        status_sistem = "SANGAT BAIK: Post-Water Change / Bening"
        status_color = (255, 255, 0) # Cyan
        rekomendasi = "Rekomendasi: Kondisi Kolam Optimal"

    # 1. Manipulasi Warna Air Dulu
    processed_frame = apply_water_filter(frame, mode=water_mode)
    
    # 2. Downscale ke 1280x720
    processed_frame = cv2.resize(processed_frame, (1280, 720))

    # 3. Deteksi YOLOv8
    results = model.predict(processed_frame, conf=0.5, iou=0.45, verbose=False)
    annotated_frame = results[0].plot()
    jumlah_lele = len(results[0].boxes)

    # 4. Gambar HUD Panel Overlay
    overlay_panel = annotated_frame.copy()
    cv2.rectangle(overlay_panel, (0, 0), (1280, 150), (15, 15, 15), -1)
    cv2.addWeighted(overlay_panel, 0.75, annotated_frame, 0.25, 0, annotated_frame)

    # Teks AI Vision (Kiri)
    cv2.putText(annotated_frame, f"AI Vision Count: {jumlah_lele} Lele", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Status: {status_sistem}", (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(annotated_frame, f"{rekomendasi}", (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Teks Telemetri Sensor (Kanan)
    sensor_x = 830
    cv2.putText(annotated_frame, f"[ TELEMETRY SENSOR ]", (sensor_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
    cv2.putText(annotated_frame, f"pH Air     : {ph_val}", (sensor_x, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Suhu Air   : {temp_val} C", (sensor_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Turbidity  : {turbidity_val} NTU", (sensor_x, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"TDS        : {tds_val} ppm", (sensor_x, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()