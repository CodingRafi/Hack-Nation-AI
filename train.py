from ultralytics import YOLO

# -------------------------------------------------------------
# LANGKAH 1: TRAINING MODEL
# -------------------------------------------------------------
# Muat model dasar
base_model = YOLO('yolov8n.pt') 

# # Latih model
# base_model.train(
#     data=r'archive/catfish.yaml',
#     epochs=200,
#     imgsz=640,
#     batch=32,
#     name='lele_model_v4'
# )
# print("Pelatihan selesai!")

# -------------------------------------------------------------
# LANGKAH 2: MEMUAT MODEL HASIL LATIHAN (BEST.PT)
# -------------------------------------------------------------
# Muat model terbaik yang baru saja selesai dilatih
trained_model = YOLO(r'runs/detect/lele_model_v4-5/weights/best.pt')

# -------------------------------------------------------------
# LANGKAH 3: TESTING / PREDIKSI PADA VIDEO
# -------------------------------------------------------------
results = trained_model.predict(
    source="lele.mp4", 
    show=True,
    conf=0.25  # Tingkat keyakinan minimal 25%
)