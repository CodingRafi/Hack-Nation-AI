from ultralytics import YOLO

# Load model lele lama kamu
model = YOLO('runs/detect/lele_model_v3/weights/best.pt')

# Generate label otomatis untuk folder gambar lele baru
model.predict(
    source='archive/train/images_new',
    conf=0.35,        # Sesuaikan threshold deteksi
    save_txt=True,    # Wajib True agar menghasilkan file .txt
    save_conf=False,
    project='dataset_baru',
    name='labels_auto'
)