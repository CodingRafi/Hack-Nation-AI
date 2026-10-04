# Hack-Nation AI — Deploy ke Replit

Engine AI pemantau kolam lele: YOLOv8 menghitung lele pada video, lalu hasilnya (jumlah lele, pH, suhu, turbidity, TDS, status, rekomendasi) dikirim ke backend Django lewat HTTP POST setiap 2 detik. Dashboard Next.js membaca data dari backend itu.

```
[ Replit: main_engine.py ] --POST /api/telemetry/--> [ Replit: Django (hackton-be) ] <--polling-- [ Vercel: Next.js ]
```

Engine adalah **proses worker** (loop terus-menerus), bukan web server. Ia tidak membuka port.

---

## ⚠️ Baca dulu: URL backend harus publik

URL backend yang saat ini dipakai:

```
https://cccd453e-9cba-410c-91b7-cb79c166160e-00-2rchn194e7yz3.pike.replit.dev/
```

adalah **private dev domain** Replit. Setiap request tanpa login Replit dialihkan ke `replit.com/__replshield` (HTTP 307), dan POST ditolak dengan `403 Expected X-Requested-With header`. Akibatnya:

- engine AI **tidak bisa** mengirim data ke URL itu, dan
- dashboard di Vercel kena **"CORS error"**. Penyebabnya redirect login tadi, bukan konfigurasi CORS Django, karena respons redirect tidak membawa header CORS.

Solusinya: **deploy (Publish) backend** di Replit. Hasilnya URL publik berakhiran `.replit.app` yang bisa diakses tanpa login. Pakai URL itu untuk `BACKEND_URL` di engine dan `NEXT_PUBLIC_API_URL` di Vercel.

Cek URL backend sudah publik (harus mengembalikan `{"status":"ok"}`, bukan HTML/redirect):

```bash
curl -i https://<url-backend-publik>/api/health/
```

---

## Isi repo

| File | Fungsi |
|---|---|
| `main_engine.py` | Entry point: YOLOv8 + filter air + analitik + kirim ke backend |
| `engine/analytics.py` | Rolling average, deteksi lonjakan pH/suhu, FCR & penghematan pakan |
| `lele.mp4` | Video sumber (diputar berulang) |
| `runs/detect/lele_model_v4-5/weights/best.pt` | Bobot model yang dipakai engine |
| `requirements.txt` | Dependensi untuk server (PyTorch CPU + OpenCV headless) |

Untuk training (`train.py`, `predict.py`, `index.py`) tidak perlu di-deploy.

> Repo ±150 MB karena menyertakan dataset `archive/` dan semua hasil training. Import dari GitHub tetap jalan, tapi lebih lama. Kalau ingin ringan, hapus `archive/` dan folder `runs/detect/lele_model*` selain `v4-5` dari repo deploy.

---

## Langkah deploy

### 1. Import repo ke Replit
1. Replit → **Create App** → **Import from GitHub**.
2. Pilih `https://github.com/CodingRafi/Hack-Nation-AI` (pastikan `main_engine.py`, `requirements.txt`, `lele.mp4`, dan `runs/detect/lele_model_v4-5/weights/best.pt` sudah di-push).
3. Pilih bahasa **Python 3.11 atau 3.12**.

### 2. Install dependensi
Di Shell Replit:

```bash
pip install -r requirements.txt
```

`requirements.txt` sengaja memakai:
- **PyTorch CPU** (`--extra-index-url .../whl/cpu`). Wheel bawaan Linux membawa CUDA berukuran beberapa GB dan akan membuat disk Replit penuh.
- **`ultralytics-opencv-headless`**. Replit tidak punya layar maupun `libGL`, jadi `opencv-python` biasa gagal dengan `ImportError: libGL.so.1`.

### 3. Isi Secrets
Tab **Secrets** (ikon gembok):

| Key | Value | Wajib |
|---|---|---|
| `BACKEND_URL` | URL backend publik, mis. `https://<nama>.replit.app` (slash di akhir boleh ada, otomatis dibuang) | ya |
| `POST_INTERVAL` | Jeda kirim data, detik (default `2`) | tidak |
| `POST_TIMEOUT` | Timeout tiap POST, detik (default `10`) | tidak |

### 4. Tes manual
```bash
python main_engine.py --headless --max-seconds 30
```

Harus berjalan tanpa error. Kalau backend tidak terjangkau, muncul `[Backend] POST /api/telemetry/ gagal: ...` (lihat Troubleshooting). Lalu cek data masuk:

```bash
curl https://<url-backend-publik>/api/telemetry/
```

### 5. Jalankan terus-menerus (Deployment)
Tombol **Run** hanya hidup selama tab Replit terbuka. Agar engine jalan 24 jam, buat **Deployment**:

1. **Deploy** → pilih tipe **Reserved VM** (proses long-running tanpa port). Tipe Autoscale tidak cocok karena mati saat tidak ada request HTTP, sedangkan engine ini tidak menerima request.
2. **Run command**: `python main_engine.py --headless`
3. Pastikan Secrets (langkah 3) ikut terbawa ke deployment.

Alternatif lewat file `.replit` di root repo:

```toml
language = "python3"
run = "python main_engine.py --headless"

[deployment]
run = ["python", "main_engine.py", "--headless"]
deploymentTarget = "gce"
```

> Nama tipe deployment di UI Replit bisa berubah. Intinya: pilih tipe VM yang selalu menyala, bukan yang berbasis request.

---

## Jalankan di komputer sendiri

```bash
python main_engine.py                  # dengan jendela video, tekan q untuk keluar
python main_engine.py --headless       # tanpa jendela
BACKEND_URL=https://<url-backend> python main_engine.py --headless
```

PowerShell: `$env:BACKEND_URL="https://<url-backend>"; python main_engine.py --headless`

Tanpa `BACKEND_URL`, engine mengirim ke `http://127.0.0.1:8000`.

---

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `POST ... gagal: ... Too many redirects` / respons HTML `replshield` / `403 Expected X-Requested-With` | `BACKEND_URL` masih private dev domain (`*.pike.replit.dev`). Pakai URL deployment publik `*.replit.app`. |
| `POST ... gagal: 400` dengan `DisallowedHost` di log backend | Host backend belum diizinkan. Set `DJANGO_ALLOWED_HOSTS` di backend (default sudah mencakup `.replit.app`, `.replit.dev`, `.repl.co`). |
| `POST ... gagal: 404` | Cek `BACKEND_URL` benar dan backend sudah memuat route `/api/telemetry/`. |
| `ImportError: libGL.so.1` | Terpasang `opencv-python` biasa. Jalankan `pip uninstall -y opencv-python opencv-contrib-python` lalu `pip install -r requirements.txt`. |
| `No space left on device` saat install | Terpasang PyTorch versi CUDA. `pip uninstall -y torch torchvision`, lalu `pip install -r requirements.txt` (versi CPU). |
| `FileNotFoundError: ...best.pt` atau `lele.mp4` | File belum ter-push ke GitHub, atau engine tidak dijalankan dari root repo. Jalankan dari folder yang berisi `main_engine.py`. |
| Dashboard tetap kosong padahal engine jalan | Cek `curl .../api/telemetry/` mengembalikan data. Kalau ada, masalah ada di FE: `NEXT_PUBLIC_API_URL` di Vercel harus URL backend publik dengan `https://`, lalu redeploy FE. |
| Engine lambat / FPS rendah | Normal di CPU Replit. Data tetap terkirim tiap `POST_INTERVAL` detik, jadi dashboard tidak terpengaruh. |

---

## Catatan

- Telemetri (pH, suhu, turbidity, TDS) dan data biomassa/FCR adalah **simulasi** berdasarkan fase waktu video (`PHASES` dan `DEMO_*` di `main_engine.py`), belum dari sensor sungguhan. Hanya jumlah lele yang berasal dari deteksi YOLO.
- Endpoint POST di backend saat ini tanpa autentikasi. Siapa pun yang tahu URL publiknya bisa mengirim data.
- Telegram tidak digunakan di proyek ini.
