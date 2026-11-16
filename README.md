# Monitoring Aktivitas Babi

Aplikasi ini mendeteksi babi dari kamera atau file video, memberi ID pada setiap babi, dan menampilkan status gerak pada dashboard web.

Pengukuran gerak memakai dense optical flow Farneback pada frame berurutan, lalu mengambil rata-rata perpindahan piksel di dalam bounding box hasil tracking. Ini mengikuti pendekatan gerak berbasis optical flow dalam Zhou et al. (2023), *Pig Movement Estimation by Integrating Optical Flow with a Multi-Object Tracking Model*, Sensors 23(23), 9499, https://doi.org/10.3390/s23239499. Artikel tersebut menemukan Skflow lebih akurat daripada implementasi optical flow OpenCV yang dievaluasi; Farneback di proyek ini adalah pendekatan OpenCV yang lebih ringan, jadi hasilnya perlu divalidasi pada video dan anotasi proyek sendiri. Nilai gerak ditampilkan dalam piksel per frame, bukan sentimeter.

## Menjalankan dashboard React

Gunakan Python 3.10 atau lebih baru dan Node.js. Dari root proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
./run_dashboard.sh
```

Script tersebut menjalankan FastAPI dan UI React sekaligus, dengan video `testing_dataset/testing1.MP4`. Untuk mencoba video lain, berikan path sebagai argumen, misalnya `./run_dashboard.sh testing_dataset/video_lain.mp4` atau `./run_dashboard.sh "/path/lengkap/video lain.mp4"`. Buka UI di `http://127.0.0.1:5173`; dokumentasi API ada di `http://127.0.0.1:8000/docs`.

Frontend berada di folder `frontend/`. Saat pengembangan, Vite meneruskan `/api`, `/health`, dan `/stream` ke FastAPI pada port 8000.

Buka `http://127.0.0.1:8000`. Pemeriksaan status tersedia di `/health`.

Model hasil training `pig/yolo11n_pig-2/weights/best.pt` sama dengan `backend/best.pt`; aplikasi memakai model `backend/best.pt` secara default. Untuk memilih model lain, set `MODEL_PATH` ke path file model.

## Konfigurasi tambahan

- `IMGSZ`: ukuran inferensi (default `640`)
- `CONF`: ambang confidence deteksi (default `0.35`)
- `INFER_FPS`: batas pemrosesan frame per detik (default `5`)
- `MASK_PATH`: path mask area deteksi, opsional
- `TRACK`: gunakan tracker bawaan YOLO saat ID stabil tidak diaktifkan (default `1`)

Contoh memakai file video lain:

```bash
CAMERAS="cam0=/path/ke/video.mp4" python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
