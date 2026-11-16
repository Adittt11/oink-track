#!/usr/bin/env python3
"""
app.py - Web server deteksi, tracking, dan analisis gerakan babi.

Jalankan dari folder backend:
    conda activate adit
    export MODEL_PATH=best.pt
    export CAMERAS="cam0=/path/ke/testing1.MP4"
    python -m uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

try:  # Mendukung `uvicorn app:app` dari backend dan `uvicorn backend.app:app` dari root.
    from .stable_ids import StableIDs, draw_boxes
    from .activity_analyzer import ActivityAnalyzer
except ImportError:
    from stable_ids import StableIDs, draw_boxes
    from activity_analyzer import ActivityAnalyzer


# ============================================================
# 1. KONFIGURASI
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent


def resolve_asset_path(value: str, default: Path) -> Path:
    """Resolve configured assets relative to cwd, then the backend directory."""
    candidate = Path(value).expanduser() if value else default
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    return cwd_candidate if cwd_candidate.exists() else BACKEND_DIR / candidate


MODEL_PATH = str(resolve_asset_path(os.getenv("MODEL_PATH", "best.pt"), BACKEND_DIR / "best.pt"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
CONF = float(os.getenv("CONF", "0.1"))
INFER_FPS = max(float(os.getenv("INFER_FPS", "5")), 0.1)
MASK_PATH = os.getenv("MASK_PATH", "")
USE_TRACK = os.getenv("TRACK", "1") == "1"
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "960"))
JPEG_Q = int(os.getenv("JPEG_Q", "70"))
DEBUG_TIMING = os.getenv("DEBUG_TIMING", "0") == "1"

ACTIVITY_MOVE_THRESHOLD = float(os.getenv("ACTIVITY_MOVE_THRESHOLD", "0.8"))
ACTIVITY_INACTIVE_SECONDS = float(os.getenv("ACTIVITY_INACTIVE_SECONDS", "5"))
ACTIVITY_OBSERVATION_SECONDS = float(os.getenv("ACTIVITY_OBSERVATION_SECONDS", "600"))

USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

if not Path(MODEL_PATH).exists():
    fallback_model = BACKEND_DIR / "yolo11n.pt"
    print(f"[warn] Model {MODEL_PATH} tidak ditemukan. Menggunakan {fallback_model}.")
    MODEL_PATH = str(fallback_model)

print(
    f"[info] Device: {'CUDA' if USE_CUDA else 'CPU'} "
    f"| Model: {MODEL_PATH} | MAX_WIDTH={MAX_WIDTH}"
)


# ============================================================
# 2. PARSING KONFIGURASI
# ============================================================

def parse_cameras() -> dict[str, str]:
    """Format CAMERAS: cam0=/path/video.mp4,cam1=rtsp://..."""
    raw = os.getenv("CAMERAS", "cam0=0")
    cameras: dict[str, str] = {}

    for item in raw.split(","):
        if "=" in item:
            name, url = item.split("=", 1)
            cameras[name.strip()] = url.strip()

    return cameras


def parse_pigs() -> dict[str, int]:
    """Format PIGS: cam0=5,cam1=8 (opsional)."""
    output: dict[str, int] = {}

    for item in os.getenv("PIGS", "").split(","):
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        try:
            number = int(value.strip())
            if number > 0:
                output[key.strip()] = number
        except ValueError:
            pass

    return output


def new_model() -> YOLO:
    model = YOLO(MODEL_PATH)

    # Warm-up model.
    dummy = np.zeros(
        (max(1, MAX_WIDTH * 9 // 16), MAX_WIDTH, 3),
        dtype=np.uint8,
    )
    model.predict(
        dummy,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )
    return model


def normalize_activity(raw: Any) -> dict[int, dict]:
    """
    Normalisasi hasil ActivityAnalyzer menjadi dictionary:
    {
        1: {"id": 1, "status": "...", "motion_ratio": 0.0,
            "inactive_seconds": 0}
    }

    Fungsi ini mendukung keluaran analyzer berbentuk dict atau list.
    Ini mencegah error saat analyzer mengembalikan list tetapi app
    mencoba memanggil .get() atau .items().
    """
    normalized: dict[int, dict] = {}

    if raw is None:
        return normalized

    if isinstance(raw, dict):
        # Bentuk dict bisa {id: data} atau satu data aktivitas.
        if any(k in raw for k in ("id", "pig_id", "track_id")):
            items = [raw]
        else:
            items = []
            for key, value in raw.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("id", key)
                    items.append(item)

    elif isinstance(raw, (list, tuple)):
        items = [item for item in raw if isinstance(item, dict)]

    else:
        return normalized

    for item in items:
        pig_id = item.get("id", item.get("pig_id", item.get("track_id")))

        if pig_id is None:
            continue

        try:
            pig_id = int(pig_id)
        except (TypeError, ValueError):
            continue

        normalized[pig_id] = dict(item)
        normalized[pig_id]["id"] = pig_id

    return normalized


# ============================================================
# 3. CLASS CAMERA
# ============================================================

class Camera:

    def __init__(
        self,
        name: str,
        url: str,
        mask: np.ndarray | None,
        max_pigs: int = 0,
    ):
        self.name = name
        self.url = int(url) if url.isdigit() else url
        self.mask = mask

        self.model = new_model()

        self.max_pigs = max_pigs
        self.stable = StableIDs(max_pigs) if max_pigs > 0 else None
        self.active = max_pigs > 0

        self.raw: np.ndarray | None = None
        self.jpeg: bytes | None = None

        self.count = 0
        self.fps = 0.0
        self.online = False
        self.stop = False

        self.lock = threading.Lock()
        self.tracker_ids: set[int] = set()

        self.activity_analyzer = ActivityAnalyzer(
            movement_threshold=ACTIVITY_MOVE_THRESHOLD,
            inactivity_seconds=ACTIVITY_INACTIVE_SECONDS,
            observation_seconds=ACTIVITY_OBSERVATION_SECONDS,
        )

        # Selalu simpan aktivitas dalam bentuk dict yang sudah dinormalisasi.
        self.activity_results: dict[int, dict] = {}

        threading.Thread(
            target=self._reader,
            daemon=True,
            name=f"{name}-reader",
        ).start()

        threading.Thread(
            target=self._infer,
            daemon=True,
            name=f"{name}-infer",
        ).start()

    # --------------------------------------------------------
    # 3A. READER VIDEO
    # --------------------------------------------------------

    def _reader(self):
        while not self.stop:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.online = False
                print(
                    f"[warn] {self.name}: sumber video tidak dapat dibuka: "
                    f"{self.url}"
                )
                cap.release()
                time.sleep(3)
                continue

            self.online = True
            print(f"[info] {self.name}: sumber video terhubung")

            while not self.stop:
                ok, frame = cap.read()

                if not ok:
                    break

                with self.lock:
                    self.raw = frame

                # Membatasi kecepatan pembacaan file lokal.
                if (
                    isinstance(self.url, str)
                    and not self.url.lower().startswith("rtsp")
                ):
                    time.sleep(0.03)

            cap.release()
            self.online = False

            if isinstance(self.url, str) and Path(self.url).is_file():
                print(
                    f"[info] {self.name}: video selesai, "
                    "mengulang dari awal"
                )

            time.sleep(1)

    # --------------------------------------------------------
    # 3B. INFERENCE DAN ANALISIS AKTIVITAS
    # --------------------------------------------------------

    def _infer(self):
        interval = 1.0 / INFER_FPS
        last = 0.0

        while not self.stop:
            now = time.time()

            if now - last < interval:
                time.sleep(0.005)
                continue

            with self.lock:
                frame = None if self.raw is None else self.raw.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            last = now
            t_start = time.time()

            h0, w0 = frame.shape[:2]

            # Resize frame jika terlalu lebar.
            if w0 > MAX_WIDTH:
                frame = cv2.resize(
                    frame,
                    (MAX_WIDTH, int(h0 * MAX_WIDTH / w0)),
                    interpolation=cv2.INTER_AREA,
                )

            # Optical Flow menggunakan frame sebelum masking.
            analysis_frame = frame.copy()

            # ------------------------------------------------
            # BELUM MEMULAI MONITORING
            # ------------------------------------------------

            if not self.active:
                preview = frame.copy()

                texts = [
                    ("Isi jumlah babi di kandang,", 36),
                    ("lalu klik 'Mulai monitoring'", 68),
                ]

                for txt, y in texts:
                    cv2.putText(
                        preview, txt, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 5, cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview, txt, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )

                ok, buf = cv2.imencode(
                    ".jpg",
                    preview,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q],
                )

                if ok:
                    self.jpeg = buf.tobytes()

                self.count = 0
                self.fps = 0.0
                time.sleep(0.2)
                continue

            # ------------------------------------------------
            # MASKING AREA KANDANG
            # ------------------------------------------------

            if self.mask is not None:
                current_mask = self.mask

                if current_mask.shape[:2] != frame.shape[:2]:
                    current_mask = cv2.resize(
                        current_mask,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                frame = frame * current_mask[..., None]

            max_det = self.max_pigs if self.max_pigs > 0 else 300
            stable = self.stable

            # ------------------------------------------------
            # YOLO DETECTION / TRACKING
            # ------------------------------------------------

            try:
                if stable is not None:
                    result = self.model.predict(
                        frame,
                        imgsz=IMGSZ,
                        conf=CONF,
                        max_det=max_det,
                        device=DEVICE,
                        verbose=False,
                    )[0]

                elif USE_TRACK:
                    result = self.model.track(
                        frame,
                        imgsz=IMGSZ,
                        conf=CONF,
                        max_det=max_det,
                        persist=True,
                        tracker=str(BACKEND_DIR / "bytetrack_pig.yaml"),
                        device=DEVICE,
                        verbose=False,
                    )[0]

                else:
                    result = self.model.predict(
                        frame,
                        imgsz=IMGSZ,
                        conf=CONF,
                        max_det=max_det,
                        device=DEVICE,
                        verbose=False,
                    )[0]

            except Exception as exc:
                print(f"[error] {self.name}: inferensi YOLO gagal: {exc}")
                time.sleep(0.2)
                continue

            # ------------------------------------------------
            # AMBIL BOUNDING BOX
            # ------------------------------------------------

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)

            detections: list[dict] = []

            # ------------------------------------------------
            # TRACKING STABLE IDS
            # ------------------------------------------------

            if stable is not None:
                ids = stable.update(frame, xyxy, confs)

                self.count = sum(
                    1 for pig_id in ids if pig_id is not None
                )

                self.tracker_ids.update(
                    int(pig_id)
                    for pig_id in ids
                    if pig_id is not None
                )

                annotated = draw_boxes(
                    frame.copy(),
                    xyxy,
                    ids,
                    confs,
                )

                for bbox, pig_id in zip(xyxy, ids):
                    if pig_id is None:
                        continue

                    detections.append({
                        "id": int(pig_id),
                        "bbox": tuple(float(v) for v in bbox),
                    })

            # ------------------------------------------------
            # TRACKING BYTE TRACK
            # ------------------------------------------------

            else:
                self.count = len(xyxy)
                ids = []

                if (
                    USE_TRACK
                    and result.boxes is not None
                    and result.boxes.id is not None
                ):
                    ids = [
                        int(value)
                        for value in result.boxes.id.cpu().tolist()
                    ]
                    self.tracker_ids.update(ids)

                annotated = result.plot(line_width=2)

                for bbox, pig_id in zip(xyxy, ids):
                    detections.append({
                        "id": int(pig_id),
                        "bbox": tuple(float(v) for v in bbox),
                    })

            # ------------------------------------------------
            # ANALISIS AKTIVITAS OPTICAL FLOW
            # ------------------------------------------------

            try:
                raw_activity = self.activity_analyzer.update(
                    analysis_frame,
                    detections,
                )
                activity = normalize_activity(raw_activity)

            except Exception as exc:
                print(
                    f"[error] {self.name}: analisis aktivitas gagal: {exc}"
                )
                activity = {}

            with self.lock:
                self.activity_results = activity.copy()

            # ------------------------------------------------
            # TAMPILKAN STATUS AKTIVITAS
            # ------------------------------------------------

            for detection in detections:
                pig_id = detection["id"]
                x1, y1, x2, y2 = [
                    int(v) for v in detection["bbox"]
                ]

                info = activity.get(pig_id, {})
                status_text = info.get(
                    "status",
                    "MENGUMPULKAN DATA",
                )

                if status_text == "BERGERAK":
                    color = (0, 200, 0)
                elif status_text == "TIDAK AKTIF":
                    color = (0, 190, 255)
                elif status_text == "PERLU OBSERVASI":
                    color = (0, 0, 255)
                else:
                    color = (255, 200, 0)

                cv2.putText(
                    annotated,
                    f"ID {pig_id}: {status_text}",
                    (max(5, x1), max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # INFORMASI KAMERA DAN ENCODE FRAME
            # ------------------------------------------------

            label = f"{self.name} | {self.count}"
            if self.max_pigs:
                label += f"/{self.max_pigs}"

            cv2.putText(
                annotated,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, buf = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q],
            )

            if ok:
                self.jpeg = buf.tobytes()

            self.fps = 1.0 / max(time.time() - t_start, 1e-3)

            if DEBUG_TIMING:
                self._debug_counter = getattr(self, "_debug_counter", 0) + 1

                if self._debug_counter % 10 == 0:
                    print(
                        f"[timing {self.name}] "
                        f"frame={w0}x{h0} | "
                        f"total={(time.time() - t_start) * 1000:.0f} ms"
                    )


# ============================================================
# 4. LOAD MASK DAN INISIALISASI KAMERA
# ============================================================

def load_mask() -> np.ndarray | None:
    if not MASK_PATH:
        return None

    mask_path = resolve_asset_path(MASK_PATH, BACKEND_DIR / MASK_PATH)

    if not mask_path.exists():
        print(f"[warn] Mask tidak ditemukan: {MASK_PATH}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("[warn] Gagal membaca mask.")
        return None

    print("[info] Mask berhasil dimuat.")
    return (mask > 0).astype(np.uint8)


mask_img = load_mask()
initial_pigs = parse_pigs()

cameras: dict[str, Camera] = {
    name: Camera(
        name,
        url,
        mask_img,
        initial_pigs.get(name, 0),
    )
    for name, url in parse_cameras().items()
}


# ============================================================
# 5. FASTAPI
# ============================================================

app = FastAPI(
    title="CCTV Pig Detection and Activity Monitoring"
)


# ============================================================
# 6. VIDEO STREAM
# ============================================================

def mjpeg(cam: Camera):
    while not cam.stop:
        with cam.lock:
            jpeg = cam.jpeg

        if jpeg is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(1.0 / max(INFER_FPS, 1))


@app.get("/stream/{name}")
def stream(name: str):
    cam = cameras.get(name)

    if cam is None:
        raise HTTPException(
            status_code=404,
            detail="kamera tidak ditemukan",
        )

    return StreamingResponse(
        mjpeg(cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# 7. API STATISTIK
# ============================================================

@app.get("/api/stats")
def stats():
    data = {}

    for name, cam in cameras.items():
        with cam.lock:
            # activity_results dipastikan dict oleh normalize_activity().
            # Konversi key ke string agar aman dikirim sebagai JSON.
            activity_copy = {
                str(pig_id): dict(info)
                for pig_id, info in cam.activity_results.items()
                if isinstance(info, dict)
            }

            count = cam.count
            fps = cam.fps
            online = cam.online
            active = cam.active
            max_pigs = cam.max_pigs
            unique_ids = len(cam.tracker_ids)

        data[name] = {
            "online": bool(online),
            "count": int(count),
            "max": int(max_pigs),
            "fps": round(float(fps), 1),
            "active": bool(active),
            "unique_ids": int(unique_ids),
            "activities": activity_copy,
        }

    return JSONResponse(content=data)


# ============================================================
# 8. MULAI MONITORING
# ============================================================

@app.post("/api/start/{name}")
def start_monitoring(name: str, n: int):
    cam = cameras.get(name)

    if cam is None:
        raise HTTPException(
            status_code=404,
            detail="kamera tidak ditemukan",
        )

    if n < 1 or n > 300:
        raise HTTPException(
            status_code=400,
            detail="jumlah babi harus 1-300",
        )

    cam.max_pigs = n
    cam.stable = StableIDs(n)
    cam.tracker_ids.clear()

    # Aman jika versi ActivityAnalyzer belum memiliki method reset().
    if hasattr(cam.activity_analyzer, "reset"):
        cam.activity_analyzer.reset()

    with cam.lock:
        cam.activity_results = {}

    cam.active = True

    print(
        f"[info] {name}: monitoring dimulai "
        f"dengan jumlah {n} babi"
    )

    return {
        "name": name,
        "max": n,
        "active": True,
    }


# ============================================================
# 9. BERHENTI MONITORING
# ============================================================

@app.post("/api/stop/{name}")
def stop_monitoring(name: str):
    cam = cameras.get(name)

    if cam is None:
        raise HTTPException(
            status_code=404,
            detail="kamera tidak ditemukan",
        )

    cam.active = False

    if hasattr(cam.activity_analyzer, "reset"):
        cam.activity_analyzer.reset()

    with cam.lock:
        cam.activity_results = {}

    print(f"[info] {name}: monitoring dihentikan")

    return {
        "name": name,
        "active": False,
    }


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": "cuda" if USE_CUDA else "cpu",
        "cameras": list(cameras.keys()),
    }


# ============================================================
# 11. DASHBOARD WEB
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    cards = "".join(
        f"""
        <div class="card">
          <div class="card-heading">
            <div><div class="eyebrow">SUMBER VIDEO</div><h2>{name}</h2></div>
            <span id="s-{name}" class="badge"><i></i> Menghubungkan</span>
          </div>
          <div class="video-stage">
            <img src="/stream/{name}" alt="Tampilan kamera {name}">
            <span class="live-tag"><i></i> LIVE</span>
          </div>
          <div class="metrics">
            <div class="metric"><span class="metric-label">Terdeteksi</span><strong id="c-{name}">0</strong><span class="metric-unit">ekor</span></div>
            <div class="metric"><span class="metric-label">Kecepatan</span><strong id="f-{name}">0</strong><span class="metric-unit">FPS</span></div>
          </div>
          <div class="controls">
            <label for="n-{name}">Jumlah babi di kandang</label>
            <div class="control-row">
              <input id="n-{name}" type="number" min="1" max="300" placeholder="Contoh: 5">
              <button id="go-{name}" onclick="startN('{name}')">Mulai monitoring</button>
              <button id="st-{name}" onclick="stopN('{name}')" class="stop-button">Berhenti</button>
            </div>
            <div id="m-{name}" class="note" aria-live="polite"></div>
          </div>
          <div class="activity-panel">
            <div class="activity-heading">
              <div><div class="eyebrow">PEMANTAUAN</div><h3>Aktivitas individu</h3></div>
              <span class="activity-count">per ID</span>
            </div>
            <div id="a-{name}" class="activity-list"><div class="empty-state">Mulai monitoring untuk melihat aktivitas babi.</div></div>
          </div>
        </div>
        """
        for name in cameras
    )

    return f"""
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitoring CCTV - Deteksi Aktivitas Babi</title>

<style>
* {{ box-sizing: border-box; }}
:root {{ --ink:#172b24; --muted:#718078; --line:#e3eae4; --green:#267453; --surface:#fff; --canvas:#f3f6f1; }}
body {{ font-family: Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; background:var(--canvas); color:var(--ink); }}
header {{ padding:24px max(24px,calc((100vw - 1440px)/2)); background:#173c2d; color:#fff; border-bottom:4px solid #9bc96c; }}
.header-inner {{ max-width:1440px; margin:auto; display:flex; align-items:center; justify-content:space-between; gap:20px; }}
.brand {{ display:flex; align-items:center; gap:14px; }}
.brand-mark {{ width:46px; height:46px; display:grid; place-items:center; border-radius:14px; background:#d7edb0; color:#173c2d; font-size:24px; font-weight:800; }}
.brand h1 {{ margin:0; font-size:clamp(1.15rem,2vw,1.55rem); letter-spacing:-.03em; }}
.brand p {{ margin:4px 0 0; color:#c9d9cf; font-size:.88rem; }}
.system-pill {{ padding:9px 13px; border:1px solid #496c5b; border-radius:999px; color:#e0eee5; font-size:.8rem; white-space:nowrap; }}
.grid {{ max-width:1440px; margin:0 auto; display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr)); align-items:start; gap:22px; padding:28px 24px 48px; }}
.card {{ min-width:0; overflow:hidden; background:var(--surface); border:1px solid var(--line); border-radius:20px; box-shadow:0 8px 28px rgba(29,59,43,.06); }}
.card-heading {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:20px 22px 16px; }}
.eyebrow {{ color:var(--muted); font-size:.66rem; font-weight:800; letter-spacing:.12em; }}
.card-heading h2 {{ margin:4px 0 0; font-size:1.25rem; letter-spacing:-.025em; }}
.video-stage {{ position:relative; margin:0 14px; overflow:hidden; aspect-ratio:16/9; background:#17231e; border-radius:13px; }}
.video-stage img {{ display:block; width:100%; height:100%; object-fit:contain; }}
.live-tag {{ position:absolute; top:12px; left:12px; display:flex; align-items:center; gap:7px; padding:6px 9px; border-radius:7px; background:rgba(14,24,19,.72); color:#fff; font-size:.65rem; font-weight:800; letter-spacing:.1em; }}
.live-tag i,.badge i {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:currentColor; }}
.badge {{ display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border-radius:999px; background:#f1f3ef; color:#6e7972; font-size:.72rem; font-weight:700; white-space:nowrap; }}
.badge.on {{ background:#e8f5ed; color:#287449; }} .badge.off {{ background:#fff0ed; color:#b44938; }}
.badge.on i {{ animation:pulse 1.8s infinite; }}
@keyframes pulse {{ 50% {{ box-shadow:0 0 0 5px rgba(40,116,73,.14); }} }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:14px 14px 0; }}
.metric {{ padding:13px 15px; border:1px solid var(--line); border-radius:12px; background:#fbfcfa; }}
.metric-label,.metric-unit {{ color:var(--muted); font-size:.76rem; }}
.metric strong {{ display:inline-block; margin-top:5px; font-size:1.55rem; letter-spacing:-.04em; }} .metric-unit {{ margin-left:5px; }}
.controls {{ padding:18px 22px 15px; }} .controls label {{ display:block; margin-bottom:8px; font-size:.82rem; font-weight:700; }}
.control-row {{ display:flex; gap:8px; }}
input,button {{ min-height:42px; padding:9px 12px; border:1px solid #d6dfd8; border-radius:9px; font:inherit; }}
input {{ width:120px; background:#fff; color:var(--ink); }} input:focus {{ outline:3px solid rgba(38,116,83,.15); border-color:var(--green); }}
button {{ cursor:pointer; background:var(--green); border-color:var(--green); color:#fff; font-size:.83rem; font-weight:750; transition:.18s ease; }}
button:hover {{ background:#1b5a40; border-color:#1b5a40; transform:translateY(-1px); }} button:focus-visible {{ outline:3px solid #a5d6ba; outline-offset:2px; }}
.stop-button {{ background:#fff; color:#a34536; border-color:#e6c8c2; }} .stop-button:hover {{ background:#fff1ee; border-color:#dca99f; color:#8c3023; }}
.note {{ min-height:18px; margin-top:8px; color:var(--muted); font-size:.76rem; }}
.activity-panel {{ margin-top:2px; padding:17px 22px 20px; border-top:1px solid var(--line); }}
.activity-heading {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }} .activity-heading h3 {{ margin:3px 0 0; font-size:1rem; }}
.activity-count {{ padding:5px 8px; border-radius:7px; background:#f1f5ef; color:#607263; font-size:.7rem; font-weight:700; }}
.empty-state {{ padding:18px 12px; border:1px dashed #d8e1d9; border-radius:10px; color:var(--muted); text-align:center; font-size:.8rem; }}
.activity-list {{ max-height:310px; overflow:auto; }}
.activity-row {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:11px 2px; border-bottom:1px solid #edf1ed; font-size:.83rem; }} .activity-row:last-child {{ border-bottom:0; }} .activity-row b {{ font-size:.83rem; }}
.status-moving {{ color:#26734a; font-weight:750; }} .status-rest {{ color:#9a6a13; font-weight:750; }} .status-alert {{ color:#b63f32; font-weight:750; }} .status-wait {{ color:#597085; font-weight:750; }}
.small {{ font-size:.72rem; color:var(--muted); }}
@media(max-width:600px) {{ header {{ padding:20px 16px; }} .system-pill {{ display:none; }} .grid {{ padding:16px 12px 32px; gap:15px; }} .card-heading {{ padding:17px 16px 13px; }} .controls,.activity-panel {{ padding-left:16px; padding-right:16px; }} .control-row {{ flex-wrap:wrap; }} input {{ flex:1 1 90px; }} .control-row button {{ flex:1 1 auto; }} }}
</style>
</head>

<body>

<header>
  <div class="header-inner">
    <div class="brand"><div class="brand-mark">P</div><div><h1>PigWatch</h1><p>Monitoring deteksi dan aktivitas kandang</p></div></div>
    <div class="system-pill">YOLO &middot; Optical Flow &middot; Real-time</div>
  </div>
</header>

<div class="grid">
{cards}
</div>

<script>
// ========================================================
// MULAI MONITORING
// ========================================================

async function startN(name) {{
  const input = document.getElementById('n-' + name);
  const number = parseInt(input.value, 10);
  const message = document.getElementById('m-' + name);

  if (!(number >= 1 && number <= 300)) {{
    message.textContent = 'Isi jumlah babi dulu (angka 1-300).';
    return;
  }}

  try {{
    const response = await fetch(
      '/api/start/' + encodeURIComponent(name) + '?n=' + number,
      {{ method: 'POST' }}
    );

    if (!response.ok) {{
      message.textContent =
        'Gagal memulai: ' + await response.text();
    }} else {{
      message.textContent = 'Monitoring dimulai.';
      input.dataset.init = '1';
    }}
  }} catch (error) {{
    message.textContent = 'Tidak dapat terhubung ke server.';
    console.error(error);
  }}
}}


// ========================================================
// BERHENTI MONITORING
// ========================================================

async function stopN(name) {{
  try {{
    const response = await fetch(
      '/api/stop/' + encodeURIComponent(name),
      {{ method: 'POST' }}
    );

    if (!response.ok) {{
      console.error('Gagal menghentikan monitoring:', await response.text());
    }}
  }} catch (error) {{
    console.error(error);
  }}
}}


// ========================================================
// MENAMPILKAN AKTIVITAS
// ========================================================

function renderActivities(container, activities) {{
  const entries = Object.entries(activities || {{}});

  if (entries.length === 0) {{
    container.textContent =
      'Belum ada ID tracking atau data aktivitas.';
    return;
  }}

  entries.sort((a, b) => Number(a[0]) - Number(b[0]));

  container.innerHTML = entries.map(([id, activity]) => {{
    let className = 'status-wait';

    if (activity.status === 'BERGERAK') {{
      className = 'status-moving';
    }} else if (activity.status === 'TIDAK AKTIF') {{
      className = 'status-rest';
    }} else if (activity.status === 'PERLU OBSERVASI') {{
      className = 'status-alert';
    }}

    const duration = Math.floor(
      Number(activity.inactive_seconds || 0)
    );
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    const motion = Number(
      activity.movement_px_per_frame ?? activity.movement_ratio ?? 0
    );
    const totalMotion = Number(activity.total_movement_px || 0);

    return `
      <div class="activity-row">
        <span>
          <b>Pig ID ${{id}}</b>
          <br>
          <span class="small">
            ${{motion.toFixed(2)}} px/frame<br>
            Total ${{totalMotion.toFixed(1)}} px
          </span>
        </span>

        <span style="text-align:right">
          <span class="${{className}}">
            ${{activity.status || 'MENGUMPULKAN DATA'}}
          </span>
          <br>
          <span class="small">
            Tidak bergerak: ${{minutes}}m ${{seconds}}s
          </span>
        </span>
      </div>
    `;
  }}).join('');
}}


// ========================================================
// UPDATE DASHBOARD
// ========================================================

async function tick() {{
  try {{
    const response = await fetch('/api/stats');

    if (!response.ok) {{
      throw new Error('HTTP ' + response.status);
    }}

    const data = await response.json();

    for (const [name, stats] of Object.entries(data)) {{
      document.getElementById('c-' + name).textContent = stats.count;
      document.getElementById('f-' + name).textContent = stats.fps;

      const input = document.getElementById('n-' + name);

      if (stats.active && !input.dataset.init) {{
        input.value = stats.max;
        input.dataset.init = '1';
      }}

      document.getElementById('go-' + name).textContent =
        stats.active ? 'Terapkan ulang' : 'Mulai monitoring';

      document.getElementById('st-' + name).style.display =
        stats.active ? 'inline-block' : 'none';

      const message = document.getElementById('m-' + name);

      if (!stats.active) {{
        if (
          !message.textContent.startsWith('Isi jumlah') &&
          !message.textContent.startsWith('Gagal')
        ) {{
          message.textContent =
            'Belum dimulai: isi jumlah babi lalu klik Mulai monitoring.';
        }}
      }} else if (stats.count < stats.max) {{
        message.textContent =
          'Terdeteksi ' + stats.count +
          ' dari ' + stats.max +
          ' (kurang ' + (stats.max - stats.count) +
          ': tertutup/terlewat?)';
      }} else {{
        message.textContent =
          'Semua ' + stats.max + ' babi terdeteksi.';
      }}

      const badge = document.getElementById('s-' + name);
      badge.textContent = !stats.online
        ? 'offline'
        : (stats.active ? 'memantau' : 'siap');

      badge.className = 'badge ' + (stats.online ? 'on' : 'off');

      renderActivities(
        document.getElementById('a-' + name),
        stats.activities
      );
    }}
  }} catch (error) {{
    console.error('Gagal mengambil statistik:', error);
  }}
}}


// Update dashboard setiap satu detik.
setInterval(tick, 1000);
tick();
</script>

</body>
</html>
"""


# ============================================================
# 12. MENJALANKAN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
