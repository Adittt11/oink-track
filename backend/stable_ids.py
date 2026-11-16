"""
stable_ids.py - Pelacak ID tetap untuk kandang dengan jumlah babi yang diketahui.

Ide: jika di kandang ada N babi, hanya ada ID 1..N. Setiap frame, deteksi dicocokkan ke
"slot" ID memakai posisi (IoU + jarak), dan warna kulit (histogram HSV). Slot yang sedang
tidak terdeteksi tetap disimpan, sehingga babi yang muncul lagi mendapat ID yang SAMA
(tidak membuat ID baru seperti tracker biasa).

Batasan: kalau dua babi bertumpuk lama dan mirip warnanya, ID bisa tertukar.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

PALETTE = [
    (66, 135, 245), (60, 200, 90), (245, 170, 50), (200, 80, 200), (50, 210, 220),
    (240, 90, 90), (150, 150, 255), (120, 220, 160), (255, 140, 190), (180, 180, 60),
]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (A,4) b: (B,4) xyxy -> (A,B)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _hist(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    # ambil bagian tengah kotak (kurangi latar belakang di tepi)
    mx, my = (x2 - x1) // 6, (y2 - y1) // 6
    crop = frame[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 6], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist.astype(np.float32)


class _Slot:
    __slots__ = ("box", "hist", "lost")

    def __init__(self, box: np.ndarray, hist: np.ndarray | None):
        self.box, self.hist, self.lost = box, hist, 0


class StableIDs:
    def __init__(self, n: int, w_iou: float = 0.35, w_dist: float = 0.30, w_app: float = 0.35,
                 new_thresh: float = 0.75):
        self.n = n
        self.w = (w_iou, w_dist, w_app)
        self.new_thresh = new_thresh
        self.slots: dict[int, _Slot] = {}

    def _cost(self, frame, boxes, hists):
        ids = list(self.slots)
        sb = np.array([self.slots[i].box for i in ids], dtype=float)
        iou = _iou_matrix(sb, boxes)
        sc = np.stack([(sb[:, 0] + sb[:, 2]) / 2, (sb[:, 1] + sb[:, 3]) / 2], 1)
        dc = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        dist = np.linalg.norm(sc[:, None] - dc[None], axis=2)
        sdiag = np.hypot(sb[:, 2] - sb[:, 0], sb[:, 3] - sb[:, 1])
        ddiag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        scale = (sdiag[:, None] + ddiag[None]) / 2
        lost = np.array([self.slots[i].lost for i in ids])[:, None]
        dist_n = np.clip(dist / (scale * (1 + 0.05 * lost) + 1e-9), 0, 1)
        app = np.full(iou.shape, 0.5)
        for a, i in enumerate(ids):
            hs = self.slots[i].hist
            if hs is None:
                continue
            for b, hd in enumerate(hists):
                if hd is not None:
                    sim = cv2.compareHist(hs, hd, cv2.HISTCMP_CORREL)
                    app[a, b] = 1.0 - float(np.clip(sim, 0.0, 1.0))
        w1, w2, w3 = self.w
        return ids, w1 * (1 - iou) + w2 * dist_n + w3 * app

    def update(self, frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray | None = None) -> list:
        """Kembalikan list ID (sejajar dengan boxes). None = deteksi dibuang (melebihi N)."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        k = len(boxes)
        if k == 0:
            for s in self.slots.values():
                s.lost += 1
            return []

        hists = [_hist(frame, b) for b in boxes]
        out: list = [None] * k
        matched_slots: set[int] = set()
        matched_dets: set[int] = set()

        if self.slots:
            ids, cost = self._cost(frame, boxes, hists)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] > self.new_thresh and len(self.slots) < self.n:
                    continue  # terlalu jauh dari semua slot & masih ada tempat -> babi baru
                sid = ids[r]
                out[c] = sid
                matched_slots.add(sid)
                matched_dets.add(c)

        # deteksi yang belum punya slot -> slot baru (urut dari kiri ke kanan), selama masih < N
        cand = list(set(range(k)) - matched_dets)
        room = self.n - len(self.slots)
        if confs is not None and len(cand) > room:  # kelebihan kandidat: pilih confidence tertinggi
            cand = sorted(cand, key=lambda j: -float(confs[j]))[:max(room, 0)]
        for c in sorted(cand, key=lambda j: boxes[j][0]):
            if len(self.slots) >= self.n:
                break
            sid = len(self.slots) + 1
            self.slots[sid] = _Slot(boxes[c].copy(), hists[c])
            out[c] = sid
            matched_slots.add(sid)

        # perbarui slot yang cocok
        ious_det = _iou_matrix(boxes, boxes) if k > 1 else np.zeros((1, 1))
        for c, sid in enumerate(out):
            if sid is None or sid not in self.slots:
                continue
            s = self.slots[sid]
            s.box = boxes[c].copy()
            s.lost = 0
            overlapped = k > 1 and (np.sum(ious_det[c] > 0.3) > 1)
            if hists[c] is not None and not overlapped:  # jangan cemari warna saat bertumpuk
                s.hist = hists[c] if s.hist is None else (0.9 * s.hist + 0.1 * hists[c]).astype(np.float32)
        for sid, s in self.slots.items():
            if sid not in matched_slots:
                s.lost += 1
        return out


def draw_boxes(frame: np.ndarray, boxes, ids, confs=None, thickness: int = 2) -> np.ndarray:
    img = frame.copy()
    for i, (b, tid) in enumerate(zip(boxes, ids)):
        if tid is None:  # deteksi yang dibuang (melebihi jumlah babi)
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        color = PALETTE[(tid - 1) % len(PALETTE)] if tid >= 1 else (180, 180, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"pig {tid}" if tid >= 1 else "pig"
        if confs is not None:
            label += f" {float(confs[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img
