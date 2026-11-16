
import cv2
import time
import numpy as np


class ActivityAnalyzer:
    def __init__(
        self,
        movement_threshold=0.8,
        inactivity_seconds=5,
        observation_seconds=600,
        max_history=7,
    ):
        self.movement_threshold = movement_threshold
        self.inactivity_seconds = inactivity_seconds
        self.observation_seconds = observation_seconds
        self.max_history = max_history

        self.prev_gray = None
        self.prev_time = None
        self.tracks = {}

    def reset(self):
        self.prev_gray = None
        self.prev_time = None
        self.tracks.clear()

    def _get_roi(self, gray, bbox):
        h, w = gray.shape[:2]

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        return roi

    def _calculate_movement(self, flow, bbox):
        """Mean dense-flow displacement inside the pig's current track box (px/frame)."""
        roi = self._get_roi(flow, bbox)
        if roi is None:
            return 0.0

        magnitude = cv2.magnitude(roi[..., 0], roi[..., 1])
        # Paper Eq. (5): average pixel displacement over the whole pig box.
        # Keeping zero/low-flow pixels avoids inflating movement from a few noisy pixels.
        return float(np.mean(magnitude))

    def _new_track(self, now):
        return {
            "last_seen": now,
            "last_movement": now,
            "start_time": now,
            "movement_ratio": 0.0,
            "total_movement_px": 0.0,
            "movement_history": [],
            "status": "MENGAMATI"
        }

    def update(self, frame, detections):
        now = time.time()

        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Frame pertama hanya menjadi referensi.
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_time = now

            results = []

            for det in detections:
                pig_id = det.get("id")
                bbox = det.get("bbox")

                if pig_id is None or bbox is None:
                    continue

                self.tracks[pig_id] = self._new_track(now)

                results.append({
                    "id": pig_id,
                    "status": "MENGAMATI",
                    "movement_ratio": 0.0,
                    "inactive_seconds": 0.0,
                    "observation_seconds": 0.0
                })

            return results

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        results = []

        for det in detections:
            pig_id = det.get("id")
            bbox = det.get("bbox")

            if pig_id is None or bbox is None:
                continue

            if pig_id not in self.tracks:
                self.tracks[pig_id] = self._new_track(now)

                results.append({
                    "id": pig_id,
                    "status": "MENGAMATI",
                    "movement_ratio": 0.0,
                    "inactive_seconds": 0.0,
                    "observation_seconds": 0.0
                })

                continue

            track = self.tracks[pig_id]
            track["last_seen"] = now

            # Hitung flow seluruh frame sekali, lalu ambil flow di dalam bbox
            # track masing-masing babi. Ini mengurangi bias dari perubahan ukuran bbox.
            movement = self._calculate_movement(flow, bbox)
            track["total_movement_px"] += movement

            history = track["movement_history"]
            history.append(movement)

            if len(history) > self.max_history:
                history.pop(0)

            average_movement = float(np.mean(history))

            # Perbarui waktu terakhir bergerak.
            if average_movement >= self.movement_threshold:
                track["last_movement"] = now

            inactive_seconds = max(
                0.0,
                now - track["last_movement"]
            )

            observation_seconds = max(
                0.0,
                now - track["start_time"]
            )

            # Penentuan status berdasarkan gerakan aktual.
            if average_movement >= self.movement_threshold:
                activity_status = "BERGERAK"

            elif inactive_seconds >= self.inactivity_seconds:
                activity_status = "TIDAK AKTIF"

            else:
                activity_status = "MENGAMATI"

            track["movement_ratio"] = average_movement
            track["status"] = activity_status

            results.append({
                "id": pig_id,
                "status": activity_status,
                "movement_ratio": round(average_movement, 4),
                "movement_px_per_frame": round(average_movement, 4),
                "total_movement_px": round(track["total_movement_px"], 2),
                "inactive_seconds": round(inactive_seconds, 2),
                "observation_seconds": round(observation_seconds, 2)
            })

        # Hapus tracking yang sudah hilang lebih dari 10 detik.
        expired_ids = [
            pig_id
            for pig_id, track in self.tracks.items()
            if now - track["last_seen"] > 10
        ]

        for pig_id in expired_ids:
            del self.tracks[pig_id]

        # Simpan frame untuk perhitungan berikutnya.
        self.prev_gray = gray.copy()
        self.prev_time = now

        return results
