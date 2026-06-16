import json
import os
from datetime import datetime
import numpy as np

H5_SAVE_DIR = os.path.join(
    os.getenv("VLA_ROOT", os.getenv("HOME", "/tmp")),
    "docs/inference_sessions"
)


class InferenceLogger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            default_root = os.getenv("VLA_ROOT", os.getenv("HOME", "/tmp"))
            log_dir = os.path.join(default_root, "docs/inference_reports")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        self._frames: list[np.ndarray] = []
        self.data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": "unknown",
            "instruction": "unknown",
            "history": [],
        }

    def start_session(self, model_name: str, instruction: str, instruction_mode=None):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        self._frames = []
        self.data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "instruction": instruction,
            "instruction_mode": instruction_mode,
            "history": [],
        }
        print(f"📝 세션 시작: {self.log_file}")

    def update_instruction(self, instruction: str):
        if hasattr(self, "data"):
            self.data["instruction"] = instruction

    def _to_rgb_array(self, image) -> np.ndarray | None:
        """PIL Image 또는 numpy array → uint8 RGB numpy array."""
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                return np.array(image.convert("RGB"), dtype=np.uint8)
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                    return image.astype(np.uint8)
        except Exception:
            pass
        return None

    def log_step(self, step_idx: int, action, latency, chunk=None, image=None, **extra):
        if not hasattr(self, "data") or self.data is None:
            self.data = {"history": []}

        step_data = {
            "step": step_idx,
            "timestamp": datetime.now().isoformat(),
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "latency_ms": latency,
        }
        if chunk is not None:
            step_data["chunk_preview"] = chunk.tolist() if isinstance(chunk, np.ndarray) else chunk

        for k, v in extra.items():
            if v is None:
                continue
            step_data[k] = v.tolist() if isinstance(v, np.ndarray) else v

        if image is not None:
            arr = self._to_rgb_array(image)
            if arr is not None:
                self._frames.append(arr)
                step_data["frame_idx"] = len(self._frames) - 1

        self.data["history"].append(step_data)

    def end_session(self, status: str = "completed") -> str | None:
        if not hasattr(self, "data") or self.data is None:
            print("⚠️ 저장할 세션 없음")
            return None

        self.data["status"] = status
        self.data["end_time"] = datetime.now().isoformat()

        if self.data["history"]:
            latencies = [
                h["latency_ms"] for h in self.data["history"]
                if isinstance(h.get("latency_ms"), (int, float))
            ]
            labels = [h.get("predicted_label") for h in self.data["history"] if h.get("predicted_label")]
            label_counts: dict = {}
            for lb in labels:
                label_counts[lb] = label_counts.get(lb, 0) + 1

            self.data["summary"] = {
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "total_steps": len(self.data["history"]),
                "n_frames": len(self._frames),
                "last_action": self.data["history"][-1]["action"],
                "action_label_counts": label_counts,
            }

        # ── JSON 저장 ────────────────────────────────────────────────────────
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"✅ JSON 저장: {self.log_file}")

        # ── H5 저장 (데이터셋 동일 포맷) ─────────────────────────────────────
        if self._frames:
            try:
                import h5py
                os.makedirs(H5_SAVE_DIR, exist_ok=True)
                h5_path = os.path.join(H5_SAVE_DIR, f"session_{self.session_id}.h5")
                imgs = np.stack(self._frames, axis=0)  # (N, H, W, 3)

                actions = []
                for h in self.data["history"]:
                    a = h.get("action", [0.0, 0.0, 0.0])
                    actions.append(a if isinstance(a, list) else list(a))

                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("observations/images", data=imgs, compression="gzip")
                    f.create_dataset("actions", data=np.array(actions, dtype=np.float32))
                    f.attrs["session_id"] = self.session_id
                    f.attrs["model_name"] = self.data.get("model_name", "")
                    f.attrs["instruction"] = self.data.get("instruction", "")
                    f.attrs["n_frames"] = len(imgs)
                    f.attrs["status"] = status

                print(f"✅ H5 저장: {h5_path}  ({len(imgs)} frames, {imgs.shape})")
                self.data["h5_path"] = h5_path
            except Exception as e:
                print(f"⚠️ H5 저장 실패: {e}")

        return self.log_file


# Singleton
_logger = InferenceLogger()


def get_logger():
    return _logger
