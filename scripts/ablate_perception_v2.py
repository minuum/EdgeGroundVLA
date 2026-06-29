"""
CH54 perception ablation v2 — 다중 시드 + 파인튜닝

설계:
  - PG2 레이블 = 정답 (pseudo GT, frame 1 bbox)
  - 평가 이미지 = frame 0 (PG2가 콜드스타트로 실패한 시점)
  - 세션 단위 80/20 split, N_SEEDS 번 반복 → mean ± std
  - 파인튜닝: OWL-v2, Kosmos-2를 train split으로 학습 → test split 평가

Usage:
  .venv/bin/python3 scripts/ablate_perception_v2.py [--seeds 5] [--finetune] [--models clip owlv2 kosmos florence]
"""

import argparse, io, json, random, time
from pathlib import Path
from copy import deepcopy

import h5py, numpy as np
from PIL import Image
import torch

SESSIONS_DIR = Path("docs/inference_sessions")
OUT_FILE     = Path("docs/v5/ablate_perception_v2.json")
DEVICE       = "cuda"
PHRASE       = "gray basket"


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_all_samples():
    """
    6/22~6/26 세션에서 (frame0_img, pg2_cx, pg2_has) 쌍 수집.
    frame 0: 평가 이미지 (PG2 콜드스타트로 항상 실패)
    frame 1: PG2 정답 레이블 (warmup 후 frame 0과 동일 시점)
    """
    samples = []
    for sp in sorted(SESSIONS_DIR.glob("session_2026062[246]*.h5")):
        with h5py.File(sp, "r") as f:
            if "grounding/bbox" not in f: continue
            bbox = f["grounding/bbox"][:]
            if len(bbox) < 2: continue
            # 정답: frame 1 PG2 결과
            pg2_cx  = float(bbox[1, 0])
            pg2_has = bool(bbox[1, 3] > 0.5)
            if not pg2_has: continue  # 정답 없으면 제외

            # 이미지: frame 0
            if "observations" in f:
                imgs = f["observations"]["images"]
            elif "images" in f:
                imgs = f["images"]
            else:
                continue
            if len(imgs) < 1: continue
            raw = imgs[0]
            try:
                if raw.ndim == 1:
                    img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
                else:
                    img = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
            except Exception:
                continue

            # 파인튜닝용: frame 1~N 이미지 + 레이블도 수집
            extra = []
            for t in range(1, min(len(bbox), len(imgs))):
                if bbox[t, 3] < 0.5: continue
                raw_t = imgs[t]
                try:
                    if raw_t.ndim == 1:
                        img_t = Image.open(io.BytesIO(bytes(raw_t))).convert("RGB")
                    else:
                        img_t = Image.fromarray(raw_t.astype(np.uint8)).convert("RGB")
                    extra.append({
                        "img": img_t,
                        "cx": float(bbox[t, 0]),
                        "cy": float(bbox[t, 1]),
                        "area": float(bbox[t, 2]),
                    })
                except Exception:
                    continue

            samples.append({
                "session": sp.stem,
                "img": img,
                "pg2_cx": pg2_cx,
                "extra_frames": extra,  # 파인튜닝용 추가 프레임
            })
    return samples


def cx_dir(cx):
    return "L" if cx < 0.4 else ("R" if cx > 0.6 else "C")


def eval_preds(preds, samples, label, lat_ms=None):
    n = len(samples)
    det  = sum(1 for p in preds if p["detected"])
    cx_ok = sum(1 for p, s in zip(preds, samples)
                if p["detected"] and abs(p["cx"] - s["pg2_cx"]) < 0.2)
    dir_ok = sum(1 for p, s in zip(preds, samples)
                 if p["detected"] and cx_dir(p["cx"]) == cx_dir(s["pg2_cx"]))
    lat_str = f"  lat={lat_ms:.0f}ms" if lat_ms is not None else ""
    print(f"  {label:<35} | det={det/n:.0%}  cx={cx_ok/n:.0%}  dir={dir_ok/n:.0%}  n={n}{lat_str}")
    return {"det": det/n, "cx": cx_ok/n, "dir": dir_ok/n}


# ── 모델 클래스 (로드/추론/파인튜닝) ──────────────────────────────────────────

class CLIPModel_:
    name = "CLIP"
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        self.proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.texts = [
            "a gray basket on the left side",
            "a gray basket in the center",
            "a gray basket on the right side",
            "no basket visible",
        ]

    def predict(self, samples):
        preds, lats = [], []
        for s in samples:
            t0 = time.time()
            inputs = self.proc(text=self.texts, images=s["img"],
                               return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image[0]
                probs  = logits.softmax(dim=0).cpu().numpy()
            lats.append((time.time()-t0)*1000)
            best = int(np.argmax(probs))
            if best == 3 or probs[best] < 0.3:
                preds.append({"detected": False, "cx": None})
            else:
                preds.append({"detected": True, "cx": [0.2, 0.5, 0.8][best]})
        return preds, float(np.mean(lats))

    def finetune(self, train_samples):
        """CLIP: 방향 분류기 linear probe (vision encoder frozen)."""
        import torch.nn as nn
        from torch.optim import AdamW

        # 특징 추출 (frozen)
        all_imgs = [s["img"] for s in train_samples]
        all_imgs += [f["img"] for s in train_samples for f in s["extra_frames"]]
        all_labels = []
        for s in train_samples:
            cx = s["pg2_cx"]
            all_labels.append(0 if cx < 0.4 else (2 if cx > 0.6 else 1))
        for s in train_samples:
            for f in s["extra_frames"]:
                cx = f["cx"]
                all_labels.append(0 if cx < 0.4 else (2 if cx > 0.6 else 1))

        feats = []
        with torch.no_grad():
            for img in all_imgs:
                inputs = self.proc(images=img, return_tensors="pt").to(DEVICE)
                feat = self.model.get_image_features(**inputs)
                feats.append(feat.squeeze(0))
        feats = torch.stack(feats)
        labels = torch.tensor(all_labels, device=DEVICE)

        probe = nn.Linear(feats.shape[1], 3).to(DEVICE)
        opt   = AdamW(probe.parameters(), lr=1e-3)
        for _ in range(50):
            loss = nn.CrossEntropyLoss()(probe(feats), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        self._probe = probe

    def predict_ft(self, samples):
        if not hasattr(self, "_probe"):
            return self.predict(samples)
        preds, lats = [], []
        for s in samples:
            t0 = time.time()
            inputs = self.proc(images=s["img"], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat  = self.model.get_image_features(**inputs)
                logit = self._probe(feat)
                cls   = int(logit.argmax())
            lats.append((time.time()-t0)*1000)
            preds.append({"detected": True, "cx": [0.2, 0.5, 0.8][cls]})
        return preds, float(np.mean(lats))


class OWLv2Model_:
    name = "OWL-v2"
    def __init__(self):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble").to(DEVICE)
        self.proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

    def _infer(self, img):
        inputs = self.proc(text=[[PHRASE]], images=img,
                           return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs)
        W = img.width
        res = self.proc.post_process_object_detection(
            out, threshold=0.1, target_sizes=[(img.height, W)])[0]
        boxes  = res["boxes"]
        scores = res["scores"].cpu().numpy()
        if len(boxes) == 0:
            return None
        best = int(np.argmax(scores))
        x1, _, x2, _ = boxes[best].cpu().tolist()
        return (x1 + x2) / 2.0 / W

    def predict(self, samples):
        preds, lats = [], []
        for s in samples:
            t0 = time.time()
            cx = self._infer(s["img"])
            lats.append((time.time()-t0)*1000)
            preds.append({"detected": cx is not None, "cx": cx})
        return preds, float(np.mean(lats))

    def finetune(self, train_samples):
        """OWL-v2: 마지막 classification head만 재학습 (few-shot)."""
        from torch.optim import AdamW
        import torch.nn.functional as F

        # 학습 데이터: frame 0 + extra_frames
        imgs, bboxes_gt = [], []
        for s in train_samples:
            imgs.append(s["img"])
            W, H = s["img"].width, s["img"].height
            cx = s["pg2_cx"]
            # normalized cx → xyxy (area 기준 20% 너비 가정)
            x1 = max(0, cx - 0.1); x2 = min(1, cx + 0.1)
            bboxes_gt.append([x1 * W, 0.2 * H, x2 * W, 0.8 * H])
            for f in s["extra_frames"]:
                imgs.append(f["img"])
                cx = f["cx"]
                x1 = max(0, cx - 0.1); x2 = min(1, cx + 0.1)
                bboxes_gt.append([x1 * W, 0.2 * H, x2 * W, 0.8 * H])

        # OWL-v2 classification head 파인튜닝 (class_head only)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.class_head.parameters():
            p.requires_grad_(True)

        opt = AdamW(self.model.class_head.parameters(), lr=5e-5)
        for epoch in range(10):
            total_loss = 0
            for img, bbox in zip(imgs, bboxes_gt):
                inputs = self.proc(text=[[PHRASE]], images=img,
                                   return_tensors="pt").to(DEVICE)
                out    = self.model(**inputs)
                scores = out.logits[0, :, 0]  # (num_queries,)
                # target: query closest to GT box gets score=1, rest=0
                W, H = img.width, img.height
                boxes_pred = out.pred_boxes[0].detach()  # (Q, 4) cx,cy,w,h normalized
                gt_cx = (bbox[0] + bbox[2]) / 2.0 / W
                dists = (boxes_pred[:, 0] - gt_cx).abs()
                pos_idx = dists.argmin()
                target = torch.zeros_like(scores)
                target[pos_idx] = 1.0
                loss = F.binary_cross_entropy_with_logits(scores, target)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"      [OWL-v2 FT] epoch={epoch+1} loss={total_loss/len(imgs):.4f}")
        # 파인튜닝 후 전체 파라미터 고정 해제
        for p in self.model.parameters():
            p.requires_grad_(True)

    def predict_ft(self, samples):
        return self.predict(samples)


class KosmosModel_:
    name = "Kosmos-2"
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        local = Path(".vlms/kosmos-2-patch14-224")
        self.model = AutoModelForVision2Seq.from_pretrained(str(local)).to(DEVICE)
        self.proc  = AutoProcessor.from_pretrained(str(local))

    def _infer(self, img):
        prompt = "<grounding><phrase>gray basket</phrase>"
        inputs = self.proc(text=prompt, images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        out = self.proc.decode(ids[0], skip_special_tokens=False)
        result = self.proc.post_process_generation(out, cleanup_and_extract=True)
        entities = result[1] if isinstance(result, tuple) and len(result) > 1 else []
        for ent in entities:
            bboxes = ent[2] if len(ent) >= 3 else []
            if bboxes:
                bbox = bboxes[0]
                return (bbox[0] + bbox[2]) / 2.0
        return None

    def predict(self, samples):
        preds, lats = [], []
        for s in samples:
            t0 = time.time()
            cx = self._infer(s["img"])
            lats.append((time.time()-t0)*1000)
            preds.append({"detected": cx is not None, "cx": cx})
        return preds, float(np.mean(lats))

    def finetune(self, train_samples):
        """Kosmos-2: LoRA-style — image projection layer만 재학습."""
        from torch.optim import AdamW

        # image_projection만 학습
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.model.embed_tokens.parameters():
            p.requires_grad_(False)
        # vision projection
        for name, p in self.model.named_parameters():
            if "image_to_text_projection" in name or "vision_model" in name:
                p.requires_grad_(True)

        opt = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5
        )
        train_imgs = [s["img"] for s in train_samples]
        train_imgs += [f["img"] for s in train_samples for f in s["extra_frames"]]
        train_cxs  = [s["pg2_cx"] for s in train_samples]
        train_cxs  += [f["cx"] for s in train_samples for f in s["extra_frames"]]

        prompt = "<grounding><phrase>gray basket</phrase>"
        for epoch in range(5):
            total = 0
            for img, cx_gt in zip(train_imgs, train_cxs):
                # 정답 bbox 문자열 생성 (Kosmos-2 포맷)
                x1 = max(0.0, cx_gt - 0.1); x2 = min(1.0, cx_gt + 0.1)
                target_text = (
                    f"<grounding><phrase>gray basket</phrase>"
                    f"<object><patch_index_{int(x1*32):04d}>"
                    f"<patch_index_{int(x2*32):04d}></object>"
                )
                inputs = self.proc(text=prompt, images=img, return_tensors="pt").to(DEVICE)
                labels_enc = self.proc.tokenizer(
                    target_text, return_tensors="pt"
                ).input_ids.to(DEVICE)
                out = self.model(**inputs, labels=labels_enc)
                loss = out.loss
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            if (epoch + 1) % 2 == 0:
                print(f"      [Kosmos-2 FT] epoch={epoch+1} loss={total/len(train_imgs):.4f}")

    def predict_ft(self, samples):
        return self.predict(samples)


class FlorenceModel_:
    name = "Florence-2"
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True).to(DEVICE)
        self.proc  = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True)

    def _infer(self, img):
        task = "<OPEN_VOCABULARY_DETECTION>"
        inputs = self.proc(text=task + PHRASE, images=img,
                           return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=128)
        out = self.proc.decode(ids[0], skip_special_tokens=False)
        result = self.proc.post_process_generation(
            out, task=task, image_size=(img.height, img.width))
        bboxes = result.get(task, {}).get("bboxes", [])
        if not bboxes: return None
        x1, _, x2, _ = bboxes[0]
        return (x1 + x2) / 2.0 / img.width

    def predict(self, samples):
        preds, lats = [], []
        for s in samples:
            t0 = time.time()
            cx = self._infer(s["img"])
            lats.append((time.time()-t0)*1000)
            preds.append({"detected": cx is not None, "cx": cx})
        return preds, float(np.mean(lats))

    def finetune(self, train_samples):
        """Florence-2: task-specific fine-tuning."""
        from torch.optim import AdamW
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.language_model.lm_head.parameters():
            p.requires_grad_(True)
        for name, p in self.model.named_parameters():
            if "image_projection" in name:
                p.requires_grad_(True)

        opt = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        task = "<OPEN_VOCABULARY_DETECTION>"
        train_imgs = [s["img"] for s in train_samples]
        train_imgs += [f["img"] for s in train_samples for f in s["extra_frames"]]
        train_cxs  = [s["pg2_cx"] for s in train_samples]
        train_cxs  += [f["cx"] for s in train_samples for f in s["extra_frames"]]

        for epoch in range(5):
            total = 0
            for img, cx_gt in zip(train_imgs, train_cxs):
                W, H = img.width, img.height
                x1 = max(0, cx_gt - 0.1) * W; x2 = min(1, cx_gt + 0.1) * W
                target = f"{PHRASE}<loc_{int(x1)}><loc_0><loc_{int(x2)}><loc_{H}>"
                inputs = self.proc(text=task + PHRASE, images=img,
                                   return_tensors="pt").to(DEVICE)
                labels = self.proc.tokenizer(
                    target, return_tensors="pt").input_ids.to(DEVICE)
                out = self.model(**inputs, labels=labels)
                loss = out.loss
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            if (epoch + 1) % 2 == 0:
                print(f"      [Florence FT] epoch={epoch+1} loss={total/len(train_imgs):.4f}")

    def predict_ft(self, samples):
        return self.predict(samples)


MODEL_REGISTRY = {
    "clip":    CLIPModel_,
    "owlv2":   OWLv2Model_,
    "kosmos":  KosmosModel_,
    "florence":FlorenceModel_,
}


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int,  default=5)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--models",   nargs="+", default=["clip","owlv2","kosmos","florence"])
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    samples = load_all_samples()
    print(f"전체 샘플: {len(samples)}개\n")

    all_results = {}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"[{model_name}] 알 수 없는 모델, 스킵")
            continue

        print(f"\n{'='*60}")
        print(f"[{model_name}] 로딩...")
        t0 = time.time()
        model_obj = MODEL_REGISTRY[model_name]()
        print(f"[{model_name}] 로딩 완료 {time.time()-t0:.1f}s")

        seed_zero_shot, seed_ft = [], []

        for seed in range(args.seeds):
            rng = random.Random(seed)
            idxs = list(range(len(samples)))
            rng.shuffle(idxs)
            n_test  = max(1, int(len(idxs) * args.test_ratio))
            test_i  = idxs[:n_test]
            train_i = idxs[n_test:]

            test_s  = [samples[i] for i in test_i]
            train_s = [samples[i] for i in train_i]

            # Zero-shot
            preds_zs, lat_zs = model_obj.predict(test_s)
            m_zs = eval_preds(preds_zs, test_s,
                              f"  [seed={seed}] zero-shot", lat_zs)
            seed_zero_shot.append(m_zs)

            # Fine-tune
            if args.finetune:
                print(f"  [seed={seed}] 파인튜닝 (train={len(train_s)}세션)...")
                model_obj.finetune(train_s)
                preds_ft, lat_ft = model_obj.predict_ft(test_s)
                m_ft = eval_preds(preds_ft, test_s,
                                  f"  [seed={seed}] fine-tuned", lat_ft)
                seed_ft.append(m_ft)

        def agg(metrics, tag):
            if not metrics: return
            for key in ["det", "cx", "dir"]:
                vals = [m[key] for m in metrics]
                print(f"  [{model_name}] {tag} {key}: {np.mean(vals):.1%} ± {np.std(vals):.1%}")

        print(f"\n  ── {model_name} 집계 ({args.seeds} seeds) ──")
        agg(seed_zero_shot, "zero-shot")
        if seed_ft:
            agg(seed_ft, "fine-tuned")

        all_results[model_name] = {
            "zero_shot": seed_zero_shot,
            "fine_tuned": seed_ft,
        }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n결과 저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
