#!/usr/bin/env python3
"""
2026-07-08 soda 발견: H5 raw 이미지가 BGR로 저장되는데 학습 로더는 RGB로 가정하고
읽음 (mobile_vla_data_collector.py: cv_bridge.compressed_imgmsg_to_cv2(..., "bgr8")로
받은 배열을 JPEG 인코딩 없이 그대로 H5에 저장 -> nav_h5_dataset_impl.py의 raw-array
분기는 변환 없이 그대로 RGB로 소비). 241/241(100%) 에피소드가 raw 저장이라 전부 영향권.

실시간 추론 경로(api_client_node.py)는 cv2.imencode로 JPEG 인코딩하는 과정에서
OpenCV가 BGR로 올바르게 처리하므로 색이 정상 — 학습 데이터만 뒤바뀐 채였음.

이 스크립트는 채널을 [:,:,::-1]로 뒤집어(BGR->RGB) vis_feat 캐시를 재생성.
정규화 버그 수정판(exp71_vis_cache_normfixed.pt)과 동일한 구조 + 색상 수정.
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_FILE_COLORFIXED = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache_colorfixed.pt"


def main():
    with open(exp71.ANN_PATH) as f:
        ann = json.load(f)
    print("FrozenCLIPV2 로드...")
    enc = exp71.FrozenCLIPV2(exp71.VLM_PATH, exp71.STAGE1_PT, DEVICE).eval()

    episodes = []
    for i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
        # BGR -> RGB 채널 반전 (soda 발견 반영)
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]][:, :, ::-1].astype("uint8"))
                    for fr in frames]
        vis = F.normalize(enc.encode_batch(pil_imgs, DEVICE), dim=-1).cpu()
        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False))) for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        episodes.append({"stem": h5_path.stem, "path_type": ep["path_type"],
                          "bboxes": bboxes, "vis": vis, "gts": gts})
        if (i + 1) % 20 == 0:
            print(f"  encoded {i+1}/{len(ann)}")

    torch.save(episodes, CACHE_FILE_COLORFIXED)
    del enc
    torch.cuda.empty_cache()
    print(f"저장 -> {CACHE_FILE_COLORFIXED} ({len(episodes)}ep)")


if __name__ == "__main__":
    main()
