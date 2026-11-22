#!/usr/bin/env python3
"""Phase 3/4 서빙 파이프라인 통합 E2E 검증 스크립트.

실제 11/13 · 11/16 로봇 카메라 사진 6장 위에서:
  1) MOD-DR-Safe (FastF2Extractor[beam1_fp16] + EgVla2FMPolicy[A3C_AG23_s0] + EgVla2Controller[S2감속+스파이크필터])
  2) VLM-E2E-SoftLoc (FastF2Extractor[softloc_fp16] + EgVla2FMPolicy[A3C_AG23_s0] + EgVla2Controller[S2감속+스파이크필터])
  3) YahBoom 구동계 어댑터 & 1프레임 카메라 면적 보정 단위 테스트
를 실제 GPU 서빙 경로 그대로 통과시켜 지연시간(ms), BBox, 출력 명령, 정지 판정을 검증한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
from robovlm_nav.serve.egvla2_inference_server import EgVla2Service
from robovlm_nav.serve.egvla2_only_logic import EgVla2Controller
from robovlm_nav.serve.egvla2_softloc_extractor import FastF2Extractor
from robovlm_nav.serve.yahboom_adapter import calibrate_near_area, serbot_to_yahboom_twist


def main() -> None:
    # 1) YahBoom 어댑터 단위 테스트
    tw_go = serbot_to_yahboom_twist(1.0, 0.35, 0.0, speed_scale=0.8)
    assert tw_go["mode"] == "drive" and tw_go["linear_x"] > 0 and tw_go["angular_z"] > 0, tw_go
    tw_rot = serbot_to_yahboom_twist(0.0, 0.0, 0.25, speed_scale=0.7)
    assert tw_rot["mode"] == "rotate" and tw_rot["angular_z"] < 0, tw_rot  # Serbot +az(우회전) -> ROS2 -angular_z
    cal = calibrate_near_area(0.25, 0.18)
    assert abs(cal["yahboom_near_area"] - 0.18) < 1e-3, cal
    print(f"[PASS] YahBoom adapter: drive={tw_go} | rotate={tw_rot} | calib={cal}")

    # 2) 실제 로봇 세션 사진 6장 로드
    scenes = json.loads((ROOT / "docs/v5/egvla2_fm/registry_decision_anatomy.json").read_text(encoding="utf-8"))
    ckpt = ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt"
    policy = EgVla2FMPolicy(ckpt, k_samples=8, speed_source="shadow")

    # 3) 두 모드(beam1_fp16, softloc_fp16) 실제 서빙 서비스(EgVla2Service) 종단 검증
    ext = FastF2Extractor(mode="beam1_fp16", local_files_only=True)
    for mode in ("beam1_fp16", "softloc_fp16"):
        ext.mode = mode
        print(f"\n=== End-to-End Serving Verification: mode={mode} (A3C_AG23_s0 + S2 Decel + Spike Filter) ===")
        lats = []
        for idx, sc in enumerate(scenes):
            img_path = ROOT / "docs/v5" / sc["img_src"]
            rgb = np.array(Image.open(img_path).convert("RGB"))
            ctrl = EgVla2Controller(near_area=0.25, near_frames=2, decel_start_area=0.14, decel_min_scale=0.35, jump_area_ratio=2.5)
            svc = EgVla2Service(ext, policy, ctrl, meta=dict(ckpt=ckpt.name, decode_mode=mode))
            svc.reset()
            resp = svc.predict_rgb(rgb, instruction=sc["ins"])
            if idx > 0:
                lats.append(resp["latency_ms"])
            bb = resp["bbox"]
            lx, ly = resp["action"]
            tw = serbot_to_yahboom_twist(lx, ly, resp["action_3d"][2], resp["action_speed_scale"])
            print(
                f"  {sc['id']:17s} | lat={resp['latency_ms']:5.1f}ms | bbox=(cx={bb['cx']:.2f}, a={bb['area']:.2f}, has={bb['has_bbox']}) "
                f"| serbot=[{lx:+.2f},{ly:+.2f}] sp={resp['action_speed_scale']:.2f} ({resp['predicted_label']}) "
                f"| yahboom=(v={tw['linear_x']:.2f}, w={tw['angular_z']:+.2f})"
            )
        print(f"  -> Mean E2E serving latency (excl. warmup): {np.mean(lats):.1f}ms")


if __name__ == "__main__":
    main()
