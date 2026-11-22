#!/usr/bin/env python3
"""3계열 아키텍처(MOD-* vs ENC-E2E-* vs VLM-E2E-*) 실시간 학습 현황 및
실제 로봇 추론 세션 6개 장면 비교 페이지 생성기.

출력:
  - docs/v5/egvla2_fm/family_arch_compare.json
  - docs/v5/family_arch_compare.html
"""
from __future__ import annotations

import copy
import html
import json
import math
import re
import sys
import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from scripts.eval_e2e_on_real_a3 import (
    PretrainedAlignedProj,
    SpatialGroundingE2E,
    stage3_tokens,
)

ANATOMY_JSON = ROOT / "docs/v5/egvla2_fm/registry_decision_anatomy.json"
BEAM_JSON = ROOT / "docs/v5/egvla2_fm/beam_bench.json"
E2E_V2_JSON = ROOT / "docs/v5/egvla2_fm/e2e_v2_results.json"
TRACK_A_JSON = ROOT / "docs/v5/egvla2_fm/track_a_dec1pass_results.json"
OUT_JSON = ROOT / "docs/v5/egvla2_fm/family_arch_compare.json"
OUT_HTML = ROOT / "docs/v5/family_arch_compare.html"
TASK337_LOG = Path(
    "/home/minum/.gemini/antigravity/brain/7631c4fd-4215-43c0-b11b-29048a50e46a/.system_generated/tasks/task-337.log"
)


@torch.no_grad()
def predict_fm_single(head_mod, bw_12: np.ndarray, feat_768: torch.Tensor, steps: int = 10) -> tuple[float, float]:
    """단일 프레임에 대해 결정적 0-초기값 Flow Matching Euler 적분으로 [lx, ly] 예측."""
    head_mod.eval()
    b = torch.tensor(bw_12, dtype=torch.float32).unsqueeze(0)
    f = feat_768.float().unsqueeze(0) if feat_768.ndim == 1 else feat_768.float()
    x = torch.zeros(1, 8, 3, dtype=torch.float32)
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((1,), s * dt, dtype=torch.float32)
        x = x + dt * head_mod(b, f, x, t)
    cmd = x[0, 0].cpu().numpy()
    return float(cmd[0]), float(cmd[1])


def cmd_stats(lx: float, ly: float) -> dict:
    mag = float(math.hypot(lx, ly))
    ang = float(math.degrees(math.atan2(ly, lx))) if mag > 1e-6 else 0.0
    return dict(
        cmd=[round(lx, 3), round(ly, 3)],
        mag=round(mag, 3),
        ang_deg=round(ang, 1),
        stop=bool(mag < 0.2),
    )


def find_frame_indices() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """11/13 및 11/16 실기 캐시에서 (sid, fi) -> 전역 인덱스 및 BBox 윈도우 반환."""
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))

    idx13, bw13_list = {}, []
    cur = 0
    for x in D13:
        sid = x["sid"]
        vis = C13[f"{sid}|vis"]
        raw = C13[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        bw = AB.window(raw, 3)
        bw[raw[:, 3] < 0.5] = 0
        bw13_list.append(bw)
        for fi in range(len(vis)):
            idx13[(sid, fi)] = cur + fi
        cur += len(vis)

    idx16, bw16_list = {}, []
    cur = 0
    for sid in S16.keys():
        vis = C16[f"{sid}|vis"]
        raw = C16[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        bw = AB.window(raw, 3)
        bw[raw[:, 3] < 0.5] = 0
        bw16_list.append(bw)
        for fi in range(len(vis)):
            idx16[(sid, fi)] = cur + fi
        cur += len(vis)

    return idx13, idx16, np.concatenate(bw13_list, 0), np.concatenate(bw16_list, 0)


def parse_task337_progress() -> dict:
    """백그라운드 Phase 2 학습(task-337) 로그에서 실시간 진행률을 파싱."""
    if TRACK_A_JSON.exists():
        return dict(state="done", pct=100, detail="Phase 2 완료 — VLM-E2E-Hidden & VLM-E2E-SoftLoc 평가 결과 반영됨")
    if not TASK337_LOG.exists():
        return dict(state="wait", pct=0, detail="로그 대기 중")
    txt = TASK337_LOG.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"train\s+(\d+)/(\d+)\s+\((\d+)s\)", txt)
    if m:
        cur, tot, sec = int(m[-1][0]), int(m[-1][1]), int(m[-1][2])
        pct = min(92, max(5, int(cur / max(1, tot) * 85)))
        return dict(
            state="running",
            pct=pct,
            detail=f"Florence-2 fp16 Decoder 1-Pass & Soft-Loc 특성 추출 중: train {cur:,}/{tot:,} ({sec}s 경과)",
        )
    if "[2/3]" in txt:
        return dict(state="running", pct=88, detail="VLM-E2E-Hidden (Track A1, 120ep) Action Expert 학습 중")
    if "[3/3]" in txt:
        return dict(state="running", pct=95, detail="VLM-E2E-SoftLoc (Track A2, 120ep) Action Expert 학습 중")
    if "Phase 2" in txt:
        return dict(state="running", pct=10, detail="Phase 2 시작 — Florence-2 모델 로드 및 특성 추출 중")
    return dict(state="running", pct=5, detail="Phase 1/2 파이프라인 실행 중")


def main() -> None:
    anatomy = json.loads(ANATOMY_JSON.read_text(encoding="utf-8"))
    beam_bench = json.loads(BEAM_JSON.read_text(encoding="utf-8")) if BEAM_JSON.exists() else {}
    e2e_v2 = json.loads(E2E_V2_JSON.read_text(encoding="utf-8")) if E2E_V2_JSON.exists() else {}
    track_a = json.loads(TRACK_A_JSON.read_text(encoding="utf-8")) if TRACK_A_JSON.exists() else {}
    prog = parse_task337_progress()

    idx13, idx16, BW13, BW16 = find_frame_indices()
    s2_npz = np.load(ROOT / "runs/e2e_cache/real_1113_1116_stage2.npz")

    # 6개 장면에 해당하는 stage2 텐서만 추출 (CPU에서 초고속 추론, GPU task-337 방해 없음)
    scene_s2 = []
    scene_bw = []
    scene_gidx = []
    for sc in anatomy:
        yr, sid, fi = sc["yr"], sc["sid"], int(sc["fi"])
        if yr == "1113":
            gi = idx13[(sid, fi)]
            scene_s2.append(s2_npz["s2_13"][gi])
            scene_bw.append(BW13[gi])
        else:
            gi = idx16[(sid, fi)]
            scene_s2.append(s2_npz["s2_16"][gi])
            scene_bw.append(BW16[gi])
        scene_gidx.append((yr, gi))
    S2_6 = torch.from_numpy(np.stack(scene_s2)).float()  # (6, 256, 512) -> stage3 입력

    # Florence-2 vision tower stage3 (CPU, local cache)
    fl_base = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()

    def make_stage3_lora(lora_sd: dict | None):
        vt = copy.deepcopy(fl_base.vision_tower)
        if lora_sd is not None:
            lcfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=r".*blocks\.3\..*(qkv|proj)",
                lora_dropout=0.05,
                bias="none",
            )
            vt = get_peft_model(vt, lcfg)
            vt.load_state_dict(lora_sd, strict=False)
            vt.eval()
            return vt.get_base_model()
        return vt.eval()

    # 1) ENC-E2E-1 (EG3-04, B1 orig — Sequential(LayerNorm(1024), Linear(1024,768)) 사용)
    ck_b1 = torch.load(ROOT / "runs/egvla2_e2e_lora/EG3_04_B1_orig.pt", map_location="cpu", weights_only=False)
    b3_b1 = make_stage3_lora(None)
    proj_b1 = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 768))
    proj_b1.load_state_dict(ck_b1["proj"])
    head_b1 = HD.PiZeroFM(12, 768)
    head_b1.load_state_dict(ck_b1["head"])
    with torch.no_grad():
        f_b1 = proj_b1(stage3_tokens(b3_b1, S2_6).mean(1))

    # 2) ENC-E2E-AG (EG3-06)
    ck_ag = torch.load(ROOT / "runs/egvla2_e2e_lora/EG3_06_E2E_LoRA_AG.pt", map_location="cpu", weights_only=False)
    b3_ag = make_stage3_lora(ck_ag["lora"])
    proj_ag = PretrainedAlignedProj(fl_base)
    proj_ag.load_state_dict(ck_ag["proj"])
    head_ag = HD.PiZeroFM(12, 768)
    head_ag.load_state_dict(ck_ag["head"])
    with torch.no_grad():
        f_ag = proj_ag(stage3_tokens(b3_ag, S2_6).mean(1))

    # 3) ENC-E2E-Aux (EG3-07, 내부 Aux BBox 동시 예측)
    ck_aux = torch.load(ROOT / "runs/egvla2_e2e_lora/EG3_07_E2E_LoRA_SpatialAux.pt", map_location="cpu", weights_only=False)
    b3_aux = make_stage3_lora(ck_aux["lora"])
    sp_aux = SpatialGroundingE2E(fl_base)
    sp_aux.load_state_dict(ck_aux["sp_mod"])
    head_aux = HD.PiZeroFM(12, 768)
    head_aux.load_state_dict(ck_aux["head"])
    with torch.no_grad():
        f_aux, aux_bb_6 = sp_aux(stage3_tokens(b3_aux, S2_6))
        aux_bb_6 = aux_bb_6.numpy()

    # 4) VLM-E2E-Hidden (EG3-08) & VLM-E2E-SoftLoc (EG3-09) — 완료 시 자동 로드
    dec_cache_path = ROOT / "runs/e2e_cache/dec1pass_softloc_cache.npz"
    a1_ckpt = ROOT / "runs/egvla2_e2e_lora/EG3_A1_Dec1Pass.pt"
    a2_ckpt = ROOT / "runs/egvla2_e2e_lora/EG3_A2_SoftLoc.pt"
    has_vlm_e2e = dec_cache_path.exists() and a1_ckpt.exists() and a2_ckpt.exists()
    vlm_a1_preds, vlm_a2_preds, vlm_sl_boxes = [], [], []
    if has_vlm_e2e:
        Z = np.load(dec_cache_path)
        m_a1 = HD.PiZeroFM(12, 768)
        m_a1.load_state_dict(torch.load(a1_ckpt, map_location="cpu", weights_only=False)["state"])
        m_a2 = HD.PiZeroFM(12, 768)
        m_a2.load_state_dict(torch.load(a2_ckpt, map_location="cpu", weights_only=False)["state"])
        for yr, gi in scene_gidx:
            y2 = yr[-2:]
            h1 = torch.from_numpy(Z[f"r{y2}_h1"][gi])
            hloc = torch.from_numpy(Z[f"r{y2}_hloc"][gi])
            sb = Z[f"r{y2}_sb"][gi].astype(np.float32)
            vlm_sl_boxes.append([round(float(v), 3) for v in sb])
            vlm_a1_preds.append(predict_fm_single(m_a1, np.zeros(12, dtype=np.float32), h1))
            vlm_a2_preds.append(predict_fm_single(m_a2, np.tile(sb, 3), hloc))

    # 장면별 통합 JSON 구축
    out_scenes = []
    for idx, sc in enumerate(anatomy):
        m_mod = sc["models"]
        zero_bw = np.zeros(12, dtype=np.float32)

        # ENC-E2E-1(B1), ENC-E2E-AG는 기존 학습과 동일하게 BBox 입력 없이(또는 0 BBox) vs 실제 BBox?
        # 참고: eval_e2e_on_real_a3.py의 EG3-04/06/07은 head에 BW(12) + LoRA visual feat(768)를 넣었는지 0을 넣었는지 확인 ->
        # run_e2e_v2_only.py에서 EG3-06은 BW_TR, EG3-07은 aux_bb.repeat(1,3)을 입력으로 학습!
        bw_cur = scene_bw[idx]
        lx_b1, ly_b1 = predict_fm_single(head_b1, bw_cur, f_b1[idx])
        lx_ag, ly_ag = predict_fm_single(head_ag, bw_cur, f_ag[idx])
        aux_b = aux_bb_6[idx]
        aux_bw12 = np.tile(aux_b, 3)
        lx_aux, ly_aux = predict_fm_single(head_aux, aux_bw12, f_aux[idx])

        models_map = {
            "MOD-A3": {
                "fam": "mod",
                "partner_id": "a3 (EG2-09)",
                "col": "#38bdf8",
                "pipe": "Florence-2 Decoder(beam=3) 문자열 BBox → PiZeroFM(BBox+Img)",
                **{k: m_mod["a3"][k] for k in ("cmd", "mag", "ang_deg", "stop")},
                "note": "현행 배포 기준선. 조향은 BBox cx 추종, 정지는 시각 외형에 의존.",
            },
            "MOD-A3-s2": {
                "fam": "mod",
                "partner_id": "a3-s2 (EG2-14)",
                "col": "#fb923c",
                "pipe": "Florence-2 Decoder(beam=3) 문자열 BBox → PiZeroFM(시드2)",
                **{k: m_mod["a3-s2"][k] for k in ("cmd", "mag", "ang_deg", "stop")},
                "note": "11/16 실기 45/45. 중앙 출발 해결했으나 근접 정지 둔화(접촉 6건).",
            },
            "MOD-A3-AG": {
                "fam": "mod",
                "partner_id": "a3-ag23 (EG2-16)",
                "col": "#4ade80",
                "pipe": "Florence-2 Decoder(beam=3/beam=1 fp16) → Area-Grounded PiZeroFM",
                **{k: m_mod["a3-ag23"][k] for k in ("cmd", "mag", "ang_deg", "stop")},
                "note": "조향(cx)+정지(area≥0.23) 동시 기하 정합. MOD-DR-Fast 결합 시 283ms.",
            },
            "ENC-E2E-1": {
                "fam": "enc",
                "partner_id": "B1 / EG3-04",
                "col": "#f87171",
                "pipe": "DaViT 마지막 단(디코더 제거) → PiZeroFM (원본 데모 라벨)",
                **cmd_stats(lx_b1, ly_b1),
                "note": "기존 B1. VLM 디코더 제거 + 원본 라벨 과적합으로 방향·정지 붕괴.",
            },
            "ENC-E2E-AG": {
                "fam": "enc",
                "partner_id": "EG3-06 (e2e-lora-ag)",
                "col": "#fbbf24",
                "pipe": "DaViT LoRA(디코더 제거) → Area-Grounded PiZeroFM",
                **cmd_stats(lx_ag, ly_ag),
                "note": "Area-Grounded로 11/13·16 시뮬 대폭 회복(8.05°). 단 VLM 디코더 미사용.",
            },
            "ENC-E2E-Aux": {
                "fam": "enc",
                "partner_id": "EG3-07 (e2e-spatial-aux)",
                "col": "#facc15",
                "pipe": f"DaViT LoRA → 내부 Aux BBox(cx={aux_b[0]:.2f}, area={aux_b[2]:.2f}) → PiZeroFM",
                "aux_bbox": [round(float(v), 3) for v in aux_b],
                **cmd_stats(lx_aux, ly_aux),
                "note": "generate() 없이 내부 BBox 예측(7.78°). VLM 디코더 미사용 대조군.",
            },
        }
        if has_vlm_e2e:
            lx1, ly1 = vlm_a1_preds[idx]
            lx2, ly2 = vlm_a2_preds[idx]
            sb = vlm_sl_boxes[idx]
            models_map["VLM-E2E-Hidden"] = {
                "fam": "vlm",
                "partner_id": "EG3-08 (Dec1Pass)",
                "col": "#c084fc",
                "pipe": "Florence-2 Enc + BART Decoder(6L) 1-Pass Hidden → PiZeroFM (BBox 없음)",
                **cmd_stats(lx1, ly1),
                "note": "π0형 순수 VLM Decoder 1-Pass E2E (문자열 파싱 0%).",
            }
            models_map["VLM-E2E-SoftLoc"] = {
                "fam": "vlm",
                "partner_id": "EG3-09 (Soft-Loc)",
                "col": "#e879f9",
                "pipe": f"Decoder <loc> Softmax 기댓값(cx={sb[0]:.2f}, area={sb[2]:.2f}) + Hidden → PiZeroFM",
                "softloc_bbox": sb,
                **cmd_stats(lx2, ly2),
                "note": "VLM Decoder 사전학습 Open-Vocab 100% 보존 + 완전 미분 E2E (283.8ms).",
            }

        out_scenes.append(
            dict(
                id=sc["id"],
                tab=sc["tab"],
                title=sc["title"],
                sid=sc["sid"],
                fi=sc["fi"],
                yr=sc["yr"],
                obj=sc["obj"],
                pos=sc["pos"],
                ins=sc["ins"],
                img_src=sc["img_src"],
                bbox=sc["bbox"],
                real_outcome=sc["real_outcome"],
                models=models_map,
            )
        )

    payload = dict(
        generated=datetime.datetime.now().isoformat(timespec="minutes"),
        progress=prog,
        has_vlm_e2e=has_vlm_e2e,
        beam_bench=beam_bench,
        e2e_v2=e2e_v2,
        track_a=track_a,
        scenes=out_scenes,
    )
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # HTML 생성
    b3_ms = beam_bench.get("beam3_fp32 (현행 기준)", {}).get("latency_med_ms", 590.7)
    b1_ms = beam_bench.get("beam1_fp16 (Track B)", {}).get("latency_med_ms", 283.0)
    b1_sp = beam_bench.get("beam1_fp16 (Track B)", {}).get("speedup_vs_beam3", 2.09)
    sl_ms = beam_bench.get("softloc_fp16 (Track A2)", {}).get("latency_med_ms", 283.8)
    sl_sp = beam_bench.get("softloc_fp16 (Track A2)", {}).get("speedup_vs_beam3", 2.08)

    ta_a1 = track_a.get("Track A1 (Decoder 1-Pass 순수 E2E)", {})
    ta_a2 = track_a.get("Track A2 (Soft-Loc 미분가능 E2E)", {})

    def fmt_ta(d: dict, k: str, suf: str = "") -> str:
        if not d or k not in d or d[k] is None:
            return '<span class="badge b-purp">Phase 2 학습 중</span>'
        v = d[k]
        if suf == "%":
            return f"<b>{v*100:.1f}%</b>"
        return f"<b>{v}{suf}</b>"

    scenes_js = json.dumps(out_scenes, ensure_ascii=False)
    gen_ts = payload["generated"]

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3계열 아키텍처 비교 — MOD vs ENC-E2E vs VLM-E2E</title>
<style>
:root{{--bg:#0b1120;--card:#111a2e;--head:#16223b;--fg:#e2e8f0;--mut:#94a3b8;--line:#233252;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc;--pink:#e879f9;--ora:#fb923c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1360px;margin:0 auto;padding:22px 18px 80px}}
h1{{font-size:1.58rem;margin:0 0 6px;color:#f8fafc}}
h2{{font-size:1.18rem;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #233252;color:#fcd34d}}
p.sub{{margin:0 0 16px;color:var(--mut);font-size:.9rem}}
a{{color:var(--acc)}}
code{{background:#1e293b;padding:2px 6px;border-radius:4px;font-size:.82em;color:#e2e8f0}}
.nav-top{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;font-size:.82rem}}
.nav-top a{{border:1px solid var(--line);background:var(--card);padding:4px 11px;border-radius:99px;text-decoration:none}}
.badge{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.73rem;font-weight:700;white-space:nowrap}}
.b-mod{{background:#0c2d48;color:#7dd3fc;border:1px solid #0284c7}}
.b-enc{{background:#3b270c;color:#fcd34d;border:1px solid #b45309}}
.b-vlm{{background:#2e1065;color:#e9d5ff;border:1px solid #9333ea}}
.b-ok{{background:#14532d;color:#86efac}}
.b-warn{{background:#451a03;color:#fcd34d}}
.b-bad{{background:#450a0a;color:#fca5a5}}
.b-purp{{background:#3b0764;color:#e9d5ff}}

.fam-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0}}
@media(max-width:980px){{.fam-grid{{grid-template-columns:1fr}}}}
.fam-card{{background:var(--card);border:2px solid var(--line);border-radius:14px;padding:15px 16px;display:flex;flex-direction:column;gap:8px}}
.fam-card.mod{{border-color:#0284c7}}.fam-card.enc{{border-color:#d97706}}.fam-card.vlm{{border-color:#a855f7}}
.fam-head{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
.fam-ttl{{font-size:1.05rem;font-weight:800;color:#fff}}
.flow-box{{background:#08101f;border:1px solid var(--line);border-radius:9px;padding:8px 10px;font-size:.79rem;line-height:1.5;color:#cbd5e1}}
.flow-box b{{color:#fff}}

.prog-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px;margin:12px 0}}
.prog-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px}}
.prog-card.active{{border:2px solid var(--purp);background:#15122b}}
.prog-card.done{{border-left:5px solid var(--ok)}}
.pbar{{height:10px;background:#0a1220;border:1px solid var(--line);border-radius:99px;overflow:hidden;margin:8px 0 6px}}
.pbar i{{display:block;height:100%;background:linear-gradient(90deg,#a855f7,#38bdf8)}}

.studio{{background:var(--card);border:2px solid var(--acc);border-radius:14px;padding:16px 18px;margin:16px 0}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}}
.tab-btn{{background:var(--head);border:1px solid var(--line);color:var(--mut);padding:6px 12px;border-radius:8px;font-size:.82rem;font-weight:700;cursor:pointer}}
.tab-btn.on{{background:var(--acc);color:#0b1120;border-color:var(--acc)}}
.tog-bar{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px;align-items:center}}
.tog-chip{{border:1px solid var(--line);background:#0d1629;color:#e2e8f0;padding:4px 10px;border-radius:99px;font-size:.77rem;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;gap:6px}}
.tog-chip.off{{opacity:.32}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}

.sim-grid{{display:grid;grid-template-columns:480px 1fr;gap:16px;align-items:start}}
@media(max-width:1020px){{.sim-grid{{grid-template-columns:1fr}}}}
.cam-card{{background:#070c17;border:1px solid var(--line);border-radius:12px;padding:10px}}
.cam-wrap{{position:relative;width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#000}}
.cam-wrap img{{width:100%;height:100%;object-fit:cover;display:block}}
.cam-wrap svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.cam-meta{{font-size:.78rem;color:var(--mut);margin-top:8px;line-height:1.55}}
.cam-meta b{{color:#fff}}

table{{width:100%;border-collapse:collapse;font-size:.82rem;background:var(--card);border-radius:10px;overflow:hidden;margin:10px 0}}
th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:var(--head);color:#cbd5e1;font-size:.77rem}}
.pipe-txt{{font-size:.76rem;color:#94a3b8;line-height:1.45}}
</style>
</head>
<body>
<main>
  <div class="nav-top">
    <a href="model_registry.html#naming">📋 모델 레지스트리 (전체 41개)</a>
    <a href="v4_vs_a3_compare.html">🟣 V4 · V4R vs A3 비교</a>
    <a href="research_story_egvla2.html#ch82">📖 CH82 실기·재생 대조</a>
    <a href="object_test_compare_1113.html">🎮 11/13 동시 재생기</a>
  </div>

  <h1>3계열 아키텍처 통합 비교 — <code>MOD-*</code> vs <code>ENC-E2E-*</code> vs <code>VLM-E2E-*</code></h1>
  <p class="sub">
    생성 {html.escape(gen_ts)} · 파트너 호환 ID(<code>a3</code>, <code>a3-ag23</code>, <code>EG3-03~09</code>)는 유지하고,
    <b>E2E 여부</b>와 <b>VLM Decoder 활용 방식</b>이 한눈에 구분되도록 3계열 축약명과 실제 로봇 세션 6개 장면의 방향·판단 과정을 나란히 비교합니다.
  </p>

  <h2>0. 3계열 아키텍처 구조 &amp; 네이밍 한눈에 보기</h2>
  <div class="fam-grid">
    <div class="fam-card mod">
      <div class="fam-head"><span class="fam-ttl">🔵 MOD-* (모듈러 2단계)</span><span class="badge b-mod">비-E2E · 문자열 파싱</span></div>
      <div class="flow-box">
        <b>① Image + Prompt</b> → <b>② Florence-2 Enc + Decoder <code>generate()</code></b> → 문자열 <code>&lt;loc&gt;</code> 파싱(BBox) → <b>③ PiZeroFM</b> → <code>[lx, ly]</code>
      </div>
      <div style="font-size:.82rem;color:var(--mut)">
        • <b>대표 모델</b>: <code>MOD-A3</code>(a3), <code>MOD-A3-AG</code>(a3-ag23), <code>MOD-DR-Fast</code>(beam=1 fp16), <code>MOD-DR-Safe</code><br>
        • <b>장점</b>: Florence-2 사전학습 Open-Vocab(의자·책상·화분) 100% 유지, 실기 검증 완료.<br>
        • <b>한계</b>: BBox 문자열 경계에서 미분 단절. (단, <code>MOD-DR-Fast</code>로 지연 <code>{b3_ms}→{b1_ms}ms</code> <b>{b1_sp}배 단축</b>)
      </div>
    </div>

    <div class="fam-card enc">
      <div class="fam-head"><span class="fam-ttl">🟡 ENC-E2E-* (인코더 단독 E2E)</span><span class="badge b-enc">E2E · 디코더 제거</span></div>
      <div class="flow-box">
        <b>① Image</b> → <b>② DaViT Vision Encoder (LoRA)</b> <i>[BART Decoder 제거]</i> → (선택: Aux BBox 헤드) → <b>③ PiZeroFM</b> → <code>[lx, ly]</code>
      </div>
      <div style="font-size:.82rem;color:var(--mut)">
        • <b>대표 모델</b>: <code>ENC-E2E-0</code>(B0), <code>ENC-E2E-1</code>(<b>구 B1</b>), <code>ENC-E2E-AG</code>(EG3-06), <code>ENC-E2E-Aux</code>(EG3-07)<br>
        • <b>장점</b>: <code>generate()</code> 제거로 초고속(~30ms), Area-Grounded 적용 시 11/13·16 시뮬 크게 회복(7.78°).<br>
        • <b>한계</b>: VLM 디코더를 쓰지 않아 300ep(바구니 1종) 외 Open-Vocab 지시어 이해에 구조적 한계.
      </div>
    </div>

    <div class="fam-card vlm">
      <div class="fam-head"><span class="fam-ttl">🟣 VLM-E2E-* (VLM 디코더 E2E)</span><span class="badge b-vlm">완전 미분 E2E · 학술 메인</span></div>
      <div class="flow-box">
        <b>① Image + Prompt</b> → <b>② Florence-2 Enc + BART Decoder(6L)</b> → <b>1-Pass Hidden</b> 또는 <b>Softmax <code>&lt;loc&gt;</code> 기댓값</b> → <b>③ PiZeroFM</b> → <code>[lx, ly]</code>
      </div>
      <div style="font-size:.82rem;color:var(--mut)">
        • <b>대표 모델</b>: <code>VLM-E2E-Hidden</code>(EG3-08, π0형), <code>VLM-E2E-SoftLoc</code>(EG3-09, 283.8ms), <code>VLM-E2E-Distill</code>(Phase 3)<br>
        • <b>장점</b>: VLM Decoder 6층을 100% 활용하면서도 문자열 파싱을 제거해 <b>완전 미분 + Open-Vocab 보존 + 2배 가속</b> 동시 달성.
      </div>
    </div>
  </div>

  <h2>1. 현재 학습 · 벤치마크 진행 현황 (Live Status)</h2>
  <div class="prog-grid">
    <div class="prog-card done">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>Phase 1 · MOD-DR-Fast &amp; Soft-Loc 벤치마크</b>
        <span class="badge b-ok">완료 ✔</span>
      </div>
      <div style="font-size:.82rem;color:var(--mut);margin-top:6px">
        11/13 다물체 + 11/16 근접 <b>330프레임</b> 실측:<br>
        • <code>MOD-DR-Fast</code>(beam=1 fp16): <code>{b3_ms}ms → {b1_ms}ms</code> (<b>{b1_sp}x</b>), 일치율 <b>99.7%</b>, <code>cx MAE 0.0016</code><br>
        • <code>VLM-E2E-SoftLoc</code>(fp16): <code>{sl_ms}ms</code> (<b>{sl_sp}x</b>), 일치율 <b>98.8%</b>, <code>cx MAE 0.0141</code>
      </div>
    </div>

    <div class="prog-card {'done' if prog['state']=='done' else 'active'}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>Phase 2 · VLM-E2E-Hidden &amp; SoftLoc 학습</b>
        <span class="badge {'b-ok' if prog['state']=='done' else 'b-purp'}">{'완료 ✔' if prog['state']=='done' else '실행 중 (task-337)'}</span>
      </div>
      <div class="pbar"><i style="width:{prog['pct']}%"></i></div>
      <div style="font-size:.82rem;color:#e2e8f0">{html.escape(prog['detail'])}</div>
      <div style="font-size:.77rem;color:var(--mut);margin-top:4px">
        완료 즉시 이 페이지를 재빌드하면 아래 6개 실기 사진 위에 보라색(<code>VLM-E2E-Hidden</code>/<code>SoftLoc</code>) 화살표가 자동 추가됩니다.
      </div>
    </div>

    <div class="prog-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>Phase 3 · VLM-E2E-Distill &amp; CH83 발행</b>
        <span class="badge b-warn">예정 ⏳</span>
      </div>
      <div style="font-size:.82rem;color:var(--mut);margin-top:6px">
        저학습률 Decoder LoRA + <code>beam=3</code> 다물체 증류(Distillation) 학습 및 <code>MOD</code> vs <code>ENC-E2E</code> vs <code>VLM-E2E</code> 파레토 프론티어 리포트(CH83) 발행.
      </div>
    </div>

    <div class="prog-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>Phase 4 · MOD-DR-Safe 실기 &amp; YahBoom 이식</b>
        <span class="badge b-warn">예정 ⏳</span>
      </div>
      <div style="font-size:.82rem;color:var(--mut);margin-top:6px">
        <code>MOD-DR-Fast</code>(283ms) + 비동기 Dual-Rate + 접근 감속(S2) + 점프 필터를 <code>soda</code>에 적용해 Serbot 정지 검증 후 YahBoom 카 이식.
      </div>
    </div>
  </div>

  <h2>2. 실제 로봇 세션 6개 장면별 모델 추론 방향 &amp; 판단 과정 비교</h2>
  <div class="studio">
    <p style="margin:0 0 8px;font-size:.86rem;color:#cbd5e1">
      실제 11/13 · 11/16 로봇 카메라 사진 위에서 <b>MOD 계열</b>, <b>ENC-E2E 계열(구 B1·AG·Aux)</b>, <b>VLM-E2E 계열</b>이 각각 어떤 과정을 거쳐 어느 방향(<code>[lx, ly]</code>)으로 주행하거나 정지하는지 직접 비교하세요.
    </p>
    <div class="tabs" id="sc-tabs"></div>
    <div class="tog-bar" id="m-togs"></div>

    <div class="sim-grid">
      <div class="cam-card">
        <div class="cam-wrap">
          <img id="cam-img" src="" alt="실기 장면">
          <svg id="cam-svg" viewBox="0 0 416 234"></svg>
        </div>
        <div class="cam-meta" id="cam-meta"></div>
      </div>
      <div>
        <div id="sc-head" style="font-size:.87rem;margin-bottom:8px;line-height:1.55"></div>
        <table>
          <thead>
            <tr>
              <th>축약명 · 파트너 ID</th>
              <th>내부 판단 과정 (파이프라인)</th>
              <th>출력 <code>[lx, ly]</code></th>
              <th>판정</th>
              <th>진단</th>
            </tr>
          </thead>
          <tbody id="cmp-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <h2>3. 3계열 전체 모델 정량 비교표 (속도 · 조향 · 출발 · 정지 · Open-Vocab)</h2>
  <table>
    <thead>
      <tr>
        <th>계열</th>
        <th>축약명 (파트너 ID)</th>
        <th>지연 (ms)</th>
        <th>A3 대비 각도차 (좌우일치)</th>
        <th>11/13 의자·책상 중앙출발</th>
        <th>11/16 원거리 오정지(&lt;.10)</th>
        <th>11/16 근접 정지(바구니/의자 ≥.25)</th>
        <th>VLM Open-Vocab</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="badge b-mod">MOD</span></td>
        <td><b>MOD-A3</b> (<code>a3</code> · EG2-09)</td>
        <td>{b3_ms}ms (1.0x)</td>
        <td>기준 0.00° (100%)</td>
        <td><span class="badge b-bad">0% / 4%</span></td>
        <td>0.0%</td>
        <td>80% / 58%</td>
        <td><span class="badge b-ok">100% 유지</span></td>
      </tr>
      <tr>
        <td><span class="badge b-mod">MOD</span></td>
        <td><b>MOD-A3-s2</b> (<code>a3-s2</code> · EG2-14)</td>
        <td>{b3_ms}ms (1.0x)</td>
        <td>0.70° (98.3%)</td>
        <td><span class="badge b-ok">100% / 100%</span></td>
        <td>0.0%</td>
        <td><span class="badge b-bad">55% / 28%</span> (접촉 6건)</td>
        <td><span class="badge b-ok">100% 유지</span></td>
      </tr>
      <tr>
        <td><span class="badge b-mod">MOD</span></td>
        <td><b>MOD-A3-AG</b> (<code>a3-ag23</code> · EG2-16)</td>
        <td>{b3_ms}ms (1.0x)</td>
        <td><b>0.54° (98.7%)</b></td>
        <td><span class="badge b-ok">100% / 100%</span></td>
        <td><b>0.0%</b></td>
        <td><span class="badge b-ok">90% / 93%</span></td>
        <td><span class="badge b-ok">100% 유지</span></td>
      </tr>
      <tr>
        <td><span class="badge b-mod">MOD</span></td>
        <td><b>MOD-DR-Fast</b> (<code>a3-dr-fast</code> · EG2-17)</td>
        <td><b>{b1_ms}ms ({b1_sp}x)</b></td>
        <td><code>MOD-A3-AG</code>와 99.7% 동일</td>
        <td><span class="badge b-ok">100% / 100%</span></td>
        <td><b>0.0%</b></td>
        <td><span class="badge b-ok">90% / 93%</span></td>
        <td><span class="badge b-ok">99.7% (cx MAE 0.0016)</span></td>
      </tr>
      <tr>
        <td><span class="badge b-enc">ENC-E2E</span></td>
        <td><b>ENC-E2E-1</b> (<code>B1</code> · EG3-04)</td>
        <td>~30ms</td>
        <td><span class="badge b-bad">25.13° (43.3%)</span></td>
        <td>붕괴</td>
        <td>—</td>
        <td><span class="badge b-bad">45% / 26%</span></td>
        <td><span class="badge b-bad">디코더 제거</span></td>
      </tr>
      <tr>
        <td><span class="badge b-enc">ENC-E2E</span></td>
        <td><b>ENC-E2E-AG</b> (<code>EG3-06</code> · <i>BBox 보조 LoRA</i>)</td>
        <td>~591ms*<br><span style="font-size:.72rem;color:#94a3b8">(외부 BBox 입력)</span></td>
        <td>8.05° (82.0%)</td>
        <td><span class="badge b-ok">100% / 100%</span></td>
        <td>0.0%</td>
        <td><span class="badge b-ok">90% / 93%</span></td>
        <td><span class="badge b-warn">외부 BBox 의존 (비순수 E2E)</span></td>
      </tr>
      <tr>
        <td><span class="badge b-enc">ENC-E2E</span></td>
        <td><b>ENC-E2E-Aux</b> (<code>EG3-07</code> · <i>내부 Aux BBox</i>)</td>
        <td>~32ms (Aux 단독)<br><span style="font-size:.72rem;color:#94a3b8">(표 수치는 BW 기준)</span></td>
        <td>7.78° (85.0%)</td>
        <td><span class="badge b-ok">100% / 100%</span></td>
        <td>0.0%</td>
        <td><span class="badge b-ok">90% / 88%</span></td>
        <td><span class="badge b-bad">바구니만 학습 (Open-Vocab X)</span></td>
      </tr>
      <tr>
        <td><span class="badge b-vlm">VLM-E2E</span></td>
        <td><b>VLM-E2E-Hidden</b> (<code>EG3-08</code> · π0형, BBox=0)</td>
        <td>~70ms (1-Pass)</td>
        <td>{fmt_ta(ta_a1, 'ang_med_vs_a3', '°')} ({fmt_ta(ta_a1, 'lat_agree_vs_a3', '%')})</td>
        <td><span class="badge b-ok">{fmt_ta(ta_a1, 'chair_center_13', '%')} / {fmt_ta(ta_a1, 'desk_center_13', '%')}</span></td>
        <td>{fmt_ta(ta_a1, 'stop_lt10_16', '%')}</td>
        <td>{fmt_ta(ta_a1, 'basket_ge25_stop_16', '%')} / <span class="badge b-bad">{fmt_ta(ta_a1, 'chair_ge25_stop_16', '%')}</span></td>
        <td><span class="badge b-vlm">Decoder 6L (기하 병목 X)</span></td>
      </tr>
      <tr>
        <td><span class="badge b-vlm">VLM-E2E</span></td>
        <td><b>VLM-E2E-SoftLoc</b> (<code>EG3-09</code> · 미분 Soft-Loc)</td>
        <td><b>{sl_ms}ms ({sl_sp}x)</b></td>
        <td><b>{fmt_ta(ta_a2, 'ang_med_vs_a3', '°')} ({fmt_ta(ta_a2, 'lat_agree_vs_a3', '%')})</b></td>
        <td><span class="badge b-ok">{fmt_ta(ta_a2, 'chair_center_13', '%')} / {fmt_ta(ta_a2, 'desk_center_13', '%')}</span></td>
        <td><b>{fmt_ta(ta_a2, 'stop_lt10_16', '%')}</b></td>
        <td><span class="badge b-ok">{fmt_ta(ta_a2, 'basket_ge25_stop_16', '%')} / {fmt_ta(ta_a2, 'chair_ge25_stop_16', '%')}</span></td>
        <td><span class="badge b-ok">98.8% (cx MAE 0.0141)</span></td>
      </tr>
    </tbody>
  </table>
</main>
<script>
const SCENES = {scenes_js};
let curSc = 0;
const visible = {{}};
const firstKeys = Object.keys(SCENES[0].models);
firstKeys.forEach(k => visible[k] = true);

function famBadge(fam) {{
  if (fam === 'mod') return '<span class="badge b-mod">MOD</span>';
  if (fam === 'enc') return '<span class="badge b-enc">ENC-E2E</span>';
  return '<span class="badge b-vlm">VLM-E2E</span>';
}}

function drawBox(svgNS, svg, bb, stroke, dash, label) {{
  if (!bb || bb[3] < 0.3) return;
  const W = 416, H = 234;
  const side = Math.sqrt(Math.max(0.002, bb[2]));
  const bw = Math.min(0.92, side * 1.15) * W;
  const bh = Math.min(0.92, side / 1.15) * H;
  const x = Math.max(2, Math.min(W - bw - 2, bb[0] * W - bw / 2));
  const y = Math.max(2, Math.min(H - bh - 2, bb[1] * H - bh / 2));
  const r = document.createElementNS(svgNS, 'rect');
  r.setAttribute('x', x); r.setAttribute('y', y); r.setAttribute('width', bw); r.setAttribute('height', bh);
  r.setAttribute('fill', 'none'); r.setAttribute('stroke', stroke); r.setAttribute('stroke-width', '2.2');
  if (dash) r.setAttribute('stroke-dasharray', dash);
  svg.appendChild(r);
  const t = document.createElementNS(svgNS, 'text');
  t.setAttribute('x', x + 3); t.setAttribute('y', Math.max(12, y - 4));
  t.setAttribute('fill', stroke); t.setAttribute('font-size', '10'); t.setAttribute('font-weight', '700');
  t.textContent = label;
  svg.appendChild(t);
}}

function drawArrow(svgNS, svg, lx, ly, col, name, isStop) {{
  const ox = 208, oy = 216;
  if (isStop) {{
    const c = document.createElementNS(svgNS, 'circle');
    c.setAttribute('cx', ox); c.setAttribute('cy', oy); c.setAttribute('r', '7');
    c.setAttribute('fill', 'none'); c.setAttribute('stroke', col); c.setAttribute('stroke-width', '2.4');
    svg.appendChild(c);
    return;
  }}
  const scale = 115;
  // 로봇 좌표계: lx=전진(화면 위), ly=좌(+)우(-) -> 화면 x = ox - ly*scale, y = oy - lx*scale
  const tx = ox - ly * scale;
  const ty = oy - lx * scale;
  const l = document.createElementNS(svgNS, 'line');
  l.setAttribute('x1', ox); l.setAttribute('y1', oy); l.setAttribute('x2', tx); l.setAttribute('y2', ty);
  l.setAttribute('stroke', col); l.setAttribute('stroke-width', '3.2'); l.setAttribute('stroke-linecap', 'round');
  svg.appendChild(l);
  const dot = document.createElementNS(svgNS, 'circle');
  dot.setAttribute('cx', tx); dot.setAttribute('cy', ty); dot.setAttribute('r', '4');
  dot.setAttribute('fill', col);
  svg.appendChild(dot);
  const lbl = document.createElementNS(svgNS, 'text');
  lbl.setAttribute('x', tx + 6); lbl.setAttribute('y', ty + 3);
  lbl.setAttribute('fill', col); lbl.setAttribute('font-size', '10'); lbl.setAttribute('font-weight', '700');
  lbl.textContent = name;
  svg.appendChild(lbl);
}}

function render() {{
  const sc = SCENES[curSc];
  document.getElementById('sc-tabs').innerHTML = SCENES.map((s, i) =>
    `<button type="button" class="tab-btn ${{i===curSc?'on':''}}" onclick="curSc=${{i}};render()">${{s.tab}}</button>`
  ).join('');

  const mkeys = Object.keys(sc.models);
  document.getElementById('m-togs').innerHTML = '<span style="font-size:.78rem;color:#94a3b8;font-weight:700">화살표 표시 토글:</span>' +
    mkeys.map(k => {{
      const m = sc.models[k];
      return `<button type="button" class="tog-chip ${{visible[k]?'':'off'}}" onclick="visible['${{k}}']=!visible['${{k}}'];render()"><i class="dot" style="background:${{m.col}}"></i>${{k}}</button>`;
    }}).join('');

  document.getElementById('cam-img').src = sc.img_src;
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.getElementById('cam-svg');
  svg.innerHTML = '';

  // Florence-2 BBox
  drawBox(svgNS, svg, sc.bbox, '#38bdf8', null, `Florence-2 BBox (cx=${{sc.bbox[0].toFixed(2)}}, area=${{sc.bbox[2].toFixed(2)}})`);
  // ENC-E2E-Aux 내부 BBox
  if (visible['ENC-E2E-Aux'] && sc.models['ENC-E2E-Aux'] && sc.models['ENC-E2E-Aux'].aux_bbox) {{
    const ab = sc.models['ENC-E2E-Aux'].aux_bbox;
    drawBox(svgNS, svg, ab, '#facc15', '5,3', `Aux BBox (cx=${{ab[0].toFixed(2)}}, a=${{ab[2].toFixed(2)}})`);
  }}
  // VLM-E2E-SoftLoc BBox
  if (visible['VLM-E2E-SoftLoc'] && sc.models['VLM-E2E-SoftLoc'] && sc.models['VLM-E2E-SoftLoc'].softloc_bbox) {{
    const sb = sc.models['VLM-E2E-SoftLoc'].softloc_bbox;
    drawBox(svgNS, svg, sb, '#e879f9', '3,3', `Soft-Loc (cx=${{sb[0].toFixed(2)}}, a=${{sb[2].toFixed(2)}})`);
  }}
  // 화살표
  mkeys.forEach(k => {{
    if (!visible[k]) return;
    const m = sc.models[k];
    drawArrow(svgNS, svg, m.cmd[0], m.cmd[1], m.col, k, m.stop);
  }});

  document.getElementById('cam-meta').innerHTML =
    `<b>세션</b>: <code>${{sc.sid}}</code> (프레임 #${{sc.fi}}) · <b>물체/위치</b>: ${{sc.obj}} / ${{sc.pos}}<br>` +
    `<b>지시어</b>: <code>"${{sc.ins}}"</code><br>` +
    `<b>실기 기록</b>: ${{sc.real_outcome}}`;

  document.getElementById('sc-head').innerHTML = `<b>${{sc.title}}</b>`;

  document.getElementById('cmp-tbody').innerHTML = mkeys.map(k => {{
    const m = sc.models[k];
    const st = m.stop ? '<span class="badge b-bad">정지 (STOP)</span>' : '<span class="badge b-ok">주행 (DRIVE)</span>';
    return `<tr>
      <td>${{famBadge(m.fam)}} <b style="color:${{m.col}}">${{k}}</b><br><span style="font-size:.75rem;color:#94a3b8">${{m.partner_id}}</span></td>
      <td class="pipe-txt">${{m.pipe}}</td>
      <td><code>[${{m.cmd[0].toFixed(2)}}, ${{m.cmd[1].toFixed(2)}}]</code><br><span style="font-size:.75rem;color:#94a3b8">크기 ${{m.mag.toFixed(2)}} · ${{m.ang_deg}}°</span></td>
      <td>${{st}}</td>
      <td style="font-size:.79rem;color:#cbd5e1">${{m.note}}</td>
    </tr>`;
  }}).join('');
}}
render();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"Generated {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB) | has_vlm_e2e={has_vlm_e2e} | prog={prog['pct']}%")


if __name__ == "__main__":
    main()
