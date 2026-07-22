#!/usr/bin/env python3
"""5~7월 실주행 세션 로그 → 경로/모델/런타임별 성공률 매트릭스 (minum 요청, 2026-07-22).

입력:
  docs/inference_reports/session_*.json  (세션별 model/runtime/history/status/summary)
  logs/episode_log.csv                   (경로검증 기록: path_type, 성공/실패, session_id)

출력(--out 디렉터리):
  session_matrix.csv     — 세션 1행씩 (runtime 4종 필드 + status + summary)
  episode_matrix.csv     — episode_log 행 + 조인된 runtime(가능 시)
  success_matrices.md    — path_type별 / checkpoint별 / grounder별 / preview별 성공률 표
"""
import argparse
import csv
import glob
import json
import os
import re
import collections


def parse_ckpt_meta(path: str) -> dict:
    """checkpoint_path 문자열에서 window/bbox_scale/exp 추출(파일명 인코딩 기반)."""
    if not path:
        return {"window": "", "bbox_scale": "", "exp": ""}
    w = re.search(r"window(\d+)", path)
    b = re.search(r"bboxscale(\d+)", path)
    exp = re.search(r"(exp\d+)", path)
    # exp73 계열은 파일명에 window/bboxscale가 없지만 배포 규격이 window6/bbox3 고정
    win = w.group(1) if w else ("6" if "exp73" in path else "")
    bs = b.group(1) if b else ("3" if "exp73" in path else "")
    return {"window": win, "bbox_scale": bs, "exp": exp.group(1) if exp else ""}


def load_sessions(reports_dir: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(reports_dir, "session_2026*.json"))):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [스킵] {os.path.basename(f)}: {e}")
            continue
        sid = d.get("session_id") or os.path.basename(f).replace("session_", "").replace(".json", "")
        rc = d.get("runtime_config") or {}
        summ = d.get("summary") or {}
        ckpt = rc.get("checkpoint_path", "")
        meta = parse_ckpt_meta(ckpt)
        out[sid] = {
            "session_id": sid,
            "timestamp": d.get("timestamp", ""),
            "model_name": d.get("model_name", ""),
            "instruction": d.get("instruction", ""),
            "instruction_mode": d.get("instruction_mode", ""),
            "status": d.get("status", ""),
            "checkpoint_path": ckpt,
            "head": rc.get("head", ""),
            "exp": meta["exp"],
            "window": meta["window"],
            "bbox_scale": meta["bbox_scale"],
            "grounder_model": rc.get("grounder_model", ""),
            "grounder_input_px": rc.get("grounder_input_px", ""),
            "preview_enabled": rc.get("preview_enabled", ""),
            "preview_hint_cx": rc.get("preview_hint_cx", ""),
            "grounding_skip_n": rc.get("grounding_skip_n", ""),
            "multi_prompt": rc.get("multi_prompt", ""),
            "git_commit": rc.get("git_commit", ""),
            "n_frames": summ.get("n_frames", summ.get("total_steps", "")),
            "avg_latency_ms": summ.get("avg_latency_ms", ""),
            "top_action": _top_action(summ.get("action_label_counts")),
            "has_runtime_config": bool(ckpt),
        }
    return out


def _top_action(counts):
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def load_episodes(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    if len(rows) < 2:
        return []
    header, data = rows[0], rows[1:]
    out = []
    for r in data:
        if len(r) < 3:
            continue
        rec = {header[i] if i < len(header) else f"col{i}": (r[i] if i < len(r) else "")
               for i in range(max(len(header), len(r)))}
        out.append(rec)
    return out


def pct(succ, tot):
    return f"{succ}/{tot} ({100*succ/tot:.0f}%)" if tot else "0/0 (—)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="docs/inference_reports")
    ap.add_argument("--episodes", default="logs/episode_log.csv")
    ap.add_argument("--out", default="docs/v5/session_matrix_export")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sessions = load_sessions(args.reports)
    episodes = load_episodes(args.episodes)
    print(f"세션 {len(sessions)}개, 에피소드기록 {len(episodes)}행 로드")

    # 1) session_matrix.csv
    sess_cols = ["session_id", "timestamp", "model_name", "instruction_mode", "status",
                 "checkpoint_path", "exp", "head", "window", "bbox_scale",
                 "grounder_model", "grounder_input_px", "preview_enabled", "preview_hint_cx",
                 "grounding_skip_n", "multi_prompt", "git_commit",
                 "n_frames", "avg_latency_ms", "top_action", "has_runtime_config"]
    with open(os.path.join(args.out, "session_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sess_cols, extrasaction="ignore")
        w.writeheader()
        for sid in sorted(sessions):
            w.writerow(sessions[sid])

    # 2) episode_matrix.csv (episode_log + runtime 조인)
    ep_extra = ["j_checkpoint", "j_exp", "j_head", "j_window", "j_bbox_scale",
                "j_grounder", "j_preview_enabled", "j_git_commit"]
    with open(os.path.join(args.out, "episode_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        base_cols = list(episodes[0].keys()) if episodes else []
        w = csv.DictWriter(f, fieldnames=base_cols + ep_extra)
        w.writeheader()
        for ep in episodes:
            sid = (ep.get("session_id") or "").strip()
            s = sessions.get(sid, {})
            ep2 = dict(ep)
            ep2.update({
                "j_checkpoint": s.get("checkpoint_path", ""),
                "j_exp": s.get("exp", ""),
                "j_head": s.get("head", ""),
                "j_window": s.get("window", ""),
                "j_bbox_scale": s.get("bbox_scale", ""),
                "j_grounder": s.get("grounder_model", ""),
                "j_preview_enabled": s.get("preview_enabled", ""),
                "j_git_commit": s.get("git_commit", ""),
            })
            w.writerow(ep2)

    # 3) success_matrices.md
    lines = ["# 실주행 성공률 매트릭스 (5~7월, soda→minum, 2026-07-22)\n",
             f"세션 {len(sessions)}개 · 경로검증 에피소드 {len(episodes)}행\n"]

    # (a) path_type별 성공률
    by_path = collections.defaultdict(lambda: [0, 0])
    for ep in episodes:
        pt = (ep.get("경로") or "").replace(" ★", "").replace("★", "").strip()
        if not pt:
            continue
        by_path[pt][1] += 1
        if (ep.get("결과") or "").strip() == "성공":
            by_path[pt][0] += 1
    lines.append("## path_type별 성공률\n")
    lines.append("| path_type | 성공/시도 |")
    lines.append("|---|---|")
    for pt in sorted(by_path):
        s, t = by_path[pt]
        lines.append(f"| {pt} | {pct(s, t)} |")

    # (b) checkpoint별 성공률(조인된 것만)
    by_ckpt = collections.defaultdict(lambda: [0, 0])
    joined = 0
    for ep in episodes:
        sid = (ep.get("session_id") or "").strip()
        s = sessions.get(sid)
        if not s or not s.get("checkpoint_path"):
            continue
        joined += 1
        key = os.path.basename(s["checkpoint_path"])
        by_ckpt[key][1] += 1
        if (ep.get("결과") or "").strip() == "성공":
            by_ckpt[key][0] += 1
    lines.append(f"\n## checkpoint별 성공률 (session_id 조인된 {joined}행)\n")
    lines.append("| checkpoint | 성공/시도 |")
    lines.append("|---|---|")
    for k in sorted(by_ckpt):
        s, t = by_ckpt[k]
        lines.append(f"| {k} | {pct(s, t)} |")

    # (c) grounder별 성공률
    by_gnd = collections.defaultdict(lambda: [0, 0])
    for ep in episodes:
        sid = (ep.get("session_id") or "").strip()
        s = sessions.get(sid)
        if not s or not s.get("grounder_model"):
            continue
        key = s["grounder_model"]
        by_gnd[key][1] += 1
        if (ep.get("결과") or "").strip() == "성공":
            by_gnd[key][0] += 1
    lines.append("\n## grounder별 성공률 (조인된 행)\n")
    lines.append("| grounder | 성공/시도 |")
    lines.append("|---|---|")
    for k in sorted(by_gnd):
        s, t = by_gnd[k]
        lines.append(f"| {k} | {pct(s, t)} |")

    # (d) checkpoint × path_type 교차표
    cross = collections.defaultdict(lambda: [0, 0])
    for ep in episodes:
        sid = (ep.get("session_id") or "").strip()
        s = sessions.get(sid)
        if not s or not s.get("checkpoint_path"):
            continue
        pt = (ep.get("경로") or "").replace(" ★", "").replace("★", "").strip()
        key = (os.path.basename(s["checkpoint_path"]), pt)
        cross[key][1] += 1
        if (ep.get("결과") or "").strip() == "성공":
            cross[key][0] += 1
    lines.append("\n## checkpoint × path_type 교차 성공률\n")
    lines.append("| checkpoint | path_type | 성공/시도 |")
    lines.append("|---|---|---|")
    for (ck, pt) in sorted(cross):
        s, t = cross[(ck, pt)]
        lines.append(f"| {ck} | {pt} | {pct(s, t)} |")

    # (e) 세션 status 분포(전체 254, path_type 없어도)
    st = collections.Counter(s["status"] for s in sessions.values())
    lines.append("\n## 세션 status 분포 (전체 세션, path_type 무관)\n")
    lines.append("| status | 세션 수 |")
    lines.append("|---|---|")
    for k, v in st.most_common():
        lines.append(f"| {k or '(빈값)'} | {v} |")

    with open(os.path.join(args.out, "success_matrices.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"완료 → {args.out}/ (session_matrix.csv, episode_matrix.csv, success_matrices.md)")


if __name__ == "__main__":
    main()
