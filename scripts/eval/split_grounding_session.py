#!/usr/bin/env python3
"""
gradio_inference_dashboard.py의 idle-gap 롤오버 버그(수정 전)로 여러 날의
그라운딩 세션이 한 jsonl 파일에 합쳐진 경우, 시간 gap 기준으로 분리한다.

각 분리된 청크는 자신의 첫 레코드 timestamp로 새 파일명을 받는다.
mp4는 영향받지 않음(_gnd_video_path가 매번 새로 생성됨) — jsonl만 문제였음.

사용:
    python3 scripts/eval/split_grounding_session.py \
        logs/grounding_sessions/gnd_20260618_172556.jsonl --gap-minutes 10
"""
import argparse
import datetime as dt
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="분리할 jsonl 경로")
    ap.add_argument("--gap-minutes", type=float, default=10.0,
                     help="이 시간(분) 이상 비면 새 세션으로 간주 (기본 10분)")
    ap.add_argument("--dry-run", action="store_true", help="분리만 미리보고 파일 안 씀")
    args = ap.parse_args()

    src = Path(args.jsonl)
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if not records:
        print("레코드 없음")
        return

    gap = dt.timedelta(minutes=args.gap_minutes)
    chunks = [[records[0]]]
    for prev, cur in zip(records, records[1:]):
        t_prev = dt.datetime.fromisoformat(prev["ts"])
        t_cur = dt.datetime.fromisoformat(cur["ts"])
        if t_cur - t_prev > gap:
            chunks.append([])
        chunks[-1].append(cur)

    print(f"{src.name}: {len(records)}건 → {len(chunks)}개 세션으로 분리 (gap={args.gap_minutes}분)")
    for chunk in chunks:
        t0 = dt.datetime.fromisoformat(chunk[0]["ts"])
        ts_name = t0.strftime("%Y%m%d_%H%M%S")
        out_path = src.parent / f"gnd_{ts_name}.jsonl"
        print(f"  {out_path.name}: {len(chunk)}건  ({chunk[0]['ts']} ~ {chunk[-1]['ts']})")
        if not args.dry_run:
            with open(out_path, "w") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if len(chunks) > 1 and not args.dry_run and out_path != src:
        print(f"\n원본 {src.name}은 그대로 둠 — 첫 청크와 내용이 다르면 직접 삭제 검토할 것")


if __name__ == "__main__":
    main()
