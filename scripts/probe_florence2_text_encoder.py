#!/usr/bin/env python3
"""CH68 68-6 우선순위 2번 — Florence-2 언어부(140.2M) 텍스트 임베딩 품질 검정.

docs/plans/plan_20260830_ch68_language_conditioning_l0l4.md 2단계.
질문: Florence-2 텍스트 인코더가 타겟 명사(basket/chair/...)를 서로 구분되는
임베딩으로 인코딩하는가? (B 경로: 헤드 concat 언어조건화의 백본 후보 검정)

CH38-4/38-5(hidden state cos_dist)와 동일한 측정 프레임: "객체명 간 분리도"를
"객체명 vs 방향/동작 어휘 간 분리도"와 비교한다. CH38 기준(객체 대비 0.36~0.44,
방향 대비 0.03~0.05, 약 10배 격차)에 준하면 "쓸 만하다"고 판정.

68-4 규칙(액션 어휘 금지) 위반 문장도 대조군으로 넣어 실제로 분리가 잘
되는지/안 되는지 같이 확인 — "curve to the left" 같은 금지 문구가 방향
어휘 자체와 얼마나 가까운 임베딩을 갖는지도 참고 정보로 남긴다.

인코더: language_model.model.encoder (BART식 encoder-decoder), 토큰 평균 풀링.

출력: docs/v5/detector/florence2_text_embedding_probe.json
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "microsoft/Florence-2-base"
OUT = ROOT / "docs/v5/detector/florence2_text_embedding_probe.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CH27 zero-shot probe와 동일 계열 객체명(허용 문구, 68-4 규칙 준수 — 타겟 명사만)
OBJECT_PHRASES = {
    "basket": "gray basket",
    "chair": "chair",
    "apple": "apple",
    "mug": "coffee mug",
    "cone": "traffic cone",
    "ball": "red ball",
}
# 68-4 허용 문구(타겟 명사 + 이동 동사, 방향/동작 어휘 없음)
ALLOWED_INSTRUCTIONS = {
    "go_chair": "go to the chair",
    "reach_basket": "reach the basket",
    "move_basket": "move toward the gray basket",
}
# 68-4 금지 문구(방향/동작 어휘 포함 — 라벨 누출 위험군, 참고용 대조군)
BANNED_INSTRUCTIONS = {
    "curve_left": "curve to the left",
    "go_straight": "go straight",
    "turn_right": "turn right to reach it",
}
# 순수 방향/동작 어휘(CH38의 "방향 어휘" 대응)
DIRECTION_WORDS = {"left": "left", "right": "right", "straight": "straight", "turn": "turn"}


def encode(encoder, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = encoder(input_ids=ids["input_ids"].to(DEVICE),
                      attention_mask=ids["attention_mask"].to(DEVICE))
    # 토큰 평균 풀링(패딩 없음, 단일 문장이라 attention_mask 전부 1)
    return out.last_hidden_state.mean(1).squeeze(0).float().cpu()


def cos_dist(a, b):
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def pairwise_mean(embs):
    keys = list(embs.keys())
    dists = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dists.append(cos_dist(embs[keys[i]], embs[keys[j]]))
    return dists


def main():
    print("[Florence-2] 로딩...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                  torch_dtype=torch.float16).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = proc.tokenizer
    encoder = model.language_model.model.encoder
    n_params = sum(p.numel() for p in model.language_model.parameters())
    print(f"[Florence-2] 언어부 파라미터 {n_params/1e6:.1f}M", flush=True)

    obj_embs = {k: encode(encoder, tokenizer, v) for k, v in OBJECT_PHRASES.items()}
    allowed_embs = {k: encode(encoder, tokenizer, v) for k, v in ALLOWED_INSTRUCTIONS.items()}
    banned_embs = {k: encode(encoder, tokenizer, v) for k, v in BANNED_INSTRUCTIONS.items()}
    dir_embs = {k: encode(encoder, tokenizer, v) for k, v in DIRECTION_WORDS.items()}

    # ① 객체명 간 분리도(서로 다른 객체 쌍 평균 cos_dist)
    obj_pairwise = pairwise_mean(obj_embs)
    obj_sep = sum(obj_pairwise) / len(obj_pairwise)

    # ② 객체명 vs 방향어휘 간 분리도
    obj_vs_dir = [cos_dist(o, d) for o in obj_embs.values() for d in dir_embs.values()]
    obj_dir_sep = sum(obj_vs_dir) / len(obj_vs_dir)

    # ③ 허용 문구(타겟 명사 포함, 방향어휘 없음) 간 분리도 — 실제 지시문끼리 얼마나 갈리는지
    allowed_pairwise = pairwise_mean(allowed_embs)
    allowed_sep = sum(allowed_pairwise) / len(allowed_pairwise)

    # ④ 참고: 금지 문구(방향어휘 포함) vs 순수 방향어휘 간 유사도 — 라벨누출 위험 문구가
    #    실제로 방향어휘와 얼마나 가까운 임베딩을 갖는지(참고 정보, 통과기준 아님)
    banned_vs_dir = {bk: {dk: cos_dist(be, de) for dk, de in dir_embs.items()}
                     for bk, be in banned_embs.items()}

    ratio = obj_dir_sep / obj_sep if obj_sep > 0 else float("inf")
    # CH38 기준: 객체 0.36~0.44 vs 방향 0.03~0.05 → 약 8~14배(양방향 다름에 유의,
    # 원 지표는 "얼마나 가까운가"라 방향값이 작을수록 판별력이 없다는 뜻 —
    # 여기서는 "분리도(거리)"라 방향값이 작으면(=가까우면=구분 안 됨) 좋은 신호.
    # 즉 CH38과 부호가 반대 방향으로 해석해야 함: 우리 obj_dir_sep가 obj_sep보다
    # 작을수록(방향어휘끼리 더 뭉쳐있을수록) 텍스트 인코더가 "판별에 무관한 축"을
    # 잘 눌러준다는 뜻 — 나쁜 신호가 아니라 좋은 신호. 판정은 obj_sep 자체가
    # 충분히 크고(객체명이 서로 잘 갈라짐), obj_dir_sep이 obj_sep 대비 작을수록 좋음.

    passed = obj_sep > 0.15 and ratio < 0.6  # 잠정 임계값 — 결과 보고 재조정 여지 있음

    result = dict(
        model=MODEL_ID, n_lang_params_M=n_params / 1e6,
        object_pairwise_sep_mean=obj_sep,
        object_pairwise_sep_all=obj_pairwise,
        object_vs_direction_sep_mean=obj_dir_sep,
        allowed_instruction_pairwise_sep_mean=allowed_sep,
        ratio_dir_over_obj=ratio,
        banned_instruction_vs_direction_word=banned_vs_dir,
        objects=list(OBJECT_PHRASES.keys()),
        allowed_instructions=list(ALLOWED_INSTRUCTIONS.values()),
        banned_instructions=list(BANNED_INSTRUCTIONS.values()),
        passed_tentative=passed,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n객체명 간 평균 분리도(cos_dist): {obj_sep:.4f}")
    print(f"객체명 vs 방향어휘 평균 분리도: {obj_dir_sep:.4f}")
    print(f"허용 지시문끼리 평균 분리도: {allowed_sep:.4f}")
    print(f"비율(방향/객체): {ratio:.3f}  (작을수록 = 방향축이 덜 부각됨 = 좋은 신호)")
    print(f"잠정 판정(임계값 obj_sep>0.15 and ratio<0.6): {'PASS' if passed else 'FAIL'}")
    print(f"\n저장 → {OUT}")


if __name__ == "__main__":
    main()
