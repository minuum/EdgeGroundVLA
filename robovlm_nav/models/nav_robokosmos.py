"""
NavRoboKosMos — instruction-conditioned Kosmos-2 backbone for navigation.

RoboKosMos의 subclass. 추가 사항:
- forward_continuous에서 lang_x word embedding token sequence를 추출 (Phase D: no detach)
- _forward_action_head에서 text_features + instruction_emb kwarg로 act_head에 전달
- MobileVLAClassificationDecoder의 cross-attention head가 text_features를 받아
  LSTM hidden output에 conditioning 적용

third_party/RoboVLMs 수정 불필요.
"""
import torch
from robovlms.model.backbone.robokosmos import RoboKosMos


class NavRoboKosMos(RoboKosMos):
    """RoboKosMos + token-level instruction conditioning via cross-attention (Phase D)."""

    def forward_continuous(self, vision_x, lang_x, attention_mask=None, text_embedding=None, **kwargs):
        """lang_x word embedding token sequence를 _text_seq_cache에 저장 후 parent 호출.

        Phase D 핵심: detach 제거 + mean pool 제거 → gradient가 text_proj까지 흐름.
        text_embedding: (bs, D) frozen embeddings for Exp18 backward-compat.
        """
        if text_embedding is not None:
            # [Exp18 backward-compat] frozen embedding 직접 사용
            self._instr_emb_cache = text_embedding
            self._text_seq_cache = None
        elif lang_x is not None:
            instr_embeds = self.word_embedding(lang_x)          # (B, T, 2048) — no detach
            # Phase D: token-level sequence for cross-attention
            self._text_seq_cache = instr_embeds                 # (B, T, 2048)
            # backward-compat: mean for legacy instr_proj path (no detach)
            self._instr_emb_cache = instr_embeds.mean(dim=1)    # (B, 2048)
        else:
            self._instr_emb_cache = None
            self._text_seq_cache = None

        return super().forward_continuous(
            vision_x, lang_x, attention_mask=attention_mask, **kwargs
        )

    def _forward_action_head(
        self, action_tokens, action_labels, action_mask, mode="train", **kwargs
    ):
        """캐시된 text features와 instruction embedding을 kwargs로 act_head에 전달."""
        instr_emb = getattr(self, "_instr_emb_cache", None)
        if instr_emb is not None:
            kwargs["instruction_emb"] = instr_emb

        # Phase D: token-level text features for cross-attention conditioning
        text_seq = getattr(self, "_text_seq_cache", None)
        if text_seq is not None:
            kwargs["text_features"] = text_seq              # (B, T, 2048)

        return super()._forward_action_head(
            action_tokens, action_labels, action_mask, mode=mode, **kwargs
        )
