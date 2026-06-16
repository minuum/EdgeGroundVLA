









import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from robovlms.train.base_trainer import BaseTrainer

class NavTrainer(BaseTrainer):
    """
    V4 학습을 위한 커스텀 트레이너.
    기본 트레이너의 그래디언트 관리 로직을 보완하고 로깅을 강화함.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_grounding_aux()
        self._init_loss_balance()
        # Upstream/Base 설정이 만든 trainable 상태를 그대로 유지하고 요약만 남긴다.
        self._log_trainable_params()

    def _init_grounding_aux(self):
        self.grounding_aux_config = self.configs.get("grounding_aux", {}) or {}
        self.grounding_bbox_head = None
        self.grounding_coarse_head = None
        self.grounding_bbox_weight = float(self.grounding_aux_config.get("lambda_bbox", 0.0))
        self.grounding_coarse_weight = float(self.grounding_aux_config.get("lambda_coarse", 0.0))
        self.grounding_bbox_loss_type = str(self.grounding_aux_config.get("bbox_loss", "smooth_l1")).lower()

        if not self.grounding_aux_config.get("enabled", False):
            return

        act_head = getattr(self.model, "act_head", None)
        if act_head is None:
            print("⚠️ [NavTrainer] grounding_aux enabled but act_head is missing.", flush=True)
            return

        hidden_size = int(getattr(act_head, "hidden_size", 1024)) * int(getattr(act_head, "latent", 1))
        mlp_hidden = int(self.grounding_aux_config.get("mlp_hidden", max(hidden_size // 2, 128)))
        dropout = float(self.grounding_aux_config.get("dropout", 0.1))

        self.grounding_bbox_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 4),
            nn.Sigmoid(),
        )
        self.grounding_coarse_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),
        )
        print(
            f"✅ [NavTrainer] grounding_aux enabled: hidden={hidden_size}, "
            f"lambda_bbox={self.grounding_bbox_weight}, lambda_coarse={self.grounding_coarse_weight}",
            flush=True,
        )

    def _init_loss_balance(self):
        self.loss_balance_mode = str(self.grounding_aux_config.get("loss_balance_mode", "fixed")).lower()
        self.loss_balance_logits = None
        self.loss_balance_names = []
        self.loss_balance_eps = float(self.grounding_aux_config.get("loss_balance_eps", 1e-6))
        self.loss_balance_ema_momentum = float(self.grounding_aux_config.get("loss_balance_ema_momentum", 0.98))
        self.loss_balance_min_share = float(self.grounding_aux_config.get("loss_balance_min_share", 0.0))
        self.loss_balance_temperature = float(self.grounding_aux_config.get("loss_balance_temperature", 1.0))
        self.loss_balance_normalize = bool(self.grounding_aux_config.get("loss_balance_normalize", True))

        if self.loss_balance_mode != "learned":
            return

        names = ["action"]
        if self.grounding_bbox_weight > 0.0:
            names.append("bbox")
        if self.grounding_coarse_weight > 0.0:
            names.append("coarse")
        self.loss_balance_names = names

        n_tasks = len(names)
        if n_tasks == 0:
            self.loss_balance_mode = "fixed"
            return

        if self.loss_balance_min_share * n_tasks >= 1.0:
            raise ValueError(
                f"loss_balance_min_share ({self.loss_balance_min_share}) must satisfy "
                f"min_share * n_tasks < 1.0 (n_tasks={n_tasks})"
            )

        init_map = self.grounding_aux_config.get("initial_task_weights", {}) or {}
        init_weights = torch.tensor(
            [float(init_map.get(name, 1.0 / n_tasks)) for name in names],
            dtype=torch.float32,
        )
        init_weights = init_weights / init_weights.sum().clamp_min(self.loss_balance_eps)
        self.loss_balance_logits = nn.Parameter(init_weights.clamp_min(self.loss_balance_eps).log())
        self.register_buffer(
            "loss_balance_ema",
            torch.ones(n_tasks, dtype=torch.float32),
        )
        print(
            "✅ [NavTrainer] learned loss balance enabled: "
            f"tasks={names}, init={init_weights.tolist()}, "
            f"normalize={self.loss_balance_normalize}, min_share={self.loss_balance_min_share}",
            flush=True,
        )

    def _get_learned_task_weights(self):
        if self.loss_balance_mode != "learned" or self.loss_balance_logits is None:
            return None

        logits = self.loss_balance_logits / max(self.loss_balance_temperature, self.loss_balance_eps)
        weights = torch.softmax(logits, dim=0)
        if self.loss_balance_min_share > 0.0:
            n_tasks = weights.numel()
            weights = weights * (1.0 - self.loss_balance_min_share * n_tasks) + self.loss_balance_min_share
        return weights

    def _combine_losses(self, base_loss, aux_loss_dict, is_training: bool):
        loss_bbox = aux_loss_dict["loss_grounding_bbox"]
        loss_coarse = aux_loss_dict["loss_grounding_coarse"]

        if self.loss_balance_mode != "learned" or self.loss_balance_logits is None:
            total = base_loss + aux_loss_dict["loss_grounding_total"]
            return total, {}

        raw_losses = [base_loss]
        if self.grounding_bbox_weight > 0.0:
            raw_losses.append(loss_bbox)
        if self.grounding_coarse_weight > 0.0:
            raw_losses.append(loss_coarse)
        raw_losses = torch.stack(raw_losses)

        ema = self.loss_balance_ema.to(device=raw_losses.device, dtype=raw_losses.dtype)
        if is_training:
            with torch.no_grad():
                detached = raw_losses.detach().clamp_min(self.loss_balance_eps)
                updated = self.loss_balance_ema_momentum * ema + (1.0 - self.loss_balance_ema_momentum) * detached
                self.loss_balance_ema.copy_(updated.to(dtype=self.loss_balance_ema.dtype))
                ema = self.loss_balance_ema.to(device=raw_losses.device, dtype=raw_losses.dtype)

        if self.loss_balance_normalize:
            normalized_losses = raw_losses / ema.clamp_min(self.loss_balance_eps)
        else:
            normalized_losses = raw_losses

        weights = self._get_learned_task_weights().to(device=raw_losses.device, dtype=raw_losses.dtype)
        total = torch.sum(weights * normalized_losses)

        metrics = {
            "loss_balance_mode_learned": raw_losses.new_tensor(1.0),
        }
        for idx, name in enumerate(self.loss_balance_names):
            metrics[f"loss_share_{name}"] = weights[idx]
            metrics[f"loss_raw_{name}"] = raw_losses[idx]
            metrics[f"loss_norm_{name}"] = normalized_losses[idx]
            metrics[f"loss_ema_{name}"] = ema[idx]

        return total, metrics

    @classmethod
    def from_checkpoint(cls, checkpoint_path, load_source="nav", variant=None):
        """
        체크포인트로부터 트레이너 인스턴스를 생성하는 팩토리 메서드.
        액션 공간 최적화(6개 클래스 등) 시 발생하는 로딩 에러(size mismatch)를 방지합니다.
        """
        print(f"🚀 [NavTrainer] Creating NavTrainer from checkpoint: {checkpoint_path}", flush=True)
        
        # 1. 트레이너 및 모델 인스턴스 초기화 (config 기반)
        instance = cls(variant)
        
        # 2. 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", {}))
        
        # module. 프리픽스 제거 (DDP 호환)
        state_dict = {k.replace("module.model.", "model.").replace("module.", ""): v for k, v in state_dict.items()}
        
        # 3. 로드 시도
        try:
            # strict=False를 기본으로 하되, 층별 데이터 로드 시도
            msg = instance.load_state_dict(state_dict, strict=False)
            print(f"✅ [NavTrainer] Full checkpoint load successful (Missing: {msg.missing_keys}, Unexpected: {msg.unexpected_keys})", flush=True)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠️ [NavTrainer] Size mismatch detected during load! Attempting to load by excluding 'act_head'.", flush=True)
                # act_head 관련 가중치만 제외하고 다시 로드 (최적화 시 architecture가 바뀌기 때문)
                filtered_state_dict = {k: v for k, v in state_dict.items() if "act_head" not in k}
                msg = instance.load_state_dict(filtered_state_dict, strict=False)
                print(f"✅ [NavTrainer] Partial load successful (Excluded act_head. Missing: {msg.missing_keys})", flush=True)
            else:
                raise e
        
        # 4. 메모리 정리 및 로깅
        del state_dict
        if hasattr(instance, "_log_trainable_params"):
            instance._log_trainable_params()
            
        return instance

    def _log_trainable_params(self):
        """Base 설정 이후의 trainable 파라미터 상태를 점검하고 기록."""
        if self.model is None:
            print("⚠️ [NavTrainer] Model is None, skipping trainable-parameter audit.", flush=True)
            return

        print(f"📊 [NavTrainer] === Parameter Gradient Check (Model ID: {id(self.model)}) ===", flush=True)
        trainable_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        for name in trainable_names[:5]:
            print(f"  [Trainable] {name}", flush=True)

        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        grad_true_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"🔍 [NavTrainer] Total trainable parameters: {trainable_params_count:,}", flush=True)
        print(f"🔍 [NavTrainer] Count of params with requires_grad=True: {grad_true_count}", flush=True)

    def _process_batch(self, batch):
        """
        BaseTrainer의 _process_batch에서 발생하는 Action slicing 에러 방지 및 리턴 규격(19개) 준수
        """
        if self.configs.get("discrete_action", False):
            # batch에서 데이터 추출
            rgb = batch["rgb"].to(self.device).to(self.dtype)
            language = batch["text"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            arm_action = batch["action"].to(self.device) 
            raw_text = batch.get("raw_text", None)
            data_source = "mobile_vla_action"

            # 19개 리턴 규격 (BaseTrainer.py:468-488 참고)
            # BaseTrainer.training_step은 arm_action_chunck(10번째)를 action_labels로 사용함
            arm_action_chunck = batch.get("action_chunck", batch["action"]).to(self.device)
            return (
                rgb,                    # 00: rgb
                None,                   # 01: hand_rgb
                None,                   # 02: attention_mask
                language,               # 03: language
                text_mask,              # 04: text_mask
                None,                   # 05: fwd_rgb_chunck
                None,                   # 06: fwd_hand_rgb_chunck
                None,                   # 07: arm_action
                None,                   # 08: gripper_action
                arm_action_chunck,      # 09: arm_action_chunck
                None,                   # 10: gripper_action_chunck
                None,                   # 11: chunck_mask
                None,                   # 12: fwd_mask
                None,                   # 13: instr_and_action_ids
                None,                   # 14: instr_and_action_labels
                None,                   # 15: instr_and_action_mask
                raw_text,               # 16: raw_text
                None,                   # 17: rel_state
                data_source             # 18: data_source
            )
        
        # 일반적인 경우 부모 클래스 호출
        return super()._process_batch(batch)

    def training_step(self, batch, batch_idx):
        """
        BaseTrainer.training_step을 오버라이딩하여 Loss 전달 과정을 투명하게 관리
        """
        text_embedding = batch.get("text_embedding", None)
        if isinstance(text_embedding, torch.Tensor):
            text_embedding = text_embedding.to(self.device).to(self.dtype)

        # 데이터 전처리
        processed_batch = self._process_batch(batch)
        
        # 모델 Forward (BaseTrainer.training_step과 동일한 인자 구성)
        prediction = self.model.forward(
            processed_batch[0],      # rgb
            processed_batch[3],      # language
            attention_mask=processed_batch[4], # text_mask
            action_labels=(processed_batch[9], processed_batch[10]), # arm_action_chunck, gripper_action_chunck
            action_mask=processed_batch[11], # chunck_mask
            text_embedding=text_embedding,
            raw_text=processed_batch[16],
            data_source=processed_batch[18]
        )

        # Loss 추출
        loss_dict = self._get_loss(prediction)
        aux_loss_dict = self._compute_grounding_aux_loss(batch)
        train_loss, balance_metrics = self._combine_losses(loss_dict["loss"], aux_loss_dict, is_training=True)
        loss_dict.update(aux_loss_dict)
        loss_dict["loss_base"] = loss_dict["loss"]
        loss_dict["loss"] = train_loss
        loss_dict.update(balance_metrics)

        # 로그 기록
        for k, v in loss_dict.items():
            if v is not None and isinstance(v, torch.Tensor):
                self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=(k=="loss"))
        
        # Gradient 체크 (디버깅)
        if hasattr(train_loss, "requires_grad") and not train_loss.requires_grad:
            print(f"❌ [NavTrainer] CRITICAL: Training loss has NO gradients! prediction keys={list(prediction.keys())}", flush=True)
            
        return train_loss

    def on_validation_epoch_start(self):
        """검증 시작 전 메모리 정리"""
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation 시에도 동일한 로직 적용 + 메모리 절약
        """
        # 확실하게 grad 계산 차단
        with torch.no_grad():
            text_embedding = batch.get("text_embedding", None)
            if isinstance(text_embedding, torch.Tensor):
                text_embedding = text_embedding.to(self.device).to(self.dtype)

            processed_batch = self._process_batch(batch)
            prediction = self.model.forward(
                processed_batch[0],      # rgb
                processed_batch[3],      # language
                attention_mask=processed_batch[4],
                action_labels=(processed_batch[9], processed_batch[10]),
                action_mask=processed_batch[11],
                text_embedding=text_embedding,
                raw_text=processed_batch[16],
                data_source=processed_batch[18]
            )
            loss_dict = self._get_loss(prediction)
            aux_loss_dict = self._compute_grounding_aux_loss(batch)
            total_loss, balance_metrics = self._combine_losses(loss_dict["loss"], aux_loss_dict, is_training=False)
            loss_dict.update(aux_loss_dict)
            loss_dict["loss_base"] = loss_dict["loss"]
            loss_dict["loss"] = total_loss
            loss_dict.update(balance_metrics)
            
            for k, v in loss_dict.items():
                if v is not None and isinstance(v, torch.Tensor):
                    # validation 시에는 on_epoch=True만 사용 (기본값)
                    self.log(f"val_{k}", v, on_step=False, on_epoch=True, sync_dist=True, prog_bar=(k=="loss"))

    def on_validation_epoch_end(self):
        """검증 종료 후 메모리 정리"""
        torch.cuda.empty_cache()

    def _get_loss(self, prediction):
        """
        NavPolicy가 리턴한 Loss를 안전하게 추출.

        배경: base_backbone._update_loss(loss, action_loss, "act") 호출 시
        NavPolicy.loss()의 키들에 suffix '_act'가 붙는다.
          예) "loss_arm_act" -> "loss_arm_act_act"
              "loss_velocity" -> "loss_velocity_act"
              "acc_arm_act"   -> "acc_arm_act_act"

        또한 _format_loss()가 모든 "loss_*" 값들을 합산해 "loss" 키로 저장한다.
        따라서 "loss" 키를 우선 사용하고, 없으면 여러 후보 키를 순서대로 탐색한다.
        """
        # 1) _format_loss()가 만들어주는 통합 "loss" 키를 우선 사용
        loss = prediction.get("loss", None)

        # 2) loss가 없거나 requires_grad=False이면 후보 키에서 재탐색
        if loss is None or (isinstance(loss, torch.Tensor) and not loss.requires_grad):
            if loss is not None:
                # print(f"[NavTrainer] Found 'loss' key but requires_grad=False. Searching alternatives...", flush=True)
                pass 
            
            candidates = [
                "loss_arm_act_act",   # _update_loss(..., "act") suffix 버전
                "loss_velocity_act",
                "loss_arm_act",
                "loss_velocity",
                "loss_arm",
            ]
            for key in candidates:
                v = prediction.get(key, None)
                if v is not None and isinstance(v, torch.Tensor):
                    # print(f"[NavTrainer] Checking candidate '{key}': requires_grad={v.requires_grad}", flush=True)
                    # 학습 중일 때만 requires_grad 체크, validation/inference 시에는 체크 안 함
                    if not self.training or v.requires_grad:
                        loss = v
                        # print(f"[NavTrainer] SUCCESS: Using '{key}' as loss (requires_grad={v.requires_grad})", flush=True)
                        break

        # 3) 여전히 None이면 경고
        if loss is None:
            print(f"❌ [NavTrainer._get_loss] No valid loss! prediction keys={list(prediction.keys())}", flush=True)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # accuracy: suffix 버전 우선, 없으면 원본 키
        acc = prediction.get("acc_arm_act_act",
              prediction.get("acc_arm_act",
              prediction.get("acc_velocity_act",
              prediction.get("acc_velocity", 0.0))))

        return {
            "loss":         loss,
            "loss_arm_act": loss,
            "acc_arm_act":  acc,
        }

    def _compute_grounding_aux_loss(self, batch):
        zero = None
        act_head = getattr(self.model, "act_head", None)
        hidden_states = getattr(act_head, "last_hidden_states", None) if act_head is not None else None

        if hidden_states is None or self.grounding_bbox_head is None or self.grounding_coarse_head is None:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_grounding_bbox": zero,
                "loss_grounding_coarse": zero,
                "loss_grounding_total": zero,
            }

        zero = hidden_states.sum() * 0.0
        aux_hidden_dtype = next(self.grounding_bbox_head.parameters()).dtype
        aux_hidden_states = hidden_states.to(aux_hidden_dtype)

        bbox_targets = batch.get("grounding_bbox", None)
        bbox_mask = batch.get("grounding_bbox_mask", None)
        bbox_weight = batch.get("grounding_bbox_weight", None)
        coarse_labels = batch.get("grounding_coarse_label", None)
        coarse_mask = batch.get("grounding_coarse_mask", None)

        loss_bbox = zero
        loss_coarse = zero

        if bbox_targets is not None and bbox_mask is not None and bbox_weight is not None and self.grounding_bbox_weight > 0.0:
            bbox_targets = bbox_targets.to(self.device).to(aux_hidden_dtype)
            bbox_mask = bbox_mask.to(self.device).bool()
            bbox_weight = bbox_weight.to(self.device).to(aux_hidden_dtype)
            pred_bbox = self.grounding_bbox_head(aux_hidden_states)

            if bbox_mask.any():
                if self.grounding_bbox_loss_type == "l1":
                    bbox_loss = F.l1_loss(pred_bbox, bbox_targets, reduction="none")
                else:
                    bbox_loss = F.smooth_l1_loss(pred_bbox, bbox_targets, reduction="none")
                bbox_loss = bbox_loss.mean(dim=-1)
                weighted_bbox_loss = bbox_loss[bbox_mask] * bbox_weight[bbox_mask]
                loss_bbox = weighted_bbox_loss.sum() / bbox_weight[bbox_mask].sum().clamp_min(1e-6)

        if coarse_labels is not None and coarse_mask is not None and bbox_weight is not None and self.grounding_coarse_weight > 0.0:
            coarse_labels = coarse_labels.to(self.device).long()
            coarse_mask = coarse_mask.to(self.device).bool()
            coarse_weights = bbox_weight.to(self.device).to(aux_hidden_dtype)
            coarse_logits = self.grounding_coarse_head(aux_hidden_states)

            if coarse_mask.any():
                coarse_loss = F.cross_entropy(
                    coarse_logits[coarse_mask],
                    coarse_labels[coarse_mask],
                    reduction="none",
                )
                weighted_coarse_loss = coarse_loss * coarse_weights[coarse_mask]
                loss_coarse = weighted_coarse_loss.sum() / coarse_weights[coarse_mask].sum().clamp_min(1e-6)

        total = self.grounding_bbox_weight * loss_bbox + self.grounding_coarse_weight * loss_coarse
        return {
            "loss_grounding_bbox": loss_bbox,
            "loss_grounding_coarse": loss_coarse,
            "loss_grounding_total": total,
        }
