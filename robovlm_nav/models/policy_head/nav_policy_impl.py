"""
Mobile VLA 전용 Policy Head
2D 속도 (linear_x, linear_y) 처리에 특화
LSTMDecoder를 기반으로 하되, gripper 없이 2D 속도만 출력
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from robovlms.model.policy_head.base_policy import BasePolicyHead, lstm_decoder, MLPTanhHead, initialize_param


class MobileVLALSTMDecoder(BasePolicyHead):
    """
    Mobile VLA 전용 LSTMDecoder
    
    특징:
    - 2D 속도 (linear_x, linear_y)만 출력 - 0.4초 동안의 이동 방향 속도 조정
    - Gripper 없음
    - BasePolicyHead.loss를 오버라이드하여 2D 속도 Loss 계산
    """
    
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        super(MobileVLALSTMDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        
        # LSTM Decoder
        self.rnn = lstm_decoder(
            in_features * latent, hidden_size * latent, num_layers, policy_rnn_dropout_p
        )
        
        # 2D 속도 출력 (gripper 없음) - 0.4초 동안의 이동 방향 속도 조정
        # action_dim=2 (linear_x, linear_y)이므로 fwd_pred_next_n * 2 차원 출력
        self.velocities = MLPTanhHead(
            self.hidden_size * latent, fwd_pred_next_n * action_dim
        )
        # [NEW] Weighting for Non-Forward Actions (C)
        self._action_weight_non_forward = kwargs.get("action_weight_non_forward", 1.0)
        
        self.hidden_state = None
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def reset(self):
        self.hidden_state = None
        self.history_memory = []
        self.last_hidden_states = None

    def forward(self, tok_seq, h_0=None, **kwargs):
        """
        Forward pass (LSTMDecoder와 동일한 구조)
        
        Args:
            tok_seq: (B, seq_len, latent_num, feature_dim) 또는 (B, seq_len, in_features * latent)
            h_0: 초기 hidden state (optional)
        
        Returns:
            velocities: (B, seq_len, fwd_pred_next_n, action_dim) - 2D 속도 (linear_x, linear_y)
            None: gripper 없음 (BasePolicyHead.loss 호환성을 위해)
        """
        # Down sample 처리 (LSTMDecoder와 동일)
        # tok_seq shape 확인 및 처리
        if len(tok_seq.shape) == 4:
            # (B, seq_len, latent_num, feature_dim)
            if self.down_sample == "pooling":
                bs, seq_len = tok_seq.shape[:2]
                tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
                tok_seq = self.global_1d_pool(
                    tok_seq.permute(0, 2, 1)
                )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
                tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
            elif self.down_sample == "resampler":
                raise NotImplementedError
            elif self.down_sample == "none":
                tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
            else:
                raise NotImplementedError
        elif len(tok_seq.shape) == 3:
            # (B, seq_len, feature_dim) - 이미 flatten된 경우
            # latent=1이므로 그대로 사용
            pass
        else:
            raise ValueError(f"Unexpected tok_seq shape: {tok_seq.shape}")

        # History memory 처리 (LSTMDecoder와 동일)
        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)
            if len(self.history_memory) <= self.history_len:
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
        else:
            self.hidden_state = h_0
            # BitsAndBytes INT8: tok_seq가 FP16일 수 있으므로 LSTM dtype과 맞춤
            if tok_seq.dtype != next(self.rnn.parameters()).dtype:
                tok_seq = tok_seq.to(next(self.rnn.parameters()).dtype)
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n

        # 2D 속도 출력 - 0.4초 동안의 이동 방향 속도 조정
        velocities = self.velocities(x)
        # (B, seq_len, fwd_pred_next_n * action_dim) -> (B, seq_len, fwd_pred_next_n, action_dim)
        velocities = rearrange(velocities, "b l (n d) -> b l n d", n=self.fwd_pred_next_n, d=self.action_dim)

        # gripper 없음 (None 반환)
        return velocities, None

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        """
        입력된 전체 액션 시퀀스에서 다중 스텝 정답을 추출하여 반환
        labels[0]: (B, Total_L, 2) 형태의 전체 액션 시퀀스
        결과: (B, L, n, 2) 형태의 레이블 (L: window_size, n: fwd_pred_next_n)
        """
        arm_labels = labels[0]
        if arm_labels is None:
            return pred_actions, labels, action_masks
            
        # pred_actions는 (velocities, None) 형태
        logits = pred_actions[0] if isinstance(pred_actions, (tuple, list)) else pred_actions
        
        bs = arm_labels.shape[0]
        L = logits.shape[1] 
        n = self.fwd_pred_next_n
        
        # MobileVLATrainer에서 이미 chunked된 labels를 보냈을 경우 처리
        if arm_labels.dim() == 4: # (B, L_full, n, 2)
            if arm_labels.size(1) != L:
                arm_labels = arm_labels[:, :L]
            return pred_actions, (arm_labels, labels[1]), action_masks

        # 만약 (B, Total_L, 2) 형태라면 직접 chunking
        chunked = []
        for t in range(L):
            # t 시점부터 t + n 시점까지의 정답 추출
            if t + n <= arm_labels.shape[1]:
                chunk_t = arm_labels[:, t : t + n] # (B, n, 2)
            else:
                # 패딩 처리
                pad_size = (t + n) - arm_labels.shape[1]
                chunk_t = arm_labels[:, t:]
                padding = torch.zeros(bs, pad_size, 2).to(arm_labels.device)
                chunk_t = torch.cat([chunk_t, padding], dim=1)
            chunked.append(chunk_t)
            
        # (B, L, n, 2) 형태로 스택
        new_arm_labels = torch.stack(chunked, dim=1).to(arm_labels.device)
        
        return pred_actions, (new_arm_labels, labels[1]), action_masks

    def loss(self, pred_action, labels, attention_mask=None):
        """
        Mobile VLA용 Loss 계산 (2D Regression)
        """
        if labels is None or labels[0] is None:
            return {"loss_arm_act": None, "loss_gripper": None, "acc_gripper": None}

        # pred_action는 (velocities, None) 형태
        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            velocities = pred_action[0]  # (B, seq_len, chunk_size, 2)
        else:
            velocities = pred_action

        # labels는 (velocity_chunck, None) 형태
        velocity_labels = labels[0]  # (B, seq_len, chunk_size, 2)

        # 2D 속도 Loss 계산 (Huber Loss)
        if attention_mask is None:
            loss_velocity = torch.nn.functional.huber_loss(velocities, velocity_labels, reduction="none")
            
            # [Directional Awareness] Weighting
            if hasattr(self, "_action_weight_non_forward") and self._action_weight_non_forward != 1.0:
                # linear_x(0)가 크고 linear_y(1)가 작은 경우를 forward로 정의
                is_forward = (velocity_labels[..., 0] > 0.5) & (torch.abs(velocity_labels[..., 1]) < 0.2)
                weights = torch.ones_like(loss_velocity)
                weights[~is_forward] = self._action_weight_non_forward
                loss_velocity = loss_velocity * weights
            
            loss_velocity = loss_velocity.mean()
        else:
            loss_velocity = torch.nn.functional.huber_loss(
                velocities, velocity_labels, reduction="none"
            )
            
            # [Directional Awareness] Weighting
            if hasattr(self, "_action_weight_non_forward") and self._action_weight_non_forward != 1.0:
                is_forward = (velocity_labels[..., 0] > 0.5) & (torch.abs(velocity_labels[..., 1]) < 0.2)
                weights = torch.ones_like(loss_velocity)
                weights[~is_forward] = self._action_weight_non_forward
                loss_velocity = loss_velocity * weights
                
            # mask 차원 맞춤
            attention_mask = attention_mask.bool()
            while attention_mask.dim() < loss_velocity.dim():
                attention_mask = attention_mask.unsqueeze(-1)
            
            if attention_mask.any():
                loss_velocity = loss_velocity[attention_mask.expand_as(loss_velocity)].mean()
            else:
                loss_velocity = loss_velocity.mean() * 0.0

        return {
            "loss_arm_act": loss_velocity,
            "loss_velocity": loss_velocity,
            "loss_gripper": None,
            "acc_arm": None,
            "acc_gripper": None,
        }


class MobileVLAClassificationDecoder(BasePolicyHead):
    """
    Mobile VLA 전용 분류(Classification) Decoder
    
    기존의 연속값(Continuous MSE/Huber) 대신 이산적인 액션 클래스를 예측합니다.
    (Forward, Slide Left, Slide Right, Diag Left, Diag Right, Stop)
    """
    
    def __init__(
        self,
        in_features,
        action_dim, # num_classes (6)
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        num_classes = kwargs.get("num_classes", 6)
        super(MobileVLAClassificationDecoder, self).__init__(in_features, num_classes, **kwargs)
        self.num_classes = num_classes
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        
        # LSTM Decoder
        self.rnn = lstm_decoder(
            in_features * latent, hidden_size * latent, num_layers, policy_rnn_dropout_p
        )
        
        # 분류 헤드 (Logits 출력)
        self.logits = nn.Linear(self.hidden_size * latent, fwd_pred_next_n * num_classes)
        
        self.hidden_state = None
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
            
        # Class Weights for cross entropy
        class_weights = kwargs.get("class_weights", None)
        if class_weights is not None:
            self.register_buffer("class_weights_tensor", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights_tensor = None

        self.label_smoothing = float(kwargs.get("label_smoothing", 0.0))
        self.prior_reg_weight = float(kwargs.get("prior_reg_weight", 0.0))
        target_class_prior = kwargs.get("target_class_prior", None)
        if target_class_prior is not None:
            prior = torch.tensor(target_class_prior, dtype=torch.float)
            if prior.numel() != action_dim:
                raise ValueError(
                    f"target_class_prior length ({prior.numel()}) must match action_dim ({action_dim})"
                )
            if torch.any(prior < 0):
                raise ValueError("target_class_prior must be non-negative")
            prior = prior / prior.sum().clamp_min(1e-8)
            self.register_buffer("target_class_prior_tensor", prior)
        else:
            self.target_class_prior_tensor = None

        # Instruction conditioning (Exp13 backward-compat: additive bias)
        instr_in_features = kwargs.get("instr_in_features", None)
        if instr_in_features is not None:
            self.instr_proj = nn.Linear(instr_in_features, in_features * latent)
        else:
            self.instr_proj = None

        initialize_param(self)

        # Phase D: cross-attention text conditioning (added after initialize_param
        # so text_gate=0.1 is not overwritten by weight init)
        text_in_features = kwargs.get("text_in_features", None)
        self.text_cross_attn_enabled = text_in_features is not None
        if self.text_cross_attn_enabled:
            lstm_out_dim = hidden_size * latent
            self.text_proj = nn.Linear(text_in_features, lstm_out_dim)
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,
                num_heads=4,
                batch_first=True,
                dropout=0.0,
            )
            # learned scalar gate: starts at 0.1 so model initially trusts vision,
            # grows as text proves useful during training
            self.text_gate = nn.Parameter(torch.tensor(0.1))

    def reset(self):
        self.hidden_state = None
        self.history_memory = []
        self.last_hidden_states = None

    def forward(self, tok_seq, h_0=None, **kwargs):
        # 강제로 그래디언트 계산 활성화
        torch.set_grad_enabled(True)
        if not tok_seq.requires_grad:
            tok_seq.requires_grad_(True)

        self.debug_tok_seq_grad = tok_seq.requires_grad
        if len(tok_seq.shape) == 4:
            if self.down_sample == "pooling":
                bs, seq_len = tok_seq.shape[:2]
                tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
                tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
                tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
            elif self.down_sample == "none":
                tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")

        # Instruction conditioning (Exp13): additive bias on LSTM input
        instruction_emb = kwargs.get("instruction_emb", None)
        if instruction_emb is not None and self.instr_proj is not None:
            # instruction_emb: (bs, embed_dim)  tok_seq: (bs, ws, in_features)
            instr_feat = self.instr_proj(instruction_emb.to(tok_seq.dtype))  # (bs, in_features)
            instr_feat = instr_feat.unsqueeze(1).expand_as(tok_seq)           # (bs, ws, in_features)
            tok_seq = tok_seq + instr_feat

        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)
            if len(self.history_memory) <= self.history_len:
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
            else:
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
        else:
            self.hidden_state = h_0
            if tok_seq.dtype != next(self.rnn.parameters()).dtype:
                tok_seq = tok_seq.to(next(self.rnn.parameters()).dtype)
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n

        self.last_hidden_states = x

        # Phase D: LSTM hidden이 text token sequence에 cross-attend
        # query: (B, ws, lstm_out_dim)  key/value: (B, T, 2048) → projected to (B, T, lstm_out_dim)
        # gate α: 초기 0.1, 학습 중 text conditioning이 유용할수록 커짐
        if self.text_cross_attn_enabled:
            text_feats = kwargs.get("text_features", None)  # (B, T, 2048)
            if text_feats is not None:
                K = self.text_proj(text_feats.to(x.dtype))  # (B, T, lstm_out_dim)
                attended, _ = self.text_cross_attn(
                    query=x, key=K, value=K,
                    need_weights=False,
                )
                x = x + self.text_gate * attended

        # 클래스별 Logits 출력
        logits = self.logits(x)
        logits = rearrange(logits, "b l (n d) -> b l n d", n=self.fwd_pred_next_n, d=self.num_classes)

        return logits, None

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        """
        입력된 전체 액션 시퀀스에서 다중 스텝 정답을 추출하여 반환
        labels[0]: (B, 17) 형태의 전체 액션 시퀀스
        결과: (B, L, n) 형태의 레이블 (L: window_size, n: fwd_pred_next_n)
        """
        arm_labels = labels[0]
        if arm_labels is None:
            return pred_actions, labels, action_masks
            
        # pred_actions는 (logits, gripper) 형태일 수 있음
        logits = pred_actions[0] if isinstance(pred_actions, (tuple, list)) else pred_actions
        
        bs = arm_labels.shape[0]
        L = logits.shape[1] 
        n = self.fwd_pred_next_n # 5
        
        # 이미 쪼개진 데이터(B, L, n)가 왔을 경우 중복 처리 방지
        if arm_labels.dim() >= 3:
            return pred_actions, labels, action_masks
            
        chunked = []
        for t in range(L):
            # t 시점부터 t + n 시점까지의 정답 추출
            # arm_labels의 길이가 충분할 때만 chunking 수행
            if t + n <= arm_labels.shape[1]:
                chunk_t = arm_labels[:, t : t + n] # (B, n)
            else:
                # 마지막 구간 패딩 (마지막 값 복사)
                last_val = arm_labels[:, -1:]
                pad_len = (t + n) - arm_labels.shape[1]
                chunk_t = torch.cat([arm_labels[:, t:], last_val.repeat(1, pad_len)], dim=1)
            chunked.append(chunk_t)
            
        # (B, L, n) 형태로 스택
        new_arm_labels = torch.stack(chunked, dim=1).to(arm_labels.device)
        
        return pred_actions, (new_arm_labels, labels[1]), action_masks

    def loss(self, pred_action, labels, attention_mask=None):
        if labels is None or labels[0] is None:
            return {"loss_velocity": None, "loss_gripper": None, "acc_velocity": None}

        logits = pred_action[0] if isinstance(pred_action, (tuple, list)) else pred_action
        class_labels = labels[0] # (B, L, n_steps)
        
        # 차원 확인 (B, L, n_steps)가 아닐 경우에만 확장 (backward compatibility)
        if class_labels.dim() == 2:
            class_labels = class_labels.unsqueeze(-1)

        # Check sequence length mismatch (L)
        l_logits = logits.size(1)
        l_labels = class_labels.size(1)
        if l_logits != l_labels:
            min_l = min(l_logits, l_labels)
            logits = logits[:, :min_l]
            class_labels = class_labels[:, :min_l]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :min_l]

        n_logits = logits.size(2)
        n_labels = class_labels.size(2)
        if n_logits != n_labels:
            min_n = min(n_logits, n_labels)
            logits = logits[:, :, :min_n]
            class_labels = class_labels[:, :, :min_n]
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask[:, :, :min_n]

        flat_logits = rearrange(logits, "b l n d -> (b l n) d")
        flat_labels = rearrange(class_labels, "b l n -> (b l n)").long()

        if attention_mask is not None:
            # attention_mask도 차원 확인 (B, L) -> (B, L, 1)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            flat_mask = rearrange(attention_mask, "b l n -> (b l n)").bool()
            
            # mask와 labels 크기 일치 확인 (간혹 BaseTrainer에서 text_mask를 넘길 경우 n=1이 아닐 수 있음)
            if flat_mask.size(0) != flat_labels.size(0):
                # 크기가 다르면 마스킹 생략 (로그 출력)
                # print(f"⚠️ [NavPolicy] Mask size {flat_mask.size(0)} mismatch with labels {flat_labels.size(0)}")
                pass
            else:
                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]

        if flat_logits.shape[0] != flat_labels.shape[0]:
            # match size 0 by trimming the larger one
            min_size = min(flat_logits.shape[0], flat_labels.shape[0])
            flat_logits = flat_logits[:min_size]
            flat_labels = flat_labels[:min_size]

        if flat_labels.size(0) == 0:
            # logits에 0을 곱해 더함으로써 grad_fn을 유지하고 값은 0이 되도록 함
            return {
                "loss_arm_act": logits.sum() * 0.0,
                "loss_velocity": logits.sum() * 0.0,
                "acc_arm_act": 0.0,
                "acc_velocity": 0.0,
            }

        ce_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            weight=self.class_weights_tensor,
            label_smoothing=self.label_smoothing,
        )

        prior_loss = None
        if self.prior_reg_weight > 0.0 and self.target_class_prior_tensor is not None:
            probs = flat_logits.softmax(dim=-1).mean(dim=0)
            prior = self.target_class_prior_tensor.to(device=probs.device, dtype=probs.dtype)
            prior = (prior + 1e-8) / (prior + 1e-8).sum()
            prior_loss = F.kl_div(probs.clamp_min(1e-8).log(), prior, reduction="sum")
            loss = ce_loss + self.prior_reg_weight * prior_loss
        else:
            loss = ce_loss
        
        # Accuracy 계산
        preds = flat_logits.argmax(dim=-1)
        acc = (preds == flat_labels).float().mean()

        return {
            "loss_arm_act": loss,
            "loss_velocity": loss,
            "loss_ce": ce_loss,
            "loss_prior_reg": prior_loss,
            "loss_gripper": None,
            "acc_arm_act": acc,
            "acc_velocity": acc,
        }
