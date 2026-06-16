# 🚀 MoNaVLA V5 Robot Deployment Checklist

**Last Updated**: 2026-04-22
**Source Server**: Billy (Training)
**Target Server**: Soda (Robot)

---

## 1. Git & Code base
- [ ] **Branch**: `inference-integration`
- [ ] **Commit Hash**: `ea5b66e24fb76d0e15e6c057c9a567683fd0accd`
- [ ] **Check**: `git pull origin inference-integration` 및 해당 커밋 체크아웃 완료 여부

## 2. Essential Files (Transfer from Billy to Soda)
### 📦 Model Checkpoints (.ckpt)
- [x] **Current local default**: `runs/v5_nav/kosmos/mobile_vla_v5_exp25/2026-04-22/v5-exp25-step3-balanced-objective/epoch_epoch=epoch=02-val_loss=val_loss=10.117.ckpt`
- [ ] **Fallback candidate**: 별도 선정 필요
  *※ 주의: /tmp/ 대신 고정된 NFS 또는 로컬 경로 사용*

### ⚙️ Configuration Files (.json)
- [x] **Current local default**: `configs/mobile_vla_v5_exp25_step3_balanced_objective.json`
- [ ] **Fallback config**: 별도 선정 필요
- [ ] **Parent Chain**: 상속된 모든 부모 config 파일 존재 확인

### 🧠 VLM Backbone
- [ ] **Path**: `.vlms/kosmos-2-patch14-224`
- [ ] **Check**: 로봇 서버에 해당 가중치 파일 존재 여부 (없으면 전송 필요)

## 3. Environment Variables (.env / .vla_env_settings)
```bash
export VLA_CHECKPOINT_PATH="/home/soda/MoNaVLA/runs/v5_nav/kosmos/mobile_vla_v5_exp25/2026-04-22/v5-exp25-step3-balanced-objective/epoch_epoch=epoch=02-val_loss=val_loss=10.117.ckpt"
export VLA_CONFIG_PATH="/home/soda/MoNaVLA/configs/mobile_vla_v5_exp25_step3_balanced_objective.json"
export VLA_API_KEY="v3_test_key"
```

## 4. Execution & Validation
- [ ] **Inference Server**: `python3 robovlm_nav/serve/inference_server.py` 가동
- [ ] **Health Check**: `curl http://localhost:8000/health` -> `{"status": "ok"}`
- [ ] **ROS Launch**: `ROS_action/start_mobile_vla.sh` 실행
- [ ] **Inference Monitor**: Gradio UI에서 텍스트 명령어에 따른 토큰 출력 확인

---

## 🎯 Selection Criteria (Based on Short-term Eval)
1. **Closed-loop Success**: 모델이 루프를 완주했는가?
2. **Non-straight Prefix@5**: 직진 편향을 깨고 조향 명령을 생성하는가?
3. **Rollout Degradation**: FPE/TLD 수치가 낮고 경로가 붕괴되지 않는가?
