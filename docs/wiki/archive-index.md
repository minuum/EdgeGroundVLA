# 아카이브 색인 — docs/*.md 스냅샷 357개 (압축 없음, 원문 그대로)

> 2025-12~2026-04 시기(research_story.html이 다루는 2026-05~ 이전)의 별개
> 프로젝트 단계 기록 — 양자화/서빙 인프라, manipulation vs navigation 비교 등.
> 2026-08-28 검토 결과 대부분 폐기된 방향의 죽은 기록으로 판단되어, 원문을
> 읽어 압축하지 않고 **제목/날짜/한줄요약만** 모았다. 실제로 필요한 파일이
> 있으면 `docs/<파일명>`으로 직접 열어서 확인한다 — 이 색인은 찾아가기용.

총 357개 파일.

| 파일 | 날짜 | 제목 | 한줄요약 |
|---|---|---|---|
| [`FINAL_PROJECT_SUMMARY.md`](../FINAL_PROJECT_SUMMARY.md) | 2024-08-22 | Final Project Summary - Mobile VLA | Mobile VLA는 로봇 제어를 위한 멀티모달 AI 시스템으로, 이미지와 텍스트 명령을 입력받아 로봇의 2D 액션(linear_x, linear_y)을 예측하는 시스템입니다. |
| [`PERFORMANCE_SUMMARY.md`](../PERFORMANCE_SUMMARY.md) | 2024-08-22 | Performance Summary - Mobile VLA | \| 메트릭 \| 목표 값 \| 현재 값 \| 달성률 \| |
| [`ROBOVLMS_CODE_ANALYSIS.md`](../ROBOVLMS_CODE_ANALYSIS.md) | 2024-10-30 | RoboVLMs 코드 분석 및 수정 내역 | - **URL**: https://github.com/Robot-VLAs/RoboVLMs |
| [`CORRECTED_CASE_ANALYSIS.md`](../CORRECTED_CASE_ANALYSIS.md) | 2025-11-06 | 실제 학습 케이스 정정 (환각 제거) | \| 학습명 \| 날짜 \| VLM Init \| Data \| Checkpoint \| 상태 \| |
| [`FINAL_SUMMARY_20251106.md`](../FINAL_SUMMARY_20251106.md) | 2025-11-06 | 🎉 Mobile VLA LoRA Fine-tuning 최종 요약 (20251106) | - ✅ RoboVLMs upstream config 기반 설정 |
| [`IMPLEMENTATION_STATUS_20251106.md`](../IMPLEMENTATION_STATUS_20251106.md) | 2025-11-06 | Mobile VLA LoRA Fine-tuning 구현 상태 (20251106) | - [x] `configs/finetune_mobile_vla_lora_20251106.json` |
| [`LORA_FINETUNING_SUMMARY.md`](../LORA_FINETUNING_SUMMARY.md) | 2025-11-06 | Mobile VLA LoRA Fine-tuning 실행 요약 | Mobile_VLA/configs/finetune_mobile_vla_lora_20251106.json |
| [`README_LORA_FINETUNING.md`](../README_LORA_FINETUNING.md) | 2025-11-06 | Mobile VLA LoRA Fine-tuning Guide (20251106 Episodes) | 20251106 날짜에 수집한 에피소드를 Kosmos VLM에 LoRA로 파인튜닝하는 가이드입니다. |
| [`PAPER_KNOWLEDGE_BASE_20251120.md`](../PAPER_KNOWLEDGE_BASE_20251120.md) | 2025-11-20 | 🔬 Mobile VLA 논문 실험 및 아키텍처 지식 베이스 | - **차원**: 7D `[x, y, z, roll, pitch, yaw, gripper]` |
| [`PAPER_ROADMAP_AND_STATUS_20251120.md`](../PAPER_ROADMAP_AND_STATUS_20251120.md) | 2025-11-20 | 📄 Mobile VLA 논문 출판 프로젝트 로드맵 & 현황 (2025-11-20) | **"소규모 데이터와 LoRA를 활용한 효율적인 Vision-Language-Action 모델 기반 모바일 로봇 주행"** |
| [`PROJECT_STRUCTURE_AND_ARCH_20251120.md`](../PROJECT_STRUCTURE_AND_ARCH_20251120.md) | 2025-11-20 | 🏗️ Mobile VLA 프로젝트 구조 및 모델 세부 아키텍처 (2025-11-20) | 지금까지 진행된 데이터 수집부터 학습, 그리고 추론 준비까지의 전체 파이프라인입니다. |
| [`TRAINING_ANALYSIS_20251120.md`](../TRAINING_ANALYSIS_20251120.md) | 2025-11-20 | 📊 Training Analysis Report (2025-11-20) | 현재 `run_lora_finetune_20251114.sh` 스크립트를 통해 Mobile VLA 모델의 LoRA 파인튜닝이 진행 중입니다. |
| [`TRAINING_RESULTS_20251120.md`](../TRAINING_RESULTS_20251120.md) | 2025-11-20 | 📊 LoRA Fine-tuning 학습 결과 (2025-11-20) | - **총 Epoch**: 10 (목표 달성 ✅) |
| [`TRAINING_STATUS_REPORT_20251120.md`](../TRAINING_STATUS_REPORT_20251120.md) | 2025-11-20 | 학습 현황 보고서 (2025-11-20) | - ❌ **학습 프로세스**: 실행 중이 아님 (완료 또는 중단됨) |
| [`ACCURATE_MODEL_ANALYSIS.md`](../ACCURATE_MODEL_ANALYSIS.md) | 2025-12-04 | 🔍 정확한 모델 구조 파악 - Frozen vs LoRA | Microsoft Kosmos-2 (원본) |
| [`ACCURATE_TODO_LIST.md`](../ACCURATE_TODO_LIST.md) | 2025-12-04 | 남은 TODO 리스트 - 환각 없는 정확한 분석 | - Context vector analysis ✅ |
| [`COMPLETE_EXPERIMENT_MATRIX.md`](../COMPLETE_EXPERIMENT_MATRIX.md) | 2025-12-04 | 전체 실험 케이스 최종 계획 및 결과 | \| Case \| VLM \| Data \| Episodes \| Status \| Val Loss \| 날짜 \| |
| [`COMPREHENSIVE_ANALYSIS.md`](../COMPREHENSIVE_ANALYSIS.md) | 2025-12-04 | Mobile-VLA 프로젝트 종합 분석 보고서 | > RoboVLMs (7DOF Manipulator VLM)을 활용하여 Mobile Robot (2DOF)을 제어 |
| [`COMPREHENSIVE_TRAINING_REPORT.md`](../COMPREHENSIVE_TRAINING_REPORT.md) | 2025-12-04 | Mobile-VLA 학습 전체 정리 및 이슈 분석 (1차 완료) | \| Case# \| 날짜 \| Model \| VLM Backbone \| VLM Freeze \| LoRA \| Action Head \| 데이터 \| Epochs \| Best Val Loss \| 상태 \| |
| [`CRITICAL_DISCOVERY.md`](../CRITICAL_DISCOVERY.md) | 2025-12-04 | 🚨 중대한 발견: RoboVLMs 모델을 사용하지 않았음! | // Mobile_VLA/configs/mobile_vla_20251203_lora.json |
| [`FINAL_EXECUTION_RESULTS.md`](../FINAL_EXECUTION_RESULTS.md) | 2025-12-04 | 최종 실행 결과 요약 | Epoch: 10/10 (100%) |
| [`PROGRESS_EVALUATION.md`](../PROGRESS_EVALUATION.md) | 2025-12-04 | 프로젝트 진행 상황 평가 (2025-12-04 15:55) | - [x] 250 left episodes 수집 |
| [`ROBOVLMS_READY.md`](../ROBOVLMS_READY.md) | 2025-12-04 | ✅ RoboVLMs 원본 모델 준비 완료! | 위치: /home/billy/.cache/huggingface/hub/models--robovlms--RoboVLMs/ |
| [`TEST_PLAN_MATRIX.md`](../TEST_PLAN_MATRIX.md) | 2025-12-04 | Mobile-VLA 모델 버전 및 테스트 계획 | \| Model \| 출처 \| 설명 \| 경로 \| 상태 \| |
| [`TRAINING_CASES_SUMMARY.md`](../TRAINING_CASES_SUMMARY.md) | 2025-12-04 | Mobile-VLA 학습 케이스별 결과 정리 (1차 정리) | 총 4개의 학습 케이스 진행 (2025-11 ~ 2025-12) |
| [`Case3_Performance_Analysis.md`](../Case3_Performance_Analysis.md) | 2025-12-05 | Case 3 Performance Analysis | **Frozen VLM + Action Head (250 Left + 250 Right)** |
| [`DAY1_PROGRESS_20251205.md`](../DAY1_PROGRESS_20251205.md) | 2025-12-05 | Day 1 진행 상황 (2025-12-05) | - **교수님 미팅 정리**: `docs/PROFESSOR_MEETING_20251205.md` |
| [`PROFESSOR_MEETING_20251205.md`](../PROFESSOR_MEETING_20251205.md) | 2025-12-05 | 교수님 미팅 (2025-12-05) - VLM Frozen vs LoRA 비교 실험 | - VLM: LoRA로 Fine-tuning (일부 학습) |
| [`PROFESSOR_QUESTIONS_TODO.md`](../PROFESSOR_QUESTIONS_TODO.md) | 2025-12-05 | 교수님 의문점 해결 TODO (우선순위별) | > VLM을 Freeze 하는 것 vs LoRA로 Fine-tuning 하는 것의 차이 |
| [`VLA_PAPERS_FROZEN_VS_FINETUNING.md`](../VLA_PAPERS_FROZEN_VS_FINETUNING.md) | 2025-12-05 | VLA 논문 조사: Frozen vs Fine-tuning 비교 | - PaLM-E, PaLI-X 같은 대형 VLM을 **robotic data로 co-fine-tune** |
| [`professor_meeting_prep_log.md`](../professor_meeting_prep_log.md) | 2025-12-06 | 교수님 미팅 준비 실행 로그 | - **스크립트**: `scripts/compare_frozen_vs_lora.py` |
| [`POETRY_EXECUTION_GUIDE.md`](../POETRY_EXECUTION_GUIDE.md) | 2025-12-07 | Poetry 환경에서 VLA 추론 실행 가이드 | - **환경 경로**: `/home/billy/.cache/pypoetry/virtualenvs/robovlms-ASlHafON-py3.10` |
| [`ACTION_CHUNKING_ANALYSIS.md`](../ACTION_CHUNKING_ANALYSIS.md) | 2025-12-09 | Action Chunking 분석 (fwd_pred_next_n) | - **RoboVLMs 원본**: 모두 `fwd_pred_next_n=10` 사용 |
| [`AUTO_TEST_MONITORING.md`](../AUTO_TEST_MONITORING.md) | 2025-12-09 | VLA 추론 테스트 자동화 실행 가이드 | 스크립트가 백그라운드에서 실행 중입니다: |
| [`DATA_AUGMENTATION_STRATEGY.md`](../DATA_AUGMENTATION_STRATEGY.md) | 2025-12-09 | Mobile VLA 데이터 증강 전략 및 계획 (2025-12-09) | 제한된 데이터셋(500 에피소드)의 한계를 극복하고 모델의 일반화 성능 향상 |
| [`DATA_INCREASE_ANALYSIS.md`](../DATA_INCREASE_ANALYSIS.md) | 2025-12-09 | 데이터 증가 원인 분석 | 데이터가 250개에서 500개로 증가한 것은 **데이터 증강(augmentation)이 아닌 필터링 조건 차이**입니다. |
| [`DENORMALIZATION_STRATEGY.md`](../DENORMALIZATION_STRATEGY.md) | 2025-12-09 | 정규화 해제 (Denormalization) 전략 문서 | 모델 출력: `[-1.0, 1.0]` → 실제 로봇: `[-1.15, 1.15]` 변환 |
| [`EXPERIMENT_PLAN_20251209.md`](../EXPERIMENT_PLAN_20251209.md) | 2025-12-09 | VLA 실험 계획 및 상태 (2025-12-09) | \| 케이스 \| Config \| 상태 \| 특징 \| |
| [`FINAL_TEST_REPORT_NO_CHUNK.md`](../FINAL_TEST_REPORT_NO_CHUNK.md) | 2025-12-09 | ✅ No Chunk 모델 추론 테스트 성공 리포트 | **Step 1: 모델 로딩** ✅ |
| [`INTEGRATION_REPORT_20251209.md`](../INTEGRATION_REPORT_20251209.md) | 2025-12-09 | VLA 추론 시스템 통합 완료 리포트 | - [x] 새 브랜치 `feature/inference-integration` 생성 |
| [`MODEL_COMPARISON_AND_ANALYSIS.md`](../MODEL_COMPARISON_AND_ANALYSIS.md) | 2025-12-09 | VLA 모델 비교 분석 및 4000 Steps 설명 | - **Epoch**: 5/10 (85% - 3385/4000) |
| [`MODEL_COMPREHENSIVE_ANALYSIS.md`](../MODEL_COMPREHENSIVE_ANALYSIS.md) | 2025-12-09 | Mobile VLA 모델 학습 종합 분석 (2025-12-09) | > **"장애물을 피해 목표 오브젝트 앞에 도착하는 것"** |
| [`OVERFITTING_ANALYSIS.md`](../OVERFITTING_ANALYSIS.md) | 2025-12-09 | ⚠️ 과적합 감지 및 평가 리포트 | Epoch 0: 0.013864  ░░░░░░░░░ |
| [`OVERFITTING_ANALYSIS_REPORT.md`](../OVERFITTING_ANALYSIS_REPORT.md) | 2025-12-09 | 과적합 분석 리포트 (환각 없는 데이터 기반) | **✅ 결론: 과적합 아님** |
| [`SHORT_TERM_ROADMAP.md`](../SHORT_TERM_ROADMAP.md) | 2025-12-09 | 단기 방향성 및 로드맵 (2025-12-09 ~ 12-11) | **12월 10일 미팅**에서 발표할 수 있는 완성된 결과 도출 |
| [`TRAINING_PROGRESS_NO_CHUNK_20251209.md`](../TRAINING_PROGRESS_NO_CHUNK_20251209.md) | 2025-12-09 | 학습 진행 상황 리포트 | \| 항목 \| 값 \| 설명 \| |
| [`TRAINING_UPDATE_1941.md`](../TRAINING_UPDATE_1941.md) | 2025-12-09 | 학습 진행 상황 업데이트 (19:41) | - **Epoch**: 3/10 (94% 완료) |
| [`inference_design_kr.md`](../inference_design_kr.md) | 2025-12-09 | VLA 추론 시스템 설계 문서 | 본 문서는 Mobile VLA(Vision-Language-Action) 모델의 추론 시스템 설계를 다룹니다. 실시간 로봇 제어를 위한 효율적인 추론 파이프라인과 액션 청크 기반 제어 전략을 제시합니다. |
| [`CASE8_TRAINING_REPORT.md`](../CASE8_TRAINING_REPORT.md) | 2025-12-10 | Case 8 학습 결과 보고서 | - **Data**: Left+Right (500 episodes) |
| [`CASE_REFINEMENT_FT_vs_NoFT.md`](../CASE_REFINEMENT_FT_vs_NoFT.md) | 2025-12-10 | 케이스 세분화: FT vs No FT | \| New ID \| Old ID \| Description \| Val Loss \| Status \| |
| [`CORE_REFOCUS.md`](../CORE_REFOCUS.md) | 2025-12-10 | 핵심 재점검 - 미팅 준비 (16:00, 1시간 27분) | - **Input**: Image + "Navigate around obstacles... on the left/right" |
| [`CURRENT_PROGRESS_20251210.md`](../CURRENT_PROGRESS_20251210.md) | 2025-12-10 | 현재 진행 상황 (2025-12-10 11:54) | - Data: L+R (500 episodes) |
| [`CURRENT_STATUS_20251210.md`](../CURRENT_STATUS_20251210.md) | 2025-12-10 | 현재 상황 요약 (2025-12-10 08:50) | - Case 5 Val Loss: **0.000532** |
| [`EXPERIMENT_DESIGN_MATRIX.md`](../EXPERIMENT_DESIGN_MATRIX.md) | 2025-12-10 | 실험 설계 매트릭스 (Experiment Design Matrix) | \| Case \| 실험명 \| 변수 추출 \| |
| [`FINAL_STATUS_20251210.md`](../FINAL_STATUS_20251210.md) | 2025-12-10 | 최종 상태 요약 (2025-12-10 03:49) | - [x] 상세 보고서 (MEETING_PREPARATION_20251210.md) |
| [`FINAL_SUMMARY_20251210.md`](../FINAL_SUMMARY_20251210.md) | 2025-12-10 | 최종 상황 요약 (2025-12-10 11:54) | - L+R (500 episodes) |
| [`LORA_COMPARISON_STATUS.md`](../LORA_COMPARISON_STATUS.md) | 2025-12-10 | LoRA vs No Training 비교 현황 분석 (환각 없음) | 모든 케이스 (Case 1-9): |
| [`MASTER_EXPERIMENT_TABLE.md`](../MASTER_EXPERIMENT_TABLE.md) | 2025-12-10 | 전체 실험 케이스 마스터 테이블 | \| Case \| Data \| Chunk \| Strategy \| 실험명 \| Config 파일 \| Checkpoint 위치 \| 학습 상태 \| Val Loss \| Train Loss \| Epochs \| 비고 \| |
| [`MEETING_20251210_FINAL.md`](../MEETING_20251210_FINAL.md) | 2025-12-10 | 미팅 발표 자료 (2025-12-10 16:00) - FINAL | - Robot → Box (장애물) → Bottle (목표) |
| [`MEETING_CHECKLIST_FINAL.md`](../MEETING_CHECKLIST_FINAL.md) | 2025-12-10 | 미팅 준비 최종 체크리스트 | - ✅ Case 9 학습 중단 |
| [`MEETING_FINAL_SUMMARY.md`](../MEETING_FINAL_SUMMARY.md) | 2025-12-10 | 미팅 자료 최종 정리 (오늘 16:00) | - ✅ **Epoch 0**: Val Loss 0.022 (NoFT, 초기 상태) |
| [`MEETING_PRESENTATION.md`](../MEETING_PRESENTATION.md) | 2025-12-10 | 미팅 발표 자료 (오늘 16:00) | 1. **배경 및 목적** (2분) |
| [`MEETING_PRESENTATION_20251210.md`](../MEETING_PRESENTATION_20251210.md) | 2025-12-10 | 12월 10일 미팅 최종 발표 자료 | - **초기 문제**: 모델이 Left/Right 방향을 전혀 구분하지 못함 (정확도 0%) |
| [`MEETING_PRESENTATION_SCRIPT_20251210.md`](../MEETING_PRESENTATION_SCRIPT_20251210.md) | 2025-12-10 | 교수님 미팅 발표 흐름 (15분 버전) | **슬라이드 1: 이전 모델의 한계** |
| [`MEETING_READY_CHECKLIST.md`](../MEETING_READY_CHECKLIST.md) | 2025-12-10 | 교수님 미팅 최종 준비 완료 | **Case 8: No Chunk + Abs Action** |
| [`PROFESSOR_EXPLANATION_GUIDE.md`](../PROFESSOR_EXPLANATION_GUIDE.md) | 2025-12-10 | 교수님 미팅 설명 가이드 (2025-12-10 16:00) | - ❌ angular_z (회전 속도) |
| [`TRAINING_FINAL_STATUS_20251210.md`](../TRAINING_FINAL_STATUS_20251210.md) | 2025-12-10 | VLA Training 최종 상태 및 다음 단계 | \| Epoch \| Val Loss \| 개선율 \| 상태 \| |
| [`URGENT_DECISION_NEEDED.md`](../URGENT_DECISION_NEEDED.md) | 2025-12-10 | 긴급 상황 정리 - 미팅 2시간 전 | **GPU Out of Memory!** |
| [`URGENT_MEETING_TODAY_16H.md`](../URGENT_MEETING_TODAY_16H.md) | 2025-12-10 | 긴급 미팅 준비 (오늘 16:00 - 2시간 10분!) | - ✅ **Epoch 0** (FT 안됨): `epoch_epoch=00-val_loss=0.022.ckpt` |
| [`VLA_TRAINING_PROGRESS_REPORT_20251210.md`](../VLA_TRAINING_PROGRESS_REPORT_20251210.md) | 2025-12-10 | VLA Training 진행사항 최종 보고서 | \| 목표 \| 상태 \| 결과 \| |
| [`PROFESSOR_MEETING_QA_20251211.md`](../PROFESSOR_MEETING_QA_20251211.md) | 2025-12-11 | 교수님 미팅 질의응답 준비 (2025-12-11) | > "데이터를 1000~3000개로 늘리면 LoRA가 의미 있을까요? 벡터 형태는 어떻게 달라지나요?" |
| [`REMAINING_TASKS_BRIEFING.md`](../REMAINING_TASKS_BRIEFING.md) | 2025-12-11 | 수요일 미팅 준비 현황 브리핑 | - [x] 교수님 미팅 정리 (`PROFESSOR_MEETING_20251205.md`) |
| [`dataset_validation_report.md`](../dataset_validation_report.md) | 2025-12-16 | Dataset Validation Report | - **Validation Rate**: 0.0% |
| [`dataset_validation_report_v2.md`](../dataset_validation_report_v2.md) | 2025-12-16 | Dataset Validation Report | - **Validation Rate**: 99.8% |
| [`CODE_CORRECTION_NEEDED.md`](../CODE_CORRECTION_NEEDED.md) | 2025-12-17 | 코드 수정 되돌리기 및 재분석 보고서 | - DOF: linear_x, **angular_z** (틀림!) |
| [`CODE_FIXES_FINAL.md`](../CODE_FIXES_FINAL.md) | 2025-12-17 | 코드 수정 최종 완료 보고서 | **데이터셋 코드** (`Mobile_VLA/src/data/mobile_vla_h5_dataset.py`): |
| [`CRITICAL_FIXES_GUIDE.md`](../CRITICAL_FIXES_GUIDE.md) | 2025-12-17 | Jetson 추론 코드 Critical Issues 수정 가이드 | 교수님 미팅 합의사항 준수: |
| [`JETSON_CODE_ANALYSIS.md`](../JETSON_CODE_ANALYSIS.md) | 2025-12-17 | Jetson 추론 코드 분석 - 교수님 미팅 계획 대조 | \| 항목 \| 합의 사항 \| 이유 \| |
| [`VLA_ALIASES_GUIDE.md`](../VLA_ALIASES_GUIDE.md) | 2025-12-17 | VLA Aliases 사용 가이드 | **작성일:** 2025-12-17 |
| [`VLA_MULTI_SERVER_SETUP.md`](../VLA_MULTI_SERVER_SETUP.md) | 2025-12-17 | VLA 멀티 서버 환경 설정 가이드 | ┌─────────────────────────────────────────────────────────┐ |
| [`WINDOW_SIZE_ANALYSIS.md`](../WINDOW_SIZE_ANALYSIS.md) | 2025-12-17 | Window Size 변경 효과 분석 | "window_size": 8, |
| [`api_server_debugging_20251217.md`](../api_server_debugging_20251217.md) | 2025-12-17 | API 서버 디버깅 결과 및 결론 | **작성일:** 2025-12-17 21:42 KST |
| [`api_server_integration_and_next_steps_20251217.md`](../api_server_integration_and_next_steps_20251217.md) | 2025-12-17 | API 서버 통합 및 다음 단계 플랜 | - ✅ **Jetson의 올바른 Action Space 적용**: 2 DOF (linear_x, angular_z) |
| [`api_server_test_report_20251217.md`](../api_server_test_report_20251217.md) | 2025-12-17 | API 서버 테스트 완료 보고서 | - ✅ `GET /` - API 정보 |
| [`chunk10_final_report_20251217.md`](../chunk10_final_report_20251217.md) | 2025-12-17 | Chunk10 학습 및 인퍼런스 테스트 완료 리포트 | - **Model:** Mobile VLA with Frozen Kosmos-2 |
| [`chunk10_training_report_20251217.md`](../chunk10_training_report_20251217.md) | 2025-12-17 | Chunk10 학습 완료 리포트 | - Model: Mobile VLA with Frozen Kosmos-2 |
| [`cleanup_result_20251217.md`](../cleanup_result_20251217.md) | 2025-12-17 | 디스크 정리 결과 보고서 | **실행일:** 2025-12-17 11:21 KST |
| [`dataset_validation_20251217.md`](../dataset_validation_20251217.md) | 2025-12-17 | Dataset Validation Report | - **Validation Rate**: 99.8% |
| [`disk_cleanup_plan_20251217.md`](../disk_cleanup_plan_20251217.md) | 2025-12-17 | 디스크 용량 분석 및 삭제 계획 | **분석일:** 2025-12-17 |
| [`experiment_status_20251217.md`](../experiment_status_20251217.md) | 2025-12-17 | 실험 진행 상황 종합 리포트 | **작성일:** 2025-12-17 19:33 KST |
| [`final_status_meeting_ready_20251217.md`](../final_status_meeting_ready_20251217.md) | 2025-12-17 | 최종 진행 상황 - 목요일 미팅 준비 완료 | **작성일:** 2025-12-17 21:43 KST |
| [`git_cleanup_analysis_20251217.md`](../git_cleanup_analysis_20251217.md) | 2025-12-17 | Git 히스토리 정리 상세 분석 | .git/ 크기: 90GB |
| [`git_cleanup_result_20251217.md`](../git_cleanup_result_20251217.md) | 2025-12-17 | Git 히스토리 정리 완료 리포트 | Git 저장소 정리가 **성공적으로 완료**되었으며, `.git/` 크기가 **90GB → 33GB**로 **63% 감소**했습니다. |
| [`git_recovery_safe_plan_20251217.md`](../git_recovery_safe_plan_20251217.md) | 2025-12-17 | Git Repository 복구 안전 계획 | runs/                                          48GB |
| [`git_recovery_success_20251217.md`](../git_recovery_success_20251217.md) | 2025-12-17 | Git Repository 복구 성공 보고서 | 현재 브랜치: feature/inference-integration |
| [`git_repository_corruption_analysis_20251217.md`](../git_repository_corruption_analysis_20251217.md) | 2025-12-17 | Git Repository Corruption 분석 보고서 | - **Push 불가**: 로컬과 리모트 간 sync가 깨진 상태 |
| [`next_steps_progress_20251217.md`](../next_steps_progress_20251217.md) | 2025-12-17 | 다음 단계 진행 상황 | - **시작 시간:** 2025-12-17 13:29:30 KST |
| [`progress_summary_20251217.md`](../progress_summary_20251217.md) | 2025-12-17 | 실험 진행 상황 요약 - 2025-12-17 20:59 | - **Chunk5**: Epoch 6, Val Loss 0.067 ⭐ (Best) |
| [`server_performance_analysis_20251217.md`](../server_performance_analysis_20251217.md) | 2025-12-17 | 안티그래비티 서버 성능 분석 리포트 | 안티그래비티 서버 진입 속도 저하의 **근본 원인은 Git 저장소의 비정상적인 비대화**입니다. `.git/` 디렉토리가 **90GB**를 차지하고 있으며, 이로 인해 파일 시스템 I/O 성능이 저하되고 있습니다. |
| [`API_INTERFACE_SPEC.md`](../API_INTERFACE_SPEC.md) | 2025-12-18 | 📡 VLA Model API Interface Specification | Billy 서버는 VLA(Vision-Language-Action) 모델을 호스팅하며, Jetson 로봇으로부터 시각 정보와 명령어를 받아 제어 신호(Action)를 반환합니다. |
| [`JETSON_COLLABORATION_NEXT_STEPS.md`](../JETSON_COLLABORATION_NEXT_STEPS.md) | 2025-12-18 | 🚀 Jetson & Billy 협업 Next Steps | 1.  **Git & Code**: Jetson 측 코드를 Pull 완료했습니다. |
| [`PROFESSOR_BRIEFING_20251218.md`](../PROFESSOR_BRIEFING_20251218.md) | 2025-12-18 | 🎓 VLA Project Update Briefing | 최근 3일(12.16 ~ 12.18) 동안 시스템 아키텍처 재구성, 데이터 검증, API 서버 구축 및 Jetson 연동 준비를 완료했습니다. |
| [`PROFESSOR_MEETING_UPDATE.md`](../PROFESSOR_MEETING_UPDATE.md) | 2025-12-18 | 🎓 VLA Project Progress Update | > 💡 **Purpose**: This document tracks the progress, key achievements, and experimental results for professor meetings. I |
| [`api_server_final_report_20251218.md`](../api_server_final_report_20251218.md) | 2025-12-18 | API 서버 통합 및 성능 테스트 결과 보고서 | API 서버 구축, 테스트, 모델 비교를 모두 성공적으로 완료했습니다. |
| [`api_server_startup_guide_20251218.md`](../api_server_startup_guide_20251218.md) | 2025-12-18 | API 서버 시작 가이드 (Billy 서버) | - `.vla_aliases` 및 `manage_api_server.sh` 경로 수정 (`/home/soda/vla` → `/home/billy/25-1kp/vla`) |
| [`experiment_plan_left_right_20251218.md`](../experiment_plan_left_right_20251218.md) | 2025-12-18 | 학습 실행 계획 업데이트 | - **시작**: 10:14 |
| [`feedback_20251218.md`](../feedback_20251218.md) | 2025-12-18 | 교수님 피드백 (2025-12-18) | - Jetson에서 실제로 들어가는 메모리가 16GB를 초과할 가능성 |
| [`final_experiment_report_20251218.md`](../final_experiment_report_20251218.md) | 2025-12-18 | 실험 결과 최종 종합 리포트 | \| 순위 \| 모델 \| 데이터 \| Action Chunking \| Best Epoch \| Val Loss \| Val RMSE \| 체크포인트 \| |
| [`phase2_phase3_plan_20251218.md`](../phase2_phase3_plan_20251218.md) | 2025-12-18 | 2단계: 추론 테스트 & 3단계: 실제 로봇 주행 계획 | Phase 1: 학습 (완료 ✅) |
| [`progress_20251218.md`](../progress_20251218.md) | 2025-12-18 | 2025-12-18 프로젝트 진행 상황 | - [x] Left navigation 데이터: 250 episodes 확인 |
| [`ssh_connection_resolution_20251218.md`](../ssh_connection_resolution_20251218.md) | 2025-12-18 | SSH 연결 문제 해결 완료 리포트 | Jetson에서 Billy 서버로 SSH 터널링을 시도했으나 연결 실패: |
| [`PTQ_batch_completion_report.md`](../PTQ_batch_completion_report.md) | 2025-12-22 | PTQ 배치 양자화 완료 보고서 | \| 모델 \| Val Loss \| 양자화 \| 파일 크기 \| 예상 메모리 \| |
| [`PTQ_batch_execution_guide.md`](../PTQ_batch_execution_guide.md) | 2025-12-22 | PTQ 배치 양자화 실행 가이드 | \| 모델명 \| Checkpoint \| Val Loss \| 설명 \| |
| [`PTQ_batch_final_report.md`](../PTQ_batch_final_report.md) | 2025-12-22 | PTQ 배치 양자화 최종 보고서 | - **QAT vs PTQ 상세 비교 분석** ✅ |
| [`QAT_vs_PTQ_complete_analysis.md`](../QAT_vs_PTQ_complete_analysis.md) | 2025-12-22 | QAT vs PTQ 완전 비교 분석 | model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') |
| [`WEEKLY_PROGRESS_20251222-24.md`](../WEEKLY_PROGRESS_20251222-24.md) | 2025-12-22 | 2025년 12월 4주차 개발 진행 보고서 | 1. ✅ **BitsAndBytes INT8 Quantization 구현 완료** |
| [`jetson_memory_budget.md`](../jetson_memory_budget.md) | 2025-12-22 | Jetson AGX Orin 16GB 메모리 구조 및 버짓 | **결론: Jetson Orin은 CPU와 GPU가 메모리를 공유합니다.** |
| [`quantization_methodology_analysis.md`](../quantization_methodology_analysis.md) | 2025-12-22 | VLA Quantization 방법론 비교 분석 | **INT8/INT4로 재학습이 필요한가, 아니면 추론만 양자화하면 되는가?** |
| [`ACTUAL_MEMORY_MEASUREMENT.md`](../ACTUAL_MEMORY_MEASUREMENT.md) | 2025-12-23 | 실제 메모리 측정 결과 - 환각 없는 사실 | "before": { "allocated_gb": 0.0 }, |
| [`API_TEST_RESULTS.md`](../API_TEST_RESULTS.md) | 2025-12-23 | API 인퍼런스 테스트 결과 보고서 | \| Test \| Endpoint \| Status \| Latency \| |
| [`DIRECTORY_AND_AUTHOR_VERIFICATION.md`](../DIRECTORY_AND_AUTHOR_VERIFICATION.md) | 2025-12-23 | 디렉토리 구조 및 코드 작성자 검증 보고서 | /home/billy/25-1kp/vla/ |
| [`MODEL_STATUS_AND_NEXT_STEPS.md`](../MODEL_STATUS_AND_NEXT_STEPS.md) | 2025-12-23 | Mobile VLA 모델 개발 상태 및 다음 단계 | \| 모델 \| Chunk Size \| Best Epoch \| Val Loss \| Status \| |
| [`QAT_RETRAINING_STATUS_20251223.md`](../QAT_RETRAINING_STATUS_20251223.md) | 2025-12-23 | QAT (Quantization-Aware Training) 재학습 계획서 | - 파일 크기: 5.4GB (예상 1.15GB vs 실제) |
| [`QAT_TRAINING_COMPLETE_20251223.md`](../QAT_TRAINING_COMPLETE_20251223.md) | 2025-12-23 | QAT 통합 모델 학습 완료 리포트 | \| 지표 \| Train \| Validation \| |
| [`QUANTIZATION_ACTUAL_STATUS.md`](../QUANTIZATION_ACTUAL_STATUS.md) | 2025-12-23 | Quantization 실제 상태 검증 - 환각 없는 분석 | $ ls -lh quantized_models/batch_ptq_20251222_200041/left_chunk5/ |
| [`QUANTIZATION_FAILURE_ANALYSIS.md`](../QUANTIZATION_FAILURE_ANALYSIS.md) | 2025-12-23 | Quantization 실패 원인 분석 및 해결 방안 | tee: quantized_models/batch_ptq_20251222_200041/left_chunk10/quantization.log: |
| [`RoboVLMs_vs_OurApproach_ActionChunking.md`](../RoboVLMs_vs_OurApproach_ActionChunking.md) | 2025-12-23 | RoboVLMs vs 우리 방식: Action Chunking 비교 분석 (Citation 포함) | **[1] RoboVLMs Paper** |
| [`STEP2_REVISED_PLAN.md`](../STEP2_REVISED_PLAN.md) | 2025-12-23 | Step 2: Quantized 모델 API 통합 계획 | batch_ptq_20251222_200041/ |
| [`ACTUAL_MEMORY_MEASUREMENT_20251224.md`](../ACTUAL_MEMORY_MEASUREMENT_20251224.md) | 2025-12-24 | Quantized Model 실제 메모리 측정 결과 | \| 모델 \| 실제 GPU 메모리 \| 파일 크기 \| |
| [`ALL_MODELS_BITSANDBYTES_TEST_20251224.md`](../ALL_MODELS_BITSANDBYTES_TEST_20251224.md) | 2025-12-24 | All Models BitsAndBytes INT8 Test Results | - ✅ **Status**: SUCCESS |
| [`ALL_MODELS_INT8_COMPLETE_20251224.md`](../ALL_MODELS_INT8_COMPLETE_20251224.md) | 2025-12-24 | 전체 모델 INT8 변환 완료 리포트 | - **API**: `torch.quantization.prepare()` + `convert()` |
| [`API_INFERENCE_TEST_COMPLETE_20251224.md`](../API_INFERENCE_TEST_COMPLETE_20251224.md) | 2025-12-24 | API Server 추론 테스트 완료 보고서 | "status": "healthy", |
| [`API_SERVER_STARTED_20251224.md`](../API_SERVER_STARTED_20251224.md) | 2025-12-24 | API Server 시작 완료 보고서 | PID: 1245940 |
| [`API_SPECIFICATION_INT8.md`](../API_SPECIFICATION_INT8.md) | 2025-12-24 | Mobile VLA API 명세서 (BitsAndBytes INT8) | \| 항목 \| FP32 (구버전) \| **INT8 (현재)** \| 개선 \| |
| [`BITSANDBYTES_ARCHITECTURE_20251224.md`](../BITSANDBYTES_ARCHITECTURE_20251224.md) | 2025-12-24 | Mobile VLA BitsAndBytes 구조 변경 다이어그램 | 기존 FP32 체크포인트 |
| [`BITSANDBYTES_COMPLETE_REPORT_20251224.md`](../BITSANDBYTES_COMPLETE_REPORT_20251224.md) | 2025-12-24 | BitsAndBytes INT8 구현 완료 보고서 | \| Model \| Val Loss \| GPU Mem \| Latency \| Status \| |
| [`BITSANDBYTES_FINAL_SUMMARY_20251224.md`](../BITSANDBYTES_FINAL_SUMMARY_20251224.md) | 2025-12-24 | ✅ BitsAndBytes INT8 구현 완료 (최종) | - [x] vlm_builder.py - quantization_config 지원 |
| [`BITSANDBYTES_IMPLEMENTATION_STATUS.md`](../BITSANDBYTES_IMPLEMENTATION_STATUS.md) | 2025-12-24 | BitsAndBytes INT8 구현 진행 상황 | pip install bitsandbytes accelerate |
| [`BITSANDBYTES_REMAINING_TASKS_20251224.md`](../BITSANDBYTES_REMAINING_TASKS_20251224.md) | 2025-12-24 | BitsAndBytes INT8 - 남은 작업 체크리스트 | - [x] BitsAndBytes INT8 코드 구현 (4 files, 31 lines) |
| [`GIT_BACKUP_SAFETY_ANALYSIS_20251224.md`](../GIT_BACKUP_SAFETY_ANALYSIS_20251224.md) | 2025-12-24 | .git_corrupted_20251217 백업 안전성 분석 | **2025-12-17 Git 정리 작업 시 생성된 백업** |
| [`GIT_CLEANUP_COMPLETE_20251224.md`](../GIT_CLEANUP_COMPLETE_20251224.md) | 2025-12-24 | Git 백업 정리 완료 보고서 | - **크기**: 33 GB |
| [`INT8_INFERENCE_TEST_20251224.md`](../INT8_INFERENCE_TEST_20251224.md) | 2025-12-24 | INT8 Inference 테스트 결과 | \|------\|------\| |
| [`JETSON_ACTION_ITEMS_20251224.md`](../JETSON_ACTION_ITEMS_20251224.md) | 2025-12-24 | Jetson 배포 - 액션 아이템 요약 | ❌ Jetson (linnaeus) 오프라인 |
| [`JETSON_CHECKPOINT_AND_STRATEGY_20251224.md`](../JETSON_CHECKPOINT_AND_STRATEGY_20251224.md) | 2025-12-24 | Jetson 배포 가이드 - 체크포인트 & 브랜치 전략 | 이름: epoch_epoch=06-val_loss=val_loss=0.067.ckpt |
| [`JETSON_DEPLOYMENT_BRIEFING_20251224.md`](../JETSON_DEPLOYMENT_BRIEFING_20251224.md) | 2025-12-24 | Jetson 로컬 온디바이스 추론 시스템 - 브리핑 | \| 분류 \| 항목 \| 상태 \| GitHub 링크 \| |
| [`JETSON_DEPLOYMENT_READY_20251224.md`](../JETSON_DEPLOYMENT_READY_20251224.md) | 2025-12-24 | Jetson 로봇 서버 배포 완료 가이드 | \| 파일 \| 용도 \| |
| [`JETSON_FINAL_DEPLOYMENT_GUIDE_20251224.md`](../JETSON_FINAL_DEPLOYMENT_GUIDE_20251224.md) | 2025-12-24 | Jetson 배포 - 최종 실행 가이드 | 1. ✅ SSH 키 복사 완료 (`soda@linnaeus`) |
| [`MOBILE_VLA_BITSANDBYTES_SUCCESS_20251224.md`](../MOBILE_VLA_BITSANDBYTES_SUCCESS_20251224.md) | 2025-12-24 | Mobile VLA + BitsAndBytes INT8 성공 보고서 | - **FP32**: 6.3 GB |
| [`PTQ_CONVERSION_COMPLETE_20251224.md`](../PTQ_CONVERSION_COMPLETE_20251224.md) | 2025-12-24 | PTQ 변환 완료 리포트 - Chunk5 Best Model | - Name: Chunk5 (통합, Left+Right) |
| [`PYTORCH_QUANTIZATION_CUDA_EVIDENCE_20251224.md`](../PYTORCH_QUANTIZATION_CUDA_EVIDENCE_20251224.md) | 2025-12-24 | PyTorch Quantization CUDA 미지원 - 공식 증거 | > "PyTorch's quantization specifically supports **x86 and ARM CPU architectures**." |
| [`PYTORCH_QUANTIZATION_TRUTH_20251224.md`](../PYTORCH_QUANTIZATION_TRUTH_20251224.md) | 2025-12-24 | 현재 상황 정리: PyTorch Quantization의 진실 | \| 항목 \| 결과 \| 설명 \| |
| [`QAT_FAILURE_DEEP_ANALYSIS_20251224.md`](../QAT_FAILURE_DEEP_ANALYSIS_20251224.md) | 2025-12-24 | QAT 실패 원인 심층 분석 리포트 | \| 시도 \| 시각 \| 에러 타입 \| 상태 \| |
| [`QUANTIZATION_ALL_ATTEMPTS_20251224.md`](../QUANTIZATION_ALL_ATTEMPTS_20251224.md) | 2025-12-24 | 양자화 시도 전체 타임라인 및 결과 | \| # \| 시도 \| 방법 \| 시각 \| 결과 \| 메모리 \| 문제 \| 해결 \| |
| [`QUANTIZATION_FINAL_COMPARISON_20251224.md`](../QUANTIZATION_FINAL_COMPARISON_20251224.md) | 2025-12-24 | Mobile VLA Quantization 최종 비교 브리핑 | \| # \| 방법 \| GPU 지원 \| 파일 크기 \| GPU 메모리 \| Latency \| 정확도 유지 \| 구현 난이도 \| VLA 사용 \| 최종 평가 \| |
| [`REMAINING_TASKS_20251224.md`](../REMAINING_TASKS_20251224.md) | 2025-12-24 | 남은 작업 체크리스트 | 1. **BitsAndBytes INT8 구현** ✅ |
| [`ROBOT_DRIVING_18STEPS_TEST_20251224.md`](../ROBOT_DRIVING_18STEPS_TEST_20251224.md) | 2025-12-24 | 실제 로봇 주행 시뮬레이션 - 18연속 추론 테스트 | - **window_size**: 8 (과거 8프레임 사용) |
| [`SERVER_STATUS_AND_CLEANUP_20251224.md`](../SERVER_STATUS_AND_CLEANUP_20251224.md) | 2025-12-24 | 서버 상태 및 디스크 정리 보고서 | ps aux \| grep uvicorn \| grep -v grep |
| [`STATIC_INT8_SUCCESS_20251224.md`](../STATIC_INT8_SUCCESS_20251224.md) | 2025-12-24 | Static INT8 Quantization 성공 리포트 | \| 항목 \| 원본 FP32 \| Dynamic Quant \| **Static INT8** \| |
| [`TRAINING_HISTORY_COMPARISON_20251224.md`](../TRAINING_HISTORY_COMPARISON_20251224.md) | 2025-12-24 | 전체 학습 히스토리 비교 분석 | \| 모델명 \| 데이터 \| Chunk \| Best Epoch \| Best Val Loss \| 크기 \| Language Task \| 비고 \| |
| [`VISUALIZATIONS_20251224.md`](../VISUALIZATIONS_20251224.md) | 2025-12-24 | Mobile VLA BitsAndBytes INT8 - 시각화 자료 모음 | PyTorch Static, QAT, Dynamic PTQ, BitsAndBytes INT8, TensorRT 등 모든 양자화 방법을 비교한 표 |
| [`VLA_QUANTIZATION_METHODS_20251224.md`](../VLA_QUANTIZATION_METHODS_20251224.md) | 2025-12-24 | VLA 논문 & GitHub: 실제 사용하는 Quantization 방법 | - ✅ **BitsAndBytes** (INT8, INT4) |
| [`DATA_COLLECTION_REPORT_20251231.md`](../DATA_COLLECTION_REPORT_20251231.md) | 2025-12-31 | 논문 작성 데이터 수집 완료 보고서 | "checkpoint": "epoch_epoch=05-val_loss=val_loss=0.284.ckpt", |
| [`FEEDBACK_ACTION_PLAN_20251231.md`](../FEEDBACK_ACTION_PLAN_20251231.md) | 2025-12-31 | 피드백 통합 실행 계획 | - **현상**: Jetson 환경에서 라이브러리 의존성 문제 발생 가능성 |
| [`JETSON_LIBRARY_REFERENCES_20251231.md`](../JETSON_LIBRARY_REFERENCES_20251231.md) | 2025-12-31 | Jetson 라이브러리 레퍼런스 및 배포 가이드 | Jetson Orin Nano (16GB)에서 VLM 배포는 **충분히 가능**하나, ARM 아키텍처 및 라이브러리 호환성 문제에 대비해야 합니다. 특히 **BitsAndBytes는 ARM에서 직접 설치 시 이슈** |
| [`MEETING_READY_20251231.md`](../MEETING_READY_20251231.md) | 2025-12-31 | 미팅 준비 완료 - 피드백 반영 종합 보고서 | \| # \| 피드백 내용 \| 우선순위 \| 상태 \| 산출물 \| |
| [`PAPER_DRAFT_EXPERIMENTS_20251231.md`](../PAPER_DRAFT_EXPERIMENTS_20251231.md) | 2025-12-31 | Experiments | We evaluate Mobile VLA on two main criteria: **Control Performance** (Accuracy) and **Deployment Efficiency** (Resource  |
| [`PAPER_DRAFT_FULL_20251231.md`](../PAPER_DRAFT_FULL_20251231.md) | 2025-12-31 | Mobile VLA: Lightweight Vision-Language-Action Model for Edge Robotic Deployment | We present **Mobile VLA**, a resource-efficient Vision-Language-Action model designed for deploying versatile robotic po |
| [`PAPER_DRAFT_METHOD_20251231.md`](../PAPER_DRAFT_METHOD_20251231.md) | 2025-12-31 | Method | We propose **Mobile VLA**, a lightweight Vision-Language-Action model designed specifically for edge robotic deployment. |
| [`PAPER_WRITING_PLAN_20251231.md`](../PAPER_WRITING_PLAN_20251231.md) | 2025-12-31 | 논문 작성 계획 (Billy 서버 기준) | 1. **RoboVLMs 경량화**: 7B → 1.6B parameters (Kosmos-2 채택) |
| [`RESOURCE_MANAGEMENT_ANALYSIS_20251231.md`](../RESOURCE_MANAGEMENT_ANALYSIS_20251231.md) | 2025-12-31 | 리소스 관리 분석 보고서 (논문용) | Mobile VLA는 **RoboVLMs 원본 모델 대비 최대 87%의 GPU 메모리를 절감**하며, **BitsAndBytes INT8 양자화를 통해 FP32 대비 71%의 메모리를 추가 절감**했습니다. |
| [`ABLATION_TEST_EPOCH1_RESULT.md`](../ABLATION_TEST_EPOCH1_RESULT.md) | 2026-01-07 | Ablation Test 결과 - Epoch 1 (2026-01-07 09:47) | - **Checkpoint**: `epoch=01, val_loss=0.354` |
| [`FINAL_SUMMARY_20260107.md`](../FINAL_SUMMARY_20260107.md) | 2026-01-07 | 최종 요약 및 다음 단계 (2026-01-07 10:00) | - Dataset → VLM → Action head 모든 경로에서 instruction 전달 확인 |
| [`GIT_COMMIT_20260107.md`](../GIT_COMMIT_20260107.md) | 2026-01-07 | Git Commit 완료 - Instruction 한국어→영어 변경 (2026-01-07 08:04) | - **Commit Hash**: `09b4000f` |
| [`GIT_COMMIT_FINAL.md`](../GIT_COMMIT_FINAL.md) | 2026-01-07 | Git Commit & Push 완료 (2026-01-07 11:50) | - **Hash**: `1bfdf24b` |
| [`INSTRUCTION_CHANGE_20260107.md`](../INSTRUCTION_CHANGE_20260107.md) | 2026-01-07 | Korean → English Instruction 변경 및 재학습 계획 (2026.01.07) | LEFT files:  Mean linear_y = +0.3194 (좌회전) ✓ |
| [`INSTRUCTION_FINAL_STATUS.md`](../INSTRUCTION_FINAL_STATUS.md) | 2026-01-07 | Instruction Mismatch - 최종 요약 (2026-01-07 09:50) | - [x] 한국어 instruction이 무시됨 확인 (테스트 완료) |
| [`INSTRUCTION_FLOW_ANALYSIS.md`](../INSTRUCTION_FLOW_ANALYSIS.md) | 2026-01-07 | Instruction Flow 분석: 한국어 vs 영어 Instruction | - **학습 데이터**: ✅ **한국어 instruction 사용** (확인됨) |
| [`INSTRUCTION_RESOLUTION_SUMMARY.md`](../INSTRUCTION_RESOLUTION_SUMMARY.md) | 2026-01-07 | Instruction Mismatch Resolution - 최종 실행 요약 (2026-01-07 07:53) | - ✅ Instruction flow 전체 분석 완료 |
| [`KOREAN_INSTRUCTION_TEST_RESULT.md`](../KOREAN_INSTRUCTION_TEST_RESULT.md) | 2026-01-07 | Korean Instruction Test 결과 (2026-01-07) | - **일시**: 2026-01-07 07:50:28 |
| [`LORA_FINETUNING_STRATEGY.md`](../LORA_FINETUNING_STRATEGY.md) | 2026-01-07 | LoRA Fine-tuning 전략 (2026-01-07) | LEFT instruction  → linear_y = -0.3274 |
| [`TODO_ENGLISH_INSTRUCTION.md`](../TODO_ENGLISH_INSTRUCTION.md) | 2026-01-07 | 영어 Instruction 재학습 현황 및 TODO (2026-01-07 09:40) | - **시작 시간**: 2026-01-07 07:53 |
| [`TRAINING_PAUSED_REPORT.md`](../TRAINING_PAUSED_REPORT.md) | 2026-01-07 | 🛑 PaliGemma-3B 학습 일시 중단 리포트 (2026-01-07 14:10) | 사용자 요청에 의해 학습을 안전하게 종료했습니다. |
| [`TRAINING_STARTED.md`](../TRAINING_STARTED.md) | 2026-01-07 | 🎉 PaliGemma-3B Mobile VLA 학습 시작! | - **Commit**: `a47d395c` |
| [`TRAINING_STATUS_CHECK.md`](../TRAINING_STATUS_CHECK.md) | 2026-01-07 | ✅ PaliGemma-3B 학습 재시작 및 문제 해결 (2026-01-07 12:12) | - `huggingface-cli`가 멈춘 것처럼 보였으나, 실제로는 **Python 라이브러리 호환성 문제**로 인해 모델 로딩 단계에서 실패/지연이 발생했습니다. |
| [`TRAINING_SUCCESS_SUMMARY.md`](../TRAINING_SUCCESS_SUMMARY.md) | 2026-01-07 | ✅ PaliGemma-3B 학습 성공적 시작 (2026-01-07 12:28) | PaliGemma-3B (2.4B) + LoRA 학습 시 발생하던 CUDA OutOfMemory 문제를 다음 조치들로 해결했습니다. |
| [`TRAINING_VS_INFERENCE_ANALYSIS.md`](../TRAINING_VS_INFERENCE_ANALYSIS.md) | 2026-01-07 | 학습 vs 추론 과정 환각 없는 분석 (2026-01-07) | \| 항목 \| Training (학습 시) \| Inference (추론 시) \| 차이점 분석 \| |
| [`WORK_LOG_20260107.md`](../WORK_LOG_20260107.md) | 2026-01-07 | 2026-01-07 작업 기록: PaliGemma-3B Mobile VLA 구축 | - **날짜**: 2026-01-07 |
| [`TRAINING_RESULT_PRETRAINED_VLM.md`](../TRAINING_RESULT_PRETRAINED_VLM.md) | 2026-01-10 | 🎉 RoboVLMs Pretrained VLM 학습 완료 보고 | \| 메트릭 \| Train \| Validation \| |
| [`robovlms_architecture_analysis.md`](../robovlms_architecture_analysis.md) | 2026-01-10 | RoboVLMs 아키텍처 종합 분석 (환각 없는 코드 기반) | > **작성일**: 2026-01-10 |
| [`COMPREHENSIVE_PROBLEM_ANALYSIS.md`](../COMPREHENSIVE_PROBLEM_ANALYSIS.md) | 2026-01-11 | 문제 종합 분석: 순차적 진단 보고서 | 1. [현재 발생한 모든 문제 정리](#1-현재-발생한-모든-문제-정리) |
| [`CONVERSATION_SUMMARY_20260111.md`](../CONVERSATION_SUMMARY_20260111.md) | 2026-01-11 | 대화 요약: 2026-01-11 | - **학습 기간**: 2026-01-10 22:57 → 2026-01-11 06:55 (약 8시간) |
| [`DATASET_STRUCTURE_CORRECT_ANALYSIS.md`](../DATASET_STRUCTURE_CORRECT_ANALYSIS.md) | 2026-01-11 | Mobile VLA 데이터셋 구조 및 학습 방법 완전 분석 | episode_20251119_080007_1box_hori_right_core_medium.h5 |
| [`HYPERPARAMETER_BRIEFING.md`](../HYPERPARAMETER_BRIEFING.md) | 2026-01-11 | Mobile VLA 핵심 하이퍼파라미터 브리핑 (대학원생용) | 1. [데이터 관련](#1-데이터-관련-파라미터) |
| [`INSTRUCTION_SPECIFIC_MODEL_STRATEGY.md`](../INSTRUCTION_SPECIFIC_MODEL_STRATEGY.md) | 2026-01-11 | "왼쪽만 되는 모델" 전략 분석 | Single Model: |
| [`INSTRUCTION_SPECIFIC_PROGRESS.md`](../INSTRUCTION_SPECIFIC_PROGRESS.md) | 2026-01-11 | Instruction-Specific Models 학습 진행 상황 | 시작: 2026-01-11 19:57 |
| [`MODEL_LEFT_PROGRESS_20260111.md`](../MODEL_LEFT_PROGRESS_20260111.md) | 2026-01-11 | Model_LEFT 학습 진행 상황 (1시간 경과) | \| Epoch \| Train Loss \| Val Loss \| Val RMSE \| 상태 \| |
| [`OXE_VS_GOOGLE_ROBOT_ANALYSIS.md`](../OXE_VS_GOOGLE_ROBOT_ANALYSIS.md) | 2026-01-11 | OXE vs Google Robot: Mobile VLA Task 적합성 분석 | \| Checkpoint \| 학습 데이터 \| 특징 \| |
| [`PROGRESS_20260111.md`](../PROGRESS_20260111.md) | 2026-01-11 | 진행 상황 보고 (2026-01-11) | - **파일**: `kosmos_ph_google-robot-post-train.pt` (6.35GB) |
| [`REALISTIC_EXPECTATION_ANALYSIS.md`](../REALISTIC_EXPECTATION_ANALYSIS.md) | 2026-01-11 | 왜 안되는가: 근본적 원인 재분석 | OpenVLA + LoRA: |
| [`ROBOVLMS_METHOD_COMPARISON.md`](../ROBOVLMS_METHOD_COMPARISON.md) | 2026-01-11 | RoboVLMs 원래 학습 방법 vs 우리 방법 비교 | "train_setup": { |
| [`TRAINING_PROCESS_VERIFICATION.md`](../TRAINING_PROCESS_VERIFICATION.md) | 2026-01-11 | 학습 과정 검증 보고서 | [Pretrained VLM] Loading from: kosmos_ph_google-robot-post-train.pt |
| [`VLA_LORA_SOLUTION_ANALYSIS.md`](../VLA_LORA_SOLUTION_ANALYSIS.md) | 2026-01-11 | VLA에서 Frozen VLM + Instruction Grounding 해결 사례 분석 | - **Frozen VLM** (Kosmos-2 scratch 또는 RoboVLMs pretrained) |
| [`ACTION_FLOW_VERIFICATION.md`](../ACTION_FLOW_VERIFICATION.md) | 2026-01-12 | Action Flow 검증: 데이터 → 학습 → Inference → ROS2 | linear_x: [0.000, 1.150] m/s |
| [`DISCRETE_VS_CONTINUOUS_DESIGN.md`](../DISCRETE_VS_CONTINUOUS_DESIGN.md) | 2026-01-12 | Continuous Regression vs Discrete Classification 설계 분석 | Linear X: 오직 2개 값만! |
| [`MANIPULATION_VS_NAVIGATION_ACTIONS.md`](../MANIPULATION_VS_NAVIGATION_ACTIONS.md) | 2026-01-12 | 6-7 DoF Manipulation vs 2 DoF Navigation: Action Space 비교 | delta_x,      # End-effector position change (m) |
| [`MODEL_RIGHT_PROGRESS_20260112.md`](../MODEL_RIGHT_PROGRESS_20260112.md) | 2026-01-12 | Model_RIGHT 학습 진행 상황 (1시간 52분 경과) | \| Epoch \| Train Loss \| Val Loss \| Val RMSE \| 상태 \| |
| [`ACTUAL_IMAGE_ANALYSIS_AND_VLM_TEST.md`](../ACTUAL_IMAGE_ANALYSIS_AND_VLM_TEST.md) | 2026-01-13 | 실제 데이터셋 이미지 분석 및 VLM 테스트 | Environment: |
| [`PRETRAINED_VLM_OBJECT_RECOGNITION_TEST.md`](../PRETRAINED_VLM_OBJECT_RECOGNITION_TEST.md) | 2026-01-13 | Pretrained VLM의 실제 데이터 객체 인식 테스트 결과 | Environment: 실내 바닥 |
| [`VLM_ANALYSIS_SUMMARY_AND_NEXT_STEPS.md`](../VLM_ANALYSIS_SUMMARY_AND_NEXT_STEPS.md) | 2026-01-13 | VLM 객체 인식 종합 분석 및 향후 방향 | - 기본 Kosmos-2는 우리 로봇 환경을 "dining area with people"로 잘못 인식 |
| [`ACTUAL_VS_VLM_COMPREHENSIVE_COMPARISON.md`](../ACTUAL_VS_VLM_COMPREHENSIVE_COMPARISON.md) | 2026-01-14 | VLM 객체 인식 종합 비교: 실제 vs VLM 응답 | Source files: |
| [`GOOGLE_ROBOT_IN_VS_OUT_DISTRIBUTION_TEST.md`](../GOOGLE_ROBOT_IN_VS_OUT_DISTRIBUTION_TEST.md) | 2026-01-14 | Google Robot VLM: In-Distribution vs Out-of-Distribution 테스트 | - Robot gripper 보임 |
| [`GOOGLE_ROBOT_VLM_ALL_TESTS_FINAL_SUMMARY.md`](../GOOGLE_ROBOT_VLM_ALL_TESTS_FINAL_SUMMARY.md) | 2026-01-14 | Google Robot VLM 최종 종합 결과 - 모든 테스트 통합 | - Manipulation (In-Dist): 5개 |
| [`GOOGLE_ROBOT_VLM_FINAL_COMPREHENSIVE_TEST.md`](../GOOGLE_ROBOT_VLM_FINAL_COMPREHENSIVE_TEST.md) | 2026-01-14 | Google Robot VLM 최종 종합 테스트 결과 | 1. **RT-1 Style**: Kitchen counter, robot arm, Coke can, blue cup, cardboard box |
| [`GOOGLE_ROBOT_VLM_TEST_FAILED.md`](../GOOGLE_ROBOT_VLM_TEST_FAILED.md) | 2026-01-14 | Google Robot Pretrained VLM 테스트 결과 | Checkpoint keys: 20개만 추출됨 |
| [`GOOGLE_ROBOT_VLM_TRAINING_DATA_AND_TEST_DATASETS.md`](../GOOGLE_ROBOT_VLM_TRAINING_DATA_AND_TEST_DATASETS.md) | 2026-01-14 | Google Robot VLM - 학습 데이터 및 테스트 가능 데이터셋 가이드 | 파일: pretrained_ckpts/checkpoints/kosmos_ph_google-robot-post-train.pt |
| [`GOOGLE_ROBOT_VS_BASE_KOSMOS2_COMPARISON.md`](../GOOGLE_ROBOT_VS_BASE_KOSMOS2_COMPARISON.md) | 2026-01-14 | Google Robot VLM vs 기본 Kosmos-2 비교 테스트 결과 | Google Robot VLM ≈ 기본 Kosmos-2 |
| [`REAL_RT1_IMAGE_TEST_RESULT.md`](../REAL_RT1_IMAGE_TEST_RESULT.md) | 2026-01-14 | 실제 RT-1 GitHub 이미지 테스트 결과 | Environment: |
| [`VLM_TEST_CORRECTION.md`](../VLM_TEST_CORRECTION.md) | 2026-01-14 | VLM 테스트 정정: 기본 Kosmos-2 vs Google Robot Pretrained | model_path = ".vlms/kosmos-2-patch14-224" |
| [`CHEAP_OBJECTS_FINAL_RECOMMENDATION.md`](../CHEAP_OBJECTS_FINAL_RECOMMENDATION.md) | 2026-01-15 | 값싼 Objects Navigation Feasibility Test - 최종 결론 | \| Rank \| Object \| Score \| Color \| Type \| Clean \| Price \| 추천 \| |
| [`OBJECT_CANDIDATES_VLM_TEST_RESULTS.md`](../OBJECT_CANDIDATES_VLM_TEST_RESULTS.md) | 2026-01-15 | Object Candidates VLM Recognition Test 결과 | \| Object Type \| Object \| Score \| Naming \| Detection \| Color \| Hallucination \| |
| [`OBJECT_TEST_REVISED_BALANCED_EVALUATION.md`](../OBJECT_TEST_REVISED_BALANCED_EVALUATION.md) | 2026-01-15 | Object Candidates VLM Test - 수정된 재평가 | Q: "Is there a blue mug?" |
| [`ROBOT_OBJECTS_ANALYSIS_AND_RECOMMENDATIONS.md`](../ROBOT_OBJECTS_ANALYSIS_AND_RECOMMENDATIONS.md) | 2026-01-15 | Robot Manipulation Datasets - Common Objects 종합 분석 | **Common Objects** (18개): |
| [`STRUCTURED_PROMPTING_PRETEST_RESULTS.md`](../STRUCTURED_PROMPTING_PRETEST_RESULTS.md) | 2026-01-15 | Structured Prompting Pre-Test Results | \| Rank \| Template \| Score \| Blue \| Mug \| Clean \| Notes \| |
| [`VLM_OBJECT_RECOGNITION_TEST_REPORT.md`](../VLM_OBJECT_RECOGNITION_TEST_REPORT.md) | 2026-01-15 | VLM Object Recognition Test Report | This report presents a comprehensive evaluation of 5 candidate objects (3 targets, 2 obstacles) for our VLM-optimized na |
| [`RED_BALL_SYSTEM_DESIGN.md`](../RED_BALL_SYSTEM_DESIGN.md) | 2026-01-19 | Red Ball Navigation System - 상세 설계 | [Camera] → [VLM Feature Extractor] → [Action Head] → [Navigation Controller] |
| [`GRAY_BASKET_VLM_TEST_RESULTS.md`](../GRAY_BASKET_VLM_TEST_RESULTS.md) | 2026-01-21 | 회색 빨랫 바구니 VLM 인식 테스트 결과 | <grounding> Is there a gray basket? |
| [`REAL_OBJECTS_VLM_TEST_RESULTS.md`](../REAL_OBJECTS_VLM_TEST_RESULTS.md) | 2026-01-21 | 실제 Objects VLM Recognition Test 결과 | - 갈색/베이지 화분 (brown/beige pot) - 앞쪽 |
| [`VLM_PROMPT_RESPONSE_DETAILED_ANALYSIS.md`](../VLM_PROMPT_RESPONSE_DETAILED_ANALYSIS.md) | 2026-01-21 | VLM 프롬프트 & 응답 상세 분석 | <grounding> Is there a brown pot on the floor? JSON: {"detected": true/false} |
| [`ROBOVLMS_vs_OURS_PROMPT_STRATEGY.md`](../ROBOVLMS_vs_OURS_PROMPT_STRATEGY.md) | 2026-01-22 | RoboVLMs Navigation Prompt Strategy Analysis | RoboVLMs(MobileVLA)는 기본적으로 **End-to-End Instruction Following**을 지향합니다. |
| [`INSTRUCTION_STRATEGY_SUMMARY.md`](../INSTRUCTION_STRATEGY_SUMMARY.md) | 2026-01-29 | Instruction Strategy - 정리 및 재확인 | 사람이 읽을 수 있고, Task를 명확히 설명하는 자연어 instruction을 H5 파일에 저장합니다. |
| [`ALL_EXPERIMENTS_INFERENCE_TEST_PLAN.md`](../ALL_EXPERIMENTS_INFERENCE_TEST_PLAN.md) | 2026-02-05 | 전체 실험 추론 테스트 계획 및 기록 | \| EXP ID \| 실험명 \| Window \| Chunk \| Visual \| Action \| Status \| Checkpoint \| |
| [`EXP06_RESAMPLER_REPORT.md`](../EXP06_RESAMPLER_REPORT.md) | 2026-02-05 | EXP-06: Visual Resampler 학습 결과 보고서 | - ✅ **10 Epochs 정상 완료** |
| [`FACTOR_ANALYSIS_AND_DESIGN.md`](../FACTOR_ANALYSIS_AND_DESIGN.md) | 2026-02-05 | Mobile VLA Factor Analysis & Experimental Design | 기존 7-DOF 매니퓰레이션(LoRobot 등) 모델을 2-DOF 모바일 내비게이션 태스크로 전환하며 수행한 구조적 최적화 내역입니다. |
| [`PROFESSOR_REPORT_20260205.md`](../PROFESSOR_REPORT_20260205.md) | 2026-02-05 | Mobile VLA 기술 진행 보고서 | 7-DOF 매니퓰레이션 중심의 RoboVLMs 아키텍처를 2-DOF 모바일 내비게이션으로 전환하여 학습 및 검증을 완료했습니다. Classification 방식에서 Regression 방식으로 전환하며 성능을 극대화 |
| [`PROFESSOR_REPORT_20260205_VERIFIED.md`](../PROFESSOR_REPORT_20260205_VERIFIED.md) | 2026-02-05 | Mobile VLA 기술 진행 보고서 (검증판) | \| 항목 \| 실제 수행 ✅ \| 계획/예정 ⏳ \| |
| [`TROUBLESHOOTING_LOG_20260205.md`](../TROUBLESHOOTING_LOG_20260205.md) | 2026-02-05 | Unified Regression Training Troubleshooting Log (2026-02-05) | \| 시각 (KST) \| 이벤트/액션 \| 설정 변화 \| 결과/상태 \| 트러블슈팅 내용 \| |
| [`training_history_20260205.md`](../training_history_20260205.md) | 2026-02-05 | Mobile VLA Training History & Evolution | \| # \| 학습 시기 \| 모델 타입 \| 데이터셋 \| Window Size \| Action Chunk \| 주요 기술 \| Train Loss \| Val Loss \| 비고 \| |
| [`EXP04_VS_EXP06_ACCURACY_COMPARISON.md`](../EXP04_VS_EXP06_ACCURACY_COMPARISON.md) | 2026-02-07 | EXP-04 vs EXP-06 정확도 비교 최종 보고서 | \| 모델 \| Perfect Match (PM) \| Directional Agreement (DA) \| 차이 \| |
| [`EXP09_10_11_TRAINING_REPORT.md`](../EXP09_10_11_TRAINING_REPORT.md) | 2026-02-07 | EXP-09, 10, 11 학습 결과 종합 보고서 | \| 실험 ID \| 주요 변경 \| 상태 \| Val Loss (Final) \| 학습 시간 \| 체크포인트 \| |
| [`ALL_EXPERIMENTS_FINAL_ANALYSIS_AND_STRATEGY.md`](../ALL_EXPERIMENTS_FINAL_ANALYSIS_AND_STRATEGY.md) | 2026-02-09 | 전체 실험 추론 테스트 결과 및 향후 전략 | \| Rank \| EXP ID \| Model \| Global PM/DA \| Initial \| Middle \| Final \| Val Loss \| |
| [`ALL_EXPERIMENTS_MASTER_LIST.md`](../ALL_EXPERIMENTS_MASTER_LIST.md) | 2026-02-09 | 전체 실험 마스터 리스트 (EXP-01 ~ EXP-17) | \| EXP \| 실험명 \| Window \| Chunk \| Visual \| 학습 완료 \| 추론 테스트 \| PM/DA \| Status \| 비고 \| |
| [`EXPERIMENT_HISTORY_AND_INSIGHTS.md`](../EXPERIMENT_HISTORY_AND_INSIGHTS.md) | 2026-02-09 | VLA 실험 히스토리 및 분석 요약 (EXP-01 ~ EXP-17) | \| Exp ID \| Model Config \| Epochs \| Success Rate \| Key Insight / Failure Mode \| Status \| |
| [`VLA_MODELS_WINDOW_CHUNK_COMPARISON.md`](../VLA_MODELS_WINDOW_CHUNK_COMPARISON.md) | 2026-02-09 | 주요 VLA 모델들의 Window/Chunk 설정 비교 분석 | \| Model \| Episode Length \| History Window \| Action Chunk \| Window/Episode 비율 \| Chunk/Episode 비율 \| Task Type \| |
| [`WINDOW_CHUNK_OPTIMIZATION_ANALYSIS.md`](../WINDOW_CHUNK_OPTIMIZATION_ANALYSIS.md) | 2026-02-09 | Window Size & Chunk Size 최적화: 대학원 수준 분석 | - **정의**: 모델이 "보는" 과거 프레임 수 |
| [`UNIFIED_REGRESSION_WIN12_LOGS.md`](../UNIFIED_REGRESSION_WIN12_LOGS.md) | 2026-02-10 | Unified Regression Win12 Experiment Logs | 본 문서는 디스크 공간 확보를 위해 `runs/unified_regression_win12` 디렉토리 내의 체크포인트를 정리하기 전, 각 실험의 성능 지표를 기록한 문서입니다. |
| [`JETSON_DEPLOYMENT_GUIDE.md`](../JETSON_DEPLOYMENT_GUIDE.md) | 2026-02-11 | 🚀 Jetson Deployment Guide for Mobile VLA | \| 구분 \| 빌리 서버 파일 경로 \| 젯슨 로컬 권장 경로 \| |
| [`PAPER_PREPARATION.md`](../PAPER_PREPARATION.md) | 2026-02-11 | Mobile VLA: Vision-Language-Action Model for Indoor Mobile Robot Navigation | - **"Mobile VLA: Adapting Vision-Language-Action Models for Real-World Indoor Mobile Robot Navigation"** |
| [`SERVER_INFERENCE_TEST_GUIDE.md`](../SERVER_INFERENCE_TEST_GUIDE.md) | 2026-02-12 | 🧪 Mobile VLA 서버 사이드 추론 테스트 가이드 | 서버 사이드 테스트는 크게 **API 기반 테스트**, **독립형(Standalone) 테스트**, **성능/양자화 테스트**의 세 가지 범주로 나뉩니다. |
| [`action_stuck_diagnostic_20260212.md`](../action_stuck_diagnostic_20260212.md) | 2026-02-12 | Mobile VLA 액션 고정 문제 진단 가이드 | [LOCAL#1] IMG@07:47:15.435 \| Latency: 5643ms \| ACTION: [1.15, 0.0] |
| [`offline_simulation_guide_20260212.md`](../offline_simulation_guide_20260212.md) | 2026-02-12 | Offline Simulation 사용 가이드 | - ✅ H5 데이터셋에서 이미지 시퀀스 로드 |
| [`status_report_20260212.md`](../status_report_20260212.md) | 2026-02-12 | Mobile VLA 추론 분석 및 현황 보고서 (2026-02-12) | 최근 추론 성능 개선 작업 중 발견된 **"액션 고정(Action Stuck)"** 현상에 대한 원인 분석과 **"첫 프레임 안전성(First-Frame Safety)"** 복구 완료 사항을 정리한 리포트입니다. |
| [`ROBOT_SERVER_DEBUG_REPORT.md`](../ROBOT_SERVER_DEBUG_REPORT.md) | 2026-02-16 | 🤖 로봇 서버 디버깅 리포트 및 수정 요청서 | 원격 서버 로그(`vla-inference-gradio`) 분석 결과입니다. |
| [`RT1_PAPER_ANALYSIS.md`](../RT1_PAPER_ANALYSIS.md) | 2026-02-16 | 🔬 RT-1 (Robotics Transformer 1) 논문 분석 보고서 | RT-1(Robotics Transformer 1)은 Google Research에서 발표한 논문으로, 자연어 명령과 이미지 입력을 통해 로봇의 동작(Action)을 직접 출력하는 **End-to-End VLA(Vi |
| [`V3_EXP01_ANALYSIS_REPORT.md`](../V3_EXP01_ANALYSIS_REPORT.md) | 2026-02-19 | V3-EXP-01 Training Log Analysis Report | V3-EXP-01 실험은 기존 V2의 Regression 방식에서 벗어나 **Classification Head**를 도입하고, **Color Jitter 및 Random Crop과 같은 Data Augmentati |
| [`turn_bias_analysis_20260225.md`](../turn_bias_analysis_20260225.md) | 2026-02-25 | Turn Bias Analysis Report | 이전 대화에서 hard-turn 에피소드에서 PM/DM 점수가 낮아 `turn_bias` 파라미터로 방향성을 강제 주입하려 했음. |
| [`augmentation_and_collection_strategy_20260228.md`](../augmentation_and_collection_strategy_20260228.md) | 2026-02-28 | Data Augmentation & Collection Strategy (VLA Model) | 현재 `mobile_vla_h5_dataset.py`에서 적용 중인 증강 기법(`ColorJitter`, `RandomCrop`)에는 두 가지 중대한 오류가 있어, 켜져 있을 경우 모델의 조향 방향감을 급감시키는 원 |
| [`camera_domain_gap.md`](../camera_domain_gap.md) | 2026-02-28 | Memory: Camera Domain Gap Analysis | * 20260129 에피소드 등 학습 데이터에 사용된 카메라는 파랗고 물빠진 색감(Mean RGB ~ [110, 137, 132])을 가지며, 실제 안정화된 로봇 카메라는 따뜻하고 명확한 색감과 어안 왜곡(Fishe |
| [`camera_domain_gap_analysis_20260228.md`](../camera_domain_gap_analysis_20260228.md) | 2026-02-28 | Camera Domain Gap Analysis Report | 현재 V3 모델(`v3-exp06`, `v3-exp07`)의 추론 과정에서 심각한 시각적 환각(Visual Hallucination) 및 예측 오차가 간헐적으로 발생하고 있습니다. 이는 16비트 혼합 정밀도(Mixe |
| [`class_distribution_analysis_20260228.md`](../class_distribution_analysis_20260228.md) | 2026-02-28 | 클래스 분포 분석 리포트 (basket_dataset_v2) | `v3-exp07` 모델의 `class_weight` 설정이 실제 데이터 분포가 아닌 직관 기반의 **Magic Number**로 설정되어 있다는 지적에 따라, 528개 전체 에피소드의 9,487 프레임에 대한 실제 |
| [`class_distribution_insight.md`](../class_distribution_insight.md) | 2026-02-28 | Memory: 클래스 분포 분석 결과 및 핵심 인사이트 | \|  cls  \|     이름     \| 프레임 수 \|      비율 \| 역수 권장 weight \| |
| [`vla_experiment_history_table.md`](../vla_experiment_history_table.md) | 2026-02-28 | VLA 모델 실험 전체 역사 테이블 | \| 커밋 해시  \| 날짜             \| 브랜치                                \| 커밋 메시지                                                  |
| [`vla_mega_comparison_v1_v3.md`](../vla_mega_comparison_v1_v3.md) | 2026-02-28 | VLA 모델 전체 히스토리 통합 비교표 (V1 ~ V3 MEGA TABLE) | \|  세대  \| 모델/실험명            \| 조향 방식 (Approach) \| 학습 방법 (Backbone / Head) \| LoRA 파라미터 (r / α) \| LR (학습률) \| 윈도우 (Window) \|  |
| [`PROJECT_ROADMAP.md`](../PROJECT_ROADMAP.md) | 2026-03-03 | RoboVLM-Nav 프로젝트 로드맵 & 관리 가이드 | > **최종 수정**: 2026-03-03 |
| [`directory_governance.md`](../directory_governance.md) | 2026-03-03 | RoboVLM-Nav 디렉토리 거버넌스 (Directory Governance) | > **최종 수정**: 2026-03-03 |
| [`prompt_instruction_design_improvement.md`](../prompt_instruction_design_improvement.md) | 2026-03-05 | VLA Prompt & Instruction 설계 개선 결과 문서 | > **작성일**: 2026-03-05 |
| [`DATASET_V3_COLLECTION_GUIDE.md`](../DATASET_V3_COLLECTION_GUIDE.md) | 2026-03-06 | basket_dataset_v3 수집 가이드 | > **기반 문서**: `docs/dataset_analysis_basket_v2_20260306.md`, `docs/training_plan_dataset_v3_20260306.md` |
| [`VLA_COMPREHENSIVE_ANALYSIS_20260306.md`](../VLA_COMPREHENSIVE_ANALYSIS_20260306.md) | 2026-03-06 | VLA 실험 전체 분석 및 다음 방향 — 학습 서버 참조용 | > **작성일**: 2026-03-06 |
| [`VLA_PROJECT_PHASE_STATUS.md`](../VLA_PROJECT_PHASE_STATUS.md) | 2026-03-06 | VLA 프로젝트 진행 상태 (2026-03-06 업데이트) | - **장애물(brown pot)**: 현재 **사용하지 않음**. |
| [`dataset_analysis_basket_v2_20260306.md`](../dataset_analysis_basket_v2_20260306.md) | 2026-03-06 | basket_dataset_v2 데이터 구조 분석 보고서 | > **작성일**: 2026-03-06 |
| [`research_progress_report_20260306.md`](../research_progress_report_20260306.md) | 2026-03-06 | Mobile VLA: 모바일 로봇 내비게이션을 위한 Vision-Language-Action 모델 연구 진행 보고서 | > **작성일**: 2026-03-06 |
| [`training_plan_dataset_v3_20260306.md`](../training_plan_dataset_v3_20260306.md) | 2026-03-06 | VLA Dataset v3 학습 계획 | > **작성일**: 2026-03-06 |
| [`antigravity_recovery_20260318.md`](../antigravity_recovery_20260318.md) | 2026-03-18 | Antigravity Recovery Report | Date: 2026-03-18 |
| [`project_status_20260327.md`](../project_status_20260327.md) | 2026-03-27 | MoNaVLA 프로젝트 현황 및 향후 계획 분석 | > **SCP 전송 완료**: `last_v4_counterfactual_weights.pth` → `soda@100.85.118.58:~/MoNaVLA/` ✅ (6.6GB, 10:45 소요) |
| [`vla_stage3_robust_analysis_20260329.md`](../vla_stage3_robust_analysis_20260329.md) | 2026-03-29 | [VLA Stage 3 Robust Final] 학습 및 성능 분석 리포트 (25.03.29) | > 본 리포트는 `logs/train_v4_stage3_robust.log`와 `mobile_vla_v4_stage3_robust.json` 설정에 기록된 **객관적 데이터**만을 기반으로 작성되었습니다. |
| [`MoNaVLA_Development_Report.md`](../MoNaVLA_Development_Report.md) | 2026-03-30 | MoNaVLA Development Analysis Report | 본 프로젝트는 **MoNaVLA** 모델을 활용하여 실내 환경에서의 로봇 내비게이션 성능을 최적화하는 것을 목표로 합니다. 기존의 VLA 모델들은 7D(Arm+Gripper) 액션에 특화되어 있으나, 내비게이션 작업 |
| [`vla_v4_nav_policy_fix_report_20260402.md`](../vla_v4_nav_policy_fix_report_20260402.md) | 2026-04-02 | 📝 VLA v4 Nav-Policy Training Fix Report (2026.04.02) | 2026년 3월 말(03.31) 진행된 **VLA v4 Stage 3** 실험 및 `mobile_vla_v4_balanced_v2.json` 설정 기반 학습 중, 특정 조건에서 모델이 중단되는 현상이 발생함. |
| [`weekly_report_20260403_20260410.md`](../weekly_report_20260403_20260410.md) | 2026-04-03 | MoNaVLA 주간 작업 기록 | **기간: 2026-04-03 (목) ~ 2026-04-10 (금)** |
| [`grounding_briefing.md`](../grounding_briefing.md) | 2026-04-09 | Grounding 조사 기술 브리핑 | > 작성일: 2026-04-09 |
| [`situation_analysis_20260411.md`](../situation_analysis_20260411.md) | 2026-04-11 | MoNaVLA 현황 분석 및 차기 TODO (2026-04-11) | > 교수님 미팅록(3/13, 3/20, 3/27) + Exp01~04 결과를 종합한 현황 점검. |
| [`v5_experiment_report_20260415.md`](../v5_experiment_report_20260415.md) | 2026-04-15 | MobileVLA V5 Experiment Status & Backbone Fix Report (2026-04-15) | 현재 MobileVLA V5 데이터셋([[NavDataset](file:///home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5)])을 기반으로 두 가지 트랙의  |
| [`AGENT_ENTRYPOINT.md`](../AGENT_ENTRYPOINT.md) | 2026-04-16 | Agent Entrypoint | 작성일: 2026-04-16 / 마지막 업데이트: 2026-04-26 |
| [`BENCHMARK_PIPELINE_DESIGN.md`](../BENCHMARK_PIPELINE_DESIGN.md) | 2026-04-16 | MoNaVLA 공통 벤치마크 파이프라인 설계 | 작성일: 2026-04-16 |
| [`MEMORY_SYNC_MAP.md`](../MEMORY_SYNC_MAP.md) | 2026-04-26 | Memory Sync Map | 작성일: 2026-04-26 |
| [`CURRENT_STATE_SNAPSHOT.md`](../CURRENT_STATE_SNAPSHOT.md) | 2026-05-29 | CURRENT_STATE_SNAPSHOT | - **수정사항**: `scripts/eval_exp59_closedloop.py` 내의 BBox 히스토리 누적 시 과거 프레임의 실제 검출 여부와 상관없이 현재 프레임 상태로 덮어쓰여지던 `has_bbox` 버그를 |
| [`MONAPI_HANDOFF_20260601.md`](../MONAPI_HANDOFF_20260601.md) | 2026-06-01 | MoNa-Pi ↔ MoNaVLA 핸드오프 가이드 | 작성: 2026-06-01 (minum 서버 기준) |
| [`MONAPI_INTEGRATION.md`](../MONAPI_INTEGRATION.md) | 2026-06-01 | MoNaVLA ↔ MonAPI 연동 가이드 | > **대상:** monapi 레포에서 MoNaVLA 추론 서버를 호출하는 쪽 |
| [`SODA_HANDOFF_20260612.md`](../SODA_HANDOFF_20260612.md) | 2026-06-12 | SODA → Minum 인수인계 (2026-06-12) | > **작성:** SODA 서버 (100.85.118.58) — 로봇 데이터 수집 담당 |
| [`STOP_DISTANCE_CALIBRATION_HANDOFF.md`](../STOP_DISTANCE_CALIBRATION_HANDOFF.md) | 2026-06-24 | STOP 거리(40~50cm) 캘리브레이션 — soda 현장 작업 핸드오프 | > 작성: 2026-06-24 (minum 서버) · 대상: soda(Jetson AGX, 실로봇) 세션 |
| [`DATASET_V6_STATUS.md`](../DATASET_V6_STATUS.md) | 2026-07-15 | 데이터셋 V6 — 명명 규정 (2026-07-15) | > 로봇서버(soda, `monavla-driving`)/학습서버(minum, `monavla-train`) 공유 문서. |
| [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) | 2026-08-16 | ══ EdgeGround-VLA Research Status — Single Source of Truth ═══════════════ | updated: "2026-08-16" |
| [`ACTION_SPACE_CORRECTION.md`](../ACTION_SPACE_CORRECTION.md) |  | CRITICAL: Action Space 환각 제거! | H5 파일 실제 데이터: |
| [`ADVANCED_SIMILARITY_METRICS_GUIDE.md`](../ADVANCED_SIMILARITY_METRICS_GUIDE.md) |  | 고급 Similarity 메트릭 가이드 | 기존 기본 메트릭(Cosine, Euclidean, Pearson, MSE)에 더해, 논문에서 검증된 **8가지 고급 메트릭**을 추가했습니다. |
| [`ALL_CASES_COMPARISON.md`](../ALL_CASES_COMPARISON.md) |  | 전체 실험 케이스 종합 비교표 | \| Case \| 모델 구성 \| 데이터 전략 \| Val Loss \| 방향 정확도 \| 상태 \| 핵심 특징 \| 장점 \| 단점 \| |
| [`API_SPECIFICATION.md`](../API_SPECIFICATION.md) |  | Mobile VLA API 명세서 | 모든 API 요청에 API Key 필요 (Health check 제외) |
| [`AUGMENTATION_COMPARISON.md`](../AUGMENTATION_COMPARISON.md) |  | Augmentation Effect Analysis (Case 3 vs Case 4) | \| **Metric** \| **Case 2 (Baseline)** \| **Case 3 (Standard)** \| **Case 4 (Mirrored)** \| |
| [`BITSANDBYTES_CHECKPOINT_EXPLANATION.md`](../BITSANDBYTES_CHECKPOINT_EXPLANATION.md) |  | BitsAndBytes INT8 - 체크포인트 설명 | checkpoint = torch.load("model.ckpt")  # FP32 그대로 |
| [`CLARIFICATION_FROZEN_VS_FINETUNING.md`](../CLARIFICATION_FROZEN_VS_FINETUNING.md) |  | [중요] 용어 정정 및 정확한 실험 정의 | "freeze_backbone": true,      // VLM Frozen ✅ |
| [`CLARIFICATION_OUR_POSITION.md`](../CLARIFICATION_OUR_POSITION.md) |  | [정정] 우리 실험의 정확한 위치 (환각 제거) | ❌ "우리 방식 = RoboFlamingo 방식" |
| [`CONTEXT_VECTOR_ACTUAL_PLAN.md`](../CONTEXT_VECTOR_ACTUAL_PLAN.md) |  | Context Vector 실제 추출 및 비교 체크리스트 | \| 항목 \| 상태 \| 비고 \| |
| [`CONTEXT_VECTOR_SHAPE_EXPLAINED.md`](../CONTEXT_VECTOR_SHAPE_EXPLAINED.md) |  | Shape [10, 8, 64, 2048] 정확한 설명 | shape: [10, 8, 64, 2048] |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) |  | Contributing Guide | We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. |
| [`DATA_IMBALANCE_CRITICAL.md`](../DATA_IMBALANCE_CRITICAL.md) |  | 데이터 분포 분석 결과 | 총 Episodes: 250 |
| [`DATA_REQUIREMENT_CALCULATION.md`](../DATA_REQUIREMENT_CALCULATION.md) |  | Mobile-VLA 필요 데이터 규모 계산 | Episodes:  ~970,000 |
| [`DUAL_STRATEGY_API_GUIDE.md`](../DUAL_STRATEGY_API_GUIDE.md) |  | Dual Strategy Inference API Guide | Mobile VLA에서 두 가지 추론 전략을 모두 지원하는 flexible API입니다. |
| [`DUAL_STRATEGY_PROGRESS.md`](../DUAL_STRATEGY_PROGRESS.md) |  | Dual Strategy Inference API | - [x] ActionBuffer 클래스 생성 (`Mobile_VLA/action_buffer.py`) |
| [`EXPERIMENT_MATRIX.md`](../EXPERIMENT_MATRIX.md) |  | Mobile-VLA 전체 실험 케이스 매트릭스 | 1. **HuggingFace RoboVLMs** 원본 모델 활용 |
| [`FOV_DATA_COLLECTION_GUIDE.md`](../FOV_DATA_COLLECTION_GUIDE.md) |  | FOV 이탈 구간 집중 데이터 수집 가이드 (Phase 2) | 현재 `v3_exp08_center_goal` 모델의 핵심 약점인 **중간 프레임 FOV 이탈 구간 조향 중단 문제** 해결을 위한 데이터 추가 수집 절차입니다. |
| [`FT_vs_NoFT_COMPARISON_TABLE.md`](../FT_vs_NoFT_COMPARISON_TABLE.md) |  | FT vs NoFT 비교표 (table_experiment_config.md 형식) | \| Checkpoint ID \| Case \| Model \| Epoch \| Fine-Tuned \| Window \| Chunk \| Data \| Val Loss \| Train Loss \| Status \| |
| [`INFERENCE_API_GUIDE.md`](../INFERENCE_API_GUIDE.md) |  | Mobile VLA Inference API 사용 가이드 | Chunk10 Epoch 8 모델을 사용한 실시간 인퍼런스 API 서버입니다. |
| [`INFERENCE_QUICKSTART.md`](../INFERENCE_QUICKSTART.md) |  | VLA 추론 시스템 빠른 시작 가이드 | 이 가이드는 Case 5 (aug_abs) 모델을 사용한 VLA 추론 시스템의 빠른 시작 방법을 제공합니다. |
| [`INFERENCE_SYNC_LOGIC.md`](../INFERENCE_SYNC_LOGIC.md) |  | Mobile VLA 추론-수집 동기화 로직 (Synchronization Logic) | RoboVLMs 기반의 Mobile VLA 시스템에서 **"데이터 수집(Data Collection)"**과 **"실시간 추론(Real-time Inference)"** 간의 시간적 불일치가 발생할 수 있습니다. 이 |
| [`INSTRUCTION_GROUNDING_EXPLAINED.md`](../INSTRUCTION_GROUNDING_EXPLAINED.md) |  | Instruction Grounding 실패 상세 분석 (대학원생용) | 우리가 수행한 실험: |
| [`JETSON_BILLY_COLLABORATION.md`](../JETSON_BILLY_COLLABORATION.md) |  | 🤝 Jetson(Soda) - Billy 서버 협업 및 통합 가이드 | 이 문서는 Jetson 로봇(Client)과 Billy 서버(Inference Server) 간의 연동을 위한 **순차적 실행 가이드**와 **필수 설정**을 공유합니다. |
| [`KOREAN_INSTRUCTION_FIX.md`](../KOREAN_INSTRUCTION_FIX.md) |  | Korean Instruction Fix - Quick Start Guide | - `Mobile_VLA/instruction_mapping.py` - 한국어 instruction 중앙 관리 모듈 |
| [`KOSMOS2_LORA_OOM_ANALYSIS.md`](../KOSMOS2_LORA_OOM_ANALYSIS.md) |  | Kosmos-2 LoRA OOM 원인 분석 (환각 없는 계산) | torch.cuda.OutOfMemoryError: CUDA out of memory. |
| [`MEETING_SLIDES_FINAL.md`](../MEETING_SLIDES_FINAL.md) |  | 미팅 자료 (15:40 발표용) - 간결 버전 | - **Input**: Camera image + "Navigate around obstacles... on the left/right" |
| [`MESSAGE_TO_JETSON.md`](../MESSAGE_TO_JETSON.md) |  | 📨 Message to Jetson Client (Soda) | Billy 서버(API Server)의 입출력 규격을 확정하여 Git에 Push했습니다 (`docs/API_INTERFACE_SPEC.md`). |
| [`NEXT_ACTIONS_PRIORITIZED.md`](../NEXT_ACTIONS_PRIORITIZED.md) |  | 추가 작업 제안 (우선순위별) | - [ ] **Loss Curve 비교** (Case 2, 3, 4, 5의 학습 곡선) |
| [`PALIGEMMA_READY.md`](../PALIGEMMA_READY.md) |  | PaliGemma-3B Mobile VLA - 생성 완료 | - **파일**: `Mobile_VLA/configs/mobile_vla_paligemma_lora.json` |
| [`PALIGEMMA_SETUP_GUIDE.md`](../PALIGEMMA_SETUP_GUIDE.md) |  | PaliGemma-3B Mobile VLA Setup Guide | PaliGemma-3B 기반 Mobile VLA를 위한 새로운 학습 구조입니다. |
| [`PROJECT_STRATEGY.md`](../PROJECT_STRATEGY.md) |  | 프로젝트 방향성 및 관리 전략 (Project Strategy & Management) | 본 문서는 VLA-driving 프로젝트의 지속 가능한 발전과 효율적인 관리를 위한 전략적 로드맵을 정의합니다. `memora`에서 `menemory`로의 메모리 시스템 전환을 기점으로, 보다 체계적인 실험 관리와  |
| [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) |  | Mobile VLA Project Summary | Mobile VLA (Vision-Language-Action) 프로젝트는 로봇 제어를 위한 멀티모달 AI 시스템입니다. 이미지와 텍스트 명령을 입력받아 로봇의 액션을 예측하는 시스템을 구현했습니다. |
| [`QAT_NECESSITY_ANALYSIS.md`](../QAT_NECESSITY_ANALYSIS.md) |  | QAT (Quantization-Aware Training) 필요성 분석 | 학습: FP32/FP16 (정상 학습) |
| [`QAT_RETRAINING_PLAN.md`](../QAT_RETRAINING_PLAN.md) |  | QAT 재학습 계획 - On-Device 목표 달성 | → Vision Encoder (INT8, frozen) ← QAT |
| [`QUANTIZATION_GUIDE.md`](../QUANTIZATION_GUIDE.md) |  | INT8/INT4 양자화 가이드 | Jetson 16GB 메모리 최적화를 위한 Mobile VLA 모델 양자화 가이드 |
| [`QUICK_REFERENCE_CARD.md`](../QUICK_REFERENCE_CARD.md) |  | 미팅 Quick Reference 카드 | **인쇄용 / 암기용** |
| [`RIGHT_DATA_COLLECTION_GUIDE.md`](../RIGHT_DATA_COLLECTION_GUIDE.md) |  | Right 데이터 수집 가이드 | 250 episodes (오른쪽 박스) 수집 |
| [`ROBOVLMS_DATASET_CLARIFICATION.md`](../ROBOVLMS_DATASET_CLARIFICATION.md) |  | RoboVLMs 데이터셋 정보 정리 | - **총 Trajectories**: ~8,000개 |
| [`RTX_BASED_DATA_CALCULATION.md`](../RTX_BASED_DATA_CALCULATION.md) |  | RT-X (Open X-Embodiment) 기반 Mobile-VLA 데이터 요구사항 | Dataset: Open X-Embodiment (RT-X) |
| [`SMALL_VLM_COMPARISON.md`](../SMALL_VLM_COMPARISON.md) |  | VLA-Suitable Small VLMs 비교 분석 | - **Parameters**: 1.6B |
| [`SSH_TUNNEL_GUIDE.md`](../SSH_TUNNEL_GUIDE.md) |  | 🔒 SSH 터널링 가이드 (Jetson ↔ Billy) | Billy 서버는 **Port 10022**에서 SSH를 listening하고 있습니다 (Port 22가 아님!). |
| [`STRATEGY_DETAILED_EXPLANATION.md`](../STRATEGY_DETAILED_EXPLANATION.md) |  | Experiment Configuration - Strategy Explanation | - **Frozen Backbone**: VLM(Kosmos-2)의 backbone은 freeze |
| [`TAILSCALE_SECURE_SETUP.md`](../TAILSCALE_SECURE_SETUP.md) |  | 🔒 Billy-Jetson 보안 접속 가이드 | ├── secrets.sh                    # ⛔ Git 무시됨 (API Key 저장) |
| [`TASK_CORRECTION.md`](../TASK_CORRECTION.md) |  | Critical Fix: Task 정의 완전 수정 (환각 제거!) | [Box] ← 장애물 (가운데 고정) |
| [`V3_DATA_COLLECTION_PLAN_TARGET_ONLY.md`](../V3_DATA_COLLECTION_PLAN_TARGET_ONLY.md) |  | VLA Phase 1.5 Data Collection Plan (장애물 없음, 목표물만) | - **목표 물체**: 회색 바구니 (Gray Basket) |
| [`V3_DATA_COLLECTION_TABLE.md`](../V3_DATA_COLLECTION_TABLE.md) |  | VLA Phase 1.5 데이터셋 V3 수집 표 (목표물 바리에이션 기반) | **총 목표 수량: 160개 에피소드** (모두 100% 수동 조작 - 타이밍 암기 방지) |
| [`V3_STATUS_REPORT.md`](../V3_STATUS_REPORT.md) |  | 📊 VLA V3 학습 현황 및 실험 이력 보고서 | 본 보고서는 Vision-Language-Action (VLA) 모델의 Phase 2 (Regression)에서 Phase 3 (Classification)으로의 전환 과정과 현재 V3-EXP-01의 학습 결과 및  |
| [`V4_to_V5_Experiment_Journey.md`](../V4_to_V5_Experiment_Journey.md) |  | MoNaVLA: Development & Experiment History (V4 to V5) | 본 프로젝트는 로봇 네비게이션을 위한 Vision-Language-Action (VLA) 모델을 개발하는 과정입니다. 이전 **V4 단계**에서는 LSTM 기반의 Regression Head를 사용해 `[linear |
| [`V5_DATA_COLLECTION_REPORT.md`](../V5_DATA_COLLECTION_REPORT.md) |  | MoNaVLA V5 데이터 수집 및 실험 정리 리포트 | MoNaVLA 프로젝트의 Phase 1.5 진입에 따라, 보다 정교한 데이터 수집 및 모델 검증 시스템을 구축하였습니다. 특히 모델의 Visual Grounding 능력을 검증하고, 데이터 수집 시 발생할 수 있는  |
| [`VERIFICATION_NO_HALLUCINATION.md`](../VERIFICATION_NO_HALLUCINATION.md) |  | 환각 없는 검증 결과 (Critical!) | Left files: |
| [`VLA_CORE_KNOWLEDGE_BASE.md`](../VLA_CORE_KNOWLEDGE_BASE.md) |  | VLA Core Knowledge Base (핵심 기술 데이터베이스) | 최근 Mobile VLA 프로젝트에서 정립된 핵심 기술 원리와 해결된 주요 이슈들을 정리한 문서입니다. 이 문서는 모델의 안정적인 학습과 성능 향상을 위한 "Core Database" 역할을 합니다. |
| [`VLA_DEPLOYMENT_CRITICAL_GUIDE.md`](../VLA_DEPLOYMENT_CRITICAL_GUIDE.md) |  | 🚨 VLA 모델 배포 및 추론 통합 가이드 (Critical Guide) | 본 문서는 **Mobile VLA (V2, EXP-17 등)** 모델을 로봇 추론 서버(Jetson/Edge)에 배포할 때 발생하는 **"환각 현상(직진 편향, 정지 불능)"**을 방지하기 위한 핵심 보정 사항을 정 |
| [`WHY_KOSMOS2_USES_MORE_MEMORY.md`](../WHY_KOSMOS2_USES_MORE_MEMORY.md) |  | 왜 Kosmos-2 (1.6B)가 PaliGemma-3B (2.4B)보다 메모리를 더 많이 쓰는가? | \| Model \| Parameters \| LoRA 메모리 예상 \| 실제 결과 \| |
| [`WINDOW_SIZE_VERIFICATION.md`](../WINDOW_SIZE_VERIFICATION.md) |  | Window Size 불일치 분석 (환각 없는 코드 검증) | \| 항목 \| Config 설정 \| Dataset 코드 \| 실제 필요 \| |
| [`action_optimization_report.md`](../action_optimization_report.md) |  | Mobile VLA Action Space Optimization Report | 기존 Mobile VLA (Kosmos-2 + Classification Head) 모델은 9종류의 행동(Stop, F, B, L, R, FL, FR, BL, BR)을 사용하였으나, 데이터셋(V3) 내 '후진(Bac |
| [`data_collection_v2_guide.md`](../data_collection_v2_guide.md) |  | [가이드] Phase 2: 9대 고정 경로 데이터 수집 및 정밀 제어 고도화 | 교수님 피드백에 따라, 모델이 환경의 기하학적 구조를 외우지 않고 '물체와의 상대적 관계'를 학습할 수 있도록 다음 9개 지점에서 데이터를 집중 수집합니다. |
| [`disk_usage_analysis.md`](../disk_usage_analysis.md) |  | MoNaVLA 디스크 사용량 분석 및 정리 우선순위 리포트 | 현재 디스크 사용량이 **95% (사용 1.6T / 가용 102G)**로 매우 여유가 없는 상태입니다. 원활한 V4 학습 및 추론을 위해 정리가 시급합니다. |
| [`experiments_v1_to_v3_comprehensive.md`](../experiments_v1_to_v3_comprehensive.md) |  | VLA V1~V3 실험 종합 기록보관소 (Comprehensive Experiment Registry) | 본 문서는 Mobile VLA 프로젝트의 첫 실험부터 현재 최고 모델 도달까지의 모든 실험(Experiment) 설정을 빠짐없이 매핑한 공식 기록입니다. |
| [`hardware_request_v1.md`](../hardware_request_v1.md) |  | [요청안] MoNaVLA 고도화 및 캡스톤 상용화 수준 달성을 위한 GPU(H100) 지원 요청 | 현재 본 팀은 Vision-Language-Action(VLA) 모델인 **MoNaVLA**를 활용하여 실내 자율 주행 로봇의 'Center-Centering' 및 'Body Rotation' 정밀 제어를 연구 중입 |
| [`inference_usage_guide.md`](../inference_usage_guide.md) |  | VLA 추론 시스템 사용 가이드 | 이 문서는 Action Chunk 기반 Mobile VLA 추론 시스템의 사용 방법을 설명합니다. |
| [`mobile_vla_v4_hard_analysis.md`](../mobile_vla_v4_hard_analysis.md) |  | Mobile-VLA v4 Hard Navigation Analysis 및 개선 리포트 | 현재 Mobile-VLA v4 모델은 정면의 시각 정보에 강하게 편향(Vision-bias)되어, 특정 명령(예: "Stop")이나 장애물 상황에서도 전진을 지속하는 문제가 발견됨. 이는 학습 데이터셋 내 '전진'  |
| [`monapi_integration_report.md`](../monapi_integration_report.md) |  | MoNaVLA ↔ MonAPI 제어 알고리즘 통합 및 검증 리포트 | MoNa-pi 레포지토리의 주행 최적화 및 제어 알고리즘(소프트 스냅핑, 자동 정지 제어, Yaw 복원 등)을 MoNaVLA의 메인 추론 서버(`inference_server.py`)에 이식하여, `monapi` 모 |
| [`status.md`](../status.md) |  | Project Status Report | This document tracks the status of the Todo items as of November 27, 2025. |
| [`v5_exp09_analysis_report.md`](../v5_exp09_analysis_report.md) |  | V5-Exp09: 8-Class Discrete Navigation Analysis Report | 본 실험(Exp09)은 로봇 내비게이션 환경에서 VLM이 이산적인 액션 클래스(8종)를 예측하는 성능을 평가하고, 특히 **목표 지향적 명령어(Center-goal instruction)**가 행동 예측의 정확도 및 |
| [`vla_inference_analysis.md`](../vla_inference_analysis.md) |  | 2. Inference Matching Table (87.0% Accuracy Analysis) | \| Frame ID \| Scene Description \| Target Ground Truth (GT) \| Model Prediction (Pred) \| Confidence Score \| Result \| |
| [`vla_inference_gradio_guide.md`](../vla_inference_gradio_guide.md) |  | MoNaVLA Gradio Inference Dashboard 가이드 | > **명령어:** `vla-inference-gradio` (별칭: `mona-inference-gradio`) |
| [`vla_v3_exp08_evaluation.md`](../vla_v3_exp08_evaluation.md) |  | V3-EXP08 Model Evaluation Report (Goal-Centric Instruction) | - **Experiment**: `mobile_vla_v3_exp08_center_goal` |