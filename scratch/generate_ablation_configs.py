# -*- coding: utf-8 -*-
# 8가지 Ablation Config 파일 및 순차 학습 구동 스크립트 생성기

import json
import os

def generate_configs():
    base_config_path = "configs/mobile_vla_paligemma_lora.json"
    if not os.path.exists(base_config_path):
        print(f"오류: 기본 설정 파일 {base_config_path} 이 존재하지 않습니다.")
        return

    # 기본 설정 파일 읽기
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 출력 디렉토리 생성
    output_dir = "configs/v5_ablation"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("scripts/train_active", exist_ok=True)

    # Ablation 설계 변수
    layer_options = {
        "top2": list(range(25, 27)), # Layer 25, 26
        "top4": list(range(23, 27)), # Layer 23 ~ 26
        "top6": list(range(21, 27)), # Layer 21 ~ 26
        "top8": list(range(19, 27))  # Layer 19 ~ 26
    }
    projector_options = {
        "proj_frozen": False,
        "proj_tuned": True
    }

    generated_files = []

    # 8가지 조합 순회
    for layer_name, layers in layer_options.items():
        for proj_name, tune_proj in projector_options.items():
            comb_name = f"v5_ablation_{layer_name}_{proj_name}"
            
            # config 깊은 복사
            new_config = json.loads(json.dumps(config))
            
            # 실험명 및 로그 경로 업데이트
            new_config["exp_name"] = comb_name
            new_config["task_name"] = f"mobile_vla_v5_{comb_name}"
            new_config["output_root"] = f"runs/mobile_vla_paligemma/{comb_name}"
            new_config["log_root"] = f"runs/mobile_vla_paligemma/{comb_name}"
            
            # train_setup 업데이트
            new_config["train_setup"]["tune_mm_projector"] = tune_proj
            # mm_projector는 tune_mm_projector와 tune_mm_mlp_adapter 둘 다 True일 때 튜닝되도록 되어 있을 수 있으므로 둘 다 동일하게 세팅
            new_config["train_setup"]["tune_mm_mlp_adapter"] = tune_proj
            
            # SigLIP 특정 레이어의 Q, K, V, Out Projection만 타깃 LoRA로 셋팅 (LM 제외)
            target_modules = []
            for l in layers:
                target_modules.extend([
                    f"vision_tower.vision_model.encoder.layers.{l}.self_attn.q_proj",
                    f"vision_tower.vision_model.encoder.layers.{l}.self_attn.k_proj",
                    f"vision_tower.vision_model.encoder.layers.{l}.self_attn.v_proj",
                    f"vision_tower.vision_model.encoder.layers.{l}.self_attn.out_proj"
                ])
            new_config["train_setup"]["lora_target_modules"] = target_modules
            
            # 로컬 데이터셋 경로 강제 지정 (minum 계정 실물 경로 연동)
            new_config["train_dataset"]["data_dir"] = "/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_V5_add_free"
            new_config["val_dataset"]["data_dir"] = "/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_V5_add_free"
            
            # Epochs 수 조절 (Ablation을 빠르게 완료하기 위해 5 epoch 설정 - 메모리와 시간 절약)
            new_config["trainer"]["max_epochs"] = 5
            
            # 파일 쓰기
            file_path = os.path.join(output_dir, f"{comb_name}.json")
            with open(file_path, "w", encoding="utf-8") as out_f:
                json.dump(new_config, out_f, indent=4, ensure_ascii=False)
            
            print(f"생성 완료: {file_path}")
            generated_files.append(file_path)

    # 8개 실험을 순차적으로 구동하는 쉘 스크립트 작성
    sh_script_path = "scripts/train_active/run_v5_ablations.sh"
    with open(sh_script_path, "w", encoding="utf-8") as sh_f:
        sh_f.write("#!/bin/bash\n")
        sh_f.write("# v5 LoRA Ablation Study 자동화 순차 구동 스크립트\n")
        sh_f.write("set -e\n")
        sh_f.write("mkdir -p logs/v5_ablation\n\n")
        
        for file in generated_files:
            base_name = os.path.splitext(os.path.basename(file))[0]
            sh_f.write(f"echo \"==================================================\"\n")
            sh_f.write(f"echo \"시작: {base_name}\"\n")
            sh_f.write(f"echo \"==================================================\"\n")
            sh_f.write(f".venv/bin/python3 third_party/RoboVLMs/main.py {file} > logs/v5_ablation/{base_name}.log 2>&1\n")
            sh_f.write(f"echo \"완료: {base_name}\"\n\n")
            
    # 실행 권한 부여
    os.chmod(sh_script_path, 0o755)
    print(f"\n순차 구동 스크립트 생성 및 실행권한 부여 완료: {sh_script_path}")

if __name__ == "__main__":
    generate_configs()
