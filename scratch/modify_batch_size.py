# -*- coding: utf-8 -*-
# configs/v5_ablation 폴더 내부의 설정 파일들에서 batch_size와 accumulate_grad_batches를 수정하는 스크립트
# GPU 활용도를 높이기 위해 batch_size를 1에서 4로 올리고, effective batch size 8을 유지하기 위해 accumulate_grad_batches를 8에서 2로 줄입니다.

import os
import json
from pathlib import Path

def modify_configs():
    config_dir = Path("/home/minum/26CS/MoNaVLA/configs/v5_ablation")
    if not config_dir.exists():
        print(f"오류: 설정 디렉토리 {config_dir} 가 존재하지 않습니다.")
        return

    json_files = sorted(list(config_dir.glob("*.json")))
    print(f"🔧 총 {len(json_files)}개의 설정 파일 수정 시작")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # batch_size 수정
            old_batch_size = data.get("batch_size")
            data["batch_size"] = 4

            # accumulate_grad_batches 수정
            old_accum = None
            if "trainer" in data:
                old_accum = data["trainer"].get("accumulate_grad_batches")
                data["trainer"]["accumulate_grad_batches"] = 2

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"✓ {file_path.name} 수정 완료:")
            print(f"  - batch_size: {old_batch_size} -> 4")
            print(f"  - accumulate_grad_batches: {old_accum} -> 2")
        except Exception as e:
            print(f"❌ {file_path.name} 수정 실패: {str(e)}")

if __name__ == "__main__":
    modify_configs()
