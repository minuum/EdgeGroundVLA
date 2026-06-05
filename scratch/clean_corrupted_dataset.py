# -*- coding: utf-8 -*-
# 손상된(Corrupted) H5 파일 스캔 및 자동 삭제/정리 스크립트

import os
import h5py
from pathlib import Path
import shutil
import argparse

def check_h5_file_integrity(h5_path):
    """H5 파일의 무결성을 검사합니다. 모든 데이터셋(images, actions 등)을 성공적으로 읽어올 수 있는지 확인합니다."""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'actions' not in f:
                return False, "No actions dataset"
            
            # 구형(images) 및 신형 V5(observations/images) 경로를 모두 지원하도록 수정
            if 'images' in f:
                images = f['images']
            elif 'observations' in f and 'images' in f['observations']:
                images = f['observations/images']
            else:
                return False, "No images dataset"
                
            # 실제로 데이터를 로딩해 봅니다 (Lazy 로딩으로 인한 필터 에러 검출 목적)
            _ = f['actions'][:]
            # 전체 프레임을 전부 한번 읽어봅니다
            for i in range(len(images)):
                _ = images[i]
                
            return True, "Integrity OK"
    except Exception as e:
        return False, f"Read Error: {str(e)}"

def clean_corrupted_dataset(dry_run=True):
    dataset_dir = Path("/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_V5_add_free")
    if not dataset_dir.exists():
        print(f"오류: 데이터셋 디렉토리 {dataset_dir} 가 없습니다.")
        return

    # 모든 H5 파일 검색
    h5_files = sorted(list(dataset_dir.glob("*.h5")))
    print("=" * 80)
    print(f"🔍 데이터 무결성 스캔 시작 (총 파일 수: {len(h5_files)}개, Dry Run: {dry_run})")
    print("=" * 80)

    corrupted_count = 0
    for idx, h5_path in enumerate(h5_files):
        success, msg = check_h5_file_integrity(h5_path)
        if not success:
            corrupted_count += 1
            print(f"❌ 손상 감지 [{idx+1}/{len(h5_files)}]: {h5_path.name} ({msg})")
            
            if dry_run:
                print(f"   └─ [Dry Run] 삭제 대상: {h5_path.name}")
                continue

            # 관련된 모든 부속 파일 삭제
            base_name = h5_path.stem
            try:
                # 1. H5 파일 삭제
                os.remove(h5_path)
                print(f"   └─ 삭제 완료: {h5_path.name}")
                
                # 2. 관련 JSON 파일 삭제
                for json_file in dataset_dir.glob(f"{base_name}*.json"):
                    os.remove(json_file)
                    print(f"   └─ 삭제 완료: {json_file.name}")
                
                # 3. 관련 CSV 파일 삭제
                for csv_file in dataset_dir.glob(f"{base_name}*.csv"):
                    os.remove(csv_file)
                    print(f"   └─ 삭제 완료: {csv_file.name}")
                
                # 4. 관련 이미지 폴더 삭제
                dir_path = dataset_dir / base_name
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    print(f"   └─ 삭제 완료: {base_name}/ (디렉토리)")
            except Exception as delete_error:
                print(f"   ⚠️ 삭제 중 오류 발생: {delete_error}")
    
    print("\n" + "=" * 80)
    if dry_run:
        print(f"스캔 완료 (Dry Run). 손상된 에피소드 총 {corrupted_count}개 감지됨 (삭제되지 않음).")
    else:
        print(f"스캔 및 정리 완료. 손상된 에피소드 총 {corrupted_count}개 감지 및 정리 완료.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrupted H5 dataset cleanup script")
    parser.add_argument("--execute", action="store_true", help="실제 삭제 작업을 진행합니다 (기본값은 Dry Run)")
    args = parser.parse_args()
    
    # execute 인자가 지정되지 않으면 dry_run=True로 실행됨
    clean_corrupted_dataset(dry_run=not args.execute)

