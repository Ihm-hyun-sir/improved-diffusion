import os
import shutil

# 원본 폴더와 대상 폴더 경로 설정
source_base_path = "sample_original_noisy_symmetric_20/"  # Class0 ~ Class9 폴더가 있는 경로
destination_path = "sample_original_noisy_symmetric_20/Full"      # Full Image 폴더 경로

# 대상 폴더가 없다면 생성
os.makedirs(destination_path, exist_ok=True)

# Class 폴더 순회 및 이미지 복사
for class_folder in range(10):  # Class0 ~ Class9
    class_folder_path = os.path.join(source_base_path, f"Class {class_folder}")
    
    # 폴더 내 파일 순회
    for file_name in os.listdir(class_folder_path):
        # 파일 경로 생성
        source_file_path = os.path.join(class_folder_path, file_name)
        destination_file_path = os.path.join(destination_path, f"{file_name}")
        
        # 파일 복사
        if os.path.isfile(source_file_path):  # 파일만 복사
            shutil.copy(source_file_path, destination_file_path)

print("이미지 복사가 완료되었습니다.")
