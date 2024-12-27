import os
import shutil

# 기존 폴더 경로
input_folder = "cifar_noisy_20/cifar_noisy_test"
output_folder = "cifar_noisy_20/sorted_images"

# 클래스 이름과 번호 매핑
class_mapping = {
    "airplane": "Class 0",
    "automobile": "Class 1",
    "bird": "Class 2",
    "cat": "Class 3",
    "deer": "Class 4",
    "dog": "Class 5",
    "frog": "Class 6",
    "horse": "Class 7",
    "ship": "Class 8",
    "truck": "Class 9"
}

# 출력 폴더 구조 생성
os.makedirs(output_folder, exist_ok=True)
for class_name in class_mapping.values():
    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

# 파일 복사
for filename in os.listdir(input_folder):
    # 파일명 형식: '{Class이름}_{파일번호}'
    if "_" in filename:
        base_name = filename.split("_")[0]  # 파일명에서 클래스 이름 추출

        # 클래스 이름을 매핑된 번호로 변환
        if base_name in class_mapping:
            target_class = class_mapping[base_name]  # 매핑된 클래스 이름 가져오기

            # 대상 클래스 폴더
            target_folder = os.path.join(output_folder, target_class)

            # 파일 복사
            source_path = os.path.join(input_folder, filename)
            target_path = os.path.join(target_folder, filename)
            shutil.copy(source_path, target_path)

print("이미지 분류가 완료되었습니다.")