import os
import torch
import torchvision.transforms as transforms
from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
from PIL import Image
from sklearn.metrics import accuracy_score

# 클래스 이름과 라벨 매핑
target_class_map = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

def load_images_and_labels(folder_path, transform, device):
    """
    폴더에서 이미지를 로드하고 라벨을 추출합니다.
    Args:
        folder_path (str): 이미지 폴더 경로.
        transform (torchvision.transforms): 이미지 전처리 함수.
        device (torch.device): Torch 장치 (CPU 또는 GPU).
    Returns:
        images (torch.Tensor): 전처리된 이미지 Tensor.
        labels (list): 이미지의 실제 라벨.
    """
    images = []
    labels = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            class_name = file_name.split("_")[0]  # 파일 이름에서 클래스 이름 추출
            label = target_class_map[class_name]  # 클래스 이름을 라벨로 매핑
            labels.append(label)
            img = Image.open(file_path).convert("RGB")
            images.append(transform(img))
    return torch.stack(images).to(device), labels

# 사용자 정의 폴더 경로
generated_images_folder = "/home/kjh012/MS_Year_Proposal_2024_Winter/improved-diffusion/sample_original_noisy_symmetric_20/Class 0"

# Torch 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 이미지 크기
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 사전 학습된 CIFAR-10 분류 모델 로드
model = resnet50(pretrained=True)
model = model.to(device)
model.eval()

for i in range(10):
    generated_images_folder = '/home/kjh012/MS_Year_Proposal_2024_Winter/improved-diffusion/sample_soft_noisy_asymmetric_20/Class ' + str(i)
    # 생성된 이미지 로드 및 라벨 추출
    generated_images, true_labels = load_images_and_labels(generated_images_folder, transform, device)

    # 생성된 이미지 분류
    with torch.no_grad():
        outputs = model(generated_images)
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

    # CAS 계산
    cas = accuracy_score(true_labels, predicted_labels)
    print(f"Classification Accuracy Score (CAS) Class {i}: {cas}")