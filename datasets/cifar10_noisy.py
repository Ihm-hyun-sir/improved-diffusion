import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def add_symmetric_noise(labels, noise_rate, num_classes=10):
    """
    대칭 노이즈 추가 함수
    Args:
        labels (numpy.ndarray): 원본 라벨
        noise_rate (float): 노이즈 비율 (0~1)
        num_classes (int): 클래스 수
    Returns:
        noisy_labels (numpy.ndarray): 노이즈가 추가된 라벨
    """
    noisy_labels = labels.copy()
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        original_label = labels[idx]
        noisy_label = np.random.choice([i for i in range(num_classes) if i != original_label])
        noisy_labels[idx] = noisy_label

    return noisy_labels

def add_asymmetric_noise(labels, noise_rate):
    """
    비대칭 노이즈 추가 함수
    Args:
        labels (numpy.ndarray): 원본 라벨
        noise_rate (float): 노이즈 비율 (0~1)
    Returns:
        noisy_labels (numpy.ndarray): 노이즈가 추가된 라벨
    """
    transition_map = {
        2: 0,  # Bird -> Airplane
        3: 5,  # Cat -> Dog
        4: 7,  # Deer -> Horse
        9: 1   # Truck -> Automobile
    }

    noisy_labels = labels.copy()
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        original_label = labels[idx]
        if original_label in transition_map:
            noisy_labels[idx] = transition_map[original_label]

    return noisy_labels

def save_images_flat(dataset, labels, save_dir):
    """
    하나의 폴더에 이미지와 라벨을 저장합니다.
    Args:
        dataset: CIFAR-10 데이터셋
        labels: 수정된 라벨 리스트
        save_dir: 이미지를 저장할 디렉토리 경로
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_counters = {class_name: 0 for class_name in CLASS_NAMES.values()}

    for idx, (image, label) in enumerate(zip(dataset.data, labels)):
        class_name = CLASS_NAMES[label]
        file_name = f"{class_name}_{class_counters[class_name]}.png"
        class_counters[class_name] += 1

        img = Image.fromarray(image)
        img.save(os.path.join(save_dir, file_name))

def save_clean_testset_flat(dataset, save_dir):
    """
    CIFAR-10 테스트 셋을 하나의 폴더에 저장합니다.
    Args:
        dataset: CIFAR-10 테스트 데이터셋
        save_dir: 이미지를 저장할 디렉토리 경로
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_counters = {class_name: 0 for class_name in CLASS_NAMES.values()}

    for idx, (image, label) in enumerate(zip(dataset.data, dataset.targets)):
        class_name = CLASS_NAMES[label]
        file_name = f"{class_name}_{class_counters[class_name]}.png"
        class_counters[class_name] += 1

        img = Image.fromarray(image)
        img.save(os.path.join(save_dir, file_name))

# CIFAR-10 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_labels = np.array(trainset.targets)

# 노이즈 비율 설정
noise_rate = 0.2  # 40% 노이즈

# 대칭 노이즈 추가
symmetric_noisy_labels = add_symmetric_noise(train_labels, noise_rate)

# 비대칭 노이즈 추가
asymmetric_noisy_labels = add_asymmetric_noise(train_labels, noise_rate)

# 데이터 저장
save_images_flat(trainset, symmetric_noisy_labels, "./symmetric_noisy_train_dataset")
save_images_flat(trainset, asymmetric_noisy_labels, "./asymmetric_noisy_train_dataset")
save_clean_testset_flat(testset, "./clean_test_dataset")

print("모든 데이터셋 저장 완료!")
