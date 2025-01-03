import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class_mapping = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck",
}

# npz 파일 로드
data = np.load('sample_original_noisy_asymmetric_20/samples_10000x32x32x3.npz')

# 특정 키로 데이터 가져오기
image_array = data['arr_0']
labels = data['arr_1']

# 저장 경로 설정
output_folder = 'sample_original_noisy_asymmetric_20/Uncond/'
import os
os.makedirs(output_folder, exist_ok=True)

# 모든 이미지를 저장
for i in range(image_array.shape[0]):
    image = Image.fromarray(image_array[i])
    image.save(f"{output_folder}{class_mapping[labels[i]]}_{i}.png")