#!/bin/bash

# 모델 경로
MODEL_PATH="/home2/ihmhyunsir/WorkingSpace/lab_proj/diffusion_imbalance/idv2/improved-diffusion/trained_model_2/model077500.pt"
# 베이스 저장 경로
BASE_SAVE_PATH="/home2/ihmhyunsir/WorkingSpace/lab_proj/diffusion_imbalance/idv2/improved-diffusion/sampled_images_2"
# 기타 플래그
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"

# 클래스 0부터 9까지 실행
for CLASS_NUM in {0..9}
do
    # 저장 경로 설정
    SAVE_PATH="${BASE_SAVE_PATH}/Class ${CLASS_NUM}/"
    
    # 스크립트 실행
    python scripts/image_sample.py \
        --model_path "$MODEL_PATH" \
        --save_path "$SAVE_PATH" \
        --use_ddim True \
        --timestep_respacing ddim50 \
        --num_samples 1000 \
        --use_classwise True \
        --class_num "$CLASS_NUM" \
        $MODEL_FLAGS $DIFFUSION_FLAGS
done
