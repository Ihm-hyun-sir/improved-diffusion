"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if not args.use_classwise:
                classes = th.randint(
                    low=0, high=10, size=(args.batch_size,), device=dist_util.dev() # 클래스 수 10 중에서 랜덤으로 선택해서 생성하도록 설정
                )
            else:
                classes = th.randint(
                    low=args.class_num, high=args.class_num+1, size=(args.batch_size,), device=dist_util.dev() # 원하는 클래스만 생성하도록 설정
                )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if not args.use_classwise:
            out_path = os.path.join(args.save_path, f"samples_{shape_str}.npz") # 이미지 저장 경로 설정
        else:
            out_path = os.path.join(args.save_path, f"samples_{shape_str}_class_{args.class_num}.npz") # 이미지 저장 경로 설정
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


    #################
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
    data = np.load(out_path)

    # 특정 키로 데이터 가져오기
    image_array = data['arr_0']
    labels = data['arr_1']
    print(labels[0])
    # 저장 경로 설정

    # 모든 이미지를 저장
    for i in range(image_array.shape[0]):
        image = Image.fromarray(image_array[i])
        image.save(f"{args.save_path}{class_mapping[labels[i]]}_{i}.png")
    #################


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_path="", # 이미지 기본 저장 경로
        use_classwise=False,
        class_num=-1, # 출력하고 싶은 클래스
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
