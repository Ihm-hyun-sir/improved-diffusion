import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from tqdm import tqdm # tqdm import
###
import random
###
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        data_len,  # 데이터 길이
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        model_path, # 모델 저장 위치
        use_fp16=False,
        epoch = 100, # Epoch 수
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.data_len=data_len # 데이터 길이 설정
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.model_path = model_path # 모델 저장 위치 설정
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
 
        self.epoch = epoch # Epoch 수 저장
        self.steps_per_epoch = self.data_len//batch_size # Epoch 마다 총 step 수 (데이터 길이 // 배치 크기)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        for epc in range(self.epoch): # 정해진 Epoch 만큼 학습
            with tqdm(total=self.steps_per_epoch, desc=f"Epoch {epc + 1}/{self.epoch}", unit="step") as pbar: #tqdm으로 진행상황 출력
                for _ in range(self.steps_per_epoch): # 1 Epoch - 전체 데이터를 배치 단위로 학습
                    # 배치 단위 학습
                    batch, cond = next(self.data)
                    self.run_step(batch, cond)

                    # 10회 step 마다 Loss 진행 상황 logging
                    if self.step % self.log_interval == 0: 
                        logger.dumpkvs()
                    
                    self.step += 1 # step 추가 (모델 저장 및 로깅에 쓰임)
                    pbar.update(1) # tqdm 업데이트
        self.save() # 모든 Epoch을 다 수행했다면 모델을 저장

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def mixup_samples(self, data, labels):
        """
        Major와 Minor 클래스를 분리한 후, 서로 Mixup.
        Args:
            data: Tensor, 배치 내 데이터 (N, C, H, W).
            labels: Tensor, 배치 내 레이블 (N,).
            alpha: Mixup 비율을 결정하는 파라미터.
        Returns:
            Mixed data와 Mixed labels.
        """
        # Major와 Minor 분리
        major_mask = (labels < 5)
        minor_mask = (labels >= 5)

        major_indices = np.nonzero(major_mask)
        minor_indices = np.nonzero(minor_mask)

        # Mixup 결과 저장용 리스트
        mixed_data = []
        mixed_labels = []

        if len(major_indices) > 0 and len(minor_indices) > 0:
            # Major -> Minor 
            for idx in range(len(data)):
                if major_mask[idx] :
                    minor_idx = random.choice(minor_indices)
                    mixed_sample = 0.1 * data[idx] + 0.9 * data[minor_idx]
                    mixed_data.append(mixed_sample.squeeze(0))
                    mixed_labels.append((labels[minor_idx] , minor_idx))
                else : 
                    major_idx = random.choice(major_indices)
                    mixed_sample = 0.9 * data[idx] + 0.1 * data[major_idx]
                    mixed_data.append(mixed_sample.squeeze(0))
                    mixed_labels.append((labels[idx] ,idx))

            # Mixed data와 labels 반환
            mixed_data = th.stack(mixed_data)
            mixed_labels = th.tensor(mixed_labels)
            return mixed_data, mixed_labels , True
        
        else:
            return data, labels , False

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        ####
        mixed_data, mixed_labels , mixup_done = self.mixup_samples(batch, cond["y"])
        ####
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            ####
            if mixup_done :
                mixed_micro = mixed_data[i : i + self.microbatch].to(dist_util.dev())
                mixed_mirco_cond = { "y" : th.tensor([ int( mixed_labels[idx][0] ) for idx in range(i, min(i+self.microbatch,len(mixed_labels)))]) }
                mixed_minor_idx = th.tensor([ int( mixed_labels[idx][1] ) for idx in range(i, min(i+self.microbatch,len(mixed_labels)))]).to(dist_util.dev())
            ####

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            # if last_batch or not self.use_ddp:
            #     print("last_batch")
            #     losses = compute_losses()
            # else:
            #     print("Nooo")
            #     with self.ddp_model.no_sync():
            #         losses = compute_losses()

            losses , gt_target = compute_losses()
            loss = (losses["loss"] * weights).mean()
            total_loss = loss

            if mixup_done :
                vb = losses['vb']

                major_mask = (micro_cond["y"] < 5)
                major_indices = major_mask.nonzero(as_tuple=True)[0]
                selected_minor_indices = mixed_minor_idx[major_indices]

                # print(micro_cond)
                # print(mixed_mirco_cond)
                # print(mixed_minor_idx)
                # print(major_indices)
                # print(selected_minor_indices)
                gt_target[major_indices] = gt_target[selected_minor_indices]
                vb[major_indices] = vb[selected_minor_indices]
                
                
                compute_mixup_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    mixed_micro,
                    t,
                    model_kwargs=mixed_mirco_cond,
                    target = gt_target,
                    vb = vb
                )
                mixup_losses , _  = compute_mixup_losses()

                mixup_loss = (mixup_losses["loss"] * weights).mean()
                total_loss += mixup_loss

            # if isinstance(self.schedule_sampler, LossAwareSampler): 사용안함
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (total_loss * loss_scale).backward()
            else:
                total_loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt" # 일반 모델
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt" # EMA 모델을 쓰면 조금 더 효율적으로 샘플링 가능
                with bf.BlobFile(bf.join(self.model_path, filename), "wb") as f: # 모델 저장 위치에 현재 모델 상태 저장 (model_XXXX.pt, ema_0.9999_XXXX.pt) 
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.model_path, f"opt{(self.step+self.resume_step):06d}.pt"), # 모델 저장 위치에 현재 Optimizer 상태 저장
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
