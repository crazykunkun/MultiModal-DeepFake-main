"""
train_distill.py — CSCL 知识蒸馏训练脚本（方案B：输出+中间特征蒸馏）
教师模型：完整 CSCL（从 checkpoint 加载，冻结）
学生模型：轻量版 CSCL_Student（CCD 2层、MLP 瘦身、consist_encoder 压缩）
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import numpy as np
import random
import time
import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist

from models.CSCL import CSCL
from models.CSCL_student import CSCL_Student
from transformers import RobertaTokenizerFast

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist()
        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)
        for j in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == j)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)
    return text_input, fake_token_pos_batch


# ============ 蒸馏损失函数 ============ #
def distillation_loss(s_out, t_out, config, temperature=4.0):
    """
    计算蒸馏损失：
    1. soft logit loss（BIC 用 KL 散度，MLC 用 MSE）
    2. 中间特征 MSE（CCD 矩阵、SCD 聚合特征、SCD 一致性分数）
    """
    # --- soft logit distillation ---
    # BIC: KL 散度
    s_bic = F.log_softmax(s_out['vl_output'] / temperature, dim=-1)
    t_bic = F.softmax(t_out['vl_output'] / temperature, dim=-1)
    loss_soft_bic = F.kl_div(s_bic, t_bic, reduction='batchmean') * (temperature ** 2)

    # MLC: MSE
    loss_soft_mlc = F.mse_loss(s_out['output_cls_img'], t_out['output_cls_img']) \
                  + F.mse_loss(s_out['output_cls_text'], t_out['output_cls_text'])

    # Bbox: MSE
    loss_soft_bbox = F.mse_loss(s_out['output_coord'], t_out['output_coord'])

    loss_soft = loss_soft_bic + loss_soft_mlc + loss_soft_bbox

    # --- 中间特征蒸馏 ---
    # CCD 一致性矩阵
    loss_feat_ccd = F.mse_loss(s_out['img_matrix_pred'], t_out['img_matrix_pred']) \
                  + F.mse_loss(s_out['text_matrix_pred'], t_out['text_matrix_pred'])

    # SCD 聚合特征
    loss_feat_scd = F.mse_loss(s_out['agger_feat_img'], t_out['agger_feat_img']) \
                  + F.mse_loss(s_out['agger_feat_text'], t_out['agger_feat_text'])

    # SCD 一致性分数
    loss_score = F.mse_loss(s_out['sim_score_img'], t_out['sim_score_img']) \
               + F.mse_loss(s_out['sim_score_text'], t_out['sim_score_text'])

    # 蒸馏超参
    alpha = config.get('distill_alpha', 1.0)
    beta = config.get('distill_beta', 0.5)
    gamma = config.get('distill_gamma', 0.5)
    delta = config.get('distill_delta', 0.5)

    loss_distill = alpha * loss_soft + beta * loss_feat_ccd + gamma * loss_feat_scd + delta * loss_score
    return loss_distill, {
        'loss_soft': loss_soft.item(),
        'loss_feat_ccd': loss_feat_ccd.item(),
        'loss_feat_scd': loss_feat_scd.item(),
        'loss_score': loss_score.item(),
    }


# ============ 蒸馏训练循环 ============ #
def train_distill(args, teacher, student, data_loader, optimizer, tokenizer,
                  epoch, device, scheduler, config, summary_writer):
    student.train()
    teacher.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_hard', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_distill', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_total', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Distill Epoch: [{}]'.format(epoch)
    print_freq = 100
    global_step = epoch * len(data_loader)

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, image_path) in enumerate(
            metric_logger.log_every(args, data_loader, print_freq, header)):

        optimizer.zero_grad()
        image = image.to(device, non_blocking=True)
        fake_image_box = fake_image_box.to(device, non_blocking=True)
        text_input = tokenizer(text, max_length=128, truncation=True,
                               add_special_tokens=True, return_attention_mask=True,
                               return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        # 教师前向（不计算梯度）
        with torch.no_grad():
            t_out = teacher.forward_with_intermediates(image, label, text_input, fake_image_box, fake_token_pos)

        # 学生前向
        s_out = student.forward_with_intermediates(image, label, text_input, fake_image_box, fake_token_pos)

        # hard label loss（学生自己的任务损失）
        loss_hard = config['loss_BIC_wgt'] * s_out['loss_BIC'] \
                  + config['loss_bbox_wgt'] * s_out['loss_bbox'] \
                  + config['loss_giou_wgt'] * s_out['loss_giou'] \
                  + config['loss_MLC_wgt'] * s_out['loss_MLC'] \
                  + config['Loss_sim_wgt'] * s_out['Loss_sim']

        # 蒸馏损失
        loss_kd, kd_info = distillation_loss(s_out, t_out, config)

        # 总损失 = hard + 蒸馏
        loss_total = loss_hard + loss_kd

        loss_total.backward()
        optimizer.step()

        metric_logger.update(loss_hard=loss_hard.item())
        metric_logger.update(loss_distill=loss_kd.item())
        metric_logger.update(loss_total=loss_total.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step += 1
        if args.log and summary_writer is not None:
            summary_writer.add_scalar('loss_hard', loss_hard.item(), global_step)
            summary_writer.add_scalar('loss_distill', loss_kd.item(), global_step)
            summary_writer.add_scalar('loss_total', loss_total.item(), global_step)

    metric_logger.synchronize_between_processes()
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# ============ 主入口 ============ #
def main_worker(gpu, args, config):
    if gpu is not None:
        args.gpu = gpu

    if args.distributed:
        init_dist(args)
    else:
        args.rank = 0
        args.world_size = 1
        args.log = True
        torch.cuda.set_device(args.gpu)
    log_dir = os.path.join(args.output_dir, 'log_distill_' + args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w'))

    summary_writer = SummaryWriter(log_dir) if args.log else None

    if args.log:
        logger.info(args)
        logger.info(config)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    max_epoch = config['schedular']['epochs']

    # ---- 数据 ----
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]
    else:
        samplers = [None, None]
    train_loader, val_loader = create_loader(
        [train_dataset, val_dataset], samplers,
        batch_size=[config['batch_size_train']] + [config['batch_size_val']],
        num_workers=[4, 4], is_trains=[True, False], collate_fns=[None, None])

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)

    # ---- 教师模型（冻结） ----
    if args.log:
        print(f"Loading teacher from {args.teacher_checkpoint}")
    teacher = CSCL(args=args, config=config)
    ckpt = torch.load(args.teacher_checkpoint, map_location='cpu')
    teacher.load_state_dict(ckpt['model'], strict=False)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- 学生模型 ----
    if args.log:
        print("Creating student model (CSCL_Student)")
    student = CSCL_Student(args=args, config=config)
    student = student.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, student)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
        student_without_ddp = student.module
    else:
        student_without_ddp = student

    if args.log:
        print("Start distillation training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        train_stats = train_distill(args, teacher, student, train_loader, optimizer,
                                    tokenizer, epoch, device, lr_scheduler, config, summary_writer)

        if args.log:
            print(f"Epoch {epoch} stats: {train_stats}")

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            with open(os.path.join(log_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': student_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(log_dir, 'student_checkpoint_latest.pth'))

        if config['schedular']['sched'] != 'cosine_in_step':
            warmup_steps = config['schedular']['warmup_epochs']
            lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'student_checkpoint_%02d.pth' % epoch))
    total_time = time.time() - start_time
    if args.log:
        print('Distillation time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train.yaml')
    parser.add_argument('--teacher_checkpoint', default='D:/study/bishe/DEMO/model/checkpoint_49.pth')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='d:/study/bishe/DEMO/model/roberta-base')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist-url', default='tcp://localhost:23459', type=str)
    parser.add_argument('--dist-backend', default='gloo', type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--log_num', default='new', type=str)
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--batch_size', default=0, type=int, help='override batch_size_train if > 0')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # 蒸馏超参（可在 config 中覆盖）
    config.setdefault('distill_alpha', 1.0)
    config.setdefault('distill_beta', 0.5)
    config.setdefault('distill_gamma', 0.5)
    config.setdefault('distill_delta', 0.5)

    if args.batch_size > 0:
        config['batch_size_train'] = args.batch_size

    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))
