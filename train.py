import warnings
# 忽略运行期告警，减少训练日志噪声（调试阶段可按需关闭）
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 关闭 tokenizer 并行提示，避免多进程下反复打印警告信息
import pdb

import argparse
# import ruamel_yaml as yaml
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
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
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.CSCL import CSCL
from transformers import RobertaTokenizerFast

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # 将 tokenizer 输出调整为模型需要的格式：去掉 SEP、重新 padding、构造伪造 token 位置
    # 目标：把“按词标注的伪造位置(fake_word_pos)”映射成“按子词 token 的位置”
    # 这样模型的 token-level 分支才能和文本输入严格对齐
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers
        
        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def get_stage_config(config, epoch):
    stages_config = config.get('stages', {})
    if not stages_config.get('enabled', False):
        total_epochs = config['schedular']['epochs']
        return {
            'name': 'default',
            'index': 0,
            'start_epoch': 0,
            'end_epoch': total_epochs,
            'freeze': {'trainable_groups': ['all']},
            'loss_weights': {
                'loss_BIC_wgt': config['loss_BIC_wgt'],
                'loss_bbox_wgt': config['loss_bbox_wgt'],
                'loss_giou_wgt': config['loss_giou_wgt'],
                'loss_MLC_wgt': config['loss_MLC_wgt'],
                'Loss_sim_wgt': config['Loss_sim_wgt'],
            },
            'optimizer': {},
            'schedular': {'epochs': total_epochs, 'warmup_epochs': config['schedular']['warmup_epochs']},
        }

    matched_stage = None
    for index, stage in enumerate(stages_config.get('definitions', [])):
        if stage['start_epoch'] <= epoch < stage['end_epoch']:
            matched_stage = dict(stage)
            matched_stage['index'] = index
            break

    if matched_stage is None:
        raise ValueError(f'No stage config found for epoch {epoch}')

    return matched_stage


def merge_stage_optimizer_config(config, stage):
    optimizer_config = dict(config['optimizer'])
    optimizer_config.update(stage.get('optimizer', {}))
    return optimizer_config


def merge_stage_scheduler_config(config, stage):
    scheduler_config = dict(config['schedular'])
    scheduler_config.update(stage.get('schedular', {}))
    return scheduler_config


def get_stage_loss_weights(config, stage):
    loss_weights = {
        'loss_BIC_wgt': config['loss_BIC_wgt'],
        'loss_bbox_wgt': config['loss_bbox_wgt'],
        'loss_giou_wgt': config['loss_giou_wgt'],
        'loss_MLC_wgt': config['loss_MLC_wgt'],
        'Loss_sim_wgt': config['Loss_sim_wgt'],
    }
    loss_weights.update(stage.get('loss_weights', {}))
    return loss_weights


def build_optimizer_scheduler_for_stage(config, model_without_ddp, stage):
    trainable_groups = stage.get('freeze', {}).get('trainable_groups', ['all'])
    trainable_summary = model_without_ddp.set_trainable_groups(trainable_groups)

    optimizer_config = merge_stage_optimizer_config(config, stage)
    scheduler_config = merge_stage_scheduler_config(config, stage)

    optimizer = create_optimizer(utils.AttrDict(optimizer_config), model_without_ddp)
    scheduler, _ = create_scheduler(utils.AttrDict(scheduler_config), optimizer)
    return optimizer, scheduler, optimizer_config, scheduler_config, trainable_summary


def build_stage_state(config, epoch):
    stage = get_stage_config(config, epoch)
    loss_weights = get_stage_loss_weights(config, stage)
    stage_state = dict(stage)
    stage_state['loss_weights'] = loss_weights
    stage_state['local_epoch'] = epoch - stage['start_epoch']
    return stage_state


def train(args, model, data_loader, optimizer, tokenizer, scaler, epoch, warmup_steps, device, scheduler, config, summary_writer, stage_info):
    # 单个 epoch 的训练流程：前向、损失加权、反向传播、指标统计与日志记录
    # 输入：
    # - data_loader: 每次返回(image, label, text, fake_image_box, fake_word_pos, W, H, image_path)
    # - tokenizer: 将原始文本转为 token
    # 输出：
    # - 当前 epoch 各项损失/学习率的全局平均值（跨卡同步后）
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # 注册需要持续追踪的训练指标（平滑窗口用于降低抖动，便于观察趋势）
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Loss_sim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100   
    step_size = 100
    local_epoch = stage_info['local_epoch']
    warmup_iterations = warmup_steps*step_size

    # 全局步数用于 tensorboard 横轴（跨 epoch 累计）
    global_step = epoch*len(data_loader)
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, image_path) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):

        # 按 iteration 调整学习率（仅 cosine_in_step 策略）
        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + local_epoch, args, config)

        optimizer.zero_grad(set_to_none=True)

        # 图像送入 GPU，文本走 tokenizer 并对齐到模型输入格式
        # text_input_adjust 会把“词级伪造标签”映射到“token级伪造标签”
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
        
        with autocast(enabled=device.type == 'cuda'):
            loss_BIC, loss_bbox, loss_giou, loss_MLC, Loss_sim = model(image, label, text_input, fake_image_box, fake_token_pos)

            loss_weights = stage_info['loss_weights']
            loss = loss_weights['loss_BIC_wgt']*loss_BIC \
                 + loss_weights['loss_bbox_wgt']*loss_bbox \
                 + loss_weights['loss_giou_wgt']*loss_giou \
                 + loss_weights['loss_MLC_wgt']*loss_MLC \
                 + loss_weights['Loss_sim_wgt']*Loss_sim
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(Loss_sim=Loss_sim.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if local_epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step':
            scheduler.step(i//step_size)

        global_step+=1
        

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],                                                                                                                                                                                              
                'loss_BIC': loss_BIC.item(),                                                                                                  
                'loss_bbox': loss_bbox.item(),                                                                                                  
                'loss_giou': loss_giou.item(),                                                                                                                                                                                               
                'loss_MLC': loss_MLC.item(),  
                'Loss_sim': Loss_sim.item(),  
                'loss': loss.item(),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # 验证流程：计算真实/伪造分类、多标签分类、框定位与 token 级别检测等指标
    # 注意：@torch.no_grad() 下不计算梯度，仅做前向推理与指标统计
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    # 逐批次前向推理并累计各任务指标所需统计量
    start_time = time.time()   
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   
    
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, image_path) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)
        # 模型一次前向返回多任务输出：
        # - logits_real_fake: 真伪二分类
        # - logits_multicls: 多标签类别预测
        # - output_coord: 预测篡改框
        # - logits_tok: token 级伪造检测

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)
        
        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    ##================= real/fake cls ========================##
    # 汇总二分类 AUC/ACC/EER
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    ##================= multi-label cls ========================##
    # 计算多标签 mAP 与整体/Top-k 指标
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)
    
    ##================= bbox cls ========================##
    # 计算框定位平均 IoU 与不同阈值下命中率
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)

    # ##================= token cls========================##
    # 根据 TP/TN/FP/FN 计算 token 级准确率、精确率、召回率与 F1
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)

    return AUC_cls, ACC_cls, EER_cls, \
           MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
           IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
           ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):
    # 训练主进程：初始化分布式环境、构建数据与模型、执行训练与验证循环、保存日志与权重
    # 这是完整训练生命周期的调度入口（每个进程/每张卡都会进入该函数）

    if gpu is not None:
        args.gpu = gpu

    if args.distributed:
        init_dist(args)
    else:
        args.rank = 0
        args.world_size = 1
        args.log = True
        if torch.cuda.is_available() and args.device == 'cuda':
            torch.cuda.set_device(args.gpu)
    # 创建当前实验日志目录，并保存运行配置
    log_dir = os.path.join(args.output_dir, 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w')) 
    
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 不同 rank 使用不同 seed（基础 seed + rank），保证多卡下随机性可控且可复现
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    if config.get('enable_tf32', True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    active_stage_name = None
    active_stage_index = None

    #### Dataset ####
    # 构建训练集/验证集；分布式训练时为训练集创建分布式采样器
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)
    
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                samplers,
                                batch_size=[config['batch_size_train']]+[config['batch_size_val']],
                                num_workers=[config.get('num_workers_train', 4), config.get('num_workers_val', 4)],
                                is_trains=[True, False],
                                collate_fns=[None, None],
                                persistent_workers=[config.get('persistent_workers', False), config.get('persistent_workers', False)],
                                prefetch_factors=[config.get('prefetch_factor', 2), config.get('prefetch_factor', 2)],
                                pin_memory=config.get('pin_memory', True))

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    #### Model ####
    # 实例化 CSCL 多模态模型并移动到目标设备
    if args.log:
        print(f"Creating CSCL")
    model = CSCL(args=args, config=config)
    model = model.to(device)

    optimizer = None
    lr_scheduler = None
    current_optimizer_config = None
    current_scheduler_config = None
    current_trainable_summary = None
    
    if args.checkpoint:
        # 可选：从 checkpoint 恢复模型；若 --resume=True 同时恢复优化器/调度器/起始 epoch
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            start_epoch = checkpoint['epoch']+1

        # model.load_state_dict(state_dict)
        if args.log:
            print('load checkpoint from %s'%args.checkpoint)
        msg = model.load_state_dict(state_dict, strict=False)
        if args.log:
            print(msg)

    model_without_ddp = model
    # 保存“未被 DDP 包装”的模型引用：
    # - 评估时直接调用原始模型
    # - 保存权重时获取干净的 state_dict
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=config.get('find_unused_parameters', False),
        )
        model_without_ddp = model.module

    initial_stage_info = build_stage_state(config, start_epoch)
    optimizer, lr_scheduler, current_optimizer_config, current_scheduler_config, current_trainable_summary = build_optimizer_scheduler_for_stage(config, model_without_ddp, initial_stage_info)
    warmup_steps = current_scheduler_config['warmup_epochs']
    active_stage_name = initial_stage_info['name']
    active_stage_index = initial_stage_info['index']

    if current_scheduler_config['sched'] == 'cosine_in_step':
        args.lr = current_optimizer_config['lr']

    scaler = GradScaler(enabled=device.type == 'cuda')

    if args.checkpoint and args.resume:
        checkpoint_stage = checkpoint.get('stage', {})
        if checkpoint_stage.get('name') == active_stage_name:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if current_scheduler_config['sched'] != 'cosine_in_step' and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        elif args.log:
            print('resume crosses stage boundary, rebuilding optimizer and scheduler for current stage')

    if args.log:
        print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        # 每个 epoch：先训练，再在验证集评估，随后记录日志并保存最新 checkpoint
        stage_info = build_stage_state(config, epoch)
        if stage_info['name'] != active_stage_name:
            optimizer, lr_scheduler, current_optimizer_config, current_scheduler_config, current_trainable_summary = build_optimizer_scheduler_for_stage(config, model_without_ddp, stage_info)
            warmup_steps = current_scheduler_config['warmup_epochs']
            active_stage_name = stage_info['name']
            active_stage_index = stage_info['index']
            if current_scheduler_config['sched'] == 'cosine_in_step':
                args.lr = current_optimizer_config['lr']
            if args.log:
                print(f"Switch to stage {active_stage_name} at epoch {epoch}, trainable params: {current_trainable_summary['trainable_params']}/{current_trainable_summary['total_params']}")

        train_stats = train(args, model, train_loader, optimizer, tokenizer, scaler, epoch, warmup_steps, device, lr_scheduler, config, summary_writer, stage_info)
        AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok \
        = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
                'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
                'EER_cls': round(EER_cls*100, 4),                                                                                                  
                'MAP': round(MAP*100, 4),                                                                                                  
                'OP': round(OP*100, 4),                                                                                                  
                'OR': round(OR*100, 4), 
                'OF1': round(OF1*100, 4), 
                'CP': round(CP*100, 4), 
                'CR': round(CR*100, 4), 
                'CF1': round(CF1*100, 4), 
                'OP_k': round(OP_k*100, 4), 
                'OR_k': round(OR_k*100, 4), 
                'OF1_k': round(OF1_k*100, 4), 
                'CP_k': round(CP_k*100, 4), 
                'CR_k': round(CR_k*100, 4), 
                'CF1_k': round(CF1_k*100, 4), 
                'IOU_score': round(IOU_score*100, 4),                                                                                                  
                'IOU_ACC_50': round(IOU_ACC_50*100, 4),                                                                                                  
                'IOU_ACC_75': round(IOU_ACC_75*100, 4),                                                                                                  
                'IOU_ACC_95': round(IOU_ACC_95*100, 4),                                                                                                  
                'ACC_tok': round(ACC_tok*100, 4),                                                                                                  
                'Precision_tok': round(Precision_tok*100, 4),                                                                                                  
                'Recall_tok': round(Recall_tok*100, 4),                                                                                                  
                'F1_tok': round(F1_tok*100, 4),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

        #============ evaluation info ============#
        val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                     "ACC_cls": "{:.4f}".format(ACC_cls*100),
                     "EER_cls": "{:.4f}".format(EER_cls*100),
                     "MAP": "{:.4f}".format(MAP*100),
                     "OP": "{:.4f}".format(OP*100),
                     "OR": "{:.4f}".format(OR*100),
                     "OF1": "{:.4f}".format(OF1*100),
                     "CP": "{:.4f}".format(CP*100),
                     "CR": "{:.4f}".format(CR*100),
                     "CF1": "{:.4f}".format(CF1*100),
                     "OP_k": "{:.4f}".format(OP_k*100),
                     "OR_k": "{:.4f}".format(OR_k*100),
                     "OF1_k": "{:.4f}".format(OF1_k*100),
                     "CP_k": "{:.4f}".format(CP_k*100),
                     "CR_k": "{:.4f}".format(CR_k*100),
                     "CF1_k": "{:.4f}".format(CF1_k*100),
                     "IOU_score": "{:.4f}".format(IOU_score*100),
                     "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                     "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                     "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                     "ACC_tok": "{:.4f}".format(ACC_tok*100),
                     "Precision_tok": "{:.4f}".format(Precision_tok*100),
                     "Recall_tok": "{:.4f}".format(Recall_tok*100),
                     "F1_tok": "{:.4f}".format(F1_tok*100),
        }
        
        if utils.is_main_process():
            # 仅主进程负责落盘，避免多进程重复写文件
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                            'stage_name': stage_info['name'],
                            'stage_index': stage_info['index'],
                            'stage_local_epoch': stage_info['local_epoch'],
                        }
            with open(os.path.join(log_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'stage': {
                        'name': stage_info['name'],
                        'index': stage_info['index'],
                        'start_epoch': stage_info['start_epoch'],
                        'end_epoch': stage_info['end_epoch'],
                        'local_epoch': stage_info['local_epoch'],
                        'trainable_groups': stage_info.get('freeze', {}).get('trainable_groups', ['all']),
                        'loss_weights': stage_info['loss_weights'],
                    },
                    'stage_optimizer_config': current_optimizer_config,
                    'stage_scheduler_config': current_scheduler_config,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'stage': {
                        'name': stage_info['name'],
                        'index': stage_info['index'],
                        'start_epoch': stage_info['start_epoch'],
                        'end_epoch': stage_info['end_epoch'],
                        'local_epoch': stage_info['local_epoch'],
                        'trainable_groups': stage_info.get('freeze', {}).get('trainable_groups', ['all']),
                        'loss_weights': stage_info['loss_weights'],
                    },
                    'stage_optimizer_config': current_optimizer_config,
                    'stage_scheduler_config': current_scheduler_config,
                }

            torch.save(save_obj, os.path.join(log_dir, 'checkpoint_latest.pth')) 

        if current_scheduler_config['sched'] != 'cosine_in_step':
            lr_scheduler.step(stage_info['local_epoch'] + warmup_steps + 1)
        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch))   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # 解析训练参数并根据 launcher 方式启动单机或多进程训练
    # 入口逻辑：
    # 1) 读取命令行参数
    # 2) 加载 YAML 配置
    # 3) 按 launcher 选择单进程或多进程启动
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train.yaml')
    parser.add_argument('--checkpoint', default=False) 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='./roberta-base')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', default='new', type=str)

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # main(args, config)
    if args.launcher == 'none':
        args.distributed = False
        main_worker(0, args, config)
    else:
        # 多 GPU 场景：为每张卡启动一个子进程，并行执行 main_worker
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))