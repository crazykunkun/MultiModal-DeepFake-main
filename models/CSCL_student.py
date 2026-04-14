"""
CSCL_Student: 轻量版学生模型，用于知识蒸馏。
相比教师模型（CSCL）的轻量化：
1. CCD Self_Interaction 层数 3→2
2. CCD/SCD build_mlp 中间维度 2x→1x
3. CCD/SCD consist_encoder 768→256→128→64 压缩为 768→128→64
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import yaml

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from .consist_modeling import get_sscore_label, get_sscore_label_text
from timm.models.layers import trunc_normal_
from .METER import METERTransformerSS
from .interaction import Self_Interaction
from .CSCL import score2posemb1d, pos2posemb2d, coords_2d
from .CSCL import get_weighted_bce_loss, get_it_bce_loss


# ============ 轻量版 CCD ============ #
class Intra_Modal_Modeling_Lite(nn.Module):
    def __init__(self, num_head, hidden_dim, input_dim, output_dim, tok_num):
        super().__init__()
        # 层数 3→2
        self.correlation_model = Self_Interaction(num_head, hidden_dim, input_dim, output_dim, layers=2)
        # consist_encoder 压缩：去掉 256 中间层
        self.consist_encoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        self.token_number = tok_num
        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.num_head = 4

    # build_mlp 瘦身：中间维度 1x（而非教师的 2x）
    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, feats, mask, pos_emb, matrix_mask=None):
        B, N, C = feats.shape
        feats = self.correlation_model(feats, mask, pos_emb)
        consist_feats = self.consist_encoder(feats)

        norms = torch.norm(consist_feats, p=2, dim=2, keepdim=True)
        normalized_vectors = consist_feats / norms
        similarity_matrix = torch.bmm(normalized_vectors, normalized_vectors.transpose(1, 2))
        similarity_matrix = torch.clamp((similarity_matrix + 1) / 2, 0, 1)

        if mask.sum() > 0:
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_unsim[~matrix_mask] = 2
            similarity_matrix_sim = similarity_matrix.clone()
            similarity_matrix_sim[~matrix_mask] = -1
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask
        else:
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_sim = similarity_matrix.clone()
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask

        topk_k = min(self.token_number, N)
        unsim_feats_index = torch.topk(similarity_matrix_unsim, topk_k, dim=-1, largest=False)[1]
        unsim_attn_mask = torch.ones([B, N, N], dtype=torch.bool).to(unsim_feats_index.device)
        batch_indices = torch.arange(B).view(B, 1, 1)
        row_indices = torch.arange(N).view(1, N, 1)
        unsim_attn_mask[batch_indices, row_indices, unsim_feats_index] = False
        unsim_attn_mask = unsim_attn_mask.repeat(self.num_head, 1, 1)

        sim_feats_index = torch.topk(similarity_matrix_sim, topk_k, dim=-1, largest=True)[1]
        sim_attn_mask = torch.ones([B, N, N], dtype=torch.bool).to(sim_feats_index.device)
        batch_indices = torch.arange(B).view(B, 1, 1)
        row_indices = torch.arange(N).view(1, N, 1)
        sim_attn_mask[batch_indices, row_indices, sim_feats_index] = False
        sim_attn_mask = sim_attn_mask.repeat(self.num_head, 1, 1)

        feats = feats + self.aggregator_mlp(self.aggregator(query=feats, key=feats, value=feats, attn_mask=sim_attn_mask)[0])
        feats = feats + self.aggregator_mlp_2(self.aggregator_2(query=feats, key=feats, value=feats, attn_mask=unsim_attn_mask)[0])

        return feats, similarity_matrix, consist_feats


# ============ 轻量版 SCD ============ #
class Extra_Modal_Modeling_Lite(nn.Module):
    def __init__(self, num_head, output_dim, tok_num):
        super().__init__()
        self.feat_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.cross_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.token_number = tok_num
        # consist_encoder 压缩
        self.consist_encoder_feat = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        self.consist_encoder_cross = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        self.cls_token_cross = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_cross = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_cross = nn.LayerNorm(output_dim)
        self.cls_token_feat = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_feat = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_feat = nn.LayerNorm(output_dim)
        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        trunc_normal_(self.cls_token_cross, std=.02)
        trunc_normal_(self.cls_token_feat, std=.02)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, feats, gloabl_feature, cross_feat, feats_mask, cross_mask):
        bs, _, _ = feats.shape
        feats = self.feat_encoder(feats)
        cross_feat = self.cross_encoder(cross_feat)

        cls_token_cross = self.cls_token_cross.expand(bs, -1, -1)
        feat_aggr_cross = self.aggregator_cross(
            query=self.norm_layer_cross(cls_token_cross),
            key=self.norm_layer_cross(cross_feat),
            value=self.norm_layer_cross(cross_feat),
            key_padding_mask=cross_mask)[0]

        feats_consist = self.consist_encoder_feat(feats)
        cross_feats_consist = self.consist_encoder_feat(feat_aggr_cross)

        norms_feat = torch.norm(feats_consist, p=2, dim=2, keepdim=True)
        norms_cross = torch.norm(cross_feats_consist, p=2, dim=2, keepdim=True)
        sim_matrix = torch.bmm(feats_consist / norms_feat, (cross_feats_consist / norms_cross).transpose(1, 2))
        sim_matrix = torch.clamp((sim_matrix + 1) / 2, 0, 1).squeeze()

        cls_token = self.cls_token_feat.expand(bs, -1, -1)
        global_feats_mask = torch.zeros(feats_mask.shape[0], 1).bool().to(feats_mask.device)
        feat_aggr = self.aggregator_feat(
            query=self.norm_layer_feat(cls_token),
            key=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)),
            value=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)),
            key_padding_mask=torch.cat([global_feats_mask, feats_mask], dim=1))[0]

        if feats_mask.sum() > 0:
            sim_score = sim_matrix.clone()
            sim_score[feats_mask] = -1
            unsim_score = sim_matrix.clone()
            unsim_score[feats_mask] = 2
        else:
            sim_score = sim_matrix.clone()
            unsim_score = sim_matrix.clone()

        topk_k = min(self.token_number, feats.shape[1])
        unsim_index = torch.topk(unsim_score, topk_k, dim=-1, largest=False)[1]
        unsim_patch = feats[torch.arange(feats.shape[0]).unsqueeze(1), unsim_index]
        sim_index = torch.topk(sim_score, topk_k, dim=-1, largest=True)[1]
        sim_patch = feats[torch.arange(feats.shape[0]).unsqueeze(1), sim_index]

        feat_aggr = feat_aggr + self.aggregator_mlp(self.aggregator(query=feat_aggr, key=sim_patch, value=sim_patch)[0])
        feat_aggr = feat_aggr + self.aggregator_mlp_2(self.aggregator_2(query=feat_aggr, key=unsim_patch, value=unsim_patch)[0])

        return feat_aggr, sim_matrix, feats_consist


# ============ 轻量版 CSCL 学生模型 ============ #
class CSCL_Student(nn.Module):
    def __init__(self, args=None, config=None):
        super().__init__()
        config_meter = yaml.load(open('configs/METER.yaml', 'r'), Loader=yaml.Loader)
        self.args = args
        text_width = config_meter['input_text_embed_size']
        vision_width = config_meter['input_image_embed_size']

        # 任务头：瘦身版 MLP（1x 中间维度）
        self.fusion_head = self.build_mlp(input_dim=text_width + text_width, output_dim=text_width)
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.cls_head_img = self.build_mlp(input_dim=text_width, output_dim=2)
        self.cls_head_text = self.build_mlp(input_dim=text_width, output_dim=2)

        # 轻量版 CCD
        self.img_intra_model = Intra_Modal_Modeling_Lite(12, 1024, vision_width, vision_width, 16)
        self.text_intra_model = Intra_Modal_Modeling_Lite(12, 1024, vision_width, vision_width, 8)

        # 轻量版 SCD
        self.img_extra_model = Extra_Modal_Modeling_Lite(12, vision_width, 16)
        self.text_extra_model = Extra_Modal_Modeling_Lite(12, vision_width, 8)

        self.emb_img_pos = nn.Sequential(
            nn.Linear(text_width * 2, text_width),
            nn.LayerNorm(text_width)
        )
        self.emb_text_pos = nn.Sequential(
            nn.Linear(text_width, text_width),
            nn.LayerNorm(text_width)
        )

        self.apply(self._init_weights)
        # 共享同一个 METER 编码器
        self.meter = METERTransformerSS(config_meter)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)
        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)
        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward_with_intermediates(self, image, label, text, fake_image_box, fake_text_pos):
        """学生蒸馏前向：返回 5 个损失 + 所有中间特征"""
        text_atts_mask_clone = text.attention_mask.clone()
        text_atts_mask_bool = text_atts_mask_clone == 0

        token_label = text.attention_mask[:, 1:].clone()
        token_label[token_label == 0] = -100
        token_label[token_label == 1] = 0
        for batch_idx in range(len(fake_text_pos)):
            fake_pos_sample = fake_text_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1

        multicls_label, real_label_pos = get_multi_label(label, image)
        sim_matrix_img, patch_label, _, _, _ = get_sscore_label(image, fake_image_box, token_label)
        sim_matrix_text, sim_matrix_text_mask = get_sscore_label_text(token_label)

        batch = {}
        batch["text_ids"] = text.input_ids
        batch["text_masks"] = text.attention_mask
        outputs = self.meter.infer(batch=batch, img=image)
        text_embeds_output = outputs['text_feats']
        image_embeds_output = outputs['image_feats']
        fusion_token = self.fusion_head(outputs['cls_feats'])

        with torch.no_grad():
            bs = image.size(0)
        itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
        itm_labels[real_label_pos] = 0
        vl_output = self.itm_head(fusion_token)
        loss_BIC = F.cross_entropy(vl_output, itm_labels)

        image_atts = torch.ones(image_embeds_output.size()[:-1], dtype=torch.long).to(image.device)
        image_atts_mask_bool = (image_atts == 0)
        patch_pos_emb = self.emb_img_pos(pos2posemb2d(coords_2d(16, 16).to(fusion_token.device).unsqueeze(0).repeat(bs, 1, 1)))
        img_patch_feat = image_embeds_output[:, 1:, :]
        img_patch_feat, img_matrix_pred, _ = self.img_intra_model(img_patch_feat, image_atts_mask_bool[:, 1:], patch_pos_emb)
        len_text = text_embeds_output.shape[1] - 1
        token_pos_emb = self.emb_text_pos(score2posemb1d(torch.arange(0, len_text, dtype=torch.float).to(text_embeds_output.device).unsqueeze(1).repeat(bs, 1, 1)))
        text_token_feat = text_embeds_output[:, 1:, :]
        text_token_feat, text_matrix_pred, _ = self.text_intra_model(text_token_feat, text_atts_mask_bool[:, 1:], token_pos_emb, sim_matrix_text_mask)

        Loss_img_matrix, _, _ = get_weighted_bce_loss(img_matrix_pred.view(-1), sim_matrix_img.float().view(-1))
        Loss_text_matrix, _, _ = get_weighted_bce_loss(text_matrix_pred[sim_matrix_text_mask].view(-1), sim_matrix_text[sim_matrix_text_mask].view(-1).float())

        agger_feat_img, sim_score_img, _ = self.img_extra_model(img_patch_feat, image_embeds_output[:, 0:1, :], text_token_feat, image_atts_mask_bool[:, 1:], text_atts_mask_bool[:, 1:])
        agger_feat_text, sim_score_text, _ = self.text_extra_model(text_token_feat, text_embeds_output[:, 0:1, :], img_patch_feat, text_atts_mask_bool[:, 1:], image_atts_mask_bool[:, 1:])

        Loss_img_score, _, _ = get_it_bce_loss(sim_score_img.view(-1), patch_label.view(-1).float())
        Loss_text_score, _, _ = get_it_bce_loss(sim_score_text.view(-1), token_label.view(-1).float())
        Loss_sim = Loss_img_score + Loss_img_matrix + Loss_text_score + Loss_text_matrix

        output_coord = self.bbox_head(agger_feat_img.squeeze(1)).sigmoid()
        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
        output_cls_img = self.cls_head_img(agger_feat_img.squeeze(1))
        loss_MLC_img = F.binary_cross_entropy_with_logits(output_cls_img, multicls_label.type(torch.float)[:, :2])
        output_cls_text = self.cls_head_text(agger_feat_text.squeeze(1))
        loss_MLC_text = F.binary_cross_entropy_with_logits(output_cls_text, multicls_label.type(torch.float)[:, 2:])
        loss_MLC = loss_MLC_img + loss_MLC_text

        return {
            'loss_BIC': loss_BIC,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_MLC': loss_MLC,
            'Loss_sim': Loss_sim,
            'img_matrix_pred': img_matrix_pred,
            'text_matrix_pred': text_matrix_pred,
            'agger_feat_img': agger_feat_img,
            'agger_feat_text': agger_feat_text,
            'sim_score_img': sim_score_img,
            'sim_score_text': sim_score_text,
            'vl_output': vl_output,
            'output_coord': output_coord,
            'output_cls_img': output_cls_img,
            'output_cls_text': output_cls_text,
        }
