import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None


class SRMConv2d(nn.Module):
    """
    Fixed SRM high-pass filtering layer.

    Input:
        x: [B, 3, H, W]
    Output:
        noise: [B, 9, H, W]
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = self._build_srm_kernels()  # [9, 1, 5, 5]
        self.register_buffer("weight", kernels, persistent=False)

    @staticmethod
    def _normalize_kernel(kernel: torch.Tensor, scale: float) -> torch.Tensor:
        return kernel / scale

    def _build_srm_kernels(self) -> torch.Tensor:
        first_order = [
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, -1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, -1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, -1, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
        ]

        second_order = [
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, -2, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, -2, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, -2, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
        ]

        third_order = [
            torch.tensor(
                [[0, 0, 0, 0, 0],
                 [0, -1, 2, -1, 0],
                 [0, 2, -4, 2, 0],
                 [0, -1, 2, -1, 0],
                 [0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            torch.tensor(
                [[-1, 2, -2, 2, -1],
                 [2, -6, 8, -6, 2],
                 [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2],
                 [-1, 2, -2, 2, -1]], dtype=torch.float32
            ),
            torch.tensor(
                [[0, 0, -1, 0, 0],
                 [0, 2, -4, 2, 0],
                 [-1, -4, 12, -4, -1],
                 [0, 2, -4, 2, 0],
                 [0, 0, -1, 0, 0]], dtype=torch.float32
            ),
        ]

        kernels = []
        for kernel in first_order:
            kernels.append(self._normalize_kernel(kernel, scale=2.0))
        for kernel in second_order:
            kernels.append(self._normalize_kernel(kernel, scale=4.0))
        for kernel in third_order:
            kernels.append(self._normalize_kernel(kernel, scale=12.0))

        stacked = torch.stack(kernels, dim=0).unsqueeze(1)  # [9, 1, 5, 5]
        return stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(f"SRMConv2d expects [B, 3, H, W], got {tuple(x.shape)}")

        channels = []
        for c in range(3):
            x_c = x[:, c:c + 1, :, :]  # [B, 1, H, W]
            noise_c = F.conv2d(x_c, self.weight, padding=2)  # [B, 9, H, W]
            channels.append(noise_c)

        noise = torch.cat(channels, dim=1)  # [B, 27, H, W]
        return noise


class ResNet18SpatialEncoder(nn.Module):
    """
    Lightweight ResNet18 backbone without FC.

    Input:
        x: [B, C_in, H, W]
    Output:
        feat: [B, 512, H/32, W/32]
    """

    def __init__(self, in_channels: int = 27, pretrained: bool = False) -> None:
        super().__init__()
        if resnet18 is not None:
            weights = "DEFAULT" if pretrained else None
            backbone = resnet18(weights=weights)
            backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.stem = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
            )
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.layer1 = self._make_stage(64, 64, stride=1)
            self.layer2 = self._make_stage(64, 128, stride=2)
            self.layer3 = self._make_stage(128, 256, stride=2)
            self.layer4 = self._make_stage(256, 512, stride=2)

    @staticmethod
    def _basic_block(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        class BasicBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = downsample

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                out = self.relu(out)
                return out

        return BasicBlock()

    def _make_stage(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            self._basic_block(in_channels, out_channels, stride=stride),
            self._basic_block(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)   # [B, 64, H/4, W/4]
        x = self.layer1(x) # [B, 64, H/4, W/4]
        x = self.layer2(x) # [B, 128, H/8, W/8]
        x = self.layer3(x) # [B, 256, H/16, W/16]
        x = self.layer4(x) # [B, 512, H/32, W/32]
        return x


class SpatialFrequencyFusion(nn.Module):
    """
    Fuse ViT patch tokens and ResNet spatial noise features.

    Inputs:
        rgb_tokens: [B, N, D_rgb]
        noise_feat_map: [B, C_noise, Hn, Wn]
    Outputs:
        fused_tokens: [B, N, D_out]
        noise_tokens: [B, N, D_out]
    """

    def __init__(self, rgb_dim: int, noise_dim: int, out_dim: int) -> None:
        super().__init__()
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rgb_tokens: torch.Tensor, noise_feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_patches, _ = rgb_tokens.shape

        rgb_tokens = self.rgb_proj(rgb_tokens)  # [B, N, D_out]

        noise_tokens = noise_feat_map.flatten(2).transpose(1, 2)  # [B, Hn*Wn, C_noise]
        noise_tokens = F.adaptive_avg_pool1d(noise_tokens.transpose(1, 2), num_patches).transpose(1, 2)  # [B, N, C_noise]
        noise_tokens = self.noise_proj(noise_tokens)  # [B, N, D_out]

        fused = torch.cat([rgb_tokens, noise_tokens], dim=-1)  # [B, N, 2*D_out]
        fused_tokens = self.fuse_proj(fused)  # [B, N, D_out]
        return fused_tokens, noise_tokens


class SpatialFrequencyDualStreamEncoder(nn.Module):
    """
    Dual-stream encoder for RGB semantics and frequency artifacts.

    Inputs:
        image: [B, 3, H, W]
        rgb_tokens: [B, N, D_rgb]
    Outputs:
        dict with:
            noise_map: [B, 27, H, W]
            noise_feat_map: [B, 512, H/32, W/32]
            fused_tokens: [B, N, D_out]
            noise_tokens: [B, N, D_out]
    """

    def __init__(self, rgb_dim: int, out_dim: int, noise_in_channels: int = 27, pretrained_resnet: bool = False) -> None:
        super().__init__()
        self.srm = SRMConv2d()
        self.noise_encoder = ResNet18SpatialEncoder(in_channels=noise_in_channels, pretrained=pretrained_resnet)
        self.fusion = SpatialFrequencyFusion(rgb_dim=rgb_dim, noise_dim=512, out_dim=out_dim)

    def forward(self, image: torch.Tensor, rgb_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        noise_map = self.srm(image)  # [B, 27, H, W]
        noise_feat_map = self.noise_encoder(noise_map)  # [B, 512, H/32, W/32]
        fused_tokens, noise_tokens = self.fusion(rgb_tokens, noise_feat_map)  # [B, N, D_out], [B, N, D_out]
        return {
            "noise_map": noise_map,
            "noise_feat_map": noise_feat_map,
            "fused_tokens": fused_tokens,
            "noise_tokens": noise_tokens,
        }


class CSCLSpatialFrequencyAdapter(nn.Module):
    """
    Adapter that refines METER visual tokens with spatial-frequency cues.

    Inputs:
        image: [B, 3, H, W]
        meter_patch_tokens: [B, N, D]
        meter_global_token: [B, 1, D]
    Outputs:
        dict with:
            refined_patch_tokens: [B, N, D]
            refined_global_token: [B, 1, D]
            fused_patch_tokens: [B, N, D]
            noise_map: [B, 27, H, W]
            noise_feat_map: [B, 512, H/32, W/32]
    """

    def __init__(self, dim: int, fusion_mode: str = "residual", pretrained_resnet: bool = False) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.dual_stream = SpatialFrequencyDualStreamEncoder(rgb_dim=dim, out_dim=dim, pretrained_resnet=pretrained_resnet)
        self.global_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        meter_patch_tokens: torch.Tensor,
        meter_global_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dual_outputs = self.dual_stream(image, meter_patch_tokens)
        fused_patch_tokens = dual_outputs["fused_tokens"]  # [B, N, D]

        if self.fusion_mode == "replace_patch_keep_cls":
            refined_patch_tokens = fused_patch_tokens  # [B, N, D]
        else:
            refined_patch_tokens = meter_patch_tokens + fused_patch_tokens  # [B, N, D]

        patch_summary = refined_patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]
        refined_global_token = self.global_proj(torch.cat([meter_global_token, patch_summary], dim=-1))  # [B, 1, D]

        return {
            "refined_patch_tokens": refined_patch_tokens,
            "refined_global_token": refined_global_token,
            "fused_patch_tokens": fused_patch_tokens,
            "noise_map": dual_outputs["noise_map"],
            "noise_feat_map": dual_outputs["noise_feat_map"],
            "noise_tokens": dual_outputs["noise_tokens"],
        }


class TokenToPatchMaxSimAligner(nn.Module):
    """
    Fine-grained token-to-patch alignment using cosine similarity + MaxSim.

    Inputs:
        visual_tokens: [B, N, D]
        text_tokens: [B, M, D]
        text_mask: [B, M] or None, 1 for valid token / 0 for pad
    Outputs:
        dict with:
            token_patch_sim: [B, M, N]
            token_maxsim: [B, M]
            token_best_patch_idx: [B, M]
            token_logits: [B, M, 2]
            grounding_feature: [B, D]
            contradiction_score: [B, 1]
            weighted_patch_feature: [B, N, D]
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.token_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_tokens = F.normalize(visual_tokens, p=2, dim=-1, eps=self.eps)  # [B, N, D]
        text_tokens = F.normalize(text_tokens, p=2, dim=-1, eps=self.eps)  # [B, M, D]

        token_patch_sim = torch.matmul(text_tokens, visual_tokens.transpose(1, 2))  # [B, M, N]
        token_maxsim, token_best_patch_idx = token_patch_sim.max(dim=-1)  # [B, M], [B, M]

        gathered_patch = torch.gather(
            visual_tokens,
            dim=1,
            index=token_best_patch_idx.unsqueeze(-1).expand(-1, -1, visual_tokens.size(-1)),
        )  # [B, M, D]

        token_pair_feature = torch.cat([text_tokens, gathered_patch], dim=-1)  # [B, M, 2D]
        token_pair_feature = self.out_proj(token_pair_feature)  # [B, M, D]
        token_gate = self.token_gate(token_pair_feature).squeeze(-1)  # [B, M]

        if text_mask is not None:
            text_mask = text_mask.float()  # [B, M]
            token_maxsim = token_maxsim.masked_fill(text_mask == 0, 0.0)  # [B, M]
            token_gate = token_gate.masked_fill(text_mask == 0, -1e4)  # [B, M]
            token_patch_sim = token_patch_sim.masked_fill(text_mask.unsqueeze(-1) == 0, -1.0)  # [B, M, N]

        token_weight = torch.softmax(token_gate, dim=-1)  # [B, M]
        if text_mask is not None:
            token_weight = token_weight * text_mask  # [B, M]
            token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # [B, M]

        grounding_feature = torch.sum(token_pair_feature * token_weight.unsqueeze(-1), dim=1)  # [B, D]
        contradiction_score = 1.0 - torch.sum(token_maxsim * token_weight, dim=-1, keepdim=True)  # [B, 1]

        patch_attention = torch.softmax(token_patch_sim.transpose(1, 2), dim=-1)  # [B, N, M]
        weighted_patch_feature = torch.matmul(patch_attention, text_tokens)  # [B, N, D]

        contradiction_feature = grounding_feature * contradiction_score + grounding_feature  # [B, D]
        contradiction_score = self.score_head(contradiction_feature).sigmoid()  # [B, 1]

        token_prob_fake = token_maxsim.clamp(0.0, 1.0)  # [B, M]
        token_prob_real = 1.0 - token_prob_fake  # [B, M]
        token_logits = torch.stack([token_prob_real, token_prob_fake], dim=-1)  # [B, M, 2]

        return {
            "token_patch_sim": token_patch_sim,
            "token_maxsim": token_maxsim,
            "token_best_patch_idx": token_best_patch_idx,
            "token_logits": token_logits,
            "grounding_feature": grounding_feature,
            "contradiction_score": contradiction_score,
            "weighted_patch_feature": weighted_patch_feature,
        }


class GroundingHead(nn.Module):
    """
    Example grounding head for bbox regression.

    Inputs:
        fused_visual_tokens: [B, N, D]
        grounding_feature: [B, D]
    Output:
        pred_box: [B, 4]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, fused_visual_tokens: torch.Tensor, grounding_feature: torch.Tensor) -> torch.Tensor:
        visual_summary = fused_visual_tokens.mean(dim=1)  # [B, D]
        fused = torch.cat([visual_summary, grounding_feature], dim=-1)  # [B, 2D]
        pred_box = self.proj(fused).sigmoid()  # [B, 4]
        return pred_box


class CSCLRefactorDemo(nn.Module):
    """
    Minimal integration demo.

    Inputs:
        image: [B, 3, H, W]
        rgb_tokens: [B, N, D]
        text_tokens: [B, M, D]
        text_mask: [B, M]
    Outputs:
        dictionary of intermediate and prediction tensors.
    """

    def __init__(self, rgb_dim: int = 768, model_dim: int = 768) -> None:
        super().__init__()
        self.dual_stream = SpatialFrequencyDualStreamEncoder(rgb_dim=rgb_dim, out_dim=model_dim)
        self.aligner = TokenToPatchMaxSimAligner(dim=model_dim)
        self.grounding_head = GroundingHead(dim=model_dim)
        self.cls_head = nn.Linear(model_dim, 2)

    def forward(
        self,
        image: torch.Tensor,
        rgb_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        dual_outputs = self.dual_stream(image, rgb_tokens)
        fused_visual_tokens = dual_outputs["fused_tokens"]  # [B, N, D]

        align_outputs = self.aligner(fused_visual_tokens, text_tokens, text_mask)
        grounding_feature = align_outputs["grounding_feature"]  # [B, D]

        logits = self.cls_head(grounding_feature)  # [B, 2]
        pred_box = self.grounding_head(fused_visual_tokens, grounding_feature)  # [B, 4]

        return {
            **dual_outputs,
            **align_outputs,
            "logits": logits,
            "pred_box": pred_box,
        }


def dummy_test() -> None:
    torch.manual_seed(42)

    batch_size = 2
    image_size = 224
    num_patches = 196
    num_tokens = 32
    dim = 768

    image = torch.randn(batch_size, 3, image_size, image_size)  # [B, 3, 224, 224]
    rgb_tokens = torch.randn(batch_size, num_patches, dim)  # [B, 196, 768]
    text_tokens = torch.randn(batch_size, num_tokens, dim)  # [B, 32, 768]
    text_mask = torch.ones(batch_size, num_tokens)  # [B, 32]
    text_mask[1, -4:] = 0

    model = CSCLRefactorDemo(rgb_dim=dim, model_dim=dim)
    outputs = model(image, rgb_tokens, text_tokens, text_mask)

    print("noise_map:", outputs["noise_map"].shape)
    print("noise_feat_map:", outputs["noise_feat_map"].shape)
    print("fused_tokens:", outputs["fused_tokens"].shape)
    print("token_patch_sim:", outputs["token_patch_sim"].shape)
    print("token_maxsim:", outputs["token_maxsim"].shape)
    print("weighted_patch_feature:", outputs["weighted_patch_feature"].shape)
    print("grounding_feature:", outputs["grounding_feature"].shape)
    print("contradiction_score:", outputs["contradiction_score"].shape)
    print("logits:", outputs["logits"].shape)
    print("pred_box:", outputs["pred_box"].shape)


if __name__ == "__main__":
    dummy_test()
