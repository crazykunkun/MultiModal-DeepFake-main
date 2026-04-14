import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashMultiheadAttention(nn.Module):
    """使用 F.scaled_dot_product_attention 的高效多头注意力（支持 Flash Attention kernel）"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        B, N, C = query.shape

        # 投影 Q/K/V
        q = F.linear(query, self.in_proj.weight[:C], self.in_proj.bias[:C])
        k = F.linear(key, self.in_proj.weight[C:2*C], self.in_proj.bias[C:2*C])
        v = F.linear(value, self.in_proj.weight[2*C:], self.in_proj.bias[2*C:])

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 构造 SDPA 兼容的 attn_mask
        sdpa_mask = None
        if attn_mask is not None:
            # attn_mask 可能是 [num_heads*B, N, N] 的 bool mask（True=屏蔽）
            if attn_mask.dim() == 3 and attn_mask.shape[0] == self.num_heads * B:
                # 转成 [B, num_heads, N, N]
                sdpa_mask = attn_mask.view(B, self.num_heads, attn_mask.shape[1], attn_mask.shape[2])
            elif attn_mask.dim() == 3 and attn_mask.shape[0] == B:
                sdpa_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                sdpa_mask = attn_mask
            # SDPA 需要 float mask（-inf 屏蔽）或 bool mask（True=参与）
            if sdpa_mask.dtype == torch.bool:
                # 原始代码中 True=屏蔽，SDPA 中 True=屏蔽（is_causal=False 时）
                # 转成 float: 屏蔽位置 -inf
                sdpa_mask = sdpa_mask.float().masked_fill(sdpa_mask, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: [B, S], True=pad
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            pad_mask = pad_mask.float().masked_fill(pad_mask, float('-inf'))
            if sdpa_mask is not None:
                sdpa_mask = sdpa_mask + pad_mask
            else:
                sdpa_mask = pad_mask

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        return out, None  # 返回 (output, attn_weights=None) 保持接口兼容


class Self_Interaction_block(nn.Module):
    def __init__(self, num_head, hidden_dim, input_dim, output_dim):
        super().__init__()

        self.self_attn = FlashMultiheadAttention(input_dim, num_head, dropout=0.0, batch_first=True)
        self.FFN = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, query, query_padding_mask, attn_mask):

        feat_after_self = query + self.dropout1(self.self_attn(query=query,
                                              key=query,
                                              value=query,
                                              key_padding_mask=query_padding_mask,
                                              attn_mask=attn_mask)[0])
        feat_after_self = self.norm1(feat_after_self)
        output = feat_after_self + self.dropout2(self.FFN(feat_after_self))
        output = self.norm2(output)
        return output

class Self_Interaction(nn.Module):
    def __init__(self, num_head, hidden_dim, input_dim, output_dim, layers=3):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(Self_Interaction_block(num_head, hidden_dim, input_dim, output_dim))

    def forward(self, query, query_padding_mask, query_pos_emb=None, attn_mask=None):
        if query_pos_emb is not None:
            for i in range(len(self.layers)):
                query = self.layers[i](query + query_pos_emb, query_padding_mask, attn_mask)
        else:
            for i in range(len(self.layers)):
                query = self.layers[i](query, query_padding_mask, attn_mask)
        return query