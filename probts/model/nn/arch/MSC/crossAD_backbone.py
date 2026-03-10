import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
# 假设 ADWT_1D 在同级目录下或已正确导入
from .ADWT_1D import ADWT_1D 
from timm.models.vision_transformer import  Mlp
from .Attention_Blocks import *
from .EncDec import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from probts.model.nn.arch.RevIN import RevIN



class patch_embedding(nn.Module):
    def __init__(self,patch_size,hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.value_embedding = nn.Linear(self.patch_size,hidden_dim, bias=False)
        self.position_embedding = SinusoidalPositionalEmbedding(hidden_dim)


    def forward(self,x,Need_pad_len,stride):
        x = F.pad(input=x, pad=(0, Need_pad_len), mode='replicate')

        x = x.unfold(dimension=-1, size=self.patch_size, step=stride)
  
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.value_embedding(x) + self.position_embedding(x) 
        return x

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.embedding_table = nn.Linear(d_model, hidden_size , bias=True)

    def forward(self, y):
        b,l_n,c = y.shape
        
        y = y.reshape(b*c,l_n)

        embeddings=self.embedding_table(y)

        return embeddings
    
class FinalLayer(nn.Module):
    """
    The final layer of PatchDN.
    """
    def __init__(self, hidden_size, patch_num, context_window):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_size*patch_num, context_window , bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)

        x = self.flatten(x)

        x = self.linear(x)


        return x

    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 确保 pe 的长度与输入匹配
        return self.pe[:, :x.size(1), :]

    
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)


        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # 返回 [B, D]
        return embeddings

def timestep_embedding(t, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)   # size: [dim/2], 一个指数衰减的曲线
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(t, "b -> b d", d=dim)
    return embedding

class TimestepEmbedder(nn.Module):

    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=128, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):

        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
    


#         return x
def modulate(x, shift, scale):

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNTransformerEncoderLayer(nn.Module):
    """
    A PatchDN block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size,num_layers, nheads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        attn_drop = block_kwargs.get('attn_dropout', 0.0)
        proj_drop = block_kwargs.get('proj_dropout', 0.0)
        self.attn = EncoderLayer(
                    attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=attn_drop), d_model=hidden_size, n_heads=nheads, proj_dropout=proj_drop),
                    d_model=hidden_size,
                    d_ff=hidden_size*4,
                    norm='layernorm',
                    dropout=0.1,
                    activation='gelu'
                )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c,attn_mask = None):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

# 检查这些变量里是否有人的 shape 是空的，或者本该是标量却变成了向量

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),attn_mask)[0]

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
# =======================================================
# 3. 核心：DTWSR 风格的扩散 Transformer (引入 AdaLN)
# =======================================================
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        # 使用 1x1 卷积学习门控权重
        # 输入通道为 2*channels (因为拼接了两个输入)
        self.gate_conv = nn.Sequential(
            nn.Linear(2 * channels, channels),
            # nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 1. 在通道维度拼接 [B, 2C, L]
        # x1 b c l 
        combined = torch.cat([x1, x2], dim=-2).permute(0,2,1) # b l c


        # 2. 生成门控权重 g: [B, C, L]
        gate = self.gate_conv(combined).permute(0,2,1) # b c l
        
        # 3. 融合特征
        out = gate * x1 + (1 - gate) * x2

        return out

class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_length: int,
        prediction_length: int,
        wavelet_level: int,
        base_patch_size: int,
        ori_patch_size: int,
        nheads: int,
        mlp_ratio: int,
        attn_drop: int,
        proj_dopr: int,
        
    ):
        super().__init__()
        
        self.c_in = input_dim
        self.d_model = hidden_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length

        # 小波变换与切片参数
        self.level = wavelet_level
        self.stride = base_patch_size // 2
        self.base_patch_size = base_patch_size

        # -------------------------------------------------------------------------
        # 2. 小波变换与 Patch 尺寸计算
        # -------------------------------------------------------------------------
        self.dwt = ADWT_1D(level=wavelet_level, learnable=False)
        self.dwt1 = ADWT_1D(level=wavelet_level, learnable=False) # 副本或用于不同阶段

        # 计算金字塔 Patch 尺寸 (LF, HF_J...HF_1)
        self.patch_sizes = [
            base_patch_size * (2 ** (wavelet_level - j - 1)) 
            for j in range(wavelet_level)
        ]
        print(f"Pyramid Patch Sizes: {self.patch_sizes}")

        # -------------------------------------------------------------------------
        # 3. 标记器 (Tokenizers / Embedding Layers)
        # -------------------------------------------------------------------------
        # 多尺度切片嵌入
        self.tokenizers = nn.ModuleList([
            patch_embedding(p, hidden_dim) for p in self.patch_sizes
        ])

        # 特殊用途的嵌入层
        self.lr_tokenizer  = patch_embedding(self.base_patch_size, hidden_dim)
        self.lf_tokenizer  = patch_embedding(self.base_patch_size, hidden_dim)
        

        # -------------------------------------------------------------------------
        # 5. 解记号器 (Detokenizers / Projection Layers)
        # -------------------------------------------------------------------------
        # 获取小波分解后的维度信息
        _, pse_hh = self.dwt._dummy_forward(self.total_length, is_dec=1)
        self._max_patch_num = self.get_max_patch_num(pse_hh=pse_hh)
        self._min_decom_length = pse_hh[-1].shape[-1]

        # 高频重构层 (针对每一层小波分解)
        self.high_freq_detokenizers = nn.ModuleList([
            FinalLayer(hidden_dim, self._max_patch_num, pse_hh[i].shape[-1]) 
            for i in range(self.level)
        ])

        # 特征与时间嵌入
        self.feat_emb = LabelEmbedder(self.total_length * 3, hidden_size=hidden_dim)
        self.time_embedding = TimestepEmbedder(hidden_dim)
        
        # self.latent_emb = LabelEmbedder(self._max_patch_num * hidden_dim,hidden_size=hidden_dim)
        # -------------------------------------------------------------------------
        # 4. Transformer 层 (Encoder/Decoder Layers)
        # -------------------------------------------------------------------------
        # 针对低频、高频和全局特征的独立 Transformer 块
        common_params = dict(
            hidden_size=hidden_dim, num_layers=1, nheads=nheads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,proj_dropout=proj_dopr,
        )

        self.LEDec_layers = nn.ModuleList([AdaLNTransformerEncoderLayer(**common_params) for _ in range(3)])
        self.HDDec_layers = nn.ModuleList([AdaLNTransformerEncoderLayer(**common_params) for _ in range(3)])
        self.Glob_layers  = nn.ModuleList([AdaLNTransformerEncoderLayer(**common_params) for _ in range(3)])

        self.ori_patch_size = ori_patch_size
        self.ori_stride = self.ori_patch_size // 2
        self.ori_patch_num = ((self.context_length + self.prediction_length) - self.ori_patch_size) //  self.ori_stride + 1

        self.ori_tokenizer = patch_embedding(self.ori_patch_size , hidden_dim)

        # 低频与全局重构层
        self.low_freq_detokenizers = FinalLayer(hidden_dim, self._max_patch_num, self._min_decom_length)
        self.global_detokenizers   = FinalLayer(hidden_dim, self.ori_patch_num, self.total_length)
        
        self.gate = GatedFusion(self.c_in)
        # self.revin = RevIN(self.c_in)
        self.level_emb = TimestepEmbedder(hidden_size=hidden_dim)

        self.latent_emb = nn.Sequential(
                        
                        nn.Linear(self.ori_patch_num, hidden_dim),
                        nn.ReLU(inplace=True),      
                        nn.Linear(hidden_dim, 1),
                        nn.Dropout(p=0.0)           
                    )
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.LEDec_layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.HDDec_layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.Glob_layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)



        nn.init.normal_(self.ori_tokenizer.value_embedding.weight, std=0.02)

        nn.init.normal_(self.lr_tokenizer.value_embedding.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.lf_tokenizer.value_embedding.weight, std=0.02)

        for tokenizer in self.tokenizers:
            nn.init.normal_(tokenizer.value_embedding.weight, std=0.02)


        nn.init.normal_(self.feat_emb.embedding_table.weight, std=0.02)

        # nn.init.normal_(self.latent_emb.embedding_table.weight, std=0.02)

        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)

        nn.init.normal_(self.level_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.level_emb.mlp[2].weight, std=0.02)


        nn.init.constant_(self.low_freq_detokenizers.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.low_freq_detokenizers.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.low_freq_detokenizers.linear.weight, 0)
        nn.init.constant_(self.low_freq_detokenizers.linear.bias, 0)

        nn.init.constant_(self.global_detokenizers.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.global_detokenizers.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.global_detokenizers.linear.weight, 0)
        nn.init.constant_(self.global_detokenizers.linear.bias, 0)

        for block in self.high_freq_detokenizers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.linear.weight, 0)
            nn.init.constant_(block.linear.bias, 0)
        
    def add_level_emb(self, tok, level_id):
        B, L, _ = tok.shape
        level_idx = torch.full(
            (B, 1), level_id,
            device=tok.device,
            dtype=torch.long
        )

        return tok + self.level_emb(level_idx)

    def get_max_patch_num(self, pse_hh):
        _max_patch_num = 0 
        for i, band in enumerate(pse_hh):
            # 确保最后一维为偶数
            if band.shape[-1] % 2 != 0:
                band = F.pad(input=band, pad=(0, 1), mode='replicate')
            current_last_dim = band.shape[-1]
            # 计算当前band的patch数量
            stride = (self.tokenizers[i].patch_size // 2)
            patch_num = (current_last_dim - self.tokenizers[i].patch_size) // stride  + 1

            # 更新最大patch数
            if patch_num > _max_patch_num:
                _max_patch_num = patch_num
        return _max_patch_num


    # def _make_mask(self, current_tokens, token_slices, mode='low'):
    #     # ... (Mask logic remains the same) ...
    #     device = current_tokens.device
    #     total_len = current_tokens.size(1)
    #     mask = torch.full((total_len, total_len), float('-inf'), device=device)
        

    #     sl_lr = token_slices['lr']
    #     sl_lf = token_slices['lf']
        

    #     if mode == 'low':
    #         # LR Block: 只能看 LR (防止由 Noisy LF 泄露信息给 Condition)
    #         mask[sl_lr, sl_lr] = 0.0
            
    #         # LF Block: 可以看 LR + LF
    #         mask[sl_lf, sl_lr] = 0.0 # LF sees LR
    #         mask[sl_lf, sl_lf] = 0.0 # LF sees LF
            

    #     elif mode == 'high':
    #         sl_hf_list = token_slices['hf_list'] # HF 切片列表
            
    #         # LR Block: 只能看 LR（不变）
    #         mask[sl_lr, sl_lr] = 0.0
            
    #         # 改动核心：LF 不能关注 LR，仅能关注自身 + 所有 HF
    #         # mask[sl_lf, sl_lf] = 0.0  # LF 自关注
    #         # 合并所有 HF 的区间
    #         hf_start = sl_hf_list[0].start
    #         hf_end = sl_hf_list[-1].stop
    #         sl_all_hf = slice(hf_start, hf_end)
    #         mask[sl_lf, sl_all_hf] = 0.0  # LF 关注所有 HF

    #         # 移除原逻辑：mask[sl_lf, sl_lr] = 0.0（禁止 LF 关注 LR）
            
    #         # HF Blocks: 规则不变（关注 LR + 所有 HF，屏蔽 LF）
    #         hf_start = sl_hf_list[0].start
    #         hf_end = sl_hf_list[-1].stop
    #         sl_all_hf = slice(hf_start, hf_end)
    #         mask[sl_all_hf, sl_lr] = 0.0
    #         mask[sl_all_hf, sl_all_hf] = 0.0
    #         mask[sl_all_hf, sl_lf] = 0.0

    #     elif mode == 'causal':
    #         l = total_len
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #         mask = torch.full((l, l), float('-inf'), device=device)
    #         mask  = torch.triu(mask, diagonal=1)
            
            
    #         mask = mask.unsqueeze(0).unsqueeze(0)

            
    #     return mask


    def _make_mask(self, current_tokens, token_slices, mode='low'):
        """
        返回 bool mask:
        True  -> masked (不可见)
        False -> visible
        """
        device = current_tokens.device
        total_len = current_tokens.size(1)

        # 一开始：全部禁止（True）
        mask = torch.ones((total_len, total_len),
                        dtype=torch.bool,
                        device=device)

        sl_lr = token_slices['lr']
        sl_lf = token_slices['lf']

        if mode == 'low':
            # LR Block: 只能看 LR
            mask[sl_lr, sl_lr] = False

            # LF Block: 可以看 LR + LF
            mask[sl_lf, sl_lr] = False
            mask[sl_lf, sl_lf] = False

        elif mode == 'high':
            sl_hf_list = token_slices['hf_list']

            # LR Block: 只能看 LR
            mask[sl_lr, sl_lr] = False

            # 所有 HF 的连续区间
            hf_start = sl_hf_list[0].start
            hf_end   = sl_hf_list[-1].stop
            sl_all_hf = slice(hf_start, hf_end)

            # LF: 只能看 HF（不能看 LR，也不能看自己）
            mask[sl_lf, sl_all_hf] = False
            
            # HF: 可以看 LR + HF，自身 OK，屏蔽 LF
            mask[sl_all_hf, sl_lr] = False
            mask[sl_all_hf, sl_all_hf] = False
            # mask[sl_all_hf, sl_lf] 保持 True（屏蔽）

        elif mode == 'causal':
            # causal：只允许看过去（含自己）
            causal = torch.triu(
                torch.ones((total_len, total_len),
                        dtype=torch.bool,
                        device=device),
                diagonal=1
            )
            mask = causal

        return mask



    def forward(self, x,t,features):
        

        # 1. 基础维度获取与时间嵌入处理
        B, L, K = x.shape
        # x = self.revin(x, 'norm')
        # 提取特征中的参考信号 (pseudo_x0) 与原始信号
        pure_x0 = features[:, :, :, -1]
        ori_x0 = pure_x0
  
        ori_x0 = (ori_x0).permute(0,2,1)

        features = features[:, :, :, :]

        # 时间步 t 的维度对齐与 Embedding
        if len(t.shape) == 0:
            t = repeat(t, " -> b 1", b=x.shape[0])
        elif len(t.shape) == 1:
            t = t.unsqueeze(-1)
        else:
            t = t[..., 0]
        
        t = repeat(t, 'b 1 -> (b k) 1', b=B, k=K)
        t_emb = self.time_embedding(t).squeeze(1)  # [B*K, D]

        # -------------------------------------------------------
        # 2. 小波分解与分箱 (DWT Decomposition)
        # -------------------------------------------------------
        # 对参考信号和原始输入进行小波分解
        x_lr, _ = self.dwt1(pure_x0.permute(0, 2, 1), is_dec=True) 
        n_ll, n_yh = self.dwt(x.permute(0, 2, 1), is_dec=True)

        # 计算 Padding 长度
        _max_patch_size = (self.base_patch_size * (2 ** (self.level)))
        
        Need_pad_len_lr = (self._max_patch_num - 1) * self.stride + self.base_patch_size - x_lr.shape[-1]
        Need_pad_len_lf = (self._max_patch_num - 1) * self.stride + self.base_patch_size - n_ll.shape[-1]
       

        # -------------------------------------------------------
        # 3. Tokenization (序列化)
        # -------------------------------------------------------
        # 原始信号、低频分支(LR, LF)的 Tokenize

        # tok_ori = self.ori_tokenizer(ori_x0, Need_pad_len_ori, stride=_max_patch_size // 2)
        

        tok_ori = self.ori_tokenizer(ori_x0, 0, stride=self.ori_stride) 
        tok_ori = tok_ori + self.add_level_emb(tok_ori,0)

        tok_lr = self.lr_tokenizer(x_lr, Need_pad_len_lr, stride=self.stride)
        tok_lf = self.lf_tokenizer(n_ll, Need_pad_len_lf, stride=self.stride)

        # 外部特征 Tokenize
        b, l, c, n = features.shape
  
        features = rearrange(features, 'b l c n -> b (l n) c')
        # features = self.revin(features,'norm')
        tok_feat = self.feat_emb(features)

        # 高频分量 (Bands) Tokenize
        band_tokens = []
        for i, band in enumerate(n_yh):
            stride = self.tokenizers[i].patch_size // 2
            need_pad = (self._max_patch_num - 1) * stride + self.tokenizers[i].patch_size - band.shape[-1]
            tok = self.tokenizers[i](band, need_pad, stride=stride) 
            tok = tok + self.add_level_emb(tok,3 + i) 
            band_tokens.append(tok)

        # -------------------------------------------------------
        # 4. 掩码准备与条件构建
        # -------------------------------------------------------
        # 拼接低频 Tokens 并记录切片信息

        # tok_lr = tok_lr + self.add_level_emb(tok_lr,1) + t_emb.unsqueeze(1)
        # tok_lf = tok_lf + self.add_level_emb(tok_lf,2) + t_emb.unsqueeze(1)

        tok_lr = tok_lr + self.add_level_emb(tok_lr,1) 
        tok_lf = tok_lf + self.add_level_emb(tok_lf,2) 


        tok_l = torch.cat((tok_lr, tok_lf), dim=1)
        len_lr, len_lf = tok_lr.shape[1], tok_lf.shape[1]
        
        slice_info = {
            'lr': slice(0, len_lr),
            'lf': slice(len_lr, len_lr + len_lf)
        }

        # 构建因果掩码和低频掩码
        mask_causal = self._make_mask(tok_ori, slice_info, mode='causal')
        mask_low = self._make_mask(tok_l, slice_info, mode='low')

        # 融合时间与特征作为 Conditioning (c_lr)
  
        c_global = t_emb + tok_feat
        

        # -------------------------------------------------------
        # 5. 多级解码 (Hierarchical Decoding)
        # -------------------------------------------------------
        # Phase 1: Global Decoding (处理原始信号 Token)
        global_x = tok_ori
        for layer in self.Glob_layers:
            global_x = layer(global_x, c_global, attn_mask=mask_causal)


        # latent_emb = self.latent_emb(rearrange(global_x,'(b c) L N -> b (L N) c',b=b,c=c))

        latent_emb = self.latent_emb(global_x.permute(0,2,1)).squeeze(-1)
 
        c_lr = c_global + latent_emb
        c_hd = c_lr


        pre_x = self._decode_branch(global_x, c_lr, branch_type="global")

        # Phase 2: Low-Energy Decoding (处理低频分支)
        out_le = tok_l
        for layer in self.LEDec_layers:
            out_le = layer(out_le, c_lr, attn_mask=mask_low)

        # Phase 3: High-Definition Decoding (合并低频与高频)
        band_tokens_cat = torch.cat(band_tokens, dim=1)  + t_emb.unsqueeze(1)

        input_hd = torch.cat((out_le, band_tokens_cat), dim=1)

        # 更新高频部分的切片信息用于构建 mask
        start_idx = len_lr + len_lf
        hf_slices = []
        for hf_tok in band_tokens:
            end_idx = start_idx + hf_tok.shape[1]
            hf_slices.append(slice(start_idx, end_idx))
            start_idx = end_idx
        slice_info['hf_list'] = hf_slices

        mask_high = self._make_mask(input_hd, slice_info, mode='high')

        out_hd = input_hd
        for layer in self.HDDec_layers:
            out_hd = layer(out_hd, c_hd, attn_mask=mask_high)

        # -------------------------------------------------------
        # 6. 信号重构 (Signal Reconstruction)
        # -------------------------------------------------------
        # 提取低频预测
        lf_pre = out_le[:, slice_info['lf'], :]
        lr_pre = out_hd[:, slice_info['lf'], :]
        ll_pre = lf_pre + lr_pre  # 融合分支预测

        # 提取并解码高频预测
        hf_cut = self._max_patch_num * 2
        hf_pre_tokens = out_hd[:, hf_cut:, :]
        
        rec_lf = self._decode_branch(ll_pre, c_lr, branch_type="low_freq",idx=1)

        # 解码每一个高频 Band
        offset = hf_cut
        rec_hfs = []
        for i, sl in enumerate(hf_slices):
            # 计算相对于 hf_pre_tokens 的新切片索引
            rel_sl = slice(sl.start - offset, sl.stop - offset)
            hf_feat = hf_pre_tokens[:, rel_sl, :]
            
            cut_length = n_yh[i].shape[-1]
            hf_rec = self._decode_branch(hf_feat, c_hd, branch_type="high_freq",idx=i)
            rec_hfs.append(hf_rec[:, :, :cut_length])

        # 逆小波变换重构
        rec_signal = self.dwt([rec_lf, rec_hfs], is_dec=False)
        
        # rec_signal = self.gate(rec_signal,pre_x)

        # rec_signal =  rec_signal +  pre_x
        # rec_signal =  rec_signal +  pre_x
        out = self.gate(rec_signal,pre_x)

        return (out.permute(0,2,1))

    def _decode_branch(self, tokens, c_lr, branch_type="global", idx=None):
            """
            通用解码函数
            branch_type: "global", "low_freq", 或 "high_freq"
            """
            # 1. 根据类型动态获取对应的 detokenizer
            if branch_type == "high_freq":
                # 高频分支需要索引 idx
                detokenizer = self.high_freq_detokenizers[idx]
            else:
                # 低频和全局分支
                attr_name = f"{branch_type}_detokenizers"
                detokenizer = getattr(self, attr_name)

            # 2. 统一执行反序列化逻辑
            x = detokenizer(tokens, c_lr)

            # 3. 统一执行维度重排
            x = rearrange(x, '(b k) l -> b k l', k=self.c_in)
            
            return x