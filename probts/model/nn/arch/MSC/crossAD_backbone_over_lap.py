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
        b,l,c = y.shape
        y = y.reshape(b*c,l)
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
    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
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
        self.attn = EncoderLayer(
                    attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=0.0), d_model=hidden_size, n_heads=nheads, proj_dropout=0.0),
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

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))[0]

        return x
# =======================================================
# 3. 核心：DTWSR 风格的扩散 Transformer (引入 AdaLN)
# =======================================================



class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_length: int,
        prediction_length: int,
        output_dim: int,
        step_emb: int,
        num_residual_blocks: int,
        num_features: int,
        target_dim: int = 1,
        residual_block: str = "s4",
        dropout: float = 0.0,
        bidirectional: bool = True,
        init_skip: bool = True,
        feature_skip: bool = True,
    ):
        super().__init__()
        
        self.c_in = input_dim
        self.d_model = hidden_dim
        wavelet_level = 3
        base_patch_size = 3
        self.level = wavelet_level
        self.base_patch_size = base_patch_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        # ... (Patch Sizes Calculation and Tokenizers remain the same) ...
        self.patch_sizes = []
        for j in range(0, wavelet_level):
            p_size = base_patch_size ** (wavelet_level - j)
            self.patch_sizes.append(p_size)
        
        print(f"Pyramid Patch Sizes (LF, HF_J...HF_1): {self.patch_sizes}")

        # 3. Tokenizers (Embedding Layers)
        self.tokenizers = nn.ModuleList([
            patch_embedding(p,hidden_dim)
            for p in self.patch_sizes
        ])
        
        self.base_patch_size = base_patch_size
        
        self.lr_tokenizer = patch_embedding(self.base_patch_size,hidden_dim)
        
        self.lf_tokenizer = patch_embedding(self.base_patch_size,hidden_dim)
        
        
        self.feat_emb = LabelEmbedder((self.context_length + self.prediction_length)*2,hidden_size=hidden_dim)


        self.dwt1 = ADWT_1D(level=wavelet_level, learnable=False)
        self.dwt = ADWT_1D(level=wavelet_level, learnable=False)

        # 4. Positional Embeddings

        # self.time_embedding = TimestepEmbedding(hidden_dim)
        self.time_embedding = TimestepEmbedder(hidden_dim)
        

        self.LEDec_layers = nn.ModuleList([
            AdaLNTransformerEncoderLayer(hidden_dim, num_layers=1,nheads=8, mlp_ratio=4.0, attn_drop=dropout) for _ in range(3)])

        self.HDDec_layers = nn.ModuleList([
            AdaLNTransformerEncoderLayer(hidden_dim, num_layers=1,nheads=8, mlp_ratio=4.0, attn_drop=dropout) for _ in range(3)])



        
        _,pse_hh = self.dwt._dummy_forward(self.context_length + self.prediction_length,is_dec=1)

        self._max_patch_num = self.get_max_patch_num(pse_hh=pse_hh)
        self._min_decom_length = pse_hh[-1].shape[-1]

        

        self.high_freq_detokenizers =  nn.ModuleList([
            FinalLayer(hidden_dim, self._max_patch_num, pse_hh[i].shape[-1]) for i in range(self.level)
        ])

        self.low_freq_detokenizers = FinalLayer(hidden_dim, self._max_patch_num, self._min_decom_length)
        

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

        for block in self.high_freq_detokenizers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.linear.weight, 0)
            nn.init.constant_(block.linear.bias, 0)
        

        nn.init.normal_(self.lr_tokenizer.value_embedding.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.lf_tokenizer.value_embedding.weight, std=0.02)

        for tokenizer in self.tokenizers:
            nn.init.normal_(tokenizer.value_embedding.weight, std=0.02)


        nn.init.normal_(self.feat_emb.embedding_table.weight, std=0.02)

        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)

        nn.init.constant_(self.low_freq_detokenizers.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.low_freq_detokenizers.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.low_freq_detokenizers.linear.weight, 0)
        nn.init.constant_(self.low_freq_detokenizers.linear.bias, 0)


    def get_max_patch_num(self, pse_hh):
        _max_patch_num = 0 
        for i, band in enumerate(pse_hh):
            # 确保最后一维为偶数
            if band.shape[-1] % 2 != 0:
                band = F.pad(input=band, pad=(0, 1), mode='replicate')
            current_last_dim = band.shape[-1]
            # 计算当前band的patch数量
            patch_num = current_last_dim // self.tokenizers[i].patch_size
            # 更新最大patch数
            if patch_num > _max_patch_num:
                _max_patch_num = patch_num
        return _max_patch_num


    def _make_mask(self, current_tokens, token_slices, mode='low'):
        # ... (Mask logic remains the same) ...
        device = current_tokens.device
        total_len = current_tokens.size(1)
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        

        sl_lr = token_slices['lr']
        sl_lf = token_slices['lf']
        

        if mode == 'low':
            # LR Block: 只能看 LR (防止由 Noisy LF 泄露信息给 Condition)
            mask[sl_lr, sl_lr] = 0.0
            
            # LF Block: 可以看 LR + LF
            mask[sl_lf, sl_lr] = 0.0 # LF sees LR
            mask[sl_lf, sl_lf] = 0.0 # LF sees LF
            

        elif mode == 'high':
            sl_hf_list = token_slices['hf_list'] # HF 切片列表
            
            # LR Block: 只能看 LR（不变）
            mask[sl_lr, sl_lr] = 0.0
            
            # 改动核心：LF 不能关注 LR，仅能关注自身 + 所有 HF
            mask[sl_lf, sl_lf] = 0.0  # LF 自关注
            # 合并所有 HF 的区间
            hf_start = sl_hf_list[0].start
            hf_end = sl_hf_list[-1].stop
            sl_all_hf = slice(hf_start, hf_end)
            mask[sl_lf, sl_all_hf] = 0.0  # LF 关注所有 HF

            # 移除原逻辑：mask[sl_lf, sl_lr] = 0.0（禁止 LF 关注 LR）
            
            # HF Blocks: 规则不变（关注 LR + 所有 HF，屏蔽 LF）
            hf_start = sl_hf_list[0].start
            hf_end = sl_hf_list[-1].stop
            sl_all_hf = slice(hf_start, hf_end)
            mask[sl_all_hf, sl_lr] = 0.0
            mask[sl_all_hf, sl_all_hf] = 0.0

            
        return mask

    def forward(self, x,t,features):
        
        B, L, K = x.shape
        


        psudo_x0 = features[:,:,:,-1]
        features = features[:,:,:,:-1]

        if len(t.shape) == 0:
            t = repeat(t, " -> b 1", b=x.shape[0])
        elif len(t.shape) == 1:
            t = t.unsqueeze(-1)
        else:
            t = t[..., 0]

        t = repeat(t,'b 1->(b k) 1',b = B,k =K)

        # t_emb = self.time_embedding(t * 10000).squeeze(1)# [B, D]
        t_emb = self.time_embedding(t).squeeze(1)# [B, D]

        # **新增：AdaLN 投影**
        # 投影出两个 Decoder (LEDec, HDDec) 所需的 scale/shift 参数

        x_lr, _ = self.dwt1(psudo_x0.permute(0,2,1), is_dec=True)########3###有问题 是否需要使用带有未来值的xlr

        n_ll, n_yh = self.dwt(x.permute(0,2,1), is_dec=True)
    
        
        
        Need_pad_len_lr = self._max_patch_num * self.base_patch_size  - x_lr.shape[-1]
        Need_pad_len_lf = self._max_patch_num * self.base_patch_size - n_ll.shape[-1]
        

        tok_lr = self.lr_tokenizer(x_lr,Need_pad_len_lr,stride = self.base_patch_size)

        tok_lf = self.lf_tokenizer(n_ll,Need_pad_len_lf,stride = self.base_patch_size)

        b,l,c,n = features.shape
        features = rearrange(features, 'b l c n -> b (l n) c')

        tok_feat = self.feat_emb(features)
       

        # **改动：不再将 t_emb 加到 Token 上，仅加 Positional Emb**
        tok_lr_emb = (tok_lr )
        tok_lf_emb = (tok_lf )
        

        band_tokens = []
        for i, band in enumerate(n_yh):
            Need_pad_len = self._max_patch_num * self.tokenizers[i].patch_size - band.shape[-1]

            tok = self.tokenizers[i](band,Need_pad_len,self.tokenizers[i].patch_size)
            
            tok = tok  # HF Token 也加 Positional Emb
            band_tokens.append(tok)

        
        tok_l = torch.cat((tok_lr_emb,tok_lf_emb),dim=1)
        
        slice_info = {'lr': slice(0, tok_lr.shape[1])}
        len_lr = tok_lr.shape[1]
        len_lf = tok_lf.shape[1]
        slice_info['lf'] = slice(len_lr, len_lr + len_lf)
        
        mask_low = self._make_mask(tok_l, slice_info, mode='low')
        
        # mask_low = torch.zeros_like(mask_low,device=x.device)
        

        c_lr = t_emb + tok_feat
   
        # c_lr = t_emb + torch.mean(tok_lr_emb,dim=1) + tok_feat

        out_le = tok_l

        for layer in self.LEDec_layers:
            out_le = layer(out_le, c_lr, attn_mask=mask_low)

        # out_le1 = out_le[:,:2,:]
        # out_le2 = out_le[:,2:,:]

        # x1 = out_le1.mean(dim=1).detach().cpu().numpy()
        # x2 = out_le2.mean(dim=1).detach().cpu().numpy()

        # X = np.vstack([x1, x2])

        # X_emb = TSNE(n_components=2, init='pca', random_state=42).fit_transform(X)
        # print(x1.shape)
        # print(x2.shape)
        # print(X.shape,'hhhhh')
        # # 3. 绘图并保存
        # B = x1.shape[0]
        # plt.scatter(X_emb[:B, 0], X_emb[:B, 1], label='Source 1', alpha=0.6)
        # plt.scatter(X_emb[B:, 0], X_emb[B:, 1], label='Source 2', alpha=0.6)
        # plt.legend()
        # plt.savefig('tsne_result_ll.png', dpi=300)

        band_tokens_cat = torch.cat(band_tokens,dim=1)

        input_hd = torch.cat((out_le,band_tokens_cat),dim=1)

        
        start_idx = len_lr + len_lf
        hf_slices = []
        for hf_tok in band_tokens:
            end_idx = start_idx + hf_tok.shape[1]
            hf_slices.append(slice(start_idx, end_idx))
            start_idx = end_idx
        slice_info['hf_list'] = hf_slices
        
        mask_high = self._make_mask(input_hd, slice_info, mode='high')
        
        # print(mask_high)
        # mask_high = torch.zeros_like(mask_high,device=x.device)


        c_hd = c_lr

        out_hd = input_hd

        for layer in self.HDDec_layers:
            out_hd = layer(out_hd, c_hd, attn_mask=mask_high)


        # feats = [out_hd[:, i:i+2, :].mean(dim=1).detach().cpu().numpy() for i in range(0, 10, 2)]
 
        # X = np.vstack(feats)  # 形状变为 (5, C)
        # print(X.shape,'xixi')
        # # 2. t-SNE 降维
        # X_emb = TSNE(n_components=2, init='pca', random_state=42, perplexity=2).fit_transform(X)

        # # 3. 绘图与保存
        # plt.figure(figsize=(8, 6))
        # colors = ['red', 'blue', 'green', 'orange', 'purple']
        # for i in range(5):
        #     start, end = i * feats[0].shape[0], (i + 1) * feats[0].shape[0]
        #     plt.scatter(X_emb[start:end, 0], X_emb[start:end, 1], 
        #                 c=colors[i], label=f'Group {i+1}', s=100, alpha=0.01)

        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.savefig('tsne_resulthh13.png', dpi=300)


        # print(out_hd.shape,'ooo')
        # exit()
        lf_pre = out_le[:,slice_info['lf'],:]
        lr_pre = out_hd[:,slice_info['lf'],:]


        ll_pre = lf_pre + lr_pre 
        hf_cut =self. _max_patch_num * 2


        hf_pre = out_hd[:,hf_cut:,:]


        rec_lf = self._decode_low_freq(ll_pre,c_lr, 1)


  
        offset = hf_cut
        new_slices = [slice(s.start - offset, s.stop - offset, s.step) for s in hf_slices]

        rec_hfs = []
        for i, sl in enumerate(new_slices):

            hf_feat = hf_pre[:, sl, :] 
            
            detok_idx = i
            cut_length = n_yh[i].shape[-1]

            hf_rec = self._decode_high_freq(hf_feat, c_hd,detok_idx)
  
            hf_rec = hf_rec[:,:,:cut_length]
            rec_hfs.append(hf_rec)


        rec_signal = self.dwt([rec_lf, rec_hfs], is_dec=False)

        return rec_signal.permute(0,2,1)

    def _decode_high_freq(self, tokens, c_lr, idx):
        # ... (Detokenizer logic remains the same) ...
        x = self.high_freq_detokenizers[idx](tokens,c_lr)

        x = rearrange(x, '(b k) l -> b k l', k=self.c_in)

        return x
    
    def _decode_low_freq(self, tokens, c_lr, idx=None):
        # ... (Detokenizer logic remains the same) ...
        x = self.low_freq_detokenizers(tokens,c_lr)

        x = rearrange(x, '(b k) l -> b k l', k=self.c_in)

        return x