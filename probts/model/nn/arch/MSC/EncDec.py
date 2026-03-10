import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x, x_attn_weights, cross_attn_weights = layer(x, cross, x_attn_mask=x_attn_mask, cross_attn_mask=cross_attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, x_attn_weights, cross_attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_attn_mask=None, cross_attn_mask=None):
        x_, x_attn_weights = self.self_attention(
            x, x, x,
            attn_mask=x_attn_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)

        x_, cross_attn_weights = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_attn_mask,
        )
        x = x + self.dropout(x_)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), x_attn_weights, cross_attn_weights
    

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None,features=None):

        for layer in self.layers:
            x,attn_weights = layer(x, attn_mask=attn_mask,features=features)

        if self.norm is not None:
            x = self.norm(x)

        return x
    

class Encoder_1(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder_1, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None,features=None):
        ys = []
        for layer in self.layers:
            x,y, attn_weights = layer(x, attn_mask=attn_mask,features=features)
            ys.append(y)

        y = torch.stack(ys).sum(0)
        if self.norm is not None:
            y = self.norm(y)
        
        return y, attn_weights
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None,features=None):
        x_, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
        )
        # x = x + self.dropout(x_)
        # x = self.norm1(x)
        
        # y = x = self.norm1(x) 
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
  
        return x_, attn_weights




class EncoderLayer_1(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(EncoderLayer_1, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None,features=None):
        x_, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)
        bs,c,length = features.shape
        num_segments = x.shape[1] // length
        features = features.reshape(bs,length,c)
        for i in range(1, num_segments):
            start_idx = i * length
            if i < num_segments-1:
                end_idx = (i + 1) * length
                x[:, start_idx:end_idx, :] = x[:, start_idx:end_idx, :] 
            else:
                
                x[:, start_idx:, :] = x[:, start_idx:, :] 

        y = x = self.norm1(x) 
        x1 = y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        x1 = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.dropout(self.conv3(y).transpose(-1, 1))

        return x1, self.norm2(y + x), attn_weights