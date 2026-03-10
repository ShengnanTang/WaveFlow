import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

import numpy as np
from torch.autograd import Function
from pytorch_wavelets import DWTForward,DWT1DForward
from pytorch_wavelets.utils import reflect


class ADWT_1D(nn.Module):
    def __init__(self, 
                 kernel_size: int = 2, 
                 level: int = 3,
                 random_init: bool = False,
                 init_wave: bool = True,
                 learnable: bool = True) -> None:
        super().__init__()
        self.level = level
        wave = pywt.Wavelet('db1')
        init_func = nn.init.trunc_normal_ if random_init else lambda x: x
        kernel_size = len(wave.dec_lo)
        # 初始化低通滤波器
        if learnable:
            self.low_row = nn.ParameterList()
            
            for _ in range(self.level):
                if init_wave:
                    weight = torch.tensor(wave.dec_lo[::-1], dtype=torch.float32)

                else:
                    weight = torch.zeros(kernel_size, dtype=torch.float32)
                self.low_row.append(nn.Parameter(init_func(weight)))
        else:
            self.low_row = []
            # init_func = nn.init.trunc_normal_ if random_init else lambda x: x
            for i in range(self.level):
                if init_wave:
                    weight = torch.tensor(wave.dec_lo[::-1], dtype=torch.float32)
                else:
                    weight = torch.zeros(kernel_size, dtype=torch.float32)

                weight = init_func(weight)
                self.register_buffer(f'low_row_{i}', weight)
                self.low_row.append(getattr(self, f'low_row_{i}'))

        # 高通滤波器（固定）
        high_dot = torch.tensor([(-1)**i for i in range(kernel_size)], dtype=torch.float32)
        self.register_buffer("high_dot", high_dot)


    def decompose(self, input):

        yh = []
        ll = input # b,c,h,w
        for j in range(self.level):
            _ll = ll[:, :, None, :]
            # 1-level Transform
            low_row = torch.reshape(self.low_row[j],shape=(1,1,1,-1))
            device = self.high_dot.device
            high_row = torch.reshape(self.high_dot*torch.flip(self.low_row[j].to(device),dims=(0,)),shape=(1,1,1,-1))
            lohi = afb1d(_ll, low_row.to(device), high_row.to(device), mode='zero', dim=3)
            low = lohi[:,::2,0].contiguous()
            highs = lohi[:,1::2,0].contiguous()
            ll = low
            yh.append(highs)
        return ll, yh
    
    def reconstruct(self, coeffs):
        yl, yh = coeffs
        ll = yl
        j = 0
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros_like(ll)
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            _ll = ll[:, :, None, :]
            h = h[:, :, None, :]
            lowi_row = torch.reshape(self.low_row[j],shape=(1,1,1,-1))
            device = self.high_dot.device
            highi_row = torch.reshape(self.high_dot*torch.flip(self.low_row[j].to(device),dims=(0,)),shape=(1,1,1,-1))
            y = sfb1d(_ll, h, lowi_row.to(device), highi_row.to(device), mode='zero', dim=3)
            ll = y[:, :, 0]
            j += 1
            
        return ll

    def forward(self, input, is_dec:bool):
        if is_dec:
            dec_coefficients = self.decompose(input=input)
            return dec_coefficients
        else:
            rec_result = self.reconstruct(coeffs=input)
            return rec_result
        
    def _dummy_forward(self,seq_len,is_dec:bool):

        input = torch.ones(1, 1, seq_len)

        l_xh = []
        if is_dec:
            ll, yh = self.decompose(input=input)
            l_xl = ll.shape[-1]

            for h in yh:
                l = h.shape[-1]
                l_xh.append(l)

            return l_xl,yh





def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d == 2 else (0, L-1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)

def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))
        