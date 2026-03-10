# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------
# Portions of this file are derived from TSDiff
# - Source: https://github.com/amazon-science/unconditional-time-series-diffusion
# - Paper: Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from probts.utils import extract
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.S4.s4_backbones import BackboneModel
from probts.utils import repeat
import sys
from ema_pytorch import EMA
from einops import rearrange
from torchdyn.core import NeuralODE
from probts.model.nn.arch.optimal_transport import OTPlanSampler
from gluonts.torch.util import lagged_sequence_values
from probts.utils import repeat
from probts.utils.variables import Prior, get_season_length
from probts.model.nn.arch.gaussian_process import Q0Dist
from typing import Union
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from probts.utils  import LongScaler
from probts.data.data_utils.time_features import get_lags

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)

def get_lags_for_freq(freq_str: str):
    if freq_str == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif freq_str == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    elif freq_str == "1D":
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]
    else:
        raise NotImplementedError(f"Lags for {freq_str} are not implemented yet.")
    return lags_seq



class TSflow_cond(Forecaster):
    def __init__(
        self,
        hidden_dim: int,
        step_emb: int,
        ema_params: dict,
        prior_params: dict,
        num_residual_blocks: int,
        use_ema: bool = False,
        timesteps: int = 4,
        solver: str = "euler",
        iterations: int = 4,
        noise_level: float = 0.5,
        guidance_scale: int = 0,
        dropout: float = 0,
        num_samples:int = 100,
        init_skip: bool = False,
        feature_skip: bool = False,
        sigmin: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        backbone_parameters = {
            "input_dim": 1,
            "hidden_dim": hidden_dim,
            "output_dim": 1,
            "step_emb": step_emb,
            "num_residual_blocks": num_residual_blocks,
            "residual_block": "s4",
            "init_skip": init_skip,
            "feature_skip": feature_skip,
            "target_dim": self.target_dim
        }
        

        self.num_samples = num_samples
        self.use_ema = use_ema
        self.iterations = iterations
        self.noise_level = noise_level
        self.guidance_scale = guidance_scale
        self.input_dim = backbone_parameters["input_dim"]
        self.lags_seq = get_lags(self.freq) if self.use_lags else [0]
        num_features = 2 + (len(self.lags_seq) if self.use_lags else 0)
        self.backbone = BackboneModel(**backbone_parameters, num_features=num_features,dropout=dropout)
        self.timesteps = timesteps
        self.solver = solver
        self.ema_backbone = EMA(self.backbone, **ema_params)

        self.prior_context_length = (
            self.context_length
            if "context_freqs" not in prior_params.keys()
            else prior_params["context_freqs"] * self.prediction_length
        )

        self.prior = Prior(prior_params["kernel"]).value

        self.frequency = get_season_length(self.freq)
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=self.prediction_length,
            freq=self.frequency,
            iso=1e-1 if self.prior != "iso" else 0,
        )
        
        self.sigmin = 0.001
        self.sigmax = 1

    
    def _extract_features(
        self,data: dict, inference:bool = False):

        
        past = self.get_inputs(data,'encode')

        past = past[:,self.context_length:,:]

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]

        prior_context = past[:, -self.prior_context_length :]

        # if isinstance(self.scaler, LongScaler):
        #     scaled_context, loc, scale = self.scaler(context, scale=mean)
        # else:
        #     _, loc, scale = self.scaler(past, context_observed)
        #     scaled_context = context / scale

        full_seq = self.get_inputs(data,'all')
        if inference:
            future = full_seq[:,-self.prediction_length:,:]
            future = torch.zeros_like(future)
        else:
            future = full_seq[:,-self.prediction_length:,:]


        x1 = torch.cat([context, future], dim=-2)

        batch_size, length, c = x1.shape

        observation_mask = torch.zeros_like(x1)
        observation_mask[:, : -self.prediction_length,:] = 1


        dist = self.q0.gp_regression(rearrange(prior_context, "b l c -> (b c) l"), self.prediction_length)

        features = []
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                long_context,
                x1,
                dim=1,
            )
            features.append(lags)
 
        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        

        fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)

        fut_std = torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1)
        fut_std = rearrange(fut_std, "(b c) ... -> b ... c", c=c)

        features.append(torch.cat([context, fut_mean], dim=-2).unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))
        x0 = torch.cat([context, fut], dim=-2)
   
        features = torch.cat(features, dim=-1)
        
        return x1, x0, observation_mask, features


    def forward_path(self, x1, x0, t):
        eps = torch.randn_like(x0)
        sig_t = (1 - t) * self.sigmax + t * self.sigmin

        psi = t * x1 + (1 - t) * x0 + sig_t * eps  # xt
        dpsi = x1 - x0 + (self.sigmin - self.sigmax) * eps  # uts
        
        return psi, dpsi
    
    def p_losses(self, x1, x0, t,features=None):
        num_dims_to_add = x1.dim() - t.dim()
        t = t.unsqueeze(-1) if num_dims_to_add == 1 else t.unsqueeze(-1).unsqueeze(-1)
        psi, dpsi = self.forward_path(x1, x0, t)
        predicted_flow = self.backbone(t,psi, features)
        loss_mse = F.mse_loss(dpsi, predicted_flow)

        return loss_mse  


    def step(self, data):


        x1, x0, _, features = self._extract_features(data,inference=False)  # x1 64 60 8  x0 64 60 8 features 64 60 8 14

        
        t = torch.rand((x1.shape[0], 1), device=x1.device)
        
        
        loss = self.p_losses(x1, x0, t, features)

        return loss

    def loss(self, batch_data):
  
        loss = self.step(batch_data)

        if torch.isnan(loss):
            print("Loss is NaN, exiting.")
            sys.exit(1)
        return loss

    def forecast(self, batch_data, num_samples):

        observation, x0, observation_mask, features = self._extract_features(batch_data,inference=True)
        
        observation = observation.to(observation.device)
        b,l,k = observation.shape
  
        repeated_observation = observation.to(observation.device).repeat_interleave(num_samples,dim=0)

        repeated_observation_mask = observation_mask.to(observation.device).repeat_interleave(num_samples, dim=0)

        repeat_features = features.to(observation.device).repeat_interleave(num_samples, dim=0)

        
        x0 = x0.to(observation.device).repeat_interleave(num_samples, dim=0)

        x0 = x0 + self.sigmax * torch.randn_like(x0)
        
        
        pred = (
            self.sample(
                noise=x0,
                features = repeat_features,
                observation=repeated_observation,
                observation_mask=repeated_observation_mask,
                guidance_scale=self.guidance_scale,
            )
            
        )

        
        
        pred = rearrange(pred, "(b n) l f -> b n l f", b=b,n=self.num_samples,f=self.target_dim)
        

        return pred[...,  - self.prediction_length :,:] 

    def get_vf(
            self,
            features,
            observation=None,  # 未输入时默认 None
            observation_mask=None,  # 未输入时默认 None
            guidance_scale: float = 0  ,
        ):
        def quantile_loss(y_prediction, y_target):

            assert y_target.shape == y_prediction.shape
            device = y_prediction.device
            batch_size_x_num_samples, _, _ = y_target.shape
            batch_size = batch_size_x_num_samples // self.num_samples
            q = torch.linspace(0.1, 0.9, self.num_samples, device=device).repeat(batch_size)
            q = q[:, None, None]
            e = y_target - y_prediction
            loss = torch.max(q * e, (q - 1) * e)
            return loss

        def score_func(t, x, model):
            with torch.enable_grad():
                x.requires_grad_(True)
                dxt = model(t,x, features)
                pred = x + (1 - t) * dxt
                E = quantile_loss(pred, observation)[observation_mask == 1].sum()
                return dxt, -torch.autograd.grad(E, x)[0]

        class vf(torch.nn.Module):
            def __init__(self, backbone, guidance_scale, sigmin, sigmax):
                super().__init__()
                self.backbone = backbone
                self.guidance_scale = guidance_scale
                self.sigmin = sigmin
                self.sigmax = sigmax

            def forward(self, t, x, args):
                

                if guidance_scale > 0:

                    dxt, score = score_func(t, x, self.backbone)
                    sig_t = (1 - t) * self.sigmax + t * self.sigmin
                    dsig_t = self.sigmin - self.sigmax
                    dxt = dxt - dsig_t * sig_t * self.guidance_scale * score

                else:

                    dxt = self.backbone(t, x, features)

                return dxt

        return vf(self.backbone if not self.use_ema else self.ema_backbone, guidance_scale, self.sigmin, self.sigmax)
    
    @torch.no_grad()
    def sample(
        self,
        noise,
        features=None,
        observation=None,
        observation_mask=None,
        guidance_scale=4,
    ):

        if self.timesteps == 0:

            return noise.to(noise.device)
        t_span = torch.linspace(0, 1, self.timesteps + 1)

        node = NeuralODE(self.get_vf(features, observation, observation_mask, guidance_scale), solver=self.solver)

        return node.trajectory(noise.to(noise.device), t_span)[-1]

