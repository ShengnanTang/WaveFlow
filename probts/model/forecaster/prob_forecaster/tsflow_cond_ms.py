

# ---------------------------------------------------------------------------------
# Portions of this file are derived from TSFlow
# - Source: https://github.com/marcelkollovieh/TSFlow/tree/main
# - Paper: PFlow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from probts.utils import extract
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.MSC.crossAD_backbone import BackboneModel
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
import torch.nn as nn
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



class TSflow_cond_ms(Forecaster):
    def __init__(
        self,
        hidden_dim: int,
        ema_params: dict,
        prior_params: dict,
        noise_observed=False, # reconstruct past
        use_ema: bool = False,
        timesteps: int = 4,
        solver: str = "euler",
        matching: str = "random",
        alpha: float = 0.005,
        iterations: int = 4,
        noise_level: float = 0.5,
        num_samples:int = 100,
        wavelet_level: int = 2,
        base_patch_size: int = 8,
        ori_patch_size: int = 16,
        nheads: int = 8,
        mlp_ratio: int = 2,
        proj_drop: float = 0.1 ,
        attn_drop: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        backbone_parameters = {
            "input_dim": self.target_dim,
            "hidden_dim": hidden_dim,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "wavelet_level": wavelet_level,
            "base_patch_size": base_patch_size,
            "ori_patch_size": ori_patch_size,
            "nheads": nheads,
            "mlp_ratio": mlp_ratio,
            "proj_dopr": proj_drop,
            "attn_drop": attn_drop

        }
        self.num_samples = num_samples
        self.alpha = alpha
        self.iterations = iterations
        self.noise_level = noise_level
        self.guidance_scale = 0
        self.input_dim = backbone_parameters["input_dim"]
        self.lags = False
        self.lags_seq = get_lags(self.freq) if self.lags else [0]
        self.backbone = BackboneModel(**backbone_parameters)
        self.timesteps = timesteps
        self.solver = solver
        self.use_ema = use_ema
        self.ema_backbone = EMA(self.backbone, **ema_params)

        self.ot_sampler = OTPlanSampler(method="exact")
        self.matching = matching

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
        self.sigmax = 1.0
        
    
    def _extract_features(
        self,data: dict, inference:bool = False):

        
        past = self.get_inputs(data,'encode')

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]

        prior_context = past[:, -self.prior_context_length :]

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

        if self.lags:
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


        if inference:
            fut_init = fut_mean   # 用均值
        else:
            fut_init = fut        # 训练用 sample

        x0 = torch.cat([context,fut_init], dim=-2)

   
        features.append(torch.cat([context, fut_mean], dim=-2).unsqueeze(-1))
        # features.append(torch.cat([context, fut_std], dim=-2).unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))

        features = torch.cat(features, dim=-1)

        
        psudo_x0 = x0.unsqueeze(-1)

        features = torch.cat([features,psudo_x0], dim=-1)
 

        return x1, x0, observation_mask, features


    def forward_path(self, x1, x0, t):
        eps = torch.randn_like(x0)
        sig_t = (1 - t) * self.sigmax + t * self.sigmin

        psi = t * x1 + (1 - t) * x0 + sig_t * eps  # xt
        dpsi = x1 - x0 + (self.sigmin - self.sigmax) * eps  # uts
        
        return psi, dpsi
   
    def p_losses(self, x1, x0, t, features=None):
        num_dims_to_add = x1.dim() - t.dim()
        t = t.unsqueeze(-1) if num_dims_to_add == 1 else t.unsqueeze(-1).unsqueeze(-1)
        psi, dpsi = self.forward_path(x1, x0, t)

        t = t.squeeze(-1).squeeze(-1)

        predicted_flow = self.backbone(psi,t,features)

        loss = F.mse_loss(dpsi, predicted_flow)

        lam_mean = 0.5
        x1_hat = psi + (1.0 - t.unsqueeze(-1).unsqueeze(-1)) * predicted_flow
        loss = loss + lam_mean * F.l1_loss(
            x1_hat[:, -self.prediction_length:, :],
            x1[:,  -self.prediction_length:, :]
        )

        return loss 


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

        repeated_observation = repeat(observation, num_samples)

        repeated_observation_mask = repeat(observation_mask, num_samples)

        repeated_observation = repeated_observation.to(batch_data.device) 
        repeated_features = repeat(features,num_samples)

        x0 = repeat(x0,num_samples)
        noise = torch.randn_like(x0)
        x0 = x0 + noise * self.sigmax
       
        pred = (
            self.sample(
                noise=x0,
                features = repeated_features,
                observation=repeated_observation,
                observation_mask=repeated_observation_mask,
                guidance_scale=self.guidance_scale,
            )
            
        )


        pred = rearrange(pred, "(b n) l f -> b n l f", n=self.num_samples,f=self.target_dim)

        return pred[...,  - self.prediction_length :,:]


    def get_vf(
            self,
            features,
            observation=None,  
            observation_mask=None,  
            guidance_scale: float = 0  ,
        ):

        class vf(torch.nn.Module):
            def __init__(self, backbone, guidance_scale, sigmin, sigmax):
                super().__init__()
                self.backbone = backbone
                self.guidance_scale = guidance_scale
                self.sigmin = sigmin
                self.sigmax = sigmax

            def forward(self, t, x, args):
                


                t = t.repeat(x.shape[0],)

                dxt = self.backbone(x, t, features)

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

