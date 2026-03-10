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



class TSflow(Forecaster):
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
        matching: str = "random",
        mode="diag",
        measure="diag",
        alpha: float = 0.005,
        iterations: int = 4,
        noise_level: float = 0.5,
        guidance_scale: int = 4,
        dropout: float = 0,
        num_samples:int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        backbone_parameters = {
            "input_dim": self.target_dim,
            "hidden_dim": hidden_dim,
            "output_dim": self.target_dim,
            "step_emb": step_emb,
            "num_residual_blocks": num_residual_blocks,
            "residual_block": "s4",
            "mode": mode,
            'measure': measure,
        }

        self.num_samples = num_samples
        self.use_ema = use_ema
        self.alpha = alpha
        self.iterations = iterations
        self.noise_level = noise_level
        self.guidance_scale = guidance_scale
        self.input_dim = backbone_parameters["input_dim"]

        self.backbone = BackboneModel(**backbone_parameters, num_features=self.target_dim,dropout=dropout)
        self.timesteps = timesteps
        self.solver = solver
        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.ot_sampler = OTPlanSampler(method="exact")
        self.matching = matching

        self.prior = Prior(prior_params["kernel"]).value

        self.frequency = get_season_length(self.freq)
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=self.prediction_length,
            freq=self.frequency,
            iso=1e-2 if self.prior != "iso" else 0,
            num_features=self.input_dim,
        )
        sigm = 0.001
        self.sigmin = sigm
        self.sigmax = 1 if self.prior != Prior.ISO.value else self.sigmin

    def fast_denoise(self, xt, t, features=None, noise=None, steps=4):
        t_span = torch.linspace(t, 1, steps + 1, device=xt.device)[:-1]
        
        node = NeuralODE(
            self.get_vf(features),
            solver=self.solver,
            sensitivity="adjoint",
        )

        return node.trajectory(xt.to(xt.device), t_span)[-1]


    def _extract_features(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        
        x = inputs[:,:, :self.target_dim]


        features = inputs.clone()
        
        if self.use_time_feat:
            features[:,self.context_length:, :self.target_dim] = 0
        else:
            features = features[:,:, :self.target_dim]
            features[:,self.context_length:] = 0
        
        observation_mask = torch.zeros_like(x, device=x.device)
        observation_mask[:,:self.context_length] = 1
        
        return x, observation_mask, features

    def prior_sample(self, input, observation, observation_mask, alpha=0.1, iterations=10, noise_level=1.0):
        
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

        noise = self.sigmax * torch.randn_like(input)

        for i in range(iterations):
            self.backbone.zero_grad()
            input.requires_grad_(True)
            # # print(input,noise,'sssssssssssssss')
            # print(f"input.grad_fn: {input.grad_fn}")  # 叶子节点应为 None（正常）
            # print(f"input.is_leaf: {input.is_leaf}")  # 应为 True（外部传入的原始张量）
            # print(f"noise.grad_fn: {noise.grad_fn}")  # 应为 None（正常，randn_like 创建的叶子节点）
            # print(f"noise.is_leaf: {noise.is_leaf}")  # 应为 True（randn_like 创建）
            # # 关键：检查是否处于 inference_mode 或 no_grad 上下文
            # print(f"torch.is_inference_mode_enabled(): {torch.is_inference_mode_enabled()}")  # 必须为 False
            # print(f"torch.is_grad_enabled(): {torch.is_grad_enabled()}")  # 必须为 True
            pred = self.fast_denoise(input + noise, torch.tensor(0.0, device=input.device))
            
            Ey = quantile_loss(pred, observation)[observation_mask == 1].sum()
            reg = self.q0.log_likelihood(input.squeeze()).sum()

            # print(Ey,reg,'ey reg')
            grad = 16 * torch.autograd.grad(Ey, input)[0] + torch.autograd.grad(reg, input)[0]
            input = input - alpha * grad + noise_level * torch.sqrt(torch.tensor(2 * alpha)) * torch.rand_like(grad)

            input.grad = None

        return input + noise
    

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

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
        
        predicted_flow = self.backbone(psi, t, features)

        loss = F.mse_loss(dpsi, predicted_flow)


        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)


        predicted_noise = self.backbone(x, t, features)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def step(self, data):

        assert self.training is True
        x1, _, _ = self._extract_features(data)

        if self.prior == "isotropic":
            x0 = torch.randn_like(x1)

        else:
            x0 = self.q0(x1.shape[0]).to(x1.device)


        if self.matching == "ot":
            # 2a. 为低频分量 (ll) 寻找匹配
            x0, x1, _ = self.ot_sampler.sample_plan(x0, x1 - 1, replace=False)       
        # if self.matching == "ot":
        #     x0, x1, _ = self.ot_sampler.sample_plan(x0, x1 - 1, replace=False)

        t = torch.rand((x1.shape[0], 1), device=x1.device)

        loss = self.p_losses(x1, x0, t, None)

        return loss

    def loss(self, batch_data):
        # [b l k 1], [b l k 2]
        # x, features, observation_mask = self._extract_features(batch_data)
        # loss_mask = 1 - observation_mask

        # t = torch.randint(
        #     0, self.timesteps, [x.shape[0]], device=x.device
        # ).long()
        
        loss = self.step(batch_data)

        if torch.isnan(loss):
            print("Loss is NaN, exiting.")
            sys.exit(1)
        return loss

    def forecast(self, batch_data, num_samples):


        observation, observation_mask, features = self._extract_features(batch_data)
        observation = observation.to(observation.device)

        repeated_observation = repeat(observation, num_samples)
        repeated_observation_mask = repeat(observation_mask, num_samples)

        batch_size, length, ch = repeated_observation.shape

        repeated_observation = repeated_observation.to(batch_data.device) - 1

        noise = self.q0(repeated_observation.shape[0]).to(batch_data.device) # 500 360 1

        noise = self.prior_sample(
            noise,
            repeated_observation,
            repeated_observation_mask,
            alpha=self.alpha,
            iterations=self.iterations,
            noise_level=self.noise_level,
        )

        pred = (
            self.sample(
                noise=noise,
                observation=repeated_observation,
                observation_mask=repeated_observation_mask,
                guidance_scale=self.guidance_scale,
            )
            + 1
        )
        pred = rearrange(pred, "(b n) l f -> b n l f", n=self.num_samples,f=self.target_dim)
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
                dxt = model(x, t, features)
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
        guidance_scale=None,
    ):

        if self.timesteps == 0:

            return noise.to(noise.device)
        
        t_span = torch.linspace(0, 1, self.timesteps + 1)

        node = NeuralODE(self.get_vf(features, observation, observation_mask, guidance_scale), solver=self.solver)

        return node.trajectory(noise.to(noise.device), t_span)[-1]

