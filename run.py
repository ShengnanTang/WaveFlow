import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import torch
import logging
from probts.data import ProbTSDataModule
from probts.model.forecast_module import ProbTSForecastModule
from probts.callbacks import MemoryCallback, TimeCallback
from probts.utils import find_best_epoch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from probts.utils.save_utils import save_exp_summary, save_csv
import time
from probts.data.data_wrapper import ProbTSBatchData
MULTI_HOR_MODEL = ['ElasTST', 'Autoformer']


torch.set_float32_matmul_precision('high')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProbTSCli(LightningCLI):
    
    def add_arguments_to_parser(self, parser):
        data_to_model_link_args = [
            "scaler",
            "train_pred_len_list", 
        ]
        data_to_forecaster_link_args = [
            "target_dim",
            "history_length",
            "context_length",
            "prediction_length",
            "train_pred_len_list", 
            "lags_list",
            "freq",
            "time_feat_dim",
            "global_mean",
            "dataset"
        ]
        for arg in data_to_model_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.{arg}", apply_on="instantiate")
        for arg in data_to_forecaster_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.forecaster.init_args.{arg}", apply_on="instantiate")

    def init_exp(self):
        config_args = self.parser.parse_args()
        self.model_args= str(dict(config_args.model.forecaster.init_args))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 格式：YYYYMMDD_HHMMSS
        unique_id = f"log_{timestamp}_{int(time.time() * 1000)}"
        
        if self.datamodule.data_manager.multi_hor:
            assert self.model.forecaster.name in MULTI_HOR_MODEL, f"Only support multi-horizon setting for {MULTI_HOR_MODEL}"
            
            self.tag = "_".join([
                self.datamodule.data_manager.dataset,
                self.model.forecaster.name,
                'TrainCTX','-'.join([str(i) for i in self.datamodule.data_manager.train_ctx_len_list]),
                'TrainPRED','-'.join([str(i) for i in self.datamodule.data_manager.train_pred_len_list]),
                'ValCTX','-'.join([str(i) for i in self.datamodule.data_manager.val_ctx_len_list]),
                'ValPRED','-'.join([str(i) for i in self.datamodule.data_manager.val_pred_len_list]),
                'seed' + str(config_args.seed_everything),
                unique_id
            ])
        else:
            self.tag = "_".join([
                self.datamodule.data_manager.dataset,
                self.model.forecaster.name,
                'CTX' + str(self.datamodule.data_manager.context_length),
                'PRED' + str(self.datamodule.data_manager.prediction_length),
                'seed' + str(config_args.seed_everything),
                unique_id
            ])
        
        log.info(f"Root dir is {self.trainer.default_root_dir}, exp tag is {self.tag}")
        
        if not os.path.exists(self.trainer.default_root_dir):
            os.makedirs(self.trainer.default_root_dir)
            
        self.save_dict = f'{self.trainer.default_root_dir}/{self.tag}'
        if not os.path.exists(self.save_dict):
            os.makedirs(self.save_dict)

        if self.model.load_from_ckpt is not None:
            # if the checkpoint file is not assigned, find the best epoch in the current folder
            if '.ckpt' not in self.model.load_from_ckpt:
                _, best_ckpt = find_best_epoch(self.model.load_from_ckpt)
                print("find best ckpt ", best_ckpt)
                self.model.load_from_ckpt = os.path.join(self.model.load_from_ckpt, best_ckpt)
            
            log.info(f"Loading pre-trained checkpoint from {self.model.load_from_ckpt}")
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.model.load_from_ckpt,
                learning_rate=config_args.model.learning_rate,
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                train_pred_len_list=self.datamodule.data_manager.train_pred_len_list,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                no_training=self.model.forecaster.no_training,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )
        
        # Set callbacks
        self.memory_callback = MemoryCallback()
        self.time_callback = TimeCallback()
        
        callbacks = [
            self.memory_callback,
            self.time_callback
        ]
        
        if not self.model.forecaster.no_training:
            if self.datamodule.dataset_val is None:  # if the validation set is empty
                monitor = "train_loss"
            else:
                # not using reweighting scheme for loss
                if self.model.sampling_weight_scheme in ['none', 'fix']:
                    monitor = 'val_CRPS'
                else:
                    monitor = 'val_weighted_ND'
            
            # Set callbacks
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=f'{self.save_dict}/ckpt',
                filename='{epoch}-{val_CRPS:.6f}',
                every_n_epochs=1,
                monitor=monitor,
                save_top_k=-1,
                save_last=True,
                enable_version_counter=False
            )

            callbacks.append(self.checkpoint_callback)

        self.set_callbacks(callbacks)

    def set_callbacks(self, callbacks):
        # Replace built-in callbacks with custom callbacks
        custom_callbacks_name = [c.__class__.__name__ for c in callbacks]
        for c in self.trainer.callbacks:
            if c.__class__.__name__ in custom_callbacks_name:
                self.trainer.callbacks.remove(c)
        for c in callbacks:
            self.trainer.callbacks.append(c)
        for c in self.trainer.callbacks:
            if c.__class__.__name__ == "ModelSummary":
                self.model_summary_callback = c

    def set_fit_mode(self):
        self.trainer.logger = TensorBoardLogger(
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='fit'
        )
    
    def set_test_mode(self):
        self.trainer.logger = CSVLogger(
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='test'
        )

        if not self.model.forecaster.no_training:
            self.ckpt = self.checkpoint_callback.best_model_path
            log.info(f"Loading best checkpoint from {self.ckpt}")
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.ckpt, 
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
                set_global_grad = self.model.set_global_grad,
            )

    def run(self):
        self.init_exp()
        print("******************training begin******************")

        if not self.model.forecaster.no_training:
            self.set_fit_mode()
            if self.datamodule.dataset_val is None:  # if the validation set is empty

                self.trainer.fit(model=self.model, train_dataloaders=self.datamodule.train_dataloader())
            else:

                self.trainer.fit(model=self.model, datamodule=self.datamodule)
            
            inference=False
        else:
            inference=True
        print("******************training end******************")
        self.set_test_mode()
        self._simple_viz()
        self.trainer.test(model=self.model, datamodule=self.datamodule)
        
        save_exp_summary(self, inference=inference)
        
        ctx_len = self.datamodule.data_manager.context_length
        if self.datamodule.data_manager.multi_hor:
            ctx_len = ctx_len[0]

        save_csv(self.model_args,self.save_dict, self.model, ctx_len)

    def _simple_viz(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        save_dir = f"{self.save_dict}/figs"
        os.makedirs(save_dir, exist_ok=True)

        
        loader = self.datamodule.test_dataloader()
        batch = next(iter(loader))

        device = next(self.model.parameters()).device
        batch_prob = ProbTSBatchData(batch, device)

        self.model.eval()
        past_std = self.model.scaler.transform(batch_prob.past_target_cdf)
        future_std = self.model.scaler.transform(batch_prob.future_target_cdf)
        batch_prob.past_target_cdf = past_std  # 关键：forecast 输入用标准化后的 past

        with torch.no_grad():
        
            out = self.model.forecaster.forecast(batch_prob,num_samples=100)

        # === 找 samples（只要这一行对上即可）===
   
        samples = out   # ← 如果报错，改成 out["forecast_samples"]

        # samples: [S, B, pred, D]
        samples = samples.detach().cpu().numpy()

        past = past_std.detach().cpu().numpy()
        future = future_std.detach().cpu().numpy()

        ctx = self.datamodule.data_manager.context_length
        prl = self.datamodule.data_manager.prediction_length

        # 只画第 0 条样本、第 0 维
        hist = past[0, -ctx:, -1]
        gt = future[0, :prl, -1]

        s = samples[0, :, :prl, -1]

        ci=0.95
        alpha = (1 - ci) / 2
        lo = np.quantile(s, alpha, axis=0)        # (L,)
        hi = np.quantile(s, 1 - alpha, axis=0)    # (L,)

        cen = s.mean(axis=0)

        # x_hist = np.arange(ctx)
        x_fut = np.arange(ctx, ctx + prl)
        total =  np.arange(ctx + prl)


        x_total = np.concatenate([hist, gt])
        plt.figure()
        plt.plot(total, x_total, label="groundtruth")
        # plt.plot(x_fut, gt, label="gt")

        plt.plot(x_fut, cen, label="pred")
        plt.fill_between(x_fut, lo, hi, alpha=0.25, label=f"{int(ci*100)}% CI")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/forecast.png", dpi=200)
        plt.close()

if __name__ == '__main__':
    cli = ProbTSCli(
        datamodule_class=ProbTSDataModule,
        model_class=ProbTSForecastModule,
        save_config_kwargs={"overwrite": True},
        run=False
    )
    cli.run()