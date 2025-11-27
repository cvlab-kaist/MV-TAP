import os
import hydra
import torch
import logging
import warnings
import model_utils
import lightning as L

from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from typing import Any, Dict, Optional

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from data.mvtap_dataset import load_train_dataset, load_eval_dataset

warnings.filterwarnings("ignore", message="No device id is provided", module="torch.distributed")
torch.set_float32_matmul_precision('high')

class MVTAPModel(L.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        max_steps: int = 100000,
        loss_name: Optional[str] = None,
        eval_loss_name: Optional[str] = 'tapir_loss', 
        optimizer_name: Optional[str] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: Optional[str] = 'OneCycleLR',
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
    ):    
        super().__init__()
        self.model = hydra.utils.instantiate(model_cfg)
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.loss = partial(model_utils.__dict__[loss_name], **({}))
        self.eval_loss = partial(model_utils.__dict__[eval_loss_name], **({}))

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs) if optimizer_kwargs else {'lr': 2e-3}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = dict(scheduler_kwargs) if scheduler_kwargs else {'max_lr': 2e-3, 'pct_start': 0.05}
        self.scheduler_kwargs['max_lr'] = self.optimizer_kwargs['lr']
        self.scheduler_kwargs['total_steps'] = max_steps + 100  
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        video = batch['video']
        queries = batch['query_points']
        intrinsic = batch['intrinsic']
        extrinsic = batch['extrinsic']
    
        output = self.model(video, queries, intrinsic, extrinsic, **self.model_forward_kwargs)
        window_len = getattr(self.model, "window_len", 1) 
        loss, loss_scalars = self.loss(batch, output, window_len)

        self.log_dict(
            {f'train/{k}': v.item() for k, v in loss_scalars.items()},
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad()
        self.manual_backward(loss)

        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt.step()
        sched.step()
        
        return loss    

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        video = batch['video']
        queries = batch['query_points'].clone().float()
        intrinsic = batch['intrinsic']
        extrinsic = batch['extrinsic']
        
        output = self.model(video, queries, intrinsic, extrinsic, **self.model_forward_kwargs)

        metrics = model_utils.eval_batch(batch, output)

        self.log_dict(
            {f"val/{k}": v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
            batch_size=1
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        video = batch['video']
        queries = batch['query_points'].clone().float()
        intrinsic = batch['intrinsic']
        extrinsic = batch['extrinsic']
        
        output = self.model(video, queries, intrinsic, extrinsic, **self.model_forward_kwargs)
        metrics = model_utils.eval_batch(batch, output)

        self.log_dict(
            {f"test/{k}": v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
            batch_size=1
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def configure_optimizers(self):
        
        base_lr = self.optimizer_kwargs.get("lr", 2e-3)
        base_wd = self.optimizer_kwargs.get("wdecay", 1e-4)
        eps = self.optimizer_kwargs.get("eps", 1e-8)


        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Param Count] Total trainable parameters: {total_params:,}")


        Optim = getattr(torch.optim, self.optimizer_name)


        optimizer = Optim(trainable_params, lr=base_lr, weight_decay=base_wd, eps=eps)
        sch_kwargs = dict(self.scheduler_kwargs)
        
        sch_kwargs["max_lr"] = base_lr
        Sched = getattr(torch.optim.lr_scheduler, self.scheduler_name)
        scheduler = Sched(optimizer, **sch_kwargs)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

@hydra.main(version_base="1.3", config_path="configs", config_name="experiment.yaml")
def main(cfg: DictConfig):
    
    # checkpoints & visualizations directory
    Path(cfg.experiment_path).mkdir(exist_ok=True, parents=True)
    seed_everything(0, workers=True)
    
    model = MVTAPModel(cfg.model, **cfg.trainer)
    
    if cfg.mode == 'eval':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    save_path = os.path.join(cfg.experiment_path, 'checkpoints')
    logger = WandbLogger(project=cfg.wandb_name, entity=cfg.get("wandb_entity"), save_dir=save_path, id=os.path.basename(cfg.experiment_path))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}-{step}-{val/average_jaccard:.4f}',
        save_last=True,
        mode='max',
        save_top_k=-1,
        monitor='val/average_jaccard',
    )   

    eval_loader = load_eval_dataset(cfg.datasets)
    
    ckpt_path = cfg.get("ckpt_path") or None
    
    if cfg.mode == 'train':
        train_loader = load_train_dataset(cfg.datasets)

        trainer = L.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
            logger=logger,
            num_sanity_val_steps=0,
            precision=cfg.precision,
            val_check_interval=cfg.val_check_interval,
            log_every_n_steps=cfg.log_every_n_steps,
            max_steps=cfg.trainer.max_steps,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
        )        
        trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    elif cfg.mode == 'eval':
        trainer = L.Trainer(strategy='ddp', logger=None, precision=cfg.precision)
        trainer.test(model, eval_loader, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")        
        

if __name__ == '__main__':
    main()