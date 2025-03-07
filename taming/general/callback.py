import os
import dotenv
dotenv.load_dotenv()

import os
import pathlib
from typing import List
import numpy as np
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

class SetupCallback(Callback):
    def __init__(self, config, exp_config, basedir):
        super().__init__()
        self.config = config
        self.exp_config = exp_config
        self.basedir = basedir
        self.ckptdir = basedir / "checkpoints"
        self.logdir = basedir / "logs"
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create directories
            os.makedirs(self.basedir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)

            # Save configs
            configs = [self.config, self.exp_config]
            for i, config in enumerate(configs):
                config_path = self.basedir / f"config_{i}.yaml"
                with open(config_path, 'w') as f:
                    f.write(config.pretty())

class ImageTextLogger(Callback):
    def __init__(self, batch_frequency, max_images):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w -> h,w,c
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            from PIL import Image
            Image.fromarray(grid).save(path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not (batch_idx % self.batch_freq == 0 and batch_idx > 0):
            return
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split="train")

        if is_train:
            pl_module.train()

        if not images:
            return
        
        self.log_local(
            trainer.logger.save_dir, 
            "train", 
            images,
            trainer.global_step, 
            trainer.current_epoch, 
            batch_idx
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx >= self.max_images:
            return
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split="val")

        if is_train:
            pl_module.train()

        if not images:
            return
        
        self.log_local(
            trainer.logger.save_dir, 
            "val", 
            images,
            trainer.global_step, 
            trainer.current_epoch, 
            batch_idx
        )
