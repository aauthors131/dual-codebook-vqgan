# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
from fvcore.nn import FlopCountAnalysis
import os
import torch
import sys
import shutil
import argparse
from pathlib import Path
import dotenv

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
os.environ['TORCH_HOME'] = 'pretrained_weights'
os.environ['WANDB_API_KEY'] = 'Your API Key'

dotenv.load_dotenv()
# import wandb

# wandb.login()

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata
#     mode= 'offline'
#     )


from taming.general import get_config_from_file, initialize_from_config, setup_callbacks

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if os.path.exists(Path('experiments')):
    #     shutil.rmtree(Path('experiments'), ignore_errors=True)
    # if not os.path.exists(Path('experiments')):
    #     os.mkdir(Path('experiments'))
    # print('renew experiments!')
    parser.add_argument('-c', '--config', type=str, default='cub_vqct')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ng', '--num_gpus', type=int, default=2)
    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('-lr', '--base_lr', type=float, default=4.5e-6)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=32)
    parser.add_argument('-m', '--max_images', type=int, default=4)
    parser.add_argument('--test', default=False, action='store_true', help="Run model in test mode")  # ✅ Add test flag
    parser.add_argument('--ckpt', type=str, default=None, help="Path to checkpoint for testing")  # ✅ Add checkpoint argument
   
    args = parser.parse_args()

    if os.path.exists(Path('experiments/{}'.format(args.config))):
        shutil.rmtree(Path('experiments/{}'.format(args.config)), ignore_errors=True)
    # if not os.path.exists(Path('experiments')):
    #     os.mkdir(Path('experiments'))
    print('renew experiments!')

    # print(list(range(args.num_gpus)))
    # print(','.join([str(i) for i in range(args.num_gpus)]))
    # set_gpu(','.join([str(i) for i in range(args.num_gpus)]))
    # set_gpu('0,1')
    set_gpu(' 0')


    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    exp_config = OmegaConf.create({"name": args.config, "epochs": args.epochs, "update_every": args.update_every,
                                   "base_lr": args.base_lr, "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    model = initialize_from_config(config.model)

    ########################################
    input_tensor = torch.randn(1, 3, 256, 256)
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    print(f"Total FLOPs: {total_flops}")




    model.learning_rate = exp_config.update_every * args.num_gpus * config.dataset.params.batch_size * exp_config.base_lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    print('setting dataset')
    data = initialize_from_config(config.dataset)
    # data.args = args
    data.prepare_data()


    # Build trainer
    print('setting training')
    

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.environ.get('LOGS_PATH', 'logs'),
        filename="epoch-{epoch:02d}-rec_loss-{val/rec_loss:.4f}",
        monitor="val/rec_loss",
        save_top_k=-1,
        mode="min",  # Save the model with the lowest validation loss
        verbose=True
    )
    # # Initialize TensorBoard logger
    # logger = TensorBoardLogger(
    #     save_dir=LOG_dir,  # Where to save logs
    #     name="cel_vqgan_experiment",  # Experiment name
    #     version="run_1"  # Run ID (increment for multiple runs)
    # )


    trainer = pl.Trainer(max_epochs=exp_config.epochs,
                         precision=16 if exp_config.use_amp else 32,
                        #  callbacks=callbacks,
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy="ddp" if args.num_nodes > 1 or args.num_gpus > 1 else None,
                         accumulate_grad_batches=exp_config.update_every,
                         limit_val_batches=1.0,
                        #  logger=logger
                        logger=pl.loggers.TensorBoardLogger("os.environ.get("USER_HOME", "")/projects/vqvae-transformer/VQCT-main/logs/ICCV/CUB-ade-3-5-25-ICCV"),
                        callbacks=[checkpoint_callback])


     # Run Training or Testing
    if args.test:
        print('🔍 Running model in test mode...')
        ckpt_path = args.ckpt if args.ckpt else "checkpoints/best_model.ckpt"  #Use provided or default checkpoint
        trainer.test(model, data, ckpt_path=ckpt_path)
    else:
        print('🚀 Starting training...')
        trainer.fit(model, data)

    # Train
    print('start training')

    trainer.fit(model, data)
    # trainer.fit(model,data,ckpt_path='checkpoint/ade_cvq/last.ckpt')