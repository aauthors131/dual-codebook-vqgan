import os
import dotenv
dotenv.load_dotenv()

import os
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint
from main import instantiate_from_config
import torchvision.utils as vutils
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize_trans_graph import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss_local, emb_loss_global , info = self.quantize(h)
        return quant, emb_loss_local, emb_loss_global, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff_local, diff_global,  info_= self.encode(input)
        dec = self.decode(quant)
        return dec, diff_local, diff_global, info_

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss_local, qloss_global, info_r = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss_local, qloss_global, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss_local, qloss_global, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    # def validation_step(self, batch, batch_idx):
    #     x = self.get_input(batch, self.image_key)
    #     xrec, qloss = self(x)
    #     aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")

    #     # discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
    #     #                                     last_layer=self.get_last_layer(), split="val")
    #     rec_loss = log_dict_ae["val/rec_loss"]
    #     self.log("val/rec_loss", rec_loss,
    #                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    #     del log_dict_ae["val/rec_loss"]
    #     # self.log("val/aeloss", aeloss,
    #     #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    #     self.log_dict(log_dict_ae)
    #     # self.log_dict(log_dict_disc)
    #     return self.log_dict

    def validation_step(self, batch, batch_idx):
        print(f" hoo hoo Validation batch ID: {batch_idx}")
        x = self.get_input(batch, self.image_key)
        xrec, qloss_local, qloss_global, info_r= self(x)
        aeloss, log_dict_ae = self.loss(qloss_local, qloss_global, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        # Save images only for epochs divisible by 10
        if (self.current_epoch % 1 == 0) and (batch_idx ==0):
            # Directories for saving images
            inputs_dir = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "val_inputs", f"epoch_{self.current_epoch}")
            rec_embed_indices_local = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "val_rec_embed_indices_local")
            rec_embed_indices_global = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "val_rec_embed_indices_global")
            reconstructions_dir = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "val_reconstructions", f"epoch_{self.current_epoch}")
            os.makedirs(inputs_dir, exist_ok=True)
            os.makedirs(reconstructions_dir, exist_ok=True)
            os.makedirs(rec_embed_indices_local, exist_ok=True)
            os.makedirs(rec_embed_indices_global, exist_ok=True)


            # Save original and reconstructed images
            x = x.detach().cpu()
            xrec = xrec.detach().cpu()
            indices_local = info_r[2].detach().cpu()
            indices_global = info_r[1].detach().cpu()
            print(f'indices local shape1:', indices_local.shape)
            print(f'indices global shape1:', indices_global.shape)
            print(f'batch :', x.shape[0])
            indices_local = indices_local.view(x.shape[0], -1)
            indices_global = indices_global.view(x.shape[0], -1)

            for i in range(x.shape[0]):
                # Save original image
                vutils.save_image(
                    x[i], 
                    os.path.join(inputs_dir, f"inputs_epoch{self.current_epoch}_batch_id{batch_idx}_image_id{i}.png"), 
                    normalize=True
                )
                # Save reconstructed image
                vutils.save_image(
                    xrec[i], 
                    os.path.join(reconstructions_dir, f"reconstructions_epoch{self.current_epoch}_batch_id{batch_idx}_image_id{i}.png"), 
                    normalize=True
                )
                # save latent indices
                # file_path = os.path.join(rec_embed_indices, f"encoding_epoch{self.current_epoch}_batch_id{batch_idx}_image_id{i}.npy")
                # np.save(file_path, indices[i])
                # save latent indices
                file_path = os.path.join(rec_embed_indices_local, f"encoding_batch_id{batch_idx}_image_id{i}.npy")
                np.save(file_path, indices_local[i])

                # save latent indices
                file_path = os.path.join(rec_embed_indices_global, f"encoding_batch_id{batch_idx}_image_id{i}.npy")
                np.save(file_path, indices_global[i])


        # Log reconstruction loss
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]

        # Log remaining metrics
        self.log_dict(log_dict_ae)

        # Return log dictionary
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        """Runs the model in evaluation mode on the test dataset."""
        print(f"🔍 Running test step for batch {batch_idx}")

        # Get input image
        x = self.get_input(batch, self.image_key)
        xrec, qloss_local, qloss_global, info_r = self(x)

        # Compute test loss
        test_loss, log_dict_test = self.loss(qloss_local, qloss_global, x, xrec, 0, self.global_step,
                                             last_layer=self.get_last_layer(), split="test")




        # Log test loss
        self.log("test/loss", test_loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

        # Optionally save test reconstructions
        inputs_dir = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "test_inputs")
        rec_embed_indices_local = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "test_rec_embed_indices_local")
        rec_embed_indices_global = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "test_rec_embed_indices_global")
        reconstructions_dir = os.path.join("os.environ.get("DATA_PATH", "")/CEL_VQGAN-trans_graph-256-512-bs8-2-20-25_newversion_2gpu", "test_reconstructions")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(reconstructions_dir, exist_ok=True)
        os.makedirs(rec_embed_indices_local, exist_ok=True)
        os.makedirs(rec_embed_indices_global, exist_ok=True)


        x = x.detach().cpu()
        xrec = xrec.detach().cpu()
        indices_local = info_r[2].detach().cpu()
        indices_global = info_r[1].detach().cpu()
        print(f'indices local shape1:', indices_local.shape)
        print(f'indices global shape1:', indices_global.shape)
        print(f'batch :', x.shape[0])
        indices_local = indices_local.view(x.shape[0], -1)
        indices_global = indices_global.view(x.shape[0], -1)


        # print(f'indices shape2:', indices.shape)
        # print(f'x shape:', x.shape)

        for i in range(x.shape[0]):
                # Save original image
                vutils.save_image(
                    x[i], 
                    os.path.join(inputs_dir, f"inputs_batch_id{batch_idx}_image_id{i}.png"), 
                    normalize=True
                )
                # Save reconstructed image
                vutils.save_image(
                    xrec[i], 
                    os.path.join(reconstructions_dir, f"reconstructions_batch_id{batch_idx}_image_id{i}.png"), 
                    normalize=True
                )
                # save latent indices
                file_path = os.path.join(rec_embed_indices_local, f"encoding_batch_id{batch_idx}_image_id{i}.npy")
                np.save(file_path, indices_local[i])

                # save latent indices
                file_path = os.path.join(rec_embed_indices_global, f"encoding_batch_id{batch_idx}_image_id{i}.npy")
                np.save(file_path, indices_global[i])


        # Log additional metrics
        self.log_dict(log_dict_test)

        return log_dict_test








    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ , b= self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        total_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss
    
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        # xrec, qloss = self(x, return_pred_indices=True)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        # discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           
    

class VQSegmentationModel_gumbel(GumbelVQ):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        total_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss
    
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
