import os
import dotenv
dotenv.load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter
from timm import create_model
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from d2l import torch as d2l



class GCN(torch.nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GAT(torch.nn.Module):
    def __init__(self, in_dim,hid_dim,out_dim, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hid_dim*heads, out_dim)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=False,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q





# class SmallGCN(nn.Module):
#     def __init__(self, input_channels=128, output_channels=128):
#         super(SmallGCN, self).__init__()
#         # Define two GCN layers for local feature extraction
#         self.conv1 = GCNConv(input_channels, 64)  # First layer reduces channels
#         self.conv2 = GCNConv(64, output_channels)  # Second layer keeps size small

#     def forward(self, x, edge_index):
#         # Apply GCN layers on input graph data
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x



# class TinyViT(nn.Module):
#     def __init__(self, embed_dim=128, num_classes=1000):
#         super(TinyViT, self).__init__()
#         # Load a pre-trained TinyViT model from timm (or you can use any other small transformer)
#         self.vit = create_model('vit_tiny_patch16_224', pretrained=True)
#         # Optionally, change the output layer for your task (e.g., classification)
#         self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

#     def forward(self, x):
#         # Extract global features
#         return self.vit(x)

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        # self.embedding = torch.load('os.environ.get("USER_HOME", "")/projects/vqvae-transformer/noun_vectors_512_64.pth')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'device = ', device)
        self.global_codebook = nn.Parameter(torch.randn(n_e, int(e_dim/2)))  # Codebook of transformer for global 
        ## self.local_codebook = nn.Parameter(torch.randn(n_e, int(e_dim/2)))  # Codebook for local

        self.local_codebook  = nn.Embedding(self.n_e, int(e_dim/2))
        self.local_codebook.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        encoder_layer = TransformerEncoderLayer(d_model= int(self.e_dim/2), nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=6)



        # # self.embedding = self.embedding.to(device)
        # self.embedding_layer = nn.Embedding(self.n_e, self.e_dim)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.e_dim, nhead=4)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        # # self.embedding = self.embedding_layer.weight


        # pretrained_embeddings = torch.load('os.environ.get("USER_HOME", "")/projects/vqvae-transformer/filtered_clip_embeddings_64_4096.pth')
        # # pretrained_embeddings = torch.load('os.environ.get("USER_HOME", "")/projects/vqvae-transformer/filtered_clip_embeddings_num_512_dim_256.pth')
        # # pretrained_embeddings = torch.load('os.environ.get("USER_HOME", "")/projects/vqvae-transformer/filtered_clip_embeddings_num_1024_dim_256.pth')
        # # pretrained_embeddings = torch.load('os.environ.get("USER_HOME", "")/projects/vqvae-transformer/codebook_512.pth')
        
        
        # self.embedding = nn.Parameter(pretrained_embeddings)

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        global_features = z[:,  :, :, :128]  
        local_features = z[:, :, :, 128:]   
        print(f'e dim size', self.e_dim)

        global_features_flattened = global_features.view(-1, int( self.e_dim/2))
        local_features_flattened = local_features.view(-1, int(self.e_dim/2))


        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        global_embedding = self.transformer(self.global_codebook)  # (B, L, D)

        # local_embedding = self.gcn(local_features_flattened, edge_index)  # (B, L, D)

        local_embedding = self.local_codebook.weight


        # transformed_vectors = self.transformer(self.embedding)
        print(f'global_embedding shape: ', global_embedding.shape)
        print(f'local_embedding shape: ', local_embedding.shape)



        d_global = torch.sum(global_features_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(global_embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', global_features_flattened, rearrange(global_embedding, 'n d -> d n'))

        d_local = torch.sum(local_features_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(local_embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', local_features_flattened, rearrange(local_embedding, 'n d -> d n'))

        
        
        
        
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices_global = torch.argmin(d_global, dim=1)
        min_encoding_indices_local = torch.argmin(d_local, dim=1)


        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q_global = self.quantize(min_encoding_indices_global,global_embedding ).view(global_features.shape)
        z_q_local = self.quantize(min_encoding_indices_local,local_embedding ).view(local_features.shape)

        z_q = torch.cat([z_q_global, z_q_local], dim=-1)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss_global = self.beta * torch.mean((z_q_global.detach()-global_features)**2) + \
                   torch.mean((z_q_global - global_features.detach()) ** 2)

            loss_local = self.beta * torch.mean((z_q_local.detach()-local_features)**2) + \
                   torch.mean((z_q_local - local_features.detach()) ** 2)
        else:
            loss_global = torch.mean((z_q_global.detach()-global_features)**2) + self.beta * \
                   torch.mean((z_q_global - global_features.detach()) ** 2)
            loss_local = torch.mean((z_q_local.detach()-local_features)**2) + self.beta * \
                   torch.mean((z_q_local - local_features.detach()) ** 2)



        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        
        
        print(f' normal gan trans z_q shape:', z_q.shape)
        return z_q, loss_local, loss_global, (perplexity,  min_encoding_indices_global, min_encoding_indices_local)
    

    def quantize(self, encoding_indices, transformed_vectors):
        # return F.embedding(encoding_indices, self.embeddings.t())
        return F.embedding(encoding_indices, transformed_vectors)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
                remap=None, unknown_index="random"):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'


        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened            
            self.embedding.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)



class NLPVectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        # nlp prior
        with open('./codebook_priors/nlp_word_knowledge.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        edges_ = edges_ + [(u, u) for u in range(n)]

        vectors = []
        for name in code_book:
            if name in graph['adj_code_book_vec'].keys():
                vectors.append(graph['adj_code_book_vec'][name])
            elif name in graph['noun_code_book_vec'].keys():
                vectors.append(graph['noun_code_book_vec'][name])

        word_vectors = torch.stack(vectors, dim=0)
        word_vectors = F.normalize(word_vectors, dim=0)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=300, out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )

        self.embedding = nn.Embedding(self.n_e-len(vectors), self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.word_vectors = word_vectors

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        # embedding 
        embedding_weight = torch.cat([self.code_book_mapping(self.word_vectors.type_as(z)), self.embedding.weight], dim=0)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = F.embedding(min_encoding_indices, embedding_weight).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class MCNLPVectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        # nlp prior
        with open('./codebook_priors/nlp_word_knowledge.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        edges_ = edges_ + [(u, u) for u in range(n)]

        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0)
        adj_vectors = F.normalize(adj_vectors, dim=0)

        noun_vectors = torch.stack(noun_vectors, dim=0)
        noun_vectors = F.normalize(noun_vectors, dim=0)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=300, out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )

        self.adj_embedding = nn.Embedding(self.n_e-len(adj_vectors), self.e_dim)
        self.adj_embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.noun_embedding = nn.Embedding(self.n_e-len(noun_vectors), self.e_dim)
        self.noun_embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.adj_vectors = adj_vectors
        self.noun_vectors = noun_vectors

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten

        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = torch.cat([self.code_book_mapping(self.adj_vectors.type_as(z)), self.adj_embedding.weight], dim=0)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices, embedding_weight).view(z_adj.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = torch.cat([self.code_book_mapping(self.noun_vectors.type_as(z)), self.noun_embedding.weight], dim=0)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])

        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class TopkNLPVectorQuantizer3(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        # nlp prior
        with open('./codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        edges_ = edges_ + [(u, u) for u in range(n)]

        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )

        self.adj_vectors = adj_vectors
        self.noun_vectors = noun_vectors

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten

        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = self.code_book_mapping(self.adj_vectors.type_as(z))

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = self.code_book_mapping(self.noun_vectors.type_as(z))

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])

        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class vqct_VectorQuantizer3(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = min_encoding_indices1.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = min_encoding_indices2.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])


        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, torch.cat([min_encoding_indices1,min_encoding_indices2.unsqueeze(1)],dim=1))
        # min_encoding_indices1 = min_encoding_indices1.reshape(b, -1, min_encoding_indices1.shape[-1])
        # min_encoding_indices2 = min_encoding_indices2.unsqueeze(1).reshape(b, -1, 1)
        # idds = torch.cat([min_encoding_indices1,min_encoding_indices2+self.adj_len],dim=-1)
        # idds = idds.permute(0, 2, 1)
        # return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, idds.reshape(-1))
    

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], 2, -1, z_q.shape[-1])
        z_q = torch.cat([z_q[:,0,: ,:], z_q[:,1,:,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class CVQVectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta=0.25, distance='l2', 
                 anchor='probrandom', first_batch=False, contras_loss=True):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # import pdb
        # pdb.set_trace()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # if self.remap is not None:
            # indices = indices.reshape(shape[0],-1) # add batch axis
            # indices = self.unmap_to_all(indices)
            # indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features




class Topk_GCN_NLPVectorQuantizer3_small(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('./codebook_priors/big_nlp_word_knowledge_clip_celehq.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 300:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 300:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = min_encoding_indices1.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = min_encoding_indices2.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])


        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, torch.cat([min_encoding_indices1,min_encoding_indices2.unsqueeze(1)+self.adj_len],dim=1))


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], -1, 2, z_q.shape[-1])
        z_q = torch.cat([z_q[:,:,0,:], z_q[:,:,1,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class GumbelQuantize_gcn(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=False,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)
        
        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.embedding_dim,out_dim=self.embedding_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.noun_len = noun_vectors.shape[0]
        self.proj_adj = nn.Conv2d(self.embedding_dim, adj_vectors.shape[0], 1)
        self.proj_noun = nn.Conv2d(self.embedding_dim, noun_vectors.shape[0], 1)
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        total_embedding_weight = self.gcn(self.data.to(z.device))
        logits_adj = self.proj_adj(z[:, :self.embedding_dim, :, :])
        logits_noun = self.proj_noun(z[:, self.embedding_dim:, :, :])
        # logits = self.proj(z)
        # if self.remap is not None:
        #     # continue only with used logits
        #     full_zeros = torch.zeros_like(logits)
        #     logits = logits[:,self.used,...]

        soft_one_hot_adj = F.gumbel_softmax(logits_adj, tau=temp, dim=1, hard=hard)
        soft_one_hot_noun = F.gumbel_softmax(logits_noun, tau=temp, dim=1, hard=hard)
        # if self.remap is not None:
        #     # go back to all entries but unused set to zero
        #     full_zeros[:,self.used,...] = soft_one_hot
        #     soft_one_hot = full_zeros
        z_q_adj = einsum('b n h w, n d -> b d h w', soft_one_hot_adj, total_embedding_weight[:self.adj_len].type_as(z))
        z_q_noun = einsum('b n h w, n d -> b d h w', soft_one_hot_noun, total_embedding_weight[self.adj_len:].type_as(z))

        # + kl divergence to the prior loss
        qy_adj = F.softmax(logits_adj, dim=1)
        qy_noun = F.softmax(logits_noun, dim=1)
        diff_adj = self.kl_weight * torch.sum(qy_adj * torch.log(qy_adj * self.adj_len + 1e-10), dim=1).mean()
        diff_noun = self.kl_weight * torch.sum(qy_noun * torch.log(qy_noun * self.noun_len + 1e-10), dim=1).mean()
        diff  = diff_adj + diff_noun
        ind = soft_one_hot_adj.argmax(dim=1)
        z_q = torch.cat([z_q_adj, z_q_noun], dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits_adj
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q

class Topk_GCN_NLPVectorQuantizer3_no_knowledge(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)
        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0) 
        self.embedding = nn.Embedding(self.code.shape[0], self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.embedding.weight
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = min_encoding_indices1.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = min_encoding_indices2.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])


        # return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, torch.cat([min_encoding_indices1,min_encoding_indices2.unsqueeze(1)+self.adj_len],dim=1))
        min_encoding_indices1 = min_encoding_indices1.reshape(b, -1, min_encoding_indices1.shape[-1])
        min_encoding_indices2 = min_encoding_indices2.unsqueeze(1).reshape(b, -1, 1)
        idds = torch.cat([min_encoding_indices1,min_encoding_indices2+self.adj_len],dim=-1)
        idds = idds.permute(0, 2, 1)
        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, idds.reshape(-1))
    

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], 2, -1, z_q.shape[-1])
        z_q = torch.cat([z_q[:,0,: ,:], z_q[:,1,:,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q



class GCN_NLPVectorQuantizer4(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(total_embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(total_embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = F.embedding(min_encoding_indices, total_embedding_weight)
        z_q = z_q.view(z.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices =  min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.indices.device())

        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], -1, 2, z_q.shape[-1])
        z_q = torch.cat([z_q[:,:,0,:], z_q[:,:,1,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    
class Topk_GCN_NLPVectorQuantizer3_glove(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = min_encoding_indices1.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = min_encoding_indices2.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])


        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, torch.cat([min_encoding_indices1,min_encoding_indices2.unsqueeze(1)],dim=1))
        # min_encoding_indices1 = min_encoding_indices1.reshape(b, -1, min_encoding_indices1.shape[-1])
        # min_encoding_indices2 = min_encoding_indices2.unsqueeze(1).reshape(b, -1, 1)
        # idds = torch.cat([min_encoding_indices1,min_encoding_indices2+self.adj_len],dim=-1)
        # idds = idds.permute(0, 2, 1)
        # return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, idds.reshape(-1))
    

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], 2, -1, z_q.shape[-1])
        z_q = torch.cat([z_q[:,0,: ,:], z_q[:,1,:,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class Topk_GCN_NLPVectorQuantizer3_baseline(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10:
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = self.adj_vectors.type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = self.beta * torch.mean((z_adj_q - z_adj.detach()) ** 2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = min_encoding_indices1.reshape(
                z_adj_q.shape[0], z_adj_q.shape[2], z_adj_q.shape[3])

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = self.noun_vectors.type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss  + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss  + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = min_encoding_indices2.reshape(
                z_noun_q.shape[0], z_noun_q.shape[2], z_noun_q.shape[3])


        return torch.cat([z_adj_q, z_noun_q], dim=1), loss, (perplexity, min_encodings, torch.cat([min_encoding_indices1,min_encoding_indices2.unsqueeze(1)+self.adj_len],dim=1))


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.cuda())

        z_q = F.embedding(indices, embedding_weight)

        z_q = z_q.reshape(shape[0], -1, 2, z_q.shape[-1])
        z_q = torch.cat([z_q[:,:,0,:], z_q[:,:,1,:]], dim=-1)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
