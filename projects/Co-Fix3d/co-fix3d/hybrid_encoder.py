import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from .encoder_utils import *
from .time_utils import T
import torchvision.models.resnet as resnet
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer
from einops import rearrange
import matplotlib.pyplot as plt
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from .attention import MultiheadFlashAttention
from .ops.msmv_sampling.wrapper import msmv_sampling
# from .diffusion.wave_mix import WavePaint
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
# from .decoder_utils  import M

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    pos = pos_tensor * scale
    dim_t = torch.arange(128, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / 128 + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu",
                 attn_dropout=None, act_dropout=None, normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        # self.input_img = input_img
        self.self_attn = MultiheadFlashAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)
        # if self.input_img:
        # self.cross_attn = MultiheadFlashAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, tgt_mask=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        # residual = tgt
        # if self.normalize_before:
        #     tgt = self.norm2(tgt)
        # # if self.input_img:
        # q = self.with_pos_embed(tgt, query_pos_embed)
        # k = self.with_pos_embed(memory, pos_embed)
        # tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)[0]
        # tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos_embed=pos_embed, query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class HybridEncoder(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 use_encoder_idx=[1],
                 point_cloud_range=None,
                 input_img=False,
                 w=90,
                 h=90,
                 bias='auto',
                 num_encoder_layers=1):
        super(HybridEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers

        self.dconv1 = ConvModule(
            hidden_dim, hidden_dim,
            stride=2, kernel_size=3, padding=1, bias=bias,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )

        self.pc_range = point_cloud_range
        self.transformer_encoder1 = TransformerEncoderLayer(hidden_dim, 8, 0.1)
        self.bev_pos = self.create_2D_grid(w, h)
        self.pos_embedding = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / x_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return nn.Parameter(coord_base, requires_grad=False)


    def forward(self, pts_feats):
        b, _, w, h = pts_feats.shape
        bev_pos_2 = gen_sineembed_for_position(self.bev_pos.repeat(b, 1, 1))

        bev_pos_2 = self.pos_embedding(bev_pos_2)

        tgt = pts_feats

        tgt = self.dconv1(tgt)

        tgt = tgt.flatten(2).permute(0, 2, 1)
        tgt = self.transformer_encoder1(tgt, query_pos_embed=bev_pos_2)

        # tgt = self.transformer_encoder2(tgt, memory, tgt_mask=None, memory_mask=None,
        #                                 pos_embed=pos_embed, query_pos_embed=bev_pos_2)
        # #
        # tgt = self.transformer_encoder3(tgt, memory, tgt_mask=None, memory_mask=None,
        #                                 pos_embed=pos_embed, query_pos_embed=bev_pos_2)

        tgt = tgt.permute(0, 2, 1).reshape(-1, self.hidden_dim, w // 2, h // 2).contiguous()
        # tgt = self.fpn_blocks1(torch.concat([tgt, hs.pop()], dim=1))

        # tgt = F.interpolate(tgt, scale_factor=2., mode='bilinear').contiguous()

        # tgt = tgt.flatten(2).permute(0, 2, 1)
        # tgt = self.transformer_encoder3(tgt, memory, tgt_mask=None, memory_mask=None,
        #                                 pos_embed=pos_embed, query_pos_embed=bev_pos_1)
        # tgt = tgt.permute(0, 2, 1).reshape(-1, self.hidden_dim, w, h)
        # tgt = self.fpn_blocks1(torch.concat([tgt, refine_feat], dim=1))
        return tgt
