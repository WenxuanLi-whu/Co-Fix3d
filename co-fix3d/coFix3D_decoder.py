# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import copy
import numpy as np
from mmengine.structures import InstanceData
import torch
from torch import nn
import torch.nn.functional as F
from .encoder_utils import *
from mmcv.cnn import ConvModule, build_conv_layer

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, rotation_3d_in_axis

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet.models.utils import multi_apply
from .decoder_utils import FFN
import math
from .ops.roiaware_pool3d import points_in_boxes_gpu
from .diffusion.wave_mix import WavePaint


def gen_sineembed_for_position_all(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, :] * scale
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=-1).flatten(-2)
    return pos_x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


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


@MODELS.register_module()
class CoFix3DDecoder(nn.Module):
    def __init__(self,
                 num_proposals=128,
                 hidden_channel=128,
                 hidden_channel_roi=512,
                 num_classes=4,
                 iter_heatmap=1,
                 # config for Transformer
                 num_decoder_layers=1,
                 num_heads=8,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 classaware_reg=False,
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_weight_heatmap=1.,
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 num_stage_proposals=None,
                 multiscale=False,
                 multistage_heatmap=False,
                 img_multistage_heatmap=3,
                 step=1,
                 reuse_first_heatmap=False,
                 extra_feat=False,
                 heatmap_box=False,
                 thin_heatmap_box=False,
                 loss_weight_separate_heatmap=0.2,
                 loss_weight_separate_bbox=0.5,
                 boxpos=None,
                 add_gt_groups=0,
                 add_gt_groups_noise='rect,1',
                 add_gt_groups_noise_box='gt',
                 gt_center_limit=None,
                 add_gt_pos_thresh=100.,
                 add_gt_pos_boxnoise_thresh=2.,
                 gt_query_loss_weight=1.,
                 bevpos=False,
                 input_pts=True,
                 input_img=True,
                 iterbev_wo_img=False,
                 mask_heatmap_mode='poscls',
                 roi_feats=0,
                 roi_dropout_rate=0.,
                 roi_expand_ratio=1.,
                 roi_based_reg=False,
                 decoder_cfg=dict(
                     type='DeformableDetrTransformerDecoder',
                     num_layers=6,
                     return_intermediate=False,
                     transformerlayers=dict(
                         type='DetrTransformerDecoderLayer',
                         attn_cfgs=[
                             dict(
                                 type='MultiheadAttention',
                                 embed_dims=128,
                                 num_heads=8,
                                 dropout=0.1),
                             dict(
                                 type='MultiScaleDeformableAttention',
                                 embed_dims=128,
                                 num_levels=1,
                                 num_points=6,
                                 num_heads=8,
                                 renorm_z=5., )
                         ],
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         ffn_cfgs=dict(
                             type='FFN',
                             embed_dims=128,
                             num_fcs=2,
                             act_cfg=dict(type='ReLU', inplace=True),
                         ),
                         operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                          'ffn', 'norm')))):
        super(CoFix3DDecoder, self).__init__()

        self.num_classes = num_classes
        self.num_proposals_ori = self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if num_stage_proposals is None:
            self.num_stage_proposals = [self.num_proposals] * self.num_decoder_layers
        else:
            self.num_stage_proposals = num_stage_proposals
        self.cumsum_proposals = np.asarray([0] + list(np.cumsum(self.num_stage_proposals)))
        self.multiscale = multiscale
        self.iter_heatmap = iter_heatmap
        self.multistage_heatmap = multistage_heatmap
        self.img_multistage_heatmap = img_multistage_heatmap
        self.reuse_first_heatmap = reuse_first_heatmap
        self.extra_feat = extra_feat
        self.boxpos = boxpos
        self.gt_query_loss_weight = gt_query_loss_weight
        self.bevpos = bevpos
        self.input_img = input_img
        self.input_pts = input_pts
        self.iterbev_wo_img = iterbev_wo_img
        self.heatmap_box = heatmap_box
        self.thin_heatmap_box = thin_heatmap_box
        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_separate_heatmap = loss_weight_separate_heatmap
        self.loss_weight_separate_bbox = loss_weight_separate_bbox
        self.step = step

        if self.extra_feat:
            self.extra_output = ConvBNReLU(hidden_channel, hidden_channel, kernel_size=3, norm_layer=nn.BatchNorm2d, activation_layer=None)
        if self.multiscale:
            self.dconv = ConvModule(
                hidden_channel, hidden_channel,
                stride=2, kernel_size=3, padding=1, bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                # act_cfg=None
            )
            self.dconv2 = ConvModule(
                hidden_channel, hidden_channel,
                stride=2, kernel_size=3, padding=1, bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                # act_cfg=None
            )

        # self.midconv = nn.Sequential(nn.Conv2d(hidden_channel, hidden_channel, 5, groups=hidden_channel, padding="same"))
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # if not (self.input_pts and self.input_img):
        #
        # else:
        #     self.fusion_bev = nn.ModuleList()
        # if self.input_pts and self.input_img:
        #     self.reduce_conv = nn.ModuleList()
        #     ff_channel = 2 * hidden_channel
        #     self.reduce_conv.append(nn.Sequential(
        #         nn.Conv2d(ff_channel, hidden_channel, 3, padding=1),
        #         nn.BatchNorm2d(hidden_channel)))
        # else:
        #     ff_channel = hidden_channel
        self.inpaint_bev = nn.ModuleList()
        for i in range(self.multistage_heatmap):
            self.inpaint_bev.append(WavePaint(
                num_modules=1,
                depth=4,
                mult=2,
                ff_channel=hidden_channel, final_dim=hidden_channel,
                dropout=0.1))
        # for i in range(self.multistage_heatmap):
        #     if not (self.input_pts and self.input_img):
        #         self.inpaint_bev.append(WavePaint(
        #             num_modules=1,
        #             depth=4,
        #             mult=2,
        #             ff_channel=ff_channel, final_dim=ff_channel,
        #             dropout=0.1))
        #     if self.input_pts and self.input_img:
        #         self.fusion_bev.append(WavePaint(
        #             num_modules=1,
        #             depth=4,
        #             mult=2,
        #             ff_channel=ff_channel, final_dim=ff_channel,
        #             dropout=0.1))
        #         self.reduce_conv.append(nn.Sequential(
        #             nn.Conv2d(ff_channel, hidden_channel, 3, padding=1),
        #             nn.BatchNorm2d(hidden_channel)))

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.gt_center_limit = gt_center_limit
        self.add_gt_pos_thresh = add_gt_pos_thresh
        self.add_gt_pos_boxnoise_thresh = add_gt_pos_boxnoise_thresh

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.mask_heatmap_mode = mask_heatmap_mode

        self.roi_feats = roi_feats
        if not roi_feats:
            assert not roi_based_reg
        self.roi_based_reg = roi_based_reg
        if isinstance(roi_expand_ratio, float):
            self.roi_expand_ratio = [roi_expand_ratio] * self.num_decoder_layers
        else:
            self.roi_expand_ratio = roi_expand_ratio

        if self.roi_feats:
            fc_list = []
            pre_channel = self.roi_feats ** 2 * hidden_channel * (3 if self.multiscale else 1)
            num_roi_layers = 3
            for i in range(num_roi_layers):
                chl = hidden_channel_roi if i < num_roi_layers - 1 else hidden_channel
                fc_list.extend([
                    nn.Linear(pre_channel, chl, bias=False),
                    nn.BatchNorm1d(chl),
                    nn.ReLU(inplace=True)
                ])
                if roi_dropout_rate > 1e-4 and i != -1:
                    fc_list.append(nn.Dropout(roi_dropout_rate))
                pre_channel = chl
            self.roi_mlp = nn.Sequential(*fc_list)

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))

            heatmap_head = nn.Sequential(*layers)
            self.heatmap_head = nn.ModuleList()
            # self.img_heatmap_head = nn.ModuleList()
            # for i in range(self.img_multistage_heatmap):
            #     self.img_heatmap_head.append(copy.deepcopy(heatmap_head))

            for i in range(self.multistage_heatmap):
                self.heatmap_head.append(copy.deepcopy(heatmap_head))


            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=False)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()

        self.decoder_cfg = decoder_cfg
        self.pos_embed_learned = nn.ModuleList()
        self.box_pos_embed_learned = nn.ModuleList()
        self.inter_reference_reg_branches = nn.ModuleList()

        for i in range(self.num_decoder_layers):
            self.decoder.append(build_transformer_layer_sequence(self.decoder_cfg))
            self.pos_embed_learned.append(nn.Sequential(
                nn.Linear(256, hidden_channel),
                nn.LayerNorm(hidden_channel),
                nn.ReLU(),
                nn.Linear(hidden_channel, hidden_channel),
                nn.LayerNorm(hidden_channel),
                nn.ReLU(),
            ))
            # self.pos_embed_learned.append(MLP(256, hidden_channel, hidden_channel, 2))
            if self.boxpos == 'xywlr':
                if i > 0:
                    self.box_pos_embed_learned.append(MLP(128 * 5, hidden_channel, hidden_channel, 2))
                else:
                    self.box_pos_embed_learned.append(None if not self.heatmap_box else MLP(128 * 5, hidden_channel, hidden_channel, 2))

        self.classaware_reg = classaware_reg
        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            if self.classaware_reg:
                for k, v in heads.items():
                    heads[k] = [v[0] * self.num_classes, v[1]]
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.add_gt_groups = add_gt_groups
        self.add_gt_groups_noise = add_gt_groups_noise
        self.add_gt_groups_noise_box = add_gt_groups_noise_box

        self.init_weights()
        self._init_assigner_sampler()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def generate_gt_groups(self, query_feat, query_pos, query_heatmap_score, lidar_feat, lidar_feat_flatten, bev_pos, heatmap,
                           gt_bboxes_3d, gt_labels_3d, dense_heatmap_boxes=None, query_box=None):

        batch_size = len(gt_bboxes_3d)
        point_cloud_range = torch.as_tensor(self.train_cfg['point_cloud_range'], device='cuda')

        # compute noised gt groups mask
        batch_valid_gt_mask = torch.zeros((batch_size, self.max_num_gts * self.add_gt_groups), dtype=bool, device=query_pos.device)
        for batch_idx in range(batch_size):
            for group_i in range(self.add_gt_groups):
                batch_valid_gt_mask[batch_idx, group_i * self.max_num_gts:group_i * self.max_num_gts + self.num_gts[batch_idx]] = 1

        # compute noised gt groups
        batch_gt_pos = []
        batch_gt_query_labels = []
        batch_gt_bev_corners = []
        batch_gt_bboxes_3d = []
        for batch_idx in range(batch_size):
            gs_box = gt_bboxes_3d[batch_idx].tensor.cuda()
            gs = F.pad(gs_box[:, :2], pad=(0, 0, 0, self.max_num_gts - self.num_gts[batch_idx]))
            gs_labels = F.pad(gt_labels_3d[batch_idx], pad=(0, self.max_num_gts - self.num_gts[batch_idx]), value=self.num_classes)  # set background class
            gs_bev_corners = F.pad(gt_bboxes_3d[batch_idx].corners.cuda().reshape(-1, 4, 2, 3)[:, :4, 0, :2], pad=(0, 0, 0, 0, 0, self.max_num_gts - self.num_gts[batch_idx]))

            gs = gs.repeat(self.add_gt_groups, 1)
            gs_labels = gs_labels.repeat(self.add_gt_groups)
            gs_bev_corners = gs_bev_corners.repeat(self.add_gt_groups, 1, 1)

            gs_box = self.bbox_coder.encode(gs_box)  # same as CenterPoint (TODO(not checked for z !!!))
            gs_box = F.pad(gs_box, pad=(0, 0, 0, self.max_num_gts - self.num_gts[batch_idx]))
            gs_box = gs_box.repeat(self.add_gt_groups, 1)

            meta_noise = (torch.rand((len(gs_bev_corners), 2), device='cuda') * 2 - 1)
            if self.add_gt_groups_noise.startswith('rect'):
                gs_bev_corners_rect = torch.cat([gs_bev_corners.min(dim=1)[0], gs_bev_corners.max(dim=1)[0]], dim=1)
                noise = float(self.add_gt_groups_noise.split(',')[1]) * meta_noise
                box_center_noise = (gs_bev_corners_rect[:, 2:4] - gs_bev_corners_rect[:, 0:2]) / 2. * noise
                gs = gs + box_center_noise
            elif self.add_gt_groups_noise.startswith('cam'):
                gs_vec = gs / (gs.norm(dim=1) + 1e-6)[:, None]
                gs_vec_orth = torch.stack([gs_vec[:, 1], -gs_vec[:, 0]], dim=1)
                gs_mat = torch.stack([gs_vec, gs_vec_orth], dim=-1)
                gs_bev_corners_trans = gs_bev_corners.matmul(gs_mat)

                gs_bev_corners_rect = torch.cat([gs_bev_corners_trans.min(dim=1)[0], gs_bev_corners_trans.max(dim=1)[0]], dim=1)
                # gs_bev_corners_rect[:, [1, 3]] = 0. # only cam
                scale = self.add_gt_groups_noise.split(',')[1]
                if '-' in scale:
                    scale = torch.as_tensor(list(map(float, scale.split('-'))), device='cuda')[None]
                else:
                    scale = float(scale)

                noise = scale * meta_noise
                box_center_noise = (gs_bev_corners_rect[:, 2:4] - gs_bev_corners_rect[:, 0:2]) / 2. * noise
                gs = gs + box_center_noise[:, None].matmul(gs_mat)[:, 0]
            elif self.add_gt_groups_noise.startswith('box'):
                pseudo_w = gs_bev_corners[:, 2] - gs_bev_corners[:, 0]
                pseudo_h = gs_bev_corners[:, 1] - gs_bev_corners[:, 0]
                pseudo_w_len = pseudo_w.norm(dim=-1)
                pseudo_h_len = pseudo_h.norm(dim=-1)
                noise = float(self.add_gt_groups_noise.split(',')[1]) * meta_noise

                box_center_noise = (pseudo_w / 2. * noise[:, 0:1]) + (pseudo_h / 2. * noise[:, 1:2])
                gs = gs + box_center_noise
            else:
                raise NotImplementedError

            gs[:, 0] = gs[:, 0].clip(min=point_cloud_range[0] + 1e-6, max=point_cloud_range[3] - 1e-5)
            gs[:, 1] = gs[:, 1].clip(min=point_cloud_range[1] + 1e-6, max=point_cloud_range[4] - 1e-5)
            gs = (gs - point_cloud_range[None, :2]) / (point_cloud_range[None, 3:5] - point_cloud_range[None, :2])
            assert self.initialize_by_heatmap
            W, H = lidar_feat.shape[-2:]  # (W, H) <---> (y, x)
            gs = gs * torch.as_tensor([H, W], dtype=gs.dtype, device='cuda')
            gs[:, 0] = gs[:, 0].clip(max=H - 1, min=0)
            gs[:, 1] = gs[:, 1].clip(max=W - 1, min=0)
            gs = gs.to(torch.int64)  # center int

            ### set the labels ####
            # determine the positive query with distance
            positive_mask = (box_center_noise.norm(dim=1) < self.add_gt_pos_thresh)
            gs_labels[torch.logical_not(positive_mask)] = self.num_classes  # set background class

            # determine the positive query with distance along box
            if self.add_gt_groups_noise.startswith('box'):
                positive_mask = noise.norm(dim=1) < self.add_gt_pos_boxnoise_thresh
                gs_labels[torch.logical_not(positive_mask)] = self.num_classes  # set background class

            batch_gt_pos.append(gs)
            batch_gt_query_labels.append(gs_labels)
            batch_gt_bboxes_3d.append(gs_box)

        batch_gt_pos = torch.stack(batch_gt_pos, dim=0)
        batch_gt_query_labels = torch.stack(batch_gt_query_labels, dim=0)
        batch_gt_bboxes_3d = torch.stack(batch_gt_bboxes_3d, dim=0)

        batch_gt_bev_pos = bev_pos.gather(index=(batch_gt_pos[:, :, 1] * H + batch_gt_pos[:, :, 0])[..., None].expand(-1, -1, bev_pos.shape[-1]), dim=1)
        batch_gt_heatmap_score = heatmap.gather(index=(batch_gt_pos[:, :, 1] * H + batch_gt_pos[:, :, 0])[:, None, :].expand(-1, self.num_classes, -1), dim=-1)
        batch_gt_query_feat = lidar_feat_flatten.gather(index=(batch_gt_pos[:, None, :, 1] * H + batch_gt_pos[:, None, :, 0]).expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
        if len(self.add_gt_groups_noise.split(',')) > 2 and self.add_gt_groups_noise.split(',')[2] == 'heatmap':
            batch_gt_heatmap_class = batch_gt_heatmap_score.argmax(dim=1)
            batch_gt_one_hot = F.one_hot(batch_gt_heatmap_class, num_classes=self.num_classes + 1).permute(0, 2, 1)
        # useles as the prediction only extracts one-hot label
        elif len(self.add_gt_groups_noise.split(',')) > 2 and self.add_gt_groups_noise.split(',')[2] == 'heatmapcls':
            batch_gt_one_hot = batch_gt_heatmap_score
        else:
            batch_gt_one_hot = F.one_hot(batch_gt_query_labels, num_classes=self.num_classes + 1).permute(0, 2, 1)
        batch_gt_one_hot = batch_gt_one_hot[:, :self.num_classes]
        batch_gt_query_cat_encoding = self.class_encoding(batch_gt_one_hot.float())
        batch_gt_query_feat += batch_gt_query_cat_encoding

        query_pos = torch.cat([query_pos, batch_gt_bev_pos * batch_valid_gt_mask[..., None].float()], dim=1)
        query_feat = torch.cat([query_feat, batch_gt_query_feat * batch_valid_gt_mask[:, None, :].float()], dim=2)
        query_heatmap_score = torch.cat([query_heatmap_score, batch_gt_heatmap_score * batch_valid_gt_mask[:, None, :].float()], dim=2)

        if dense_heatmap_boxes is not None:
            if self.add_gt_groups_noise_box == 'pred':
                batch_gt_heatmap_box = dense_heatmap_boxes.gather(index=
                                                                  (batch_gt_query_labels.clip(max=self.num_classes - 1) * H * W + batch_gt_pos[:, :, 1] * H + batch_gt_pos[:, :, 0]
                                                                   )[:, None].expand(-1, dense_heatmap_boxes.shape[1], -1), dim=-1)
            elif self.add_gt_groups_noise_box == 'gt':
                batch_gt_heatmap_box = batch_gt_bboxes_3d.transpose(1, 2)
                # x y shift
                batch_gt_heatmap_box[:, :2, :] = (batch_gt_bev_pos * batch_valid_gt_mask[..., None].float()).transpose(1, 2)
            elif self.add_gt_groups_noise_box == 'gtnoise':
                batch_gt_heatmap_box = batch_gt_bboxes_3d.transpose(1, 2)
                meta_noise = (torch.rand(batch_gt_heatmap_box[:, 2:, :].shape, device='cuda') * 2 - 1)

                # x y shift
                batch_gt_heatmap_box[:, :2, :] = (batch_gt_bev_pos * batch_valid_gt_mask[..., None].float()).transpose(1, 2)  # x, y
                # length along x y z
                batch_gt_heatmap_box[:, [2], :] = batch_gt_heatmap_box[:, [2], :] + meta_noise[:, [0]] * batch_gt_heatmap_box[:, [5]].exp()  # z
                batch_gt_heatmap_box[:, [3, 4, 5], :] = torch.log(batch_gt_heatmap_box[:, [3, 4, 5], :].exp() * (1 + meta_noise[:, [1, 2, 3]]).clip(min=0.1, max=3.) + 1e-6)  # l' w' h'
                batch_gt_heatmap_box[:, 6:7, :] = torch.sin(meta_noise[:, 4:5] * np.pi)  # sin
                batch_gt_heatmap_box[:, 7:8, :] = torch.cos(meta_noise[:, 4:5] * np.pi)  # cos
                if self.bbox_coder.code_size == 10:
                    batch_gt_heatmap_box[:, 8:10, :] = batch_gt_heatmap_box[:, 8:10, :] * (1 + meta_noise[:, 6:8])

            # set bg box = 0 as it will interact in transformer
            batch_gt_heatmap_box *= (batch_gt_query_labels != self.num_classes).float()[:, None]

            query_box = torch.cat([query_box, batch_gt_heatmap_box * batch_valid_gt_mask[:, None, :].float()], dim=2)
            return query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels, query_box
        else:
            return query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels

    def forward(self, pts_inputs, img_inputs, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, **input_kwargs):
        self.num_proposals = self.num_proposals_ori  # reset proposals
        lidar_feat = pts_inputs
        batch_size, C, H, W = lidar_feat.shape
        dense_heatmap_boxes = None
        query_box = None
        heatmap_train = []
        query_labels = []
        query_feats = []
        query_poses = []
        query_heatmap_scores = []
        multistage_masks = []
        padding = self.nms_kernel_size // 2
        acc_masks = torch.ones([batch_size, self.num_classes, H, W], device=lidar_feat.device)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        for i in range(self.multistage_heatmap):
            if self.step==1 or self.step ==3:
                refine_feat = self.inpaint_bev[i](lidar_feat)
            else:
                refine_feat = lidar_feat

            dense_heatmap = self.heatmap_head[i](refine_feat)
            heatmap_train.append(dense_heatmap)
            multistage_masks.append(acc_masks.clone())

            heatmap = dense_heatmap.detach().sigmoid()
            heatmap = heatmap * acc_masks.view(*heatmap.shape)
            local_max = torch.zeros_like(heatmap)

            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
            top_proposals = torch.topk(heatmap.view(batch_size, -1), k=self.num_proposals, dim=-1, largest=True, sorted=False).indices

            # masking by pooling
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat.view(batch_size, lidar_feat.shape[1], -1).gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat.shape[1], -1), dim=-1)
            query_labels.append(top_proposals_class)

            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())

            query_feat += query_cat_encoding
            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            query_heatmap_score = heatmap.view(batch_size, self.num_classes, -1).gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)

            query_feats.append(query_feat)
            query_poses.append(query_pos)
            query_heatmap_scores.append(query_heatmap_score)

            selected_mask = acc_masks.new_zeros(batch_size, self.num_classes * heatmap.shape[-1])
            selected_mask.scatter_(index=top_proposals, dim=1, src=torch.ones_like(top_proposals, dtype=acc_masks.dtype))
            selected_mask = selected_mask.reshape(*dense_heatmap.shape)
            # masking by pooling
            selected_mask_kernel = F.max_pool2d(selected_mask, kernel_size=self.nms_kernel_size, stride=1, padding=self.nms_kernel_size // 2)
            if self.test_cfg['dataset'] == 'nuScenes':  ## for Pedestrian & Traffic_cone in nuScenes
                selected_mask_kernel[:, 8:10] = F.max_pool2d(selected_mask[:, 8:10], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                selected_mask_kernel[:, 1:3] = F.max_pool2d(selected_mask[:, 1:3], kernel_size=1, stride=1, padding=0)
            acc_masks = acc_masks * (1. - selected_mask_kernel).view(*acc_masks.shape)

        self.query_labels = torch.cat(query_labels, dim=1)
        query_feat = torch.cat(query_feats, dim=2)
        query_pos = torch.cat(query_poses, dim=1)
        query_heatmap_score = torch.cat(query_heatmap_scores, dim=2)

        if self.num_decoder_layers == 2:
            self.num_proposals = self.num_proposals_ori * self.multistage_heatmap

        if self.multiscale:
            bev_pos_2 = self.create_2D_grid(lidar_feat.shape[2] // 2, lidar_feat.shape[2] // 2).repeat(batch_size, 1, 1).to(lidar_feat.device) * 2
            bev_pos_4 = self.create_2D_grid(lidar_feat.shape[2] // 4, lidar_feat.shape[2] // 4).repeat(batch_size, 1, 1).to(lidar_feat.device) * 4

        if self.training:
            self.num_gts = [i.shape[0] for i in gt_labels_3d]
            self.max_num_gts = max(self.num_gts)

        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  #
        query_labels = self.query_labels
        if self.training and self.add_gt_groups > 0:
            updated_gt_queries = self.generate_gt_groups(
                query_feat, query_pos, query_heatmap_score,
                lidar_feat, lidar_feat_flatten, bev_pos, heatmap.view(batch_size, heatmap.shape[1], -1),
                gt_bboxes_3d, gt_labels_3d, dense_heatmap_boxes=dense_heatmap_boxes, query_box=query_box)
            if dense_heatmap_boxes is None:
                query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels = updated_gt_queries
            else:
                query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels, query_box = updated_gt_queries

            query_labels = torch.cat([query_labels, batch_gt_query_labels], dim=1)

        if self.extra_feat:
            extra_lidar_feat = self.extra_output(lidar_feat)
        else:
            extra_lidar_feat = lidar_feat

        multiscale_inputs = [extra_lidar_feat]
        if self.multiscale:
            multiscale_inputs.append(self.dconv(multiscale_inputs[-1]))
            multiscale_inputs.append(self.dconv2(multiscale_inputs[-1]))
        multiscale_inputs_flatten = torch.cat([i.flatten(2, 3) for i in multiscale_inputs], dim=-1)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            if self.training:
                num_proposals_new = self.num_proposals + self.max_num_gts * self.add_gt_groups
            else:
                num_proposals_new = self.num_proposals

            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            ################## Deformable Parameters #############
            if not self.multiscale:
                W, H = lidar_feat.shape[-2:]
                spatial_shapes = torch.as_tensor([[W, H]], dtype=torch.long, device='cuda')
                level_start_index = torch.as_tensor([0, ], dtype=torch.long, device='cuda')
            else:
                spatial_shapes = torch.as_tensor([i.shape[2:] for i in multiscale_inputs], dtype=torch.long, device='cuda')
                level_start_index = torch.as_tensor([0, *(torch.cumsum(torch.prod(spatial_shapes, dim=1), dim=0)[:-1])], dtype=torch.long, device='cuda')

                # lidar feat
                lidar_feat_flatten = multiscale_inputs_flatten
                if self.bevpos and i == 0:
                    # bev pos
                    bev_pos = torch.cat([bev_pos, bev_pos_2, bev_pos_4], dim=1)

            if self.training and self.add_gt_groups > 0:
                # [batch_size, num_queries, num_keys]
                attn_masks = torch.ones((batch_size, num_proposals_new, num_proposals_new), dtype=bool, device='cuda')

                # all(query) sees query(key)
                attn_masks[:, :, :self.num_proposals] = 0
                attn_masks[:, self.num_proposals:, self.num_proposals:] = torch.logical_not(batch_valid_gt_mask[:, None] & batch_valid_gt_mask[:, :, None])
                attn_masks = attn_masks[:, None, :, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            else:
                attn_masks = None

            kwargs = dict(
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=torch.ones((batch_size, 1, 2), device='cuda'),
                key_padding_mask=None,  # key_padding_mask,
                attn_masks=attn_masks,
            )

            ################### Transformer Inputs ##############
            reference_points = query_pos / torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
            query_sine_pos = gen_sineembed_for_position(reference_points[:, :, :2])
            query_pos_embed = self.pos_embed_learned[i](query_sine_pos)  # bs, nq, 256
            if self.boxpos is not None and query_box is not None:
                # if self.boxpos == 'xywlr':  # actually boxdim3 + sincos2
                #     extra_box_pos = query_box[:, 3:8].transpose(1, 2)
                # extra_box_sine_pos = gen_sineembed_for_position_all(extra_box_pos).flatten(-2)
                query_box_embed = self.box_pos_embed_learned[i]
                query_pos_embed += query_box_embed

            # stage 1: query_pos = int+0.5, query_box = abs xy (z1)
            # stage 234..: query_pos = int+0.5, query_box = abs xy (z2)

            if self.bevpos:
                bev_reference_points = bev_pos / torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
                bev_sine_pos = gen_sineembed_for_position(bev_reference_points[:, :, :2])
                bev_pos_embed = self.pos_embed_learned[i](bev_sine_pos)  # bs, nq, 256
                pos_lidar_feat_flatten = lidar_feat_flatten + bev_pos_embed.transpose(1, 2)  # TODO( multiple addition for bev pos embedding )
            else:
                pos_lidar_feat_flatten = lidar_feat_flatten

            if self.roi_feats and query_box is not None:
                rot, dim, center, height, vel = query_box[:, 6:8], query_box[:, 3:6], query_box[:, 0:2], query_box[:, 2:3], query_box[:, 8:]
                std_boxes = self.bbox_coder.decode_box(rot.clone(), dim.clone() * self.roi_expand_ratio[i], center.clone(), height.clone(), vel.clone())

                std_boxes = std_boxes.reshape(batch_size * num_proposals_new, std_boxes.shape[-1])
                lidar_std_boxes = LiDARInstance3DBoxes(std_boxes, box_dim=std_boxes.shape[-1])  # TODO(z is not checked)
                grid_points = self.get_dense_grid_points(std_boxes, batch_size * num_proposals_new, self.roi_feats)
                grid_points = torch.cat([grid_points, grid_points.new_ones(*grid_points.shape[:2], 1)], dim=-1)
                grid_points = rotation_3d_in_axis(grid_points, std_boxes[:, 6], axis=2)
                grid_points = grid_points[..., :2]
                grid_points = grid_points + std_boxes[:, None, :2]
                grid_points = grid_points.view(batch_size, num_proposals_new, self.roi_feats ** 2, 2)

                if self.test_cfg['dataset'] == 'nuScenes':
                    pc_range = torch.as_tensor([-54, -54, -5.0, 54, 54, 3.0], device='cuda')
                else:
                    pc_range = torch.as_tensor([-75.2, -75.2, -2, 75.2, 75.2, 4], device='cuda')
                grid_points = (grid_points - pc_range[:2]) / (pc_range[3:5] - pc_range[:2])
                grid_points = grid_points * 2. - 1.
                grid_points = grid_points.clip(min=-2., max=2.)

                if not self.multiscale:
                    roi_feat = F.grid_sample(lidar_feat_flatten.view(batch_size, -1, *lidar_feat.shape[-2:]), grid_points)  # W, H
                else:
                    ms_roi_feat = []
                    for feat in multiscale_inputs:
                        roi_feat = F.grid_sample(feat, grid_points, mode='bilinear')
                        ms_roi_feat.append(roi_feat)
                    roi_feat = torch.cat(ms_roi_feat, dim=1)
                roi_feat = roi_feat.permute(0, 2, 1, 3).reshape(batch_size * num_proposals_new, lidar_feat_flatten.shape[1] * (3 if self.multiscale else 1) * self.roi_feats ** 2)
                roi_feat = self.roi_mlp(roi_feat)
                roi_feat = roi_feat.view(batch_size, num_proposals_new, lidar_feat_flatten.shape[1]).transpose(1, 2)
                query_feat += roi_feat

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            # query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            query_feat, reference_points = self.decoder[i](
                query=query_feat.permute(2, 0, 1),
                key=None,
                value=pos_lidar_feat_flatten.permute(2, 0, 1),
                query_pos=query_pos_embed.permute(1, 0, 2),
                reference_points=reference_points,
                **kwargs)

            query_feat = query_feat.permute(1, 2, 0)
            query_pos = reference_points * torch.flip(spatial_shapes[:1], dims=(1,))[:, None]

            # Prediction

            res_layer = self.prediction_heads[i](query_feat.clone())
            if self.classaware_reg:
                for k in ['center', 'height', 'dim', 'rot']:
                    res_layer[k] = res_layer[k].view(batch_size, self.num_classes, -1, num_proposals_new)
                    res_layer[k] = res_layer[k].gather(index=query_labels[:, None, None, :].expand(-1, -1, res_layer[k].shape[2], -1).clip(0, self.num_classes - 1), dim=1)[:, 0]

            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

            if self.roi_based_reg and query_box is not None:  # only for bev
                res_layer['dim'][:, :2] = res_layer['dim'][:, :2] + query_box[:, 3:5].detach()
                res_layer['rot'] = res_layer['rot'] + query_box[:, 6:8].detach()
                # res_layer['vel'] = res_layer['vel'] + query_box[:, 8:].detach()

            query_box = [res_layer['center'], res_layer['height'], res_layer['dim'], res_layer['rot']]
            if 'vel' in res_layer:
                query_box.append(res_layer['vel'])
            query_box = torch.cat(query_box, dim=1).detach()
            ret_dicts.append(res_layer)

        if self.training and self.add_gt_groups > 0:
            ret_dicts[0]['batch_valid_gt_mask'] = batch_valid_gt_mask
            ret_dicts[0]['batch_gt_query_labels'] = batch_gt_query_labels
        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = query_heatmap_score  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = heatmap_train  # TODO(DeepInteraction only train image dense heatmap)

        ret_dicts[0]['last_bev_feature'] = lidar_feat  #
        if self.multistage_heatmap:
            ret_dicts[0]['multistage_masks'] = multistage_masks
            # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key in ['dense_heatmap', 'query_heatmap_score', 'multistage_masks', 'last_bev_feature']:
                new_res[key] = ret_dicts[0][key]
            elif key in ['gt_query_bbox_targets']:
                new_res[key] = ret_dicts[0][key]
            elif key in ['batch_valid_gt_mask', 'batch_gt_query_labels']:
                new_res[key] = ret_dicts[0][key]
            else:
                if self.training and self.add_gt_groups > 0:
                    new_res[key] = torch.cat([
                        ret_dict[key][:, :, :(-self.max_num_gts * self.add_gt_groups)]
                        for i, ret_dict in enumerate(ret_dicts)], dim=-1)
                    new_res[key + '_gtgroups'] = torch.cat([
                        ret_dict[key][:, :, -self.max_num_gts * self.add_gt_groups:]
                        for i, ret_dict in enumerate(ret_dicts)], dim=-1)
                else:
                    new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
        return [[new_res]]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            # heatmap_score = torch.cat(res_tuple[8], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        query_heatmap_score = copy.deepcopy(preds_dict['query_heatmap_score'].detach())[:, :, :(-self.max_num_gts * self.add_gt_groups)]
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        assign_result_list = []
        for idx_layer in range(self.num_decoder_layers):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]
            # if idx_layer == self.num_decoder_layers - 1:
            #     sec_match = True
            # else:
            #     sec_match = False
            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result, sec_matchs = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg, sec_match=False)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError

            # if idx_layer == self.num_decoder_layers - 1:
            #     if self.initialize_by_heatmap:
            #         device = gt_labels_3d.device
            #         grid_size = torch.tensor(self.train_cfg['grid_size'])
            #         pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            #         voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            #         feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            #         heatmap = bboxes_tensor_layer.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            #         heatmap_score = bboxes_tensor_layer.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            #
            #         #neg
            #         final_preds = score_layer.max(1, keepdims=False).indices
            #         final_scores = score_layer.max(1, keepdims=False).values
            #         sec_matched_row_inds = sec_matchs[0]
            #         sec_matched_col_inds = sec_matchs[1]
            #
            #         sec_pred_box = bboxes_tensor_layer[sec_matched_row_inds]
            #         sec_pred_scores = final_scores[0][sec_matched_row_inds]
            #         sec_pred_labels = final_preds[0][sec_matched_row_inds]
            #         sec_gt_boxes = gt_bboxes_tensor[sec_matched_col_inds]
            #
            #         sec_valid_gt_idxs = (sec_gt_boxes[:, :2] - sec_pred_box[:, :2]).norm(dim=1) > self.gt_center_limit
            #
            #         sec_pred_labels = sec_pred_labels[sec_valid_gt_idxs]
            #         sec_pred_box = sec_pred_box[sec_valid_gt_idxs]
            #         sec_gt_boxes[:, :2][sec_valid_gt_idxs]
            #         sec_pred_scores = sec_pred_scores[sec_valid_gt_idxs]
            #         for idx in range(len(sec_pred_labels)):
            #             neg_cur_box = sec_pred_box[idx]
            #             neg_cur_cls = sec_pred_labels[idx]
            #             neg_cur_score = sec_pred_scores[idx].item()
            #             neg_width = neg_cur_box[3]
            #             neg_length = neg_cur_box[4]
            #             neg_width = neg_width / voxel_size[0] / self.train_cfg['out_size_factor']
            #             neg_length = neg_length / voxel_size[1] / self.train_cfg['out_size_factor']
            #             if neg_width > 0 and neg_length > 0:
            #                 radius = gaussian_radius((neg_length, neg_width), min_overlap=self.train_cfg['gaussian_overlap'])
            #                 radius = max(self.train_cfg['min_radius'], int(radius))
            #                 neg_x, neg_y = neg_cur_box[0], neg_cur_box[1]
            #                 neg_coor_x = (neg_x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
            #                 neg_coor_y = (neg_y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
            #                 neg_center = torch.tensor([neg_coor_x, neg_coor_y], dtype=torch.float32, device=device)
            #                 neg_center_int = neg_center.to(torch.int32)
            #                 draw_heatmap_gaussian(heatmap_score[neg_cur_cls], neg_center_int, radius, k=-neg_cur_score)
            #
            #         #positive
            #         #positi
            #         hp_ious = assign_result.max_overlaps[assign_result.gt_inds > 0]
            #         hp_ious = torch.clamp(hp_ious, min=0.0, max=1.0)
            #         for idx in range(len(gt_labels_3d)):
            #             cur_pos_box = gt_bboxes_tensor[assign_result.gt_inds[assign_result.gt_inds > 0] - 1][idx]
            #             cur_pos_cls = gt_labels_3d[assign_result.gt_inds[assign_result.gt_inds > 0] - 1][idx]
            #             pos_width = cur_pos_box[3]
            #             pos_length = cur_pos_box[4]
            #             pos_width = pos_width / voxel_size[0] / self.train_cfg['out_size_factor']
            #             pos_length = pos_length / voxel_size[1] / self.train_cfg['out_size_factor']
            #             if pos_width > 0 and pos_length > 0:
            #                 radius = gaussian_radius((pos_length, pos_width), min_overlap=self.train_cfg['gaussian_overlap'])
            #                 radius = max(self.train_cfg['min_radius'], int(radius))
            #                 pos_x, pos_y = cur_pos_box[0], cur_pos_box[1]
            #                 pos_coor_x = (pos_x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
            #                 pos_coor_y = (pos_y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
            #                 pos_center = torch.tensor([pos_coor_x, pos_coor_y], dtype=torch.float32, device=device)
            #                 pos_center_int = pos_center.to(torch.int32)
            #                 hp_iou = hp_ious[idx]
            #                 draw_heatmap_gaussian(heatmap_score[cur_pos_cls], pos_center_int, radius, k=hp_iou)
            #                 draw_heatmap_gaussian(heatmap[cur_pos_cls], pos_center_int, radius)

            if self.gt_center_limit is not None:
                pos_gt_bboxes_tensor = gt_bboxes_tensor[assign_result.gt_inds[assign_result.gt_inds > 0] - 1]
                pos_bboxes_tensor_layer = bboxes_tensor_layer[assign_result.gt_inds > 0]
                invalid_gt_idxs = (pos_gt_bboxes_tensor[:, :2] - pos_bboxes_tensor_layer[:, :2]).norm(dim=1) > self.gt_center_limit
                assign_result.gt_inds[torch.nonzero(assign_result.gt_inds > 0)[:, 0][invalid_gt_idxs]] = 0

            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        # sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # heatmap_score=None
        # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)  # heatmap of shape [180, 180], draw at (x,y) with radius = 2 mostly

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])

        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            # if len(preds_dict['dense_heatmap'])>1:
            #     if 'multistage_masks' in preds_dict:
            #         multistage_masks = torch.cat(preds_dict['multistage_masks'], dim=0)
            #         heatmap = heatmap.repeat(len(preds_dict['dense_heatmap']), 1, 1, 1) * multistage_masks  # mask the ignore gt
            #     else:
            #         heatmap = heatmap.repeat(len(preds_dict['dense_heatmap']), 1, 1, 1)
            #     preds_dict['dense_heatmap'] = torch.cat(preds_dict['dense_heatmap'], dim=0)
            #
            #     loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, weight=(multistage_masks if 'multistage_masks' in preds_dict else None),
            #                                      avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            # else:
            #     loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap'][0]), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            # loss_dict['loss_heatmap'] = loss_heatmap * self.loss_weight_heatmap
            for idx_layer in range(self.multistage_heatmap):
                prefix = f'layer_{idx_layer}'
                # if 'multistage_masks' not in preds_dict:
                #     loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap'][idx_layer]).float(), heatmap,
                #                                      avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
                #     loss_img_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap'][idx_layer + 1]).float(), heatmap,
                #                                          avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
                #     loss_dict[f'{prefix}_loss_heatmap'] = 0.5 * loss_heatmap * self.loss_weight_heatmap
                #     loss_dict[f'{prefix}_loss_img_heatmap'] = 0.5 * loss_img_heatmap * self.loss_weight_heatmap
                # else:
                tmp_heatmap = heatmap * preds_dict['multistage_masks'][idx_layer]
                # if idx_layer == 0:
                #     avg_factor = max(tmp_heatmap.eq(1).float().sum().item(), 1)
                #     layer_weight = 1.0
                # elif idx_layer == 1:
                #     avg_factor = max(tmp_heatmap.eq(1).float().sum().item(), 1)
                #     layer_weight = 0.15

                loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap'][idx_layer]).float(), tmp_heatmap,
                                                 avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
                # loss_img_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap'][2 * idx_layer + 1]).float(), heatmap * preds_dict['multistage_masks'][idx_layer],
                #                                      avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
                loss_dict[f'{prefix}_loss_heatmap'] = loss_heatmap * self.loss_weight_heatmap
                # loss_dict[f'{prefix}_loss_img_heatmap'] = 0.5 * loss_img_heatmap * self.loss_weight_heatmap
        # compute loss for each layer


        for idx_layer in range(self.num_decoder_layers):
            prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]

            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        if self.add_gt_groups > 0:
            num_gt_groups_layers = self.num_decoder_layers

            batch_valid_gt_mask = preds_dict['batch_valid_gt_mask'].float()
            batch_gt_query_labels = preds_dict['batch_gt_query_labels'].repeat(1, num_gt_groups_layers)

            # bbox
            preds_gt_query = torch.cat([preds_dict['center' + '_gtgroups'], preds_dict['height' + '_gtgroups'], preds_dict['rot' + '_gtgroups'], preds_dict['dim' + '_gtgroups']], dim=1)
            if 'vel' in preds_dict.keys():
                preds_gt_query = torch.cat([preds_gt_query, preds_dict['vel' + '_gtgroups']], dim=1)
            preds_gt_query = preds_gt_query.permute(0, 2, 1)

            # labels
            score_gt_query = preds_dict['heatmap' + '_gtgroups'].permute(0, 2, 1).reshape(-1, self.num_classes)
            gt_query_bbox_targets = torch.zeros((preds_gt_query.shape[0], self.max_num_gts, self.bbox_coder.code_size), dtype=preds_gt_query.dtype, device=preds_gt_query.device)
            for batch_idx, g in enumerate(gt_bboxes_3d):
                gt_query_bbox_targets[batch_idx, :len(g.tensor)] = self.bbox_coder.encode(g.tensor)
            positive_mask = (batch_gt_query_labels != self.num_classes)

            # bbox loss
            gt_query_bbox_targets = gt_query_bbox_targets.repeat(1, self.add_gt_groups * num_gt_groups_layers, 1)
            gt_query_bbox_weights = batch_valid_gt_mask[:, :, None].repeat(1, num_gt_groups_layers, gt_query_bbox_targets.shape[-1])
            gt_query_reg_weights = gt_query_bbox_weights * gt_query_bbox_weights.new_tensor(code_weights)
            gt_query_reg_weights = gt_query_reg_weights * positive_mask[..., None].float()
            loss_dict['gt_query_loss_box'] = self.loss_bbox(preds_gt_query, gt_query_bbox_targets, gt_query_reg_weights,
                                                            avg_factor=max(sum(self.num_gts) * self.add_gt_groups * num_gt_groups_layers, 1)) * self.gt_query_loss_weight

            # labels loss
            gt_query_label_weights = batch_valid_gt_mask.repeat(1, num_gt_groups_layers).reshape(-1)
            loss_dict['gt_query_loss_cls'] = self.loss_cls(score_gt_query, batch_gt_query_labels.reshape(-1), gt_query_label_weights,
                                                           avg_factor=max(sum(self.num_gts) * self.add_gt_groups * num_gt_groups_layers, 1)) * self.gt_query_loss_weight

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            # all three has shape of [N, num_classes, num_proposals]
            # one_hot extracts only the target class score
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories

                if len(boxes3d) > 200:
                    # print('get more than 200 boxes, save top 200 only !!!')
                    inds = scores.argsort(descending=True)[:300]
                    boxes3d = boxes3d[inds]
                    scores = scores[inds]
                    labels = labels[inds]

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = img_metas[0].box_type_3d(
                    boxes3d, box_dim=boxes3d.shape[-1])
                temp_instances.scores_3d = scores
                temp_instances.labels_3d = labels.int()

                ret_layer.append(temp_instances)
            return ret_layer

    def get_heatmap_targets(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_heatmap_targets_single, gt_bboxes_3d, gt_labels_3d)
        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # transpose anno_boxes
        anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # transpose inds
        inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]
        # transpose inds
        masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_heatmap_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx in range(len(self.heatmap_tasks)):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                                     x - pc_range[0]
                             ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                                     y - pc_range[1]
                             ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:5]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points


def _gather_feat(feat, ind, mask=None):
    """Gather feature map.

    Given feature map and index, return indexed feature map.

    Args:
        feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
        ind (torch.Tensor): Index of the ground truth boxes with the
            shape of [B, max_obj].
        mask (torch.Tensor): Mask of the feature map with the shape
            of [B, max_obj]. Default: None.

    Returns:
        torch.Tensor: Feature map after gathering with the shape
            of [B, max_obj, 10].
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat