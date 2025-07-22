# modify from https://github.com/mit-han-lab/bevfusion

import torch
from torch import nn
from mmdet3d.registry import MODELS
from .ops.msmv_sampling.wrapper import msmv_sampling
from .checkpoint import checkpoint as cp
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def make_sample_points(query_bbox, offset, res=[0.6, 0.6, 0.2]):
    '''
    query_bbox: [B, Q, 3]
    offset: [B, Q, num_points, 4], normalized by stride
    '''

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3]
    delta_xyz[..., :2] = res[0] * delta_xyz[..., :2]
    delta_xyz[..., 2] = res[2] * delta_xyz[..., 2]
    sample_xyz = xyz[:, :, None, :] + delta_xyz  # [B, Q, P, 3]
    return sample_xyz


@MODELS.register_module()
class DepthLSSTransform(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            position_range,
            image_size=[1600, 640],
            num_outs=5,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__()
        self.in_channels = in_channels
        self.C = out_channels
        self.image_size = image_size
        self.position_range = position_range

        self.G = 4
        self.P = 8
        self.camC = 64
        self.res = [0.6, 0.6, 0.4]
        self.grid = [180, 180, 20]
        self.num_levels = num_outs
        self.img_scale_weights = nn.Linear(in_channels, self.G * self.P * self.num_levels)
        self.sampling_offset = nn.Linear(in_channels, self.G * self.P * 3)
        # self.voxel_weight = nn.Linear(in_channels, self.grid[2])
        self.aggregate_points = nn.Sequential(nn.Linear(self.P * self.C, self.C))
        self.bev_encode = nn.Sequential(
            nn.Conv2d(self.C * self.grid[2] , self.camC* self.grid[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.camC* self.grid[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.camC* self.grid[2], 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

        self.init_weights()
        self.frustum = self.create_frustum(x_size=self.grid[0], y_size=self.grid[1], z_size=self.grid[2])

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.G * self.P, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def create_frustum(self, x_size=180, y_size=180, z_size=5):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size], [0, z_size - 1, z_size]]
        # NOTE: modified
        batch_x, batch_y, batch_z = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) * self.res[0] + self.position_range[0]
        batch_y = (batch_y + 0.5) * self.res[1] + self.position_range[1]
        batch_z = (batch_z + 0.5) * self.res[2] + self.position_range[2]
        coord_base = torch.cat([batch_x[None], batch_y[None], batch_z[None]], dim=0).permute(1, 2, 3, 0)  # [w,h,l,3]

        return nn.Parameter(coord_base, requires_grad=False)

    def inner_forward(self, img_feature, pts_feature, pst_feat, img_metas, **kwargs):
        points = pst_feat['points']
        B, N, G, P = len(points), 6, self.G, self.P
        eps = 1e-5
        p_count_list = []
        pos_list = []
        coords = self.frustum.clone()
        for i in range(len(points)):
            cur_points = points[i]
            x_img = ((cur_points[..., 0] - self.position_range[0]) / self.res[0]).int()
            y_img = ((cur_points[..., 1] - self.position_range[1]) / self.res[1]).int()
            z_img = ((cur_points[..., 2] - self.position_range[2]) / self.res[2]).int()
            x_img = x_img.clamp(max=self.grid[0] - 1, min=0)
            y_img = y_img.clamp(max=self.grid[1] - 1, min=0)
            z_img = z_img.clamp(max=self.grid[2] - 1, min=0)
            grid_voxel = torch.zeros([self.grid[0], self.grid[1], self.grid[2]], device=cur_points.device)
            grid_voxel[x_img, y_img, z_img] = 1
            pos = torch.nonzero(grid_voxel)
            pos = torch.cat([pos, i * torch.ones_like(pos[..., :1])], dim=-1)
            pos_list.append(pos)
            p_count_list.append(pos.shape[0])

        Q = max(p_count_list)
        max_index = p_count_list.index(Q)
        _pos = []
        for j in range(len(points)):
            if j != Q:
                cur_pos = torch.zeros_like(pos_list[max_index])
                cur_pos[:pos_list[j].shape[0], :] = pos_list[j]
            else:
                cur_pos = pos_list[j]
            _pos.append(cur_pos)
        inst_vox = torch.stack(_pos)

        lidar2img = [torch.stack((m['lidar2img'])) for m in img_metas]
        lidar2img = torch.stack(lidar2img).to(coords.device).float()

        pts_feat = pts_feature[0].permute(0, 2, 3, 1)
        pts_f = [pts_feat[i, inst_vox[i, :, 1], inst_vox[i, :, 0]] for i in range(B)]
        pts_f = torch.stack(pts_f)

        sampling_off = self.sampling_offset(pts_f)
        sampling_off = sampling_off.reshape(B, Q, self.G * self.P, 3)
        scale_weights = self.img_scale_weights(pts_f).reshape(B,  Q, self.G * self.P, self.num_levels)

        coord = [coords[inst_vox[i, :, 0], inst_vox[i, :, 1], inst_vox[i, :, 2]] for i in range(B)]
        coord = torch.stack(coord)
        sampling_offset = make_sample_points(coord, sampling_off, res=self.res)

        lidar2img = lidar2img[:, :, None, None, :, :]

        lidar2img = lidar2img.expand(B, N, Q, G * P, 4, 4)

        sample_points = torch.cat([sampling_offset, torch.ones_like(sampling_offset[..., :1])], dim=-1)
        sample_points = sample_points[:, None, :, :, :, None]
        sample_points = sample_points.expand(B, N, Q, G * P, 4, 1)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)

        homo = sample_points_cam[..., 2:3]
        homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
        sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
        image_w, image_h = self.image_size[0], self.image_size[1]

        sample_points_cam[..., 0] /= image_w
        sample_points_cam[..., 1] /= image_h

        valid_mask = ((homo > eps) \
                      & (sample_points_cam[..., 1:2] > 0.0)
                      & (sample_points_cam[..., 1:2] < 1.0)
                      & (sample_points_cam[..., 0:1] > 0.0)
                      & (sample_points_cam[..., 0:1] < 1.0)
                      ).squeeze(-1).float()  # [B, N, Q, GP]

        valid_mask = valid_mask.permute(0, 2, 3, 1)  # [B,  Q, GP, N]
        sample_points_cam = sample_points_cam.permute(0, 2, 3, 1, 4)  # [B,  Q, GP, N, 2]

        i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
        i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)

        i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
        i_batch = i_batch.view(B, 1, 1, 1).expand(B, Q, G * P, 1)
        i_query = i_query.view(1, Q, 1, 1).expand(B, Q, G * P, 1)
        i_point = i_point.view(1, 1, G * P, 1).expand(B, Q, G * P, 1)
        i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, Q, GP, 1]

        sample_points_cam = sample_points_cam[i_batch, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
        valid_mask = valid_mask[i_batch, i_query, i_point, i_view]  # [B, Q, GP, 1]

        sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / 5], dim=-1)

        # plt.figure(figsize=(800, 49))
        # view_mapping = [1, 2, 0, 4, 5, 3]
        #
        # for view_id in range(6):
        #     filenames = img_metas[0]['img_path'][view_id]
        #     img = Image.open(filenames)
        #     img = img.crop((0, 260, 1600, 900))
        #     ax = plt.subplot(6, 6, view_mapping[view_id]+1)
        #     ax.imshow(img)
        #     ax.axis('off')
        #     ax.set_xlim(0, 1600)
        #     ax.set_ylim(640, 0)
        #     for query_id in range(Q):
        #         xyz = sample_points_cam[0, query_id,:,0,:].cpu().numpy()
        #         sample_points_cam[sample_points_cam[...,2]==0]
        #         # [32, 3]
        #         mask = valid_mask[0, query_id,:,0].cpu().numpy()  # [32]
        #         mask = np.round(mask).astype(bool)
        #         cmask = (xyz[:,2] == view_id/5)
        #         mask = mask&(cmask)
        #         cx = xyz[:, 0] * 1600
        #         cy = xyz[:, 1] * 640
        #         cz = xyz[:, 2]
        #
        #         cz[np.where(cz <= 0)] = 1e8
        #         cz = np.log(60 / cz ** 0.8) * 2.4
        #         cx, cy, cz = cx[mask], cy[mask], cz[mask]
        #         if len(cz) == 0:
        #             continue
        #
        #         ax.scatter(cx, cy, s=1, color='C%d' % (query_id % 5))
        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.01, wspace=0.01)
        # plt.savefig('out_img/sp_d.jpg', dpi=1000)
        # plt.close()
        sample_points_cam = sample_points_cam.reshape(B, Q, G, P, 1, 3)
        sample_points_cam = sample_points_cam.permute(0, 2, 1, 3, 4, 5)  # [B,  G, Q, P, 1, 3]
        sample_points_cam = sample_points_cam.reshape(B * G, Q, P, 3)

        scale_weights = scale_weights.reshape(B, Q, G, P, self.num_levels)
        scale_weights = scale_weights.permute(0, 2, 1, 3, 4)
        scale_weights = scale_weights.reshape(B * G, Q, P, self.num_levels)

        sample_points_cam = sample_points_cam.contiguous().float()
        scale_weights = scale_weights.contiguous().float()
        final = msmv_sampling(img_feature, sample_points_cam, scale_weights)
        C = final.shape[2]  # [BG, Q, C, P]
        final = final.reshape(B, G, Q, C, P)
        final = final.permute(0, 2, 4, 1, 3)
        final = final.flatten(2)
        final = self.aggregate_points(final)
        ffeat = torch.zeros([B, self.grid[2], self.grid[1], self.grid[0], G * C], device=scale_weights.device)
        pos_t = torch.cat(pos_list, dim=0)

        fi = [final[i, :p_count_list[i]] for i in range(B)]
        fi = torch.cat(fi, dim=0)
        ffeat[pos_t[:, 3], pos_t[:, 2], pos_t[:, 1], pos_t[:, 0]] = fi
        ffeat = ffeat.permute(0, 4, 1, 2, 3)
        ffeat = ffeat.reshape(B, -1, self.grid[1], self.grid[0])
        ffeat = self.bev_encode(ffeat)
        # voxel_weight = voxel_weight.permute(0, 3, 1, 2)[:, None]
        # ffeat = (ffeat * voxel_weight).sum(2)
        # ffeat = ffeat + pts_feature[0]
        return ffeat

    def forward(self, img_feature, pts_feature, pst_feat, img_metas, **kwargs):
        with torch.autocast('cuda', enabled=False, dtype=torch.float32):
            for lvl, feat in enumerate(img_feature):
                B, N, GC, H, W = feat.shape  # [B, N, GC, H, W]
                G, C = 4, GC // 4
                feat = feat.reshape(B, N, G, C, H, W)
                feat = feat.permute(0, 2, 1, 4, 5, 3)  # [B,  G, N, H, W, C]
                feat = feat.reshape(B * G, N, H, W, C)  # [BTG, C, N, H, W]
                img_feature[lvl] = feat.contiguous().float()
        if self.training and img_feature[0].requires_grad and False:
            return cp(self.inner_forward, img_feature, pts_feature, pst_feat, img_metas, **kwargs, use_reentrant=False)
        else:
            return self.inner_forward(img_feature, pts_feature, pst_feat, img_metas, **kwargs)
