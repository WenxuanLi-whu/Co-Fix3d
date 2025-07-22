from mmdet3d.registry import HOOKS
from mmengine.hooks import Hook
from typing import Optional, Sequence, Union
import torch
from collections import OrderedDict
DATA_BATCH = Optional[Union[dict, tuple, list]]
@HOOKS.register_module()
class LoadMyCheckpointHook(Hook):

    priority = 'NORMAL'

    def __init__(self,
                 load_img_from=None,load_lss_from=None) -> None:
        self.load_img_from = load_img_from
        self.load_lss_from = load_lss_from

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations before saving the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        flag =False
        for key in  checkpoint['state_dict'].keys():
            if key.startswith('img_backbone.'):
                flag = True
                break
        if flag:
            return

        if self.load_img_from is not None:
            img_dict = torch.load(self.load_img_from)
            ckpt = img_dict["state_dict"]
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    new_v = v
                    new_k = k.replace('backbone.', 'img_backbone.')
                    new_ckpt[new_k] = new_v
                if k.startswith('img_backbone'):
                    new_ckpt[k] = v
                if k.startswith('img_neck'):
                    new_ckpt[k] = v
                if k.startswith('pts_fusion_layer.cam_lss'):
                    new_ckpt[k] = v

        # new_ckpt = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     if k.startswith('pts_middle_encoder.conv_out.0.weight'):
        #         v = v.permute(0, 1, 4, 2, 3).contiguous()
        #     if k.startswith('imgpts_neck'):
        #         new_v = v
        #         new_k = k.replace('imgpts_neck.', 'pts_fusion_layer.')
        #         new_ckpt[new_k] = new_v
        #     else:
        #         new_ckpt[k] = v
        # checkpoint['state_dict'].clear()
        checkpoint['state_dict'].update(new_ckpt)
