# Copyright (c) DuCongju. All rights reserved.
from typing import Optional, Sequence, Tuple, Union  # typing 是python3.5中开始新增的专用于类型注解(type hints)的模块，为python程序提供静态类型检查

import numpy as np
import torch
from mmengine.logging import MessageHub
from torch import Tensor

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from .ihl_core_head import IHL_Core_Head

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class IHL_Head(IHL_Core_Head):
    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 lambda_t: int = -1,
                 debias: bool = False,
                 beta: float = 1.0,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 has_final_layer: bool = True,
                 input_transform: str = 'select',
                 input_index: Union[int, Sequence[int]] = -1,
                 align_corners: bool = False,
                 loss: ConfigType = dict(
                     type='MultipleLossWrapper',
                     losses=[
                         dict(type='RLELoss', use_target_weight=True),
                         dict(type='JSDiscretLoss', use_target_weight=True)
                     ]),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        super().__init__(
            in_channels=in_channels,
            in_featuremap_size=in_featuremap_size,
            num_joints=num_joints,
            debias=debias,
            beta=beta,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            has_final_layer=has_final_layer,
            input_transform=input_transform,
            input_index=input_index,
            align_corners=align_corners,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg)

        self.lambda_t = lambda_t

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords0, pred_heatmaps = self.forward(inputs)
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])

        pred_coords = pred_coords0[:, :, :2]

        input_list = [pred_coords0, pred_heatmaps]
        target_list = [keypoint_labels, gt_heatmaps]

        # calculate losses
        losses = dict()

        loss_list = self.loss_module(input_list, target_list, keypoint_weights)

        loss = loss_list[0] + loss_list[1]

        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch >= self.lambda_t:
                loss = loss_list[0]

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses
