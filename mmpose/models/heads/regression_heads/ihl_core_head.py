# Copyright (c) DuCongju. All rights reserved.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates, flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from .. import HeatmapHead
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class IHL_Core_Head(BaseHead):
    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
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
                     type='SmoothL1Loss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.debias = debias
        self.beta = beta
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:

            self.heatmap_size = tuple(
                [s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.simplebaseline_head = HeatmapHead(
                in_channels=in_channels,
                out_channels=num_joints,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                has_final_layer=has_final_layer,
                input_transform=input_transform,
                input_index=input_index,
                align_corners=align_corners)

            if has_final_layer:
                in_channels = num_joints
            else:
                in_channels = deconv_out_channels[-1]

        else:
            in_channels = self._get_in_channels()
            self.simplebaseline_head = None

            if has_final_layer:
                cfg = dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=num_joints,
                    kernel_size=1)
                self.final_layer = build_conv_layer(cfg)
            else:
                self.final_layer = None

            if self.input_transform == 'resize_concat':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[0]
            elif self.input_transform == 'select':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[input_index]

        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        W, H = self.heatmap_size
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H, 1) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

        self.fc = nn.Linear(in_channels, self.num_joints * 2)

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _linear_expectation_x(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""
        B, N, H, W = heatmaps.shape

        heatmaps_temp = heatmaps.mul(linspace).reshape(B, N, -1)

        expectation = torch.sum(heatmaps_temp, dim=2, keepdim=True)
        expectation_temp = expectation.unsqueeze(-1).repeat(1, 1, 1, W)

        vars_temp = ((linspace - expectation_temp) ** 2).repeat(1, 1, H, 1)
        vars = vars_temp.mul(heatmaps).reshape(B, N, -1)
        vars = torch.sum(vars, dim=2, keepdim=True)

        return expectation, vars
    
    def _linear_expectation_y(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""
        B, N, H, W = heatmaps.shape

        heatmaps_temp = heatmaps.mul(linspace).reshape(B, N, -1)

        expectation = torch.sum(heatmaps_temp, dim=2, keepdim=True)
        expectation_temp = expectation.unsqueeze(-1).repeat(1, 1, H, 1)

        vars_temp = ((linspace - expectation_temp) ** 2).repeat(1, 1, 1, W)
        vars = vars_temp.mul(heatmaps).reshape(B, N, -1)
        vars = torch.sum(vars, dim=2, keepdim=True)

        return expectation, vars

    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""

        _, N, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, N, H * W)

        min_feat, _ = torch.min(featmaps, dim=2, keepdim=True)
        featmaps_input = featmaps - min_feat
        heatmaps = featmaps_input / featmaps_input.sum(dim=2, keepdim=True)

        return heatmaps.reshape(-1, N, H, W)

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        if self.simplebaseline_head is None:
            feats = self._transform_inputs(feats)  # Transform multi scale features into the network input.
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.simplebaseline_head(feats)

        heatmaps = self._flat_softmax(feats * self.beta)

        pred_x, var_x = self._linear_expectation_x(heatmaps, self.linspace_x)
        pred_y, var_y = self._linear_expectation_y(heatmaps, self.linspace_y)

        if self.debias:
            B, N, H, W = feats.shape
            C = feats.reshape(B, N, H * W).exp().sum(dim=2).reshape(B, N, 1)
            pred_x = C / (C - 1) * (pred_x - 1 / (2 * C))
            pred_y = C / (C - 1) * (pred_y - 1 / (2 * C))

        coords = torch.cat([pred_x, pred_y, var_x, var_y], dim=-1)

        return coords, heatmaps

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _batch_heatmaps = self.forward(_feats)

            _batch_coords_flip, _batch_heatmaps_flip = self.forward(
                _feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode='heatmap',
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_coords, batch_heatmaps = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)
        batch_coords1 = batch_coords[:, :, :, :2]
        preds = self.decode(batch_coords1)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords0, _ = self.forward(inputs)
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        pred_coords = pred_coords0[:, :, :2]

        # calculate losses
        losses = dict()

        loss = self.loss_module(pred_coords0, keypoint_labels, keypoint_weights)

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

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to load weights of deconv layers from
        :class:`HeatmapHead` into `simplebaseline_head`.

        The hook will be automatically registered during initialization.
        """

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k.lstrip(prefix)

            k_new = _k
            k_parts = k.split('.')
            if self.simplebaseline_head is not None:
                if k_parts[0] == 'conv_layers':
                    k_new = (
                        prefix + 'simplebaseline_head.deconv_layers.' +
                        '.'.join(k_parts[1:]))
                elif k_parts[0] == 'final_layer':
                    k_new = prefix + 'simplebaseline_head.' + k

            state_dict[k_new] = v
