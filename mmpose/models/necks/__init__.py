# Copyright (c) OpenMMLab. All rights reserved.
from .fpn import FPN
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck
from .texture_neck import AttentionPooling

__all__ = ['GlobalAveragePooling', 'PoseWarperNeck', 'FPN', 'AttentionPooling']