# Copyright (c) OpenMMLab. All rights reserved.
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .ihl_core_head import IHL_Core_Head
from .ihl_head import IHL_Head

__all__ = [
    'RegressionHead',
    'IntegralRegressionHead',
    'DSNTHead',
    'RLEHead',
    'IHL_Head',
    'IHL_Core_Head',
]
