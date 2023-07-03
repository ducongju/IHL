# Copyright (c) OpenMMLab. All rights reserved.
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss, KLDiscretLoss2, JSDiscretLossPeak, JSDiscretLossPeak2, WassersteinLoss, SinkhornDistance, NMSLoss
from .heatmap_loss import AdaptiveWingLoss
from .loss_wrappers import MultipleLossWrapper
from .mse_loss import (CombinedTargetMSELoss, KeypointMSELoss,
                       KeypointOHKMMSELoss)
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss, RLELoss2, RLELoss3,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'CombinedTargetMSELoss',
    'HeatmapLoss', 'AELoss', 'MultiLossFactory', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss', 'RLELoss2', 'RLELoss3',
    'KLDiscretLoss', 'KLDiscretLoss2', 'MultipleLossWrapper', 'JSDiscretLoss', 'JSDiscretLossPeak', 'JSDiscretLossPeak2', 
    'WassersteinLoss', 'SinkhornDistance', 'NMSLoss'
]
