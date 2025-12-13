import torch
import torch.nn as nn
from torchvision import models

class RotationNet(nn.Module):
    """
    PoseResNet backbone with a small quaternion regression head.

    This class wraps a pretrained ResNet-50 backbone and attaches a small
    regression head that predicts a normalized quaternion (w, x, y, z).

    Parameters
    - `pretrained` (bool): if True, load ImageNet pretrained weights for ResNet-50.
    - `freeze_scale` (float in [0, 1]): fraction of the backbone modules to freeze
        (set `requires_grad=False`). For example, `freeze_scale=0.5` freezes approximately
        the first half of the backbone modules. The value is validated and must
        be between 0 and 1 inclusive.

    Notes
    - Freezing is applied per-backbone-module (the immediate children of the
        assembled `self.features` Sequential), not per-parameter. The mapping of
        modules to logical layers (layer1, layer2, ...) depends on torchvision's
        ResNet implementation.

    Example
    >>> model = PoseResNet(pretrained=True, freeze_scale=0.5)
    >>> out = model(torch.randn(1, 3, 224, 224))  # out.shape == (1, 4)
    """

    def __init__(self, pretrained: bool = True, freeze_scale: float = 0.0) -> None:
        super(RotationNet, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze approximately half of the ResNet backbone modules (first half)
        # This sets requires_grad=False for parameters in those modules.
        backbone_modules = list(self.features.children())
        num_modules = len(backbone_modules)
        # Validate freeze_scale: must be between 0 and 1 inclusive
        if not (0.0 <= freeze_scale <= 1.0):
            raise ValueError(f"freeze_scale must be between 0 and 1 inclusive; got {freeze_scale}")

        num_to_freeze = int(num_modules * freeze_scale)
        for i in range(num_to_freeze):
            for p in backbone_modules[i].parameters():
                p.requires_grad = False
        print(f"PoseResNet: froze {num_to_freeze}/{num_modules} backbone modules")
        # The output size of the ResNet50 backbone is 2048.
        feature_dim = 2048
        
        # Regression Head 
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # Quaternion (w, x, y, z)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): input image batch, shape (N, 3, H, W).

        Returns:
            torch.Tensor: normalized quaternion predictions with shape (N, 4).
        """
        x = self.features(x)
        x = x.flatten(1)

        q = self.regressor(x)

        # Normalize: a quaternion should have unit norm
        q = torch.nn.functional.normalize(q, p=2, dim=1)

        return q