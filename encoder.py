import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# class CnnEncoder(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()

#         # Una CNN semplice (stile ResNet ridotta)
        
#         self.conv = nn.Sequential(

#             # Block 1
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.GELU(),
#             nn.Conv2d(64, 64, 3, stride=2, padding=1),   # ↓ invece di MaxPool
#             # nn.Dropout2d(0.1),

#             # Block 2
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.GELU(),
#             nn.Conv2d(128, 128, 3, stride=2, padding=1),
#             # nn.Dropout2d(0.1),

#             # Block 3
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.GELU(),
#             nn.Conv2d(256, 256, 3, stride=2, padding=1),
#             #nn.Dropout2d(0.1),

#             # Block 4 (kernel grande)
#             nn.Conv2d(256, d_model, 5, padding=2),
#             nn.BatchNorm2d(d_model),
#             nn.GELU(),
#             nn.Conv2d(d_model, d_model, 3, stride=2, padding=1),
#         )

#     def forward(self, x):
#         feat = self.conv(x)            # [B, d_model, 4, 20]
#         feat = feat.flatten(2)         # [B, d_model, 80]
#         feat = feat.permute(0, 2, 1)   # [B, 80, d_model]
#         return feat


class CnnEncoder(nn.Module):
    """
    CNN-based encoder module.

    This module uses a modified ResNet-18 backbone to extract feature maps from images.
    The output is a sequence of feature vectors suitable for feeding into
    subsequent layers like a Transformer decoder.
    """

    def __init__(self, d_model=256, pretrained=True):
        """
        Initializes the CNN encoder.

        Args:
            d_model (int): Output feature dimension after projection.
            pretrained (bool): If True, loads ResNet-18 with pretrained weights.
        """
        super().__init__()

        # Load ResNet-18 model, optionally with pretrained weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Extract the initial convolution, batch norm, ReLU, and maxpool layers
        self.stem = nn.Sequential(
            resnet.conv1,  # Initial convolution
            resnet.bn1,    # Batch normalization
            nn.ReLU(inplace=True),
            resnet.maxpool # Max pooling to reduce spatial size
        )

        # Take ResNet residual layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify stride in the last residual block to maintain spatial resolution
        self.layer4[0].conv1.stride = (1, 1)
        if self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (1, 1)

        # 1x1 convolution to project ResNet features to the desired dimension
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Flattened and permuted feature sequence of shape
                          (batch_size, seq_len, d_model).
        """
        # Pass through initial convolutional stem
        x = self.stem(x)

        # Pass through the residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Project features to d_model channels
        x = self.proj(x)

        # Flatten spatial dimensions into a sequence
        x = x.flatten(2)       # shape: (batch_size, d_model, H*W)
        x = x.permute(0, 2, 1) # shape: (batch_size, seq_len, d_model)
        return x