import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.layers import Act, Norm

class ourConv3D(nn.Module):
    def __init__(self, dropout_prob, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(dropout_prob),

            # Second conv block
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(dropout_prob),

            # Third conv block
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(dropout_prob),
        )

        # Make it independent of spatial size: pool to 1×1×1
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier head only depends on channel dim (256), not D/H/W
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 256, 1, 1, 1) -> (B, 256)
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)        # (B, 256, D', H', W')
        x = self.global_pool(x)     # (B, 256, 1, 1, 1)
        x = self.classifier(x)      # (B, num_classes)
        return x
    




# BASED ON MONAI'S UNet IMPLEMENTATION
class UNetEncoderClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_prob: float = 0.0,
        spatial_dims: int = 3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size: int = 3,
        num_res_units: int = 2,
        act: str | tuple = Act.PRELU,
        norm: str | tuple = Norm.INSTANCE,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):
        """
        Encoder-only UNet-like classifier for 3D volumes:
        in -> downsampling blocks -> bottleneck -> global pool -> MLP
        """
        super().__init__()

        if len(channels) < 2:
            raise ValueError("len(channels) must be at least 2.")
        if len(strides) != len(channels) - 1:
            raise ValueError("len(strides) must be len(channels) - 1.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.adn_ordering = adn_ordering

        # ----- build encoder (down path + bottom) -----
        encoder_blocks = nn.ModuleList()

        # Top block: in_channels -> channels[0] with stride strides[0]
        in_c = in_channels
        out_c = channels[0]
        stride = strides[0]
        encoder_blocks.append(
            self._make_down_block(in_c, out_c, stride)
        )

        # Intermediate down blocks: channels[i-1] -> channels[i] with stride strides[i]
        for i in range(1, len(channels) - 1):
            in_c = channels[i - 1]
            out_c = channels[i]
            stride = strides[i]
            encoder_blocks.append(
                self._make_down_block(in_c, out_c, stride)
            )

        # Bottom block: channels[-2] -> channels[-1] with stride 1 (no further downsampling)
        encoder_blocks.append(
            self._make_down_block(channels[-2], channels[-1], stride=1)
        )

        # Wrap in Sequential for simplicity
        self.encoder = nn.Sequential(*encoder_blocks)

        # ----- global pooling + classifier head -----
        feature_dim = channels[-1]  # last encoder channel size

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, C, 1, 1, 1) -> (B, C)
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def _make_down_block(self, in_c: int, out_c: int, stride: int) -> nn.Module:
        """
        Roughly matches UNet._get_down_layer from MONAI.
        Uses ResidualUnit if num_res_units > 0, else a simple Convolution block.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                self.spatial_dims,
                in_c,
                out_c,
                strides=stride,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout_prob,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        else:
            return Convolution(
                self.spatial_dims,
                in_c,
                out_c,
                strides=stride,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout_prob,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, D, H, W)
        """
        x = self.encoder(x)         # (B, C_enc, D', H', W'), C_enc = channels[-1]
        x = self.global_pool(x)     # (B, C_enc, 1, 1, 1)
        x = self.classifier(x)      # (B, num_classes)
        return x
