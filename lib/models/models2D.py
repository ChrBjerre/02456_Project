import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, densenet121

class ourConv2D(nn.Module):
    def __init__(self, dropout_prob=0.2, in_channels=3, num_classes=12, num_views = 6, pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_prob),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_prob),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_prob),
        )

        # Make it independent of spatial size: pool to 1×1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(256*num_views, 512), # since we have 6 viewpoints
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, V, C, H, W = x.shape  # V = 6 views
        x = x.view(B * V, C, H, W)

        # Run each view through the shared CNN
        feats = self.features(x)                 # (B*6, 256, h', w')
        feats = self.global_pool(feats)          # (B*6, 256, 1, 1)

        # Reshape back into (B, 6, 256)
        feats = feats.view(B, V, 256)

        # Concat all views -> (B, 1536)
        fused = feats.reshape(B, -1)

        # Classify
        out = self.classifier(fused)
        return out


class LateFusionResNet18(nn.Module):
    def __init__(self, num_classes=12, pretrained=False, dropout_prob=0.2, num_views = 6):
        super().__init__()

        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove last FC layer
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # output = (B,512,1,1)
        self.feature_dim = base.fc.in_features  # 512
        

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * num_views, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, V, H, W)
        B, V, C, H, W = x.shape

        # Add channel dimension
        x = x.view(B * V, C, H, W)

        # Run through encoder
        f = self.encoder(x)       
        f = f.view(B, V, -1)      

        fused = f.reshape(B, -1)  

        return self.classifier(fused)
    


class LateFusionDenseNet121(nn.Module):
    def __init__(self, num_classes=12, pretrained=False, dropout_prob=0.2, num_views = 6):
        super().__init__()

        base = densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = base.features
        self.feature_dim = 1024

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * num_views, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, V, C, H, W = x.shape

        x = x.view(B * V, C, H, W)

        f = self.encoder(x)
        f = nn.functional.adaptive_avg_pool2d(f, 1)
        f = f.view(B, V, -1)

        fused = f.reshape(B, -1)
        return self.classifier(fused)
    
#### FROM MONAI SOURCE CODE:
# --------------------------
# Basic UNet block
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# --------------------------
# UNet Encoder only
# --------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Only return deepest features
        x = self.down1(x)
        x = self.down2(self.pool(x))
        x = self.down3(self.pool(x))
        x = self.down4(self.pool(x))
        x = self.pool(x)          # final bottleneck (B, 512, h, w)
        return x


# --------------------------
# Multi-view UNet encoder LATE FUSION
# --------------------------
class LateFusionUNetEncoder(nn.Module):
    def __init__(self, in_channels = 3, num_views=6, dropout_prob=0.2, num_classes = 12, pretrained=False):
        super().__init__()
        self.num_views = num_views

        self.encoder = UNetEncoder(in_channels=in_channels)

        # fuse 6 × bottleneck features
        self.fusion_bottleneck = DoubleConv(512 * num_views, 1024)
        self.dropout = nn.Dropout2d(dropout_prob)

        # final projection to feature vector
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, 1024, 1, 1)
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes)
    )



    def forward(self, x):
        # x: (B, V, C, H, W)
        B, V, C, H, W = x.shape
        assert V == self.num_views

        # Flatten views into batch: (B*V, C, H, W)
        x = x.view(B * V, C, H, W)

        # Encode all (B*V) images at once
        feats = self.encoder(x)                # (B*V, 512, h, w)
        _, Cenc, h, w = feats.shape           # Cenc should be 512

        # Reshape back: (B, V, 512, h, w) and fuse over views
        feats = feats.view(B, V * Cenc, h, w) # (B, V*512, h, w)

        fused = self.fusion_bottleneck(feats) # (B, 1024, h, w)
        fused = self.dropout(fused)

        feature_vec = self.fc(fused)          # (B, 1024)
        out = self.classifier(feature_vec)    # (B, num_classes)

        return out
