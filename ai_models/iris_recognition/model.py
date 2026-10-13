import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class IrisRecognitionCNN(nn.Module):
    """
    Deep Biometric Metric Feature Extractor for Iris Recognition.
    Maps input Daugman polar normalized iris strips (or iris crops)
    into a scale-invariant, L2-normalized 512-dimensional feature embedding space.
    """
    def __init__(self, embedding_dim=512):
        super(IrisRecognitionCNN, self).__init__()

        print(f"Initializing High-Precision Deep Metric Encoder (Embedding Dim: {embedding_dim})...")

        # Use MobileNetV3-Large pre-trained backbone for rich feature extraction
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Project 960 features -> 512-D embedding space
        self.fc = nn.Sequential(
            nn.Linear(960, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, 3, H, W]
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        embedding = self.fc(feat)

        # Crucial for Cosine Similarity & Open-Set Biometric Matching: L2 Normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding