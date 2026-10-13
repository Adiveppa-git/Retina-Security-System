import torch
import torch.nn as nn
import torchvision.models as models


class AntiSpoofCNN(nn.Module):
    """
    Anti-Spoofing Liveness Classifier.
    Analyzes high-frequency specular reflections and micro-textures
    to classify input image into:
    Class 1: REAL HUMAN EYE
    Class 0: FAKE / SPOOF (Photo, LCD Screen, Printed Image)
    """
    def __init__(self):
        super(AntiSpoofCNN, self).__init__()

        print("Initializing Anti-Spoofing Liveness CNN Model...")
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [Fake, Real]
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits