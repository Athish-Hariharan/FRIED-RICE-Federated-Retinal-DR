import torch.nn as nn
import torchvision.models as models

class ResNetDR(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetDR, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
