Centralized Training — ResNet18 + Weighted Loss
Dataset

APTOS 2019

Train: 2930 images

Validation: 366 images

Class distribution:

0: 1434

1: 300

2: 808

3: 154

4: 234

Model

ResNet18 (ImageNet pretrained)

Final FC modified to 5 classes

Training Config

Loss: Weighted CrossEntropy

Optimizer: SGD (lr=0.001, momentum=0.9)

Batch size: 16

Epochs: 30

Image size: 224×224

Best Validation Accuracy

85.79%

Macro F1

0.73

Key Observations

Minority recall improved significantly

Overfitting gap ≈ 12%

Architecture depth critical

Transfer learning highly beneficial
