import torch
import torch.nn as nn
import torch.optim as optim

class FederatedClient:
    def __init__(self, model, dataloader, device, class_weights=None, lr=0.0005):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )

    def train(self, epochs=1):
        self.model.train()

        for _ in range(epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()
