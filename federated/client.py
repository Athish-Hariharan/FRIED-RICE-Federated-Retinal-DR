import torch
import torch.nn as nn
import torch.optim as optim

class FederatedClient:
    def __init__(self, model, dataloader, device, lr=0.001):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

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
