import torch
import torch.nn as nn
import torch.optim as optim

class FederatedClient:

    def __init__(self, model, dataloader, device,
                 class_weights=None,
                 mu=0.0):  # mu = proximal coefficient

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.mu = mu

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9
        )

    def train(self, global_model, epochs=1):

        self.model.train()

        global_weights = {
            name: param.clone().detach()
            for name, param in global_model.state_dict().items()
        }

        for _ in range(epochs):

            for images, labels in self.dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # ----- FedProx proximal term -----
                if self.mu > 0:
                    prox_term = 0.0
                    for param, global_param in zip(
                            self.model.parameters(),
                            global_model.parameters()
                        ):
                        prox_term += torch.norm(param - global_param.detach()) ** 2

                    loss += (self.mu / 2) * prox_term
                # ----------------------------------

                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()
