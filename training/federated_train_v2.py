import os
import json
import torch
from torch.utils.data import DataLoader

from data.unified_dataset import UnifiedDRDataset
from federated.partition import dirichlet_partition
from federated.client import FederatedClient
from federated.server import fed_avg
from models.resnet_model import ResNetDR
from training.transforms import train_transforms, val_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/kaggle/working/logs"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_checkpoint(model, round_num):
    torch.save(model.state_dict(), f"{SAVE_DIR}/checkpoint_round_{round_num}.pth")


def save_log(log_data):
    with open(f"{SAVE_DIR}/training_log.json", "w") as f:
        json.dump(log_data, f, indent=4)


def main():

    dataset = UnifiedDRDataset(
        root_dir="/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy",
        transform=train_transforms
    )

    clients_data = dirichlet_partition(dataset, num_clients=10, alpha=0.5)

    clients = []
    client_sizes = []

    for subset in clients_data:
        loader = DataLoader(subset, batch_size=16, shuffle=True)

        model = ResNetDR().to(DEVICE)

        clients.append(FederatedClient(model, loader, DEVICE))
        client_sizes.append(len(subset))

    global_model = ResNetDR().to(DEVICE)

    rounds = 20
    logs = {"round_accuracy": []}

    for r in range(rounds):

        client_weights = []

        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            weights = client.train(epochs=1)
            client_weights.append(weights)

        global_model = fed_avg(global_model, client_weights, client_sizes)

        # Dummy validation (replace later)
        acc = torch.rand(1).item() * 100

        logs["round_accuracy"].append(acc)

        print(f"Round {r+1} Acc: {acc:.2f}")

        save_checkpoint(global_model, r+1)
        save_log(logs)


if __name__ == "__main__":
    main()
