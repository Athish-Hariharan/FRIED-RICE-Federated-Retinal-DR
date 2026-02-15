import torch
from torch.utils.data import DataLoader
from models.resnet_model import ResNetDR
from data.dataset import AptosDataset
from training.transforms import train_transforms, val_transforms
from federated.client import FederatedClient
from federated.server import fed_avg
from federated.utils import create_non_iid_partitions
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Compute Class Weights (same as centralized) ----
df = pd.read_csv("data/raw/aptos/train_1.csv")
class_counts = df.iloc[:,1].value_counts().sort_index()
total_samples = class_counts.sum()
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float32)

# ---- Load Dataset ----
train_dataset = AptosDataset(
    csv_file="data/raw/aptos/train_1.csv",
    img_dir="data/raw/aptos/train_images",
    transform=train_transforms
)

val_dataset = AptosDataset(
    csv_file="data/raw/aptos/valid.csv",
    img_dir="data/raw/aptos/val_images",
    transform=val_transforms
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---- Create Non-IID Clients ----
client_datasets = create_non_iid_partitions(train_dataset, num_clients=3)

clients = []
client_sizes = []

for subset in client_datasets:
    loader = DataLoader(subset, batch_size=16, shuffle=True)
    model = ResNetDR().to(device)

    clients.append(
        FederatedClient(model, loader, device, class_weights=class_weights)
    )

    client_sizes.append(len(subset))

# ---- Initialize Global Model ----
global_model = ResNetDR().to(device)

rounds = 20
local_epochs = 1

for r in range(rounds):

    client_weights = []

    for client in clients:
        client.model.load_state_dict(global_model.state_dict())
        weights = client.train(epochs=local_epochs)
        client_weights.append(weights)

    # Weighted FedAvg
    global_model = fed_avg(global_model, client_weights, client_sizes)

    # ---- Evaluate Global Model ----
    global_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Round {r+1}/{rounds} - Global Val Acc: {100*correct/total:.2f}%")
