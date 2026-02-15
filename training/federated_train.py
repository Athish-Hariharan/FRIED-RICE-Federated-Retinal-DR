import torch
from torch.utils.data import DataLoader
from models.resnet_model import ResNetDR
from data.dataset import AptosDataset
from training.transforms import train_transforms, val_transforms
from federated.client import FederatedClient
from federated.server import fed_avg
from federated.utils import create_non_iid_partitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full dataset
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

# Create non-IID clients
client_datasets = create_non_iid_partitions(train_dataset, num_clients=3)
clients = []

for subset in client_datasets:
    loader = DataLoader(subset, batch_size=16, shuffle=True)
    model = ResNetDR().to(device)
    clients.append(FederatedClient(model, loader, device))

# Initialize global model
global_model = ResNetDR().to(device)

rounds = 10
local_epochs = 1

for r in range(rounds):

    client_weights = []

    for client in clients:
        client.model.load_state_dict(global_model.state_dict())
        weights = client.train(epochs=local_epochs)
        client_weights.append(weights)

    global_model = fed_avg(global_model, client_weights)

    # Evaluate global model
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
