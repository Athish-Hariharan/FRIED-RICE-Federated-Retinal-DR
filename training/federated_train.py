import torch
from torch.utils.data import DataLoader
from models.resnet_model import ResNetDR
from data.dataset import AptosDataset
from training.transforms import train_transforms, val_transforms
from federated.client import FederatedClient
from federated.server import fed_avg
from federated.utils import create_dirichlet_partitions

import pandas as pd
import argparse
import json
import os
import matplotlib.pyplot as plt


# ===============================
# Argument Parser
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--mu", type=float, default=0.0)
parser.add_argument("--log_name", type=str, required=True)
args = parser.parse_args()

alpha = args.alpha
mu = args.mu
log_name = args.log_name


# ===============================
# Setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("logs/results", exist_ok=True)
os.makedirs("logs/plots", exist_ok=True)


# ===============================
# Compute Class Weights
# ===============================
df = pd.read_csv("data/raw/aptos/train_1.csv")
class_counts = df.iloc[:, 1].value_counts().sort_index()
total_samples = class_counts.sum()
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float32)


# ===============================
# Load Dataset
# ===============================
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


# ===============================
# Dirichlet Non-IID Partition
# ===============================
client_datasets = create_dirichlet_partitions(
    train_dataset,
    num_clients=3,
    alpha=alpha
)

clients = []
client_sizes = []

for subset in client_datasets:
    loader = DataLoader(subset, batch_size=16, shuffle=True)
    model = ResNetDR().to(device)

    clients.append(
        FederatedClient(
            model,
            loader,
            device,
            class_weights=class_weights,
            mu=mu
        )
    )

    client_sizes.append(len(subset))


# ===============================
# Initialize Global Model
# ===============================
global_model = ResNetDR().to(device)

rounds = 20
local_epochs = 1

round_accuracies = []


# ===============================
# Federated Training Loop
# ===============================
for r in range(rounds):

    client_weights = []

    for client in clients:
        client.model.load_state_dict(global_model.state_dict())
        weights = client.train(global_model, epochs=local_epochs)
        client_weights.append(weights)

    global_model = fed_avg(global_model, client_weights, client_sizes)

    # ---- Evaluate ----
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

    acc = 100 * correct / total
    round_accuracies.append(acc)

    print(f"Round {r+1}/{rounds} - Global Val Acc: {acc:.2f}%")


# ===============================
# Save JSON Results
# ===============================
# ---- Save Results ----
os.makedirs("logs/results", exist_ok=True)

results = {
    "method": args.method,
    "alpha": args.alpha,
    "mu": args.mu,
    "round_accuracies": round_accuracies,
    "best_accuracy": max(round_accuracies),
    "final_accuracy": round_accuracies[-1]
}

filename = f"logs/results/{args.method}_alpha_{args.alpha}"
if args.method == "fedprox":
    filename += f"_mu_{args.mu}"

filename += ".json"

with open(filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {filename}")

# ===============================
# Plot Accuracy vs Round
# ===============================
plt.figure()
plt.plot(range(1, len(round_accuracies) + 1), round_accuracies)
plt.xlabel("Round")
plt.ylabel("Validation Accuracy (%)")
plt.title(f"Alpha={alpha}, Mu={mu}")
plt.grid(True)

plt.savefig(f"logs/plots/{log_name}_round_curve.png")
plt.close()


print("\nTraining Complete.")
print(f"Best Accuracy: {max(round_accuracies):.2f}%")
print(f"Final Accuracy: {round_accuracies[-1]:.2f}%")
print(f"Saved log: logs/results/{log_name}.json")
print(f"Saved plot: logs/plots/{log_name}_round_curve.png")
