import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from data.dataset import AptosDataset
from training.transforms import train_transforms, val_transforms
from models.baseline import BaselineDR
from models.resnet_model import ResNetDR

# -----------------------------
# Data Loaders
# -----------------------------
def get_dataloaders(batch_size=16):

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# -----------------------------
# Training Function
# -----------------------------
def train_model(class_weights, epochs=30, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(batch_size=16)

    model = ResNetDR().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_val_acc = 0

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []

    for epoch in range(epochs):

        # ---------------------
        # Training
        # ---------------------
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        train_acc = 100 * correct / total

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_weighted.pth")

    print("\nBest Validation Accuracy:", best_val_acc)

    # -----------------------------
    # Load Best Model for Final Evaluation
    # -----------------------------
    model.load_state_dict(torch.load("best_model_weighted.pth"))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print("\nFinal Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_plot.png")

    # -----------------------------
    # Plot Loss
    # -----------------------------
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("loss_plot.png")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("data/raw/aptos/train_1.csv")
    class_counts = df.iloc[:,1].value_counts().sort_index()

    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)

    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)

    print("Class Weights:", class_weights)

    train_model(class_weights, epochs=30, lr=0.001)
