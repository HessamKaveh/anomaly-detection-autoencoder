import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_train_loader
from model import ConvAutoencoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_train_loader(batch_size=128)

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    history = {"train_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Reconstruction Loss: {train_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "autoencoder.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Autoencoder Training Loss (Normal Class Only)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curve.png"))
    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    train()
