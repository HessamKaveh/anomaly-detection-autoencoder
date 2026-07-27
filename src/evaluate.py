import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import get_test_loader, NORMAL_CLASS
from model import ConvAutoencoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "autoencoder.pt")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, _ = get_test_loader(batch_size=128)

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    errors, labels_binary = [], []
    sample_images, sample_recons, sample_errors, sample_is_anomaly = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            per_sample_error = ((outputs - images) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()

            errors.extend(per_sample_error.tolist())
            is_anomaly = (labels != NORMAL_CLASS).int().numpy()
            labels_binary.extend(is_anomaly.tolist())

            if len(sample_images) < 8:
                for i in range(min(8 - len(sample_images), images.size(0))):
                    sample_images.append(images[i].cpu())
                    sample_recons.append(outputs[i].cpu())
                    sample_errors.append(per_sample_error[i])
                    sample_is_anomaly.append(is_anomaly[i])

    errors = np.array(errors)
    labels_binary = np.array(labels_binary)

    auc = roc_auc_score(labels_binary, errors)
    print(f"ROC-AUC (normal=0 vs anomaly=other digits): {auc:.4f}")

    fpr, tpr, _ = roc_curve(labels_binary, errors)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))

    # هیستوگرام خطای بازسازی برای نرمال در برابر anomaly
    plt.figure(figsize=(6, 4))
    plt.hist(errors[labels_binary == 0], bins=50, alpha=0.6, label="Normal (digit 0)")
    plt.hist(errors[labels_binary == 1], bins=50, alpha=0.6, label="Anomaly (other digits)")
    plt.xlabel("Reconstruction Error (MSE)"); plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))

    # نمایش چند نمونه: تصویر اصلی / بازسازی‌شده / خطا
    fig, axes = plt.subplots(2, len(sample_images), figsize=(2 * len(sample_images), 4))
    for i in range(len(sample_images)):
        axes[0, i].imshow(sample_images[i].squeeze(), cmap="gray")
        tag = "Anomaly" if sample_is_anomaly[i] else "Normal"
        axes[0, i].set_title(f"{tag}\nerr={sample_errors[i]:.4f}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(sample_recons[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_reconstructions.png"))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"roc_auc": float(auc)}, f, indent=2)

    print("Saved ROC curve, error distribution, and sample reconstructions.")


if __name__ == "__main__":
    evaluate()
