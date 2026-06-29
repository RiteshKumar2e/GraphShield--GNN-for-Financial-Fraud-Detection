import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print("Class Distribution:")
    for cls, cnt in zip(unique, counts):
        label = "Illicit" if cls == 1 else "Licit"
        print(f"  {label} ({cls}): {cnt:,} ({cnt/total*100:.1f}%)")
    return dict(zip(unique.tolist(), counts.tolist()))


def plot_class_distribution(y, save_path=None):
    unique, counts = np.unique(y, return_counts=True)
    labels = ["Licit" if c == 0 else "Illicit" for c in unique]
    colors = ["steelblue" if c == 0 else "crimson" for c in unique]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, counts, color=colors)
    plt.title("Class Distribution")
    plt.ylabel("Count")
    for i, cnt in enumerate(counts):
        plt.text(i, cnt + cnt * 0.01, str(cnt), ha="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
