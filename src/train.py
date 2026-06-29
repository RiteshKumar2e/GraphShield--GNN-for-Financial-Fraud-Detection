import torch
import torch.nn.functional as F
import numpy as np


def compute_class_weights(data):
    """Inverse-frequency class weights to handle imbalance."""
    y = data.y[data.train_mask].numpy()
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, data, optimizer, class_weights=None, device="cpu"):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask].to(device),
        weight=class_weights.to(device) if class_weights is not None else None,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def train_model(model, data, epochs=200, lr=0.001, weight_decay=5e-4, use_class_weights=True, device="cpu"):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights = compute_class_weights(data) if use_class_weights else None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, optimizer, class_weights, device)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(
                out[data.val_mask],
                data.y[data.val_mask],
                weight=class_weights.to(device) if class_weights is not None else None,
            ).item()
        history["val_loss"].append(val_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, history
