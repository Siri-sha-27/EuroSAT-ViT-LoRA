# utils.py
import torch
from tqdm.auto import tqdm


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for ONE epoch.
    Returns: (loss, accuracy)
    """
    model.train()
    correct, total = 0, 0
    loss_sum = 0.0

    for batch in tqdm(loader, desc="Training"):
        x = batch["pixel_values"].to(device)
        y = torch.tensor(batch["label"]).to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=x)
        loss = criterion(outputs.logits, y)
        loss.backward()
        optimizer.step()

        preds = outputs.logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    return loss_sum / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Eval pass → returns accuracy.
    """
    model.eval()
    correct, total = 0, 0

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = torch.tensor(batch["label"]).to(device)

        outputs = model(pixel_values=x)
        preds = outputs.logits.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total
