# train_full.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from datasets import load_dataset
from torchvision import transforms

from dataset import EuroSATDataset
from utils import train_one_epoch, evaluate, count_trainable_params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("blanchon/EuroSAT_RGB")
    train_ds, val_ds = dataset["train"], dataset["validation"]

    label_names = train_ds.features["label"].names
    num_labels = len(label_names)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_pt = EuroSATDataset(train_ds, transform)
    val_pt   = EuroSATDataset(val_ds, transform)

    train_loader = DataLoader(train_pt, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_pt, batch_size=64, shuffle=False)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    total_p, train_p = count_trainable_params(model)
    print("Trainable params:", train_p)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")

    model.save_pretrained("vit_full_finetuned_eurosat")


if __name__ == "__main__":
    main()
