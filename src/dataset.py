# dataset.py
import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):
    """
    A PyTorch Dataset wrapper for the EuroSAT RGB dataset loaded via HuggingFace.
    Ensures safe conversion to PIL images and applies transforms.
    """

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]

        # CASE 1: HF dict with raw bytes
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")

        # CASE 2: numpy array
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype("uint8")).convert("RGB")

        # CASE 3: list (convert to numpy)
        elif isinstance(img, list):
            img = Image.fromarray(np.array(img).astype("uint8")).convert("RGB")

        # CASE 4: PIL image
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = example["label"]
        return {"pixel_values": img, "label": label}
