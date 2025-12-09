# gradio_app.py
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr
from transformers import ViTForImageClassification
from peft import PeftModel, LoraConfig, get_peft_model
from torchvision import transforms
from datasets import load_dataset


def load_labels():
    ds = load_dataset("blanchon/EuroSAT_RGB")
    return ds["train"].features["label"].names


label_names = load_labels()
num_labels = len(label_names)

# Same transform as training
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_lora_model():
    base = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["query", "key", "value"]
    )

    model = get_peft_model(base, lora_cfg)
    model.load_adapter("vit_lora_finetuned_eurosat")  # load your fine-tuned weights
    return model.eval()


model = load_lora_model()


def predict(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = F.softmax(logits, dim=1)[0]

    top3 = torch.topk(probs, 3)
    return {label_names[idx]: float(top3.values[i]) for i, idx in enumerate(top3.indices)}


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="EuroSAT Land-Cover Classification (LoRA ViT)",
)

app.launch()
