<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/Transformers-4.44+-orange" />
  <img src="https://img.shields.io/badge/LoRA-PEFT-green" />
  <img src="https://img.shields.io/badge/Model-ViT--Base--Patch16--224-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-success" />
</p>
#  🌍  Parameter-Efficient Fine-Tuning of Vision Transformers on EuroSAT Using LoRA  
### A Complete Study of Accuracy, Efficiency, and Practical Deployment

This project investigates **Parameter-Efficient Fine-Tuning (PEFT)** for **Vision Transformers (ViTs)** using **Low-Rank Adaptation (LoRA)**.  
We fine-tune the **ViT-Base (Patch16-224)** model on the **EuroSAT RGB remote-sensing dataset** and compare:

-  **Full Fine-Tuning** (all 85M+ parameters updated)  
- **LoRA Fine-Tuning** (only ~0.5% parameters updated)  

The goal is to determine whether LoRA can deliver competitive accuracy while drastically reducing computation, memory usage, and trainable parameters — making ViTs practical for **low-resource and edge-device scenarios**.

---

#  1. Motivation

Vision Transformers perform extremely well, but:

- They require **expensive GPUs**  
- Full fine-tuning updates **tens of millions of parameters**  
- Storing separate fine-tuned models for multiple tasks becomes infeasible  
- Training time & memory usage grow rapidly with scale  

**Low-Rank Adaptation (LoRA)** solves this by:

- Freezing the backbone  
- Injecting small *rank-r* trainable matrices into attention layers  
- Training only these lightweight modules  

This enables:

-  Faster training  
- Lower memory usage  
- Much smaller task-specific weights  
- On-device fine-tuning possibilities  

This project verifies these benefits empirically using a real dataset.

---

#  2. Dataset Overview — EuroSAT RGB

We use the **EuroSAT RGB** dataset via HuggingFace (`blanchon/EuroSAT_RGB`):

- 27,000 satellite images  
- Resolution: 64×64 px  
- 10 land-cover classes  
- Based on Sentinel-2 satellite imagery  
- Class examples:  
  - Residential  
  - Industrial  
  - River  
  - SeaLake  
  - Pasture  
  - Forest  
  - Highway  
  - PermanentCrop  
  - HerbaceousVegetation  
  - AnnualCrop  

The dataset represents a **domain shift** from the ImageNet pretraining data.  
This makes it an ideal test case for PEFT methods like LoRA.

---

#  3. Research Objective

We evaluate whether **LoRA** can:

- Achieve accuracy comparable to *full fine-tuning*  
- Reduce the number of trainable parameters dramatically  
- Reduce training time  
- Maintain generalization to unseen EuroSAT test samples  
- Enable lightweight deployment through a **Gradio inference app**

This replicates real-world constraints where **compute or memory may be limited**.

---

#  4. Methodology

## 4.1 Model: Vision Transformer (ViT-Base)

- 12 Transformer encoder blocks  
- Hidden size 768  
- 12 attention heads  
- Pretrained on ImageNet-1k  
- Classification head replaced to fit 10 EuroSAT labels  

---

## 4.2 Fine-Tuning Approaches

###  **Full Fine-Tuning**
- Entire ViT backbone unfrozen  
- All weights updated (~85M params)  
- Highest computational cost  

###  **LoRA Fine-Tuning**
- Backbone frozen  
- LoRA rank-8 adapters inserted into:
  - Query projection  
  - Key projection  
  - Value projection  

Only **~442K parameters** are trainable.

**LoRA Config:**

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
