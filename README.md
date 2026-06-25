# MMOC: Multimodal Oral Cancer Detection

> **Cross-modal integration of clinical information and visual concepts for cancer recognition from oral photographs**  
> 

---

## Overview

**MMOC** (*Multimodal Oral Cancer detection*) integrates structured clinical risk factors and predefined visual concepts into a Swin Transformer backbone for binary malignancy classification from oral photographs. Three contributions drive the design:

1. **Hierarchical cross-modal injection** — visual concepts injected at Layer 0 (56×56, C=96) via multi-head cross-attention; clinical risk factors fused at Layer 8 (14×14, C=384) via a sigmoid-gated module. Depths are motivated by CKA analysis and validated via a 12×12 layer-pair ablation.
2. **Dual-stream encoding** — lightweight MLP encoders project visual concept scores and tabular clinical variables into token sequences compatible with the backbone at each injection point.
3. **Cross-modal knowledge distillation** — a vision-only Swin-T student is trained via a joint CE + KL + feature distillation objective to inherit cross-modal correlations, enabling image-only inference at deployment.

---

## Datasets

| Dataset | Domain | Images | Malignant | Clinical | Concepts |
|---------|--------|--------|-----------|----------|----------|
| **OMPAA** *(proprietary)* | Oral | 2,967 | 10.6% | ✓ | ✓ |
| **OralCon** | Oral | 2,517 | 3.7% | ✓ | ✓ |
| **ISIC** | Dermatology | 25,331 | 32.0% | ✓ | — |
| **SkinCon** | Dermatology | 3,886 | 15.9% | — | ✓ |

Clinical features: age, sex, tobacco/alcohol history, anatomical location, medical history, induration.  
Visual concepts: border sharpness, color multiplicity, everted edges, surrounding tissue alteration.

---

## Results

### OMPAA

| Model | Modalities | AUC (%) | BACC (%) | Sen. (%) |
|-------|-----------|---------|---------|---------|
| Swin-T | Img | 87.10 | 82.85 | 71.08 |
| TabPFN | Cli + Con | 91.44 | 86.19 | 75.50 |
| **MMOC Teacher** | **Img + Cli + Con** | **94.50** | **90.91** | **84.34** |
| **MMOC Student** | **Img only** | **89.20** | **85.30** | **77.11** |

### Cross-domain & small-lesion generalization


All results: 5-fold stratified cross-validation, patient-level separation (60/20/20)

---

## Installation & Usage

```bash
git clone https://github.com/jerome-de-chauveron/MMOC-cancer-detection
cd MMOC-cancer-detection
pip install -r requirements.txt   # Python ≥ 3.9, PyTorch ≥ 2.0
```

```bash
# Teacher (multimodal)
python train_teacher.py --dataset ompaa --concept_layer 0 --clinical_layer 8 --epochs 45 --lr 5e-6

# Student distillation
python train_student.py --teacher_ckpt checkpoints/mmoc_teacher.pth --temperature 4.0 --alpha 0.7

# Inference (image only)
python predict.py --model_ckpt checkpoints/mmoc_student.pth --image path/to/lesion.jpg
```

---

## Key Hyperparameters

| Component | Value |
|-----------|-------|
| Backbone | Swin-Tiny, stages [2,2,6,2] |
| Concept injection | MCA, Layer 0 |
| Clinical injection | Gated addition, Layer 8 |
| Distillation loss weights | α=0.3 |
| Distillation temperature | T = 4 |
| Epoch (teacher)| 45|
| Epoch (teacher)|40|
| LR (teacher)| 1e-6|
| LR (student)| 1e-5|
| Optimizer| AdamW|


---



---

## License

Released under the [MIT License](LICENSE). The OMPAA dataset is proprietary and subject to a separate data-sharing agreement.
