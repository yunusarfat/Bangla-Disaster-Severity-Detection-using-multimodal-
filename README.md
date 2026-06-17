# 🌪️ Disaster Severity Classification — Multimodal Deep Learning

A multimodal deep learning system that classifies disaster severity from **images + text** using CLIP (vision) and BanglaBERT (text) with Cross-Modal Attention fusion. Built for the **Datathon competition** and optimized to run on **Kaggle**.

---

## 🧠 Model Architecture

```
Image ──► CLIP ViT-B/32 ──► img_feat (768)  ─┐
                                               ├──► CrossModalAttention ──► fused (512)
Text  ──► BanglaBERT    ──► txt_feat (768)  ─┘         │
                                                         │
Category ──► Embedding (32) ────────────────────────────┤
                                                         ▼
                                              Concat [768 + 768 + 512 + 32]
                                                         │
                                              Classifier Head (MLP)
                                                         │
                                              Severity Prediction (5 classes)
```

### Key Components

| Component | Detail |
|---|---|
| Vision Encoder | `openai/clip-vit-base-patch32` |
| Text Encoder | `csebuetnlp/banglabert` |
| Fusion | Cross-Modal Attention (bidirectional) |
| Category | Learned disaster-type embedding (8 categories) |
| Classifier | 3-layer MLP with LayerNorm + GELU + Dropout |

### Why Cross-Modal Attention?
Simple concatenation treats image and text as independent. Cross-Modal Attention lets each modality **query the other** — the image can suppress or amplify text signals and vice versa. This is critical for disaster severity where a blurry image paired with alarming text should still predict high severity.

---

## 📊 Severity Labels

| Label | Description |
|---|---|
| `Minimal` | Little to no visible damage |
| `Mild` | Minor damage, limited impact |
| `Moderate` | Significant damage, some casualties possible |
| `Severe` | Major destruction, high impact |
| `Catastrophic` | Complete devastation |

## 🗂️ Disaster Categories

`Drought` · `Earthquake` · `Flood` · `Human Damage` · `Landslides` · `Non Disaster` · `Tropical Storm` · `Wildfire`

---

## 📁 Project Structure

```
├── best_notebook_kaggle.ipynb   # Main training notebook (Kaggle-ready)
├── README.md
└── requirements.txt             # (optional) for local runs
```

### Expected Dataset Structure

```
your-kaggle-dataset/
├── train.csv
├── validation.csv
├── test.csv
└── images/
    ├── train/
    ├── validation/
    └── test/
```

### CSV Format

```
image_id, context, category, image_name, label
```

---

## 🚀 Running on Kaggle

### Step 1 — Upload your dataset
Add your dataset via **Notebook Settings → Add Data**. Note the dataset slug (the part after `/kaggle/input/`).

### Step 2 — (Recommended) Pre-download models
To avoid internet dependency during training, pre-download models locally and upload as a separate Kaggle dataset:

```python
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32').save_pretrained('./clip_model')
CLIPModel.from_pretrained('openai/clip-vit-base-patch32').save_pretrained('./clip_model')
AutoTokenizer.from_pretrained('csebuetnlp/banglabert').save_pretrained('./bangla_model')
AutoModel.from_pretrained('csebuetnlp/banglabert').save_pretrained('./bangla_model')
```

### Step 3 — Configure paths
In **Cell 2** of the notebook, set:

```python
DATASET_SLUG = 'your-dataset-name'   # ← your data slug
CLIP_MODEL   = '/kaggle/input/your-models-dataset/clip_model'    # if pre-downloaded
BANGLA_MODEL = '/kaggle/input/your-models-dataset/bangla_model'  # if pre-downloaded
```

### Step 4 — Enable GPU & Internet
- **Settings → Accelerator → GPU T4 x2** (or P100)
- **Settings → Internet → On** (needed if not pre-downloading models)

### Step 5 — Run All
Click **Run → Run All** and let it go. Outputs saved to `/kaggle/working/`.

---

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 30 | With early stopping |
| Batch size | 16 | Effective 32 with grad accumulation |
| Backbone LR | 1e-5 | Lower for pretrained weights |
| Head LR | 5e-4 | Higher for new layers |
| Max text length | 192 tokens | Captures full context |
| Image size | 224×224 | CLIP standard |
| Label smoothing | 0.2 | Reduces overconfidence |
| Mixup alpha | 0.4 | Image-level mixup augmentation |
| Patience | 6 epochs | Early stopping |
| Unfreeze epoch | 15 | Full backbone unfreeze mid-training |

---

## 🔧 Training Strategy

### Class Imbalance
- **Offline oversampling** — minority classes duplicated to reach `TARGET_COUNT=790` samples
- **WeightedRandomSampler** with square-root smoothing — avoids over-focusing on rarest class
- **Stronger augmentation** on synthetic (duplicated) rows via `aug_strong`

### Progressive Unfreezing
- **Epochs 1–14**: Bottom backbone layers frozen, only top layers + head train
- **Epoch 15+**: All layers unfrozen with reduced backbone LR (`1e-6`) to avoid catastrophic forgetting

### Augmentation Pipeline (Training)
`RandomResizedCrop` → `HorizontalFlip` → `ColorJitter` → `GaussNoise` → `MotionBlur` → `RandomRain` → `CoarseDropout` → `Rotate`

### Test-Time Augmentation (TTA)
5 TTA passes averaged at inference: original, horizontal flip, random crop, color jitter, crop+flip.

### Pseudo-Labeling
High-confidence test predictions (≥ 0.70 probability) are added back into training for additional fine-tuning rounds.

---

## 📦 Dependencies

```
torch
transformers
timm
albumentations>=2.0
scikit-learn
pandas
numpy
Pillow
tqdm
ftfy
regex
langdetect
accelerate
```

Install all:
```bash
pip install transformers accelerate timm albumentations langdetect ftfy regex scikit-learn
```

---

## 📤 Output Files

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint by validation F1 |
| `best_model_pseudo.pt` | Best checkpoint after pseudo-labeling |
| `test_probs_clip.npy` | Raw softmax probabilities for ensemble |
| `test_image_ids.npy` | Image IDs aligned with probabilities |
| `submission_clip_only.csv` | Final submission file |

---

## 📈 Evaluation Metrics

- **Primary**: Weighted F1-score
- **Secondary**: Macro F1-score, Accuracy
- Full `classification_report` printed per class after training

---

## 💡 Tips for Better Results

- Use **GPU T4 x2** on Kaggle for faster training
- Pre-download HuggingFace models to avoid rate limits / timeouts
- If val F1 plateaus early, try reducing `PATIENCE` or increasing `LABEL_SMOOTH`
- Ensemble `test_probs_clip.npy` with other model runs for a leaderboard boost

---

## 📄 License

This project is for research and competition purposes.
