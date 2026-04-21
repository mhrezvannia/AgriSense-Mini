# AgriSense Mini
## Robust Multi-Modal Representation Learning for Agricultural Image + Metadata Classification

### Overview
AgriSense Mini is a research-engineering portfolio project focused on agricultural classification from two complementary input streams: leaf imagery and structured field metadata. The repository compares classical tabular baselines, deep metadata models, image-only encoders, and a late-fusion multi-modal model built with PyTorch. The goal is to demonstrate disciplined modeling, evaluation, and communication rather than inflate the project into a paper or a production platform.

### Why this project matters
Agricultural ML systems rarely operate in clean i.i.d. settings. Appearance can shift with season, imaging condition, and field context, while metadata can be incomplete or only weakly predictive on its own. This project frames multi-modal learning as a practical representation problem: learn image features, learn metadata representations, fuse them, and test whether fusion improves both predictive performance and robustness under a mild distribution shift.

### Key Features
- PyTorch image-only classifier with a compact ResNet encoder
- PyTorch metadata MLP and fused image + metadata model
- scikit-learn metadata baseline with transparent feature importance
- Reusable CSV + folder dataset pipeline with missing metadata handling
- Robustness-aware evaluation with an explicit mild-shift holdout split
- Calibration, confidence reporting, confusion matrices, and comparison tables
- Lightweight Grad-CAM for image inspection and global metadata feature importance
- Streamlit demo for interactive sample inspection

### Project Structure
```text
agrisense_mini/
├── app/                  # Streamlit demo and UI helpers
├── configs/              # YAML configuration
├── data/                 # Sample agricultural-style images and metadata CSV
├── notebooks/            # Lightweight demo notebook
├── src/
│   ├── data/             # Dataset, preprocessing, split logic
│   ├── models/           # Image, metadata, and fusion architectures
│   ├── training/         # Training entrypoints and trainer utilities
│   ├── evaluation/       # Metrics, calibration, robustness, and plotting
│   ├── explainability/   # Grad-CAM and metadata importance
│   └── utils/            # Config, IO, seeds, sample-data generation
├── outputs/              # Saved models, figures, and reports
├── tests/                # Smoke tests for core components
├── requirements.txt
├── run_demo.py
└── README.md
```

### Dataset
The repository includes a small synthetic agricultural-style sample dataset generated locally with Pillow. Images resemble simple leaf health patterns across three classes: `healthy`, `leaf_spot`, and `rust`. The metadata CSV contains categorical fields such as region, humidity band, season, soil type, imaging condition, and sensor view.

This metadata is synthetic and intended for portfolio demonstration only. The README and code make this explicit to avoid overstating scientific realism. The project structure is designed so the same pipeline can be pointed at a real agricultural folder dataset plus CSV with minimal changes.

### Models
`scikit-learn metadata baseline`
Uses logistic regression on imputed, one-hot encoded metadata to provide a transparent tabular baseline and feature-importance signal.

`PyTorch metadata-only model`
Uses a compact MLP encoder over preprocessed metadata features to show a neural tabular baseline parallel to the classical model.

`PyTorch image-only model`
Uses a ResNet-based encoder plus projection head and classifier to learn image embeddings from agricultural imagery.

`PyTorch fused model`
Encodes the image and metadata separately, concatenates the learned embeddings, and predicts with a fused classifier head. The model also exposes modality-specific logits for practical inspection of image vs metadata confidence.

### Evaluation
Evaluation covers:
- Accuracy, macro precision, macro recall, and macro F1
- Confusion matrices for each split
- ROC-AUC when the split permits stable computation
- Expected calibration error and calibration curves
- Comparison across `sklearn_metadata`, `torch_metadata`, `image_model`, and `fusion_model`
- A mild robustness check by holding out `imaging_condition=overcast` samples as a shift split

The main intended takeaway is modest and credible: multi-modal fusion can improve predictive stability and confidence behavior relative to single-modality baselines, especially when image appearance shifts.

### Demo
Install dependencies and run the full workflow:

```bash
pip install -r requirements.txt
python -m src.training.train_metadata --config configs/default.yaml
python -m src.training.train_image --config configs/default.yaml
python -m src.training.train_fusion --config configs/default.yaml
python -m src.evaluation.reporting --config configs/default.yaml
python run_demo.py
```

The Streamlit app supports:
- Sample selection or image upload
- Editable metadata fields
- Model selection across baseline and deep models
- Prediction display with confidence
- Metadata feature-importance visualization
- Grad-CAM overlay for the image-only model
- Fusion confidence breakdown via modality-specific logits

### Results
The bundled synthetic sample dataset was trained locally to produce example artifacts under `outputs/`. On the standard test split:

| Model | Accuracy | Macro F1 | ECE |
| --- | ---: | ---: | ---: |
| `sklearn_metadata` | 0.444 | 0.381 | 0.238 |
| `torch_metadata` | 0.444 | 0.367 | 0.094 |
| `image_model` | 0.667 | 0.556 | 0.455 |
| `fusion_model` | 0.667 | 0.536 | 0.296 |

On the mild shift split (`imaging_condition=overcast`), the image-only model dropped from `0.667` to `0.333` accuracy, while the fused model held at `0.667`. This is a small synthetic experiment, not a scientific claim, but it does support the project’s intended message: combining modalities can improve resilience when visual appearance shifts.

Report files are written to `outputs/reports/`, figures to `outputs/figures/`, and model artifacts to `outputs/models/`. The combined comparison table is saved to `outputs/reports/comparison_table.csv`, and the shift-gap summary is saved to `outputs/reports/robustness_summary.json`.

### CV-Ready Summary
Designed and implemented a multi-modal agricultural AI pipeline combining PyTorch-based image and fusion models with scikit-learn baselines for metadata-driven classification. Evaluated representation quality, robustness under mild distribution shift, confidence behavior, and lightweight explainability, and built a Streamlit demo for interactive model inspection.

### Future Work
- Replace synthetic metadata with real agronomic or geospatial covariates
- Add self-supervised or contrastive pretraining for stronger image representations
- Extend robustness evaluation to stronger out-of-distribution and missing-modality settings
- Explore richer fusion mechanisms such as gated fusion or cross-attention
- Investigate domain-specific foundation-model adaptation for agricultural imagery

### License
MIT
