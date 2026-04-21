# AgriSense Mini
## Robust Multi-Modal Representation Learning for Agricultural Image + Metadata Classification

### Overview
AgriSense Mini is a portfolio-grade research-engineering project for multimodal agricultural classification. It combines image inputs and structured metadata, compares classical and deep baselines, and evaluates not just prediction quality but also calibration, mild distribution shift, and practical model inspection.

This repository is intentionally positioned between a toy tutorial and a paper-scale research codebase. The emphasis is on clean PyTorch implementation, multimodal reasoning, strong evaluation discipline, and a polished demo that is easy to explain in an interview.

### Why this project matters
Agricultural computer vision rarely operates in clean i.i.d. settings. Leaf appearance changes with season, lighting, imaging setup, and environmental context. Metadata can be incomplete, weak on its own, or only useful in combination with image evidence. This project treats multimodal learning as a representation problem:

- learn image embeddings from leaf imagery
- learn metadata embeddings from structured tabular signals
- fuse both modalities in a shared classifier
- compare fusion against single-modality baselines
- inspect confidence and robustness under a mild shift

### Key Features
- PyTorch image-only classifier with a ResNet-based encoder
- PyTorch metadata-only MLP baseline
- PyTorch late-fusion model for image + metadata classification
- scikit-learn metadata baseline with transparent feature importance
- CSV + folder dataset pipeline with missing-metadata handling
- Train / validation / test / mild-shift split support
- Accuracy, precision, recall, macro F1, ROC-AUC, calibration, and confusion matrices
- Lightweight explainability with Grad-CAM and metadata importance plots
- Streamlit demo for interactive model inspection
- Reliability guard in the UI for visibly shifted or out-of-scope uploaded images

### Project Structure
```text
agrisense_mini/
├── app/                  # Streamlit demo and UI helpers
├── configs/              # YAML configuration
├── data/                 # Sample images and metadata CSV
├── notebooks/            # Lightweight demo notebook
├── src/
│   ├── data/             # Dataset, preprocessing, split logic
│   ├── evaluation/       # Metrics, calibration, robustness, reporting, plots
│   ├── explainability/   # Grad-CAM and metadata feature importance
│   ├── models/           # Image, metadata, and fusion models
│   ├── training/         # Training entrypoints and trainer utilities
│   └── utils/            # Config, IO, seeds, synthetic sample data generation
├── outputs/              # Saved models, figures, and reports
├── tests/                # Smoke tests for core components
├── requirements.txt
├── run_demo.py
└── README.md
```

### Dataset
The bundled dataset is synthetic and generated locally with Pillow. It contains three leaf-health classes:

- `healthy`
- `leaf_spot`
- `rust`

The metadata file contains structured agricultural-style fields such as:

- `region`
- `humidity_band`
- `temperature_band`
- `season`
- `soil_type`
- `imaging_condition`
- `sensor_view`

The current sample dataset is designed for a credible portfolio demonstration, not for scientific benchmarking. The images and metadata are synthetic, and the repository is explicit about that. The main value of the dataset is to exercise the full multimodal pipeline and support clear engineering and evaluation narratives.

### Models
`scikit-learn metadata baseline`
Uses logistic regression on imputed and one-hot encoded metadata. This provides a transparent baseline and interpretable metadata importance signal.

`PyTorch metadata-only model`
Uses an MLP over preprocessed metadata features. This shows a neural metadata baseline parallel to the classical model.

`PyTorch image-only model`
Uses a ResNet18-based encoder with a projection head and classifier. The backbone can use pretrained ImageNet weights for a stronger representation starting point.

`PyTorch fused model`
Uses separate encoders for image and metadata, concatenates the embeddings, and predicts with a shared head. It also exposes modality-specific logits for practical confidence inspection in the demo.

### Evaluation
The project evaluates four model families:

1. `sklearn_metadata`
2. `torch_metadata`
3. `image_model`
4. `fusion_model`

The evaluation stack includes:

- accuracy
- macro precision / recall / F1
- confusion matrix
- ROC-AUC where applicable
- expected calibration error
- calibration curves
- comparison tables across models and splits
- a mild robustness check using `imaging_condition=overcast` as a held-out shift split

This is not presented as a claim of state-of-the-art performance. The intended message is narrower and more credible: multimodal pipelines should be evaluated for both predictive quality and behavior under modest shift, and the engineering should make those tradeoffs visible.

### Demo
The Streamlit app presents the project as an academic inspection dashboard rather than a product UI. It supports:

- sample selection or image upload
- editable metadata fields
- model selection across metadata, image-only, and fusion variants
- confidence profiles over classes
- metadata feature-importance plots
- Grad-CAM overlays for image-only predictions
- modality confidence breakdowns for the fusion model
- warnings for uploaded images that are visually shifted relative to the training distribution

### Installation
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### How to Run
Train the full pipeline:

```bash
.\.venv\Scripts\python.exe -m src.training.train_metadata --config configs/default.yaml
.\.venv\Scripts\python.exe -m src.training.train_image --config configs/default.yaml
.\.venv\Scripts\python.exe -m src.training.train_fusion --config configs/default.yaml
.\.venv\Scripts\python.exe -m src.evaluation.reporting --config configs/default.yaml
```

Launch the demo:

```bash
.\.venv\Scripts\python.exe run_demo.py
```

Run tests:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

### Results
The current bundled run on the synthetic sample dataset produces the following standard test metrics:

| Model | Accuracy | Macro F1 | ECE |
| --- | ---: | ---: | ---: |
| `sklearn_metadata` | 0.833 | 0.833 | 0.199 |
| `torch_metadata` | 0.625 | 0.570 | 0.267 |
| `image_model` | 1.000 | 1.000 | 0.0003 |
| `fusion_model` | 1.000 | 1.000 | 0.0007 |

On the mild shift split:

| Model | Accuracy | Macro F1 | ECE |
| --- | ---: | ---: | ---: |
| `sklearn_metadata` | 0.619 | 0.611 | 0.221 |
| `torch_metadata` | 0.667 | 0.665 | 0.313 |
| `image_model` | 1.000 | 1.000 | 0.0004 |
| `fusion_model` | 1.000 | 1.000 | 0.0011 |

These numbers should be interpreted carefully:

- they come from a synthetic dataset bundled for demonstration
- they show that the image branch is currently much stronger than the metadata branch
- they are useful for illustrating the pipeline and evaluation logic
- they are not evidence of real-world agricultural generalization

Saved reports are written to:

- `outputs/models/`
- `outputs/figures/`
- `outputs/reports/comparison_table.csv`
- `outputs/reports/robustness_summary.json`

### Reliability Notes
The Streamlit app includes a simple similarity-based warning for uploaded images that do not visually resemble the training distribution. This is useful for highlighting that a classifier can still output a class label even when the input is visibly shifted or out of scope.

For example:

- a salad image may receive a class prediction, but that prediction is not meaningful because the input is outside the intended domain
- a real leaf image may still trigger a warning if it differs significantly from the synthetic training distribution

This is presented as a practical model-inspection feature, not as a full out-of-distribution detection system.


### Future Work
- Replace the synthetic sample set with a real agricultural image dataset plus real metadata
- Add stronger missing-modality and out-of-distribution evaluation
- Explore self-supervised or contrastive pretraining for the image encoder
- Test richer fusion mechanisms such as gated fusion or cross-attention
- Extend the pipeline toward temporal, geospatial, or domain-specific foundation-model settings

### License
MIT
