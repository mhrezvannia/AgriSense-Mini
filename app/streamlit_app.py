from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app.ui_helpers import (
    apply_theme,
    benchmark_figure,
    calibration_figure,
    clean_metric_name,
    feature_importance_figure,
    get_top_line,
    render_hero,
    render_metadata_snapshot,
    render_modality_breakdown,
    render_prediction_card,
    render_probability_bars,
    render_shift_callout,
    render_warning_panel,
    section_card_end,
    section_card_start,
)
from src.data.dataset import default_image_transform
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.fusion_model import FusionClassifier
from src.models.image_model import ImageClassifier
from src.models.metadata_model import MetadataClassifier
from src.utils.config import load_config
from src.utils.io import load_joblib, load_json, load_torch_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "default.yaml")
MODEL_OPTIONS = ["Fusion", "Image Only", "Torch Metadata", "Sklearn Metadata"]


@dataclass
class PredictionBundle:
    label: str
    confidence: float
    probabilities: dict[str, float]
    extras: dict[str, Any]


@dataclass
class DomainAssessment:
    nearest_similarity: float
    centroid_similarity: float
    threshold: float
    is_out_of_domain: bool


def _artifact_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


@st.cache_resource
def load_preprocessor():
    path = _artifact_path("outputs", "models", "metadata_preprocessor.joblib")
    return load_joblib(path) if path.exists() else None


@st.cache_resource
def load_sklearn_model():
    path = _artifact_path("outputs", "models", "sklearn_metadata.joblib")
    return load_joblib(path) if path.exists() else None


@st.cache_resource
def load_torch_model(model_key: str):
    checkpoint_paths = {
        "image": _artifact_path("outputs", "models", "image_model.pt"),
        "torch_metadata": _artifact_path("outputs", "models", "torch_metadata.pt"),
        "fusion": _artifact_path("outputs", "models", "fusion_model.pt"),
    }
    checkpoint_path = checkpoint_paths[model_key]
    if not checkpoint_path.exists():
        return None
    checkpoint = load_torch_checkpoint(checkpoint_path)
    class_names = checkpoint["class_names"]
    if model_key == "image":
        model = ImageClassifier(
            num_classes=len(class_names),
            embedding_dim=CONFIG["model"]["image_embedding_dim"],
            dropout=CONFIG["model"]["dropout"],
        )
    elif model_key == "torch_metadata":
        model = MetadataClassifier(
            input_dim=checkpoint["metadata_dim"],
            hidden_dims=CONFIG["model"]["metadata_hidden_dims"],
            num_classes=len(class_names),
            dropout=CONFIG["model"]["dropout"],
        )
    else:
        model = FusionClassifier(
            metadata_input_dim=checkpoint["metadata_dim"],
            num_classes=len(class_names),
            image_embedding_dim=CONFIG["model"]["image_embedding_dim"],
            metadata_hidden_dims=CONFIG["model"]["metadata_hidden_dims"],
            fusion_hidden_dims=CONFIG["model"]["fusion_hidden_dims"],
            dropout=CONFIG["model"]["dropout"],
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@st.cache_data
def load_metadata() -> pd.DataFrame:
    return pd.read_csv(_artifact_path(CONFIG["paths"]["metadata_csv"]))


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    path = _artifact_path("outputs", "reports", "comparison_table.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_robustness_summary() -> dict[str, dict[str, float]]:
    path = _artifact_path("outputs", "reports", "robustness_summary.json")
    return load_json(path) if path.exists() else {}


@st.cache_data
def load_image_report() -> dict[str, Any]:
    path = _artifact_path("outputs", "reports", "image_reports.json")
    return load_json(path) if path.exists() else {}


def prepare_metadata_input(metadata_row: dict[str, Any], preprocessor) -> torch.Tensor:
    metadata_frame = pd.DataFrame([metadata_row])
    metadata_features = preprocessor.transform(metadata_frame)
    return torch.tensor(metadata_features, dtype=torch.float32)


def prepare_image_input(image: Image.Image) -> torch.Tensor:
    transform = default_image_transform(CONFIG["data"]["image_size"], train=False)
    return transform(image.convert("RGB")).unsqueeze(0)


def _normalized_embeddings(model: ImageClassifier, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(batch, return_embedding=True)
        embeddings = outputs["embedding"]
    return torch.nn.functional.normalize(embeddings, dim=1)


@st.cache_resource
def load_reference_bank():
    model = load_torch_model("image")
    metadata = load_metadata()
    if model is None or metadata.empty:
        return None

    tensors: list[torch.Tensor] = []
    for image_name in metadata["image_filename"].tolist():
        image_path = ROOT / CONFIG["paths"]["image_dir"] / image_name
        image = Image.open(image_path).convert("RGB")
        tensors.append(prepare_image_input(image))

    batch = torch.cat(tensors, dim=0)
    embeddings = _normalized_embeddings(model, batch)
    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix.fill_diagonal_(-1.0)
    nearest_scores = similarity_matrix.max(dim=1).values.cpu().numpy()
    centroid = torch.nn.functional.normalize(embeddings.mean(dim=0, keepdim=True), dim=1)
    threshold = float(np.percentile(nearest_scores, 10))

    return {
        "embeddings": embeddings.cpu(),
        "centroid": centroid.cpu(),
        "threshold": threshold,
        "nearest_scores": nearest_scores,
    }


def assess_image_domain(image: Image.Image) -> DomainAssessment | None:
    model = load_torch_model("image")
    bank = load_reference_bank()
    if model is None or bank is None:
        return None

    query = prepare_image_input(image)
    embedding = _normalized_embeddings(model, query).cpu()
    similarities = torch.matmul(bank["embeddings"], embedding.squeeze(0))
    nearest_similarity = float(similarities.max().item())
    centroid_similarity = float(torch.matmul(bank["centroid"], embedding.squeeze(0)).item())
    threshold = float(bank["threshold"])
    is_out_of_domain = nearest_similarity < threshold
    return DomainAssessment(
        nearest_similarity=nearest_similarity,
        centroid_similarity=centroid_similarity,
        threshold=threshold,
        is_out_of_domain=is_out_of_domain,
    )


def predict_with_model(model_name: str, image: Image.Image, metadata_row: dict[str, Any]) -> PredictionBundle | None:
    class_names = CONFIG["model"]["class_names"]
    preprocessor = load_preprocessor()
    image_tensor = prepare_image_input(image)
    metadata_tensor = prepare_metadata_input(metadata_row, preprocessor) if preprocessor is not None else None

    if model_name == "Sklearn Metadata":
        model = load_sklearn_model()
        if model is None or metadata_tensor is None:
            return None
        probabilities = model.predict_proba(metadata_tensor.numpy())[0]
        extras: dict[str, Any] = {}
    elif model_name == "Torch Metadata":
        model = load_torch_model("torch_metadata")
        if model is None or metadata_tensor is None:
            return None
        with torch.no_grad():
            logits = model(metadata_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        extras = {}
    elif model_name == "Image Only":
        model = load_torch_model("image")
        if model is None:
            return None
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        extras = {}
    else:
        model = load_torch_model("fusion")
        if model is None or metadata_tensor is None:
            return None
        with torch.no_grad():
            outputs = model(image_tensor, metadata_tensor, return_parts=True)
            probabilities = torch.softmax(outputs["logits"], dim=1).squeeze(0).numpy()
            image_conf = torch.softmax(outputs["image_logits"], dim=1).max().item()
            metadata_conf = torch.softmax(outputs["metadata_logits"], dim=1).max().item()
        extras = {"image_confidence": image_conf, "metadata_confidence": metadata_conf}

    prediction_index = int(torch.tensor(probabilities).argmax().item())
    return PredictionBundle(
        label=class_names[prediction_index],
        confidence=float(probabilities[prediction_index]),
        probabilities={label: float(prob) for label, prob in zip(class_names, probabilities)},
        extras=extras,
    )


def get_selected_image(uploaded_file, image_path: Path) -> Image.Image:
    if uploaded_file is None:
        return Image.open(image_path).convert("RGB")
    image_bytes = BytesIO(uploaded_file.getvalue())
    return Image.open(image_bytes).convert("RGB")


def render_gradcam_panel(image: Image.Image, model_name: str) -> None:
    if model_name != "Image Only":
        return
    model = load_torch_model("image")
    if model is None:
        st.info("Image-only checkpoint not found.")
        return
    image_tensor = prepare_image_input(image)
    target_layer = model.encoder.target_layer
    gradcam = GradCAM(model=model, target_layer=target_layer)
    heatmap = gradcam.generate(image_tensor)
    overlay = overlay_heatmap(image=image, heatmap=heatmap)
    st.image(overlay, caption="Grad-CAM overlay for the selected image", use_container_width=True)


def render_overview_metrics(comparison_table: pd.DataFrame) -> None:
    if comparison_table.empty:
        return
    test_rows = comparison_table.loc[comparison_table["split"] == "test"].copy()
    shift_rows = comparison_table.loc[comparison_table["split"] == "test_shift"].copy()
    best_test = get_top_line(test_rows, "accuracy")
    best_shift = get_top_line(shift_rows, "accuracy")
    best_f1 = get_top_line(test_rows, "f1_macro")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        if best_test:
            st.metric("Best Test Accuracy", f"{best_test[1]:.2%}", best_test[0])
    with metric_cols[1]:
        if best_shift:
            st.metric("Best Shift Accuracy", f"{best_shift[1]:.2%}", best_shift[0])
    with metric_cols[2]:
        if best_f1:
            st.metric("Best Test Macro F1", f"{best_f1[1]:.2%}", best_f1[0])


def main() -> None:
    apply_theme()
    render_hero()

    metadata_frame = load_metadata()
    comparison_table = load_comparison_table()
    robustness_summary = load_robustness_summary()
    image_report = load_image_report()
    render_overview_metrics(comparison_table)

    st.markdown("### Interactive Sample Inspection")
    control_col, preview_col = st.columns([0.9, 1.1], gap="large")

    with control_col:
        section_card_start(
            "Experiment Controls",
            "Select a reference sample, optionally replace the image, and compare models under the same metadata context.",
        )
        sample_options = metadata_frame["sample_id"].tolist()
        selected_id = st.selectbox("Sample ID", sample_options, index=0)
        model_name = st.selectbox("Model", MODEL_OPTIONS, index=0)
        uploaded_file = st.file_uploader("Optional image upload", type=["png", "jpg", "jpeg"])
        selected_row = metadata_frame.loc[metadata_frame["sample_id"] == selected_id].iloc[0].to_dict()
        section_card_end()

        section_card_start(
            "Editable Metadata",
            "Synthetic metadata fields can be changed to inspect how the tabular signal affects fused and metadata-only predictions.",
            alt=True,
        )
        editable_metadata: dict[str, Any] = {}
        for column in CONFIG["data"]["categorical_columns"]:
            options = sorted(value for value in metadata_frame[column].dropna().unique().tolist())
            default_value = selected_row.get(column)
            default_index = options.index(default_value) if default_value in options else 0
            editable_metadata[column] = st.selectbox(
                column.replace("_", " ").title(),
                options,
                index=default_index,
                key=f"meta_{column}",
            )
        run_prediction = st.button("Run Model Inspection", type="primary", use_container_width=True)
        section_card_end()

    with preview_col:
        section_card_start(
            "Sample Preview",
            "Image context and metadata snapshot used for the current model inspection pass.",
        )
        image_path = ROOT / CONFIG["paths"]["image_dir"] / selected_row["image_filename"]
        image = get_selected_image(uploaded_file=uploaded_file, image_path=image_path)
        st.image(
            image,
            caption=f"{selected_row['image_filename']}  |  Ground truth: {selected_row['label']}",
            use_container_width=True,
        )
        render_metadata_snapshot(
            {
                "sample_id": selected_row["sample_id"],
                "label": selected_row["label"],
                "region": editable_metadata["region"],
                "season": editable_metadata["season"],
                "imaging_condition": editable_metadata["imaging_condition"],
                "sensor_view": editable_metadata["sensor_view"],
            }
        )
        section_card_end()

    if run_prediction:
        prediction = predict_with_model(model_name=model_name, image=image, metadata_row=editable_metadata)
        domain_assessment = assess_image_domain(image)
        if prediction is None:
            st.error("Model artifacts are missing. Train the models first using the scripts in `src/training/`.")
            return

        if domain_assessment and domain_assessment.is_out_of_domain:
            render_warning_panel(
                "Out-of-Domain Warning",
                "This image does not look similar to the leaf-style images used to train the demo models. "
                "The class probabilities are therefore not reliable, even if one class appears with moderate confidence.",
            )

        st.markdown("### Prediction Analysis")
        result_col, detail_col = st.columns([0.9, 1.1], gap="large")
        with result_col:
            render_prediction_card(prediction.label, prediction.confidence, model_name)
            st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
            section_card_start("Confidence Profile", "Relative confidence over the class space.", alt=True)
            render_probability_bars(prediction.probabilities)
            st.pyplot(calibration_figure(prediction.probabilities))
            if domain_assessment:
                st.caption(
                    f"Nearest training-image similarity: {domain_assessment.nearest_similarity:.2f} | "
                    f"reference threshold: {domain_assessment.threshold:.2f}"
                )
            section_card_end()

        with detail_col:
            section_card_start(
                "Interpretation Layer",
                "Practical inspection outputs rather than claims of full interpretability.",
            )
            if prediction.extras:
                render_modality_breakdown(
                    image_confidence=float(prediction.extras["image_confidence"]),
                    metadata_confidence=float(prediction.extras["metadata_confidence"]),
                    fused_confidence=float(prediction.confidence),
                )
            elif model_name == "Sklearn Metadata":
                feature_path = _artifact_path("outputs", "reports", "sklearn_metadata", "feature_importance.json")
                if feature_path.exists():
                    feature_data = load_json(feature_path)
                    st.pyplot(feature_importance_figure(feature_data["feature_names"][:8], feature_data["importances"][:8]))
            elif model_name == "Image Only":
                render_gradcam_panel(image=image, model_name=model_name)
            else:
                st.info("No additional inspection panel is available for the selected model.")
            section_card_end()

        if model_name == "Fusion":
            image_only = predict_with_model("Image Only", image=image, metadata_row=editable_metadata)
            torch_meta = predict_with_model("Torch Metadata", image=image, metadata_row=editable_metadata)
            if image_only and torch_meta:
                st.markdown("### Model Comparison Snapshot")
                compare_cols = st.columns(3)
                with compare_cols[0]:
                    render_prediction_card(image_only.label, image_only.confidence, "Image Only")
                with compare_cols[1]:
                    render_prediction_card(torch_meta.label, torch_meta.confidence, "Torch Metadata")
                with compare_cols[2]:
                    render_prediction_card(prediction.label, prediction.confidence, "Fusion")

        if domain_assessment and not image_report:
            st.info("Robustness artifacts are unavailable, so only sample-level reliability checks are shown.")

    st.markdown("### Evaluation Snapshot")
    eval_col, benchmark_col = st.columns([0.8, 1.2], gap="large")
    with eval_col:
        section_card_start(
            "Benchmark Reading Guide",
            "Standard test and mild-shift test performance based on saved reports in outputs/reports.",
        )
        st.markdown(
            """
            <div class="ag-subtle">
                The benchmark panel is intended to support an interview-friendly narrative:
                metadata alone is informative but limited, image models can be stronger yet shift-sensitive,
                and fusion can stabilize performance under mild distribution change.
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_card_end()

        if not comparison_table.empty:
            filtered = comparison_table.loc[comparison_table["split"].isin(["test", "test_shift"])].copy()
            filtered["model"] = filtered["model"].map(clean_metric_name)
            st.markdown('<div class="ag-table-caption">Saved comparison metrics from the most recent training run.</div>', unsafe_allow_html=True)
            st.dataframe(
                filtered[["model", "split", "accuracy", "f1_macro", "ece"]]
                .rename(columns={"f1_macro": "macro_f1"})
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

    with benchmark_col:
        if not comparison_table.empty:
            filtered = comparison_table.loc[comparison_table["split"].isin(["test", "test_shift"])].copy()
            st.pyplot(benchmark_figure(filtered))

    robustness_cols = st.columns(4)
    robustness_keys = ["sklearn_metadata", "torch_metadata", "image_model", "fusion_model"]
    for column, key in zip(robustness_cols, robustness_keys):
        with column:
            summary = robustness_summary.get(key, {})
            render_shift_callout(
                clean_metric_name(key),
                summary.get("accuracy_drop"),
                summary.get("f1_drop"),
            )


if __name__ == "__main__":
    main()
