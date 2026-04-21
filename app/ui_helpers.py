from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ACCENT_GREEN = "#2E6F57"
ACCENT_BLUE = "#2E5B9A"
INK = "#123127"
MUTED = "#577267"
BORDER = "#D7E4DE"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F4F8F6"


def apply_theme() -> None:
    st.set_page_config(page_title="AgriSense Mini", page_icon="🌿", layout="wide")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top right, rgba(112, 172, 146, 0.16), transparent 22%),
                    linear-gradient(180deg, #f8fbf9 0%, #fcfdfc 100%);
                color: {INK};
            }}
            .main .block-container {{
                max-width: 1260px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
            h1, h2, h3 {{
                color: {INK};
                letter-spacing: -0.02em;
            }}
            p, label, .stMarkdown, .stCaption {{
                color: {INK};
            }}
            div[data-testid="stMetric"] {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                padding: 0.9rem 1rem;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(18, 49, 39, 0.05);
            }}
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                border-radius: 18px;
            }}
            .ag-shell {{
                padding: 0.3rem 0 0.8rem 0;
            }}
            .ag-hero {{
                background:
                    linear-gradient(135deg, rgba(46, 111, 87, 0.10), rgba(46, 91, 154, 0.08)),
                    {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 28px;
                padding: 1.6rem 1.7rem;
                box-shadow: 0 18px 42px rgba(18, 49, 39, 0.07);
            }}
            .ag-eyebrow {{
                color: {ACCENT_BLUE};
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.75rem;
                font-weight: 700;
                margin-bottom: 0.75rem;
            }}
            .ag-subtitle {{
                color: {MUTED};
                max-width: 62rem;
                font-size: 1rem;
                line-height: 1.55;
                margin-top: 0.6rem;
            }}
            .ag-pill-row {{
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            }}
            .ag-pill {{
                border: 1px solid {BORDER};
                background: rgba(255,255,255,0.72);
                color: {INK};
                padding: 0.45rem 0.7rem;
                border-radius: 999px;
                font-size: 0.84rem;
            }}
            .ag-card {{
                border: 1px solid {BORDER};
                border-radius: 22px;
                padding: 1rem 1.05rem;
                background: {SURFACE};
                box-shadow: 0 12px 28px rgba(18, 49, 39, 0.05);
                height: 100%;
            }}
            .ag-card.alt {{
                background: {SURFACE_ALT};
            }}
            .ag-section-title {{
                font-size: 0.82rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {ACCENT_BLUE};
                font-weight: 700;
                margin-bottom: 0.65rem;
            }}
            .ag-subtle {{
                color: {MUTED};
                font-size: 0.92rem;
            }}
            .ag-prediction {{
                font-size: 1.6rem;
                font-weight: 700;
                margin: 0.2rem 0 0.35rem 0;
            }}
            .ag-model-tag {{
                color: {ACCENT_GREEN};
                font-weight: 700;
                font-size: 0.85rem;
            }}
            .ag-divider {{
                height: 1px;
                background: {BORDER};
                margin: 0.95rem 0;
            }}
            .ag-list {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem 0.8rem;
            }}
            .ag-meta-item {{
                background: rgba(255,255,255,0.72);
                border: 1px solid {BORDER};
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
            }}
            .ag-meta-label {{
                color: {MUTED};
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.2rem;
            }}
            .ag-meta-value {{
                color: {INK};
                font-weight: 600;
            }}
            .ag-bar {{
                margin-bottom: 0.7rem;
            }}
            .ag-bar-head {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                margin-bottom: 0.28rem;
                color: {INK};
            }}
            .ag-bar-track {{
                background: #E6EFEB;
                height: 10px;
                border-radius: 999px;
                overflow: hidden;
            }}
            .ag-bar-fill {{
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, {ACCENT_GREEN}, {ACCENT_BLUE});
            }}
            .ag-table-caption {{
                color: {MUTED};
                font-size: 0.9rem;
                margin-top: -0.2rem;
                margin-bottom: 0.5rem;
            }}
            .stButton > button {{
                border-radius: 14px;
                border: 1px solid {ACCENT_GREEN};
                background: {ACCENT_GREEN};
                color: white;
                font-weight: 700;
                padding: 0.65rem 1rem;
            }}
            .stSelectbox label, .stFileUploader label {{
                font-weight: 700;
                color: {INK};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_card_start(title: str, subtle: str | None = None, alt: bool = False) -> None:
    card_class = "ag-card alt" if alt else "ag-card"
    subtle_html = f'<div class="ag-subtle" style="margin-top:0.35rem;">{subtle}</div>' if subtle else ""
    st.markdown(
        f"""
        <div class="{card_class}" style="margin-bottom:0.85rem;padding-bottom:0.8rem;">
            <div class="ag-section-title">{title}</div>
            {subtle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card_end() -> None:
    return None


def render_hero() -> None:
    st.markdown(
        """
        <div class="ag-shell">
            <div class="ag-hero">
                <div class="ag-eyebrow">Multi-Modal Agricultural AI</div>
                <h1 style="margin:0;">AgriSense Mini</h1>
                <div style="font-size:1.18rem;font-weight:600;margin-top:0.35rem;">
                    Robust Multi-Modal Representation Learning for Agricultural Image + Metadata Classification
                </div>
                <div class="ag-subtitle">
                    An academic-style inspection interface for comparing metadata-only, image-only, and fused models,
                    with confidence behavior, mild-shift robustness, and lightweight explainability.
                </div>
                <div class="ag-pill-row">
                    <div class="ag-pill">PyTorch + scikit-learn</div>
                    <div class="ag-pill">Image + Metadata Fusion</div>
                    <div class="ag-pill">Calibration + Robustness</div>
                    <div class="ag-pill">Streamlit Research Demo</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(label: str, confidence: float, model_name: str) -> None:
    st.markdown(
        f"""
        <div class="ag-card">
            <div class="ag-model-tag">{model_name}</div>
            <div class="ag-prediction">{label.replace('_', ' ').title()}</div>
            <div class="ag-subtle">Top-class confidence</div>
            <div style="font-size:1.35rem;font-weight:700;margin-top:0.2rem;">{confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_snapshot(metadata_row: dict[str, str]) -> None:
    items: list[str] = []
    for key, value in metadata_row.items():
        items.append(
            "<div class=\"ag-meta-item\">"
            f"<div class=\"ag-meta-label\">{key.replace('_', ' ')}</div>"
            f"<div class=\"ag-meta-value\">{str(value).replace('_', ' ').title()}</div>"
            "</div>"
        )
    grid_html = "<div class=\"ag-list\">" + "".join(items) + "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)


def render_probability_bars(probabilities: dict[str, float]) -> None:
    bars = []
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        bars.append(
            f"""
            <div class="ag-bar">
                <div class="ag-bar-head">
                    <span>{label.replace('_', ' ').title()}</span>
                    <span>{probability:.1%}</span>
                </div>
                <div class="ag-bar-track">
                    <div class="ag-bar-fill" style="width:{probability * 100:.1f}%"></div>
                </div>
            </div>
            """
        )
    st.markdown("".join(bars), unsafe_allow_html=True)


def render_modality_breakdown(image_confidence: float, metadata_confidence: float, fused_confidence: float) -> None:
    frame = pd.DataFrame(
        {
            "Signal": ["Image head", "Metadata head", "Fused prediction"],
            "Confidence": [image_confidence, metadata_confidence, fused_confidence],
        }
    )
    fig, axis = plt.subplots(figsize=(5.7, 2.9))
    colors = [ACCENT_BLUE, ACCENT_GREEN, "#1E3A5F"]
    axis.barh(frame["Signal"], frame["Confidence"], color=colors, height=0.56)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_title("Modality Contribution Snapshot")
    axis.grid(axis="x", alpha=0.18)
    for idx, value in enumerate(frame["Confidence"]):
        axis.text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def feature_importance_figure(feature_names: Sequence[str], importances: Sequence[float]):
    fig, axis = plt.subplots(figsize=(6.5, 3.8))
    axis.barh(list(feature_names)[::-1], list(importances)[::-1], color=ACCENT_BLUE)
    axis.set_xlabel("Importance")
    axis.set_title("Metadata Feature Importance")
    axis.grid(axis="x", alpha=0.18)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def benchmark_figure(frame: pd.DataFrame):
    display = frame.copy()
    display["model"] = display["model"].str.replace("_", " ").str.title()
    split_order = {"test": 0, "test_shift": 1}
    display["split_order"] = display["split"].map(split_order)
    display = display.sort_values(by=["split_order", "accuracy"]).reset_index(drop=True)

    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    colors = [ACCENT_GREEN if split == "test" else ACCENT_BLUE for split in display["split"]]
    labels = [f"{model}\n{split.replace('_', ' ')}" for model, split in zip(display["model"], display["split"])]
    axis.bar(labels, display["accuracy"], color=colors, alpha=0.92)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Benchmark Snapshot")
    axis.grid(axis="y", alpha=0.18)
    axis.tick_params(axis="x", rotation=24)
    for idx, value in enumerate(display["accuracy"]):
        axis.text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def render_shift_callout(model_name: str, accuracy_drop: float | None, f1_drop: float | None) -> None:
    accuracy_text = "n/a" if accuracy_drop is None else f"{accuracy_drop:+.2f}"
    f1_text = "n/a" if f1_drop is None else f"{f1_drop:+.2f}"
    st.markdown(
        f"""
        <div class="ag-card alt">
            <div class="ag-section-title">Shift Sensitivity</div>
            <div style="font-weight:700;margin-bottom:0.35rem;">{model_name}</div>
            <div class="ag-subtle">Change from standard test split to mild-shift test split.</div>
            <div class="ag-divider"></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                <span>Accuracy gap</span><strong>{accuracy_text}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>Macro F1 gap</span><strong>{f1_text}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_metric_name(model_name: str) -> str:
    return model_name.replace("_", " ").title()


def get_top_line(frame: pd.DataFrame, metric: str) -> tuple[str, float] | None:
    if frame.empty or metric not in frame.columns:
        return None
    row = frame.sort_values(metric, ascending=False).iloc[0]
    return clean_metric_name(str(row["model"])), float(row[metric])


def calibration_figure(probabilities: dict[str, float]):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(5.4, 3.0))
    axis.plot(positions, values, marker="o", linewidth=2.2, color=ACCENT_GREEN)
    axis.fill_between(positions, values, color="#B9D8CB", alpha=0.5)
    axis.set_xticks(positions, [label.replace("_", " ").title() for label in labels])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Confidence")
    axis.set_title("Class Confidence Profile")
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig


def render_warning_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ag-card alt" style="border-color:#E3C98A;background:#FFF9EC;">
            <div class="ag-section-title" style="color:#8A5A00;">{title}</div>
            <div class="ag-subtle" style="color:#6B5632;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
