from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MetadataSchema:
    categorical_columns: list[str]
    numeric_columns: list[str]


class MetadataPreprocessor:
    """Reusable sklearn-based tabular preprocessing for metadata inputs."""

    def __init__(self, categorical_columns: Sequence[str], numeric_columns: Sequence[str]) -> None:
        self.schema = MetadataSchema(list(categorical_columns), list(numeric_columns))
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "MetadataPreprocessor":
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self.transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.schema.numeric_columns),
                ("categorical", categorical_pipeline, self.schema.categorical_columns),
            ],
            remainder="drop",
        )
        self.transformer.fit(frame)
        self.feature_names_ = self._extract_feature_names()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("MetadataPreprocessor must be fit before calling transform.")
        transformed = self.transformer.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return transformed.astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    def get_feature_names(self) -> list[str]:
        if not self.feature_names_:
            raise RuntimeError("Feature names are unavailable before fitting the preprocessor.")
        return self.feature_names_

    def _extract_feature_names(self) -> list[str]:
        if self.transformer is None:
            return []
        feature_names: list[str] = []
        if self.schema.numeric_columns:
            feature_names.extend(self.schema.numeric_columns)
        if self.schema.categorical_columns:
            categorical_transformer = self.transformer.named_transformers_["categorical"]
            one_hot = categorical_transformer.named_steps["onehot"]
            categorical_feature_names = one_hot.get_feature_names_out(self.schema.categorical_columns)
            feature_names.extend(categorical_feature_names.tolist())
        return feature_names

