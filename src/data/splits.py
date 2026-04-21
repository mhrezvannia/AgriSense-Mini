from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    frame: pd.DataFrame,
    label_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    shift_column: str | None = None,
    shift_values: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits and an optional mild-shift test split."""
    working = frame.copy()
    shift_mask = pd.Series(False, index=working.index)
    if shift_column and shift_values:
        shift_mask = working[shift_column].isin(list(shift_values))
    shift_frame = working.loc[shift_mask].reset_index(drop=True)
    base_frame = working.loc[~shift_mask].reset_index(drop=True)

    train_frame, temp_frame = train_test_split(
        base_frame,
        test_size=val_size + test_size,
        stratify=base_frame[label_col],
        random_state=random_state,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_size,
        stratify=temp_frame[label_col],
        random_state=random_state,
    )
    return {
        "train": train_frame.reset_index(drop=True),
        "val": val_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
        "test_shift": shift_frame.reset_index(drop=True),
    }

