from __future__ import annotations

from collections.abc import Iterable
import pandas as pd


# Common numeric placeholders used when a sensor/value is missing or invalid.
# Extend this set only when the dataset/domain confirms another impossible value.
DEFAULT_SENTINEL_VALUES = {-999999, -99999, -9999, -999, 9999, 99999, 999999}


def sentinel_value(
    df: pd.DataFrame,
    sentinel_values: Iterable[int | float] | None = None,
    min_count: int = 5,
) -> dict[str, dict[int | float, int]]:
    """Find repeated known sentinel numbers in numeric columns.

    Uses Series.value_counts() as required. Returns only sentinel candidates
    occurring at least ``min_count`` times, grouped by column.

    The caller should confirm candidates against the dataset's valid ranges;
    unusual values are not automatically deleted or replaced.
    """
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    candidates = set(
        DEFAULT_SENTINEL_VALUES if sentinel_values is None else sentinel_values
    )
    findings: dict[str, dict[int | float, int]] = {}

    for column in df.select_dtypes(include="number").columns:
        counts = df[column].value_counts(dropna=False)
        repeated = {
            value: int(counts.get(value, 0))
            for value in candidates
            if counts.get(value, 0) >= min_count
        }
        if repeated:
            findings[str(column)] = repeated

    return findings
