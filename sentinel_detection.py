"""Sentinel-value detection for MP268.

Integration note:
Copy DEFAULT_SENTINEL_VALUES and sentinel_value() into src/mp268/checks.py.
The existing findings.py looks for sentinel_value in mp268.checks first.

This detector does not flag merely frequent values such as 0. It reports
configured sentinel candidates only when they repeat at least MIN_COUNT times.
"""

from __future__ import annotations

import pandas as pd

# 9999 is the expected staff sentinel in the MP268 training data.
DEFAULT_SENTINEL_VALUES = {
    -999999, -99999, -9999, -999, 9999, 99999, 999999,
}

MIN_COUNT = 2


def sentinel_value(train: pd.DataFrame) -> dict:
    """Return repeated sentinel candidates by numeric column.

    Example return: {"staff": {9999: 45}}
    Returns {} when no configured sentinel repeats MIN_COUNT times.
    """
    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame")

    findings: dict = {}

    for column in train.select_dtypes(include="number").columns:
        counts = train[column].value_counts(dropna=True)
        suspicious = {
            value: int(count)
            for value, count in counts.items()
            if value in DEFAULT_SENTINEL_VALUES and count >= MIN_COUNT
        }
        if suspicious:
            findings[str(column)] = suspicious

    return findings
