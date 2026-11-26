"""Unit tests for the MP268 sentinel-value detector.

Expected project integration:
- sentinel_value() is defined in src/mp268/checks.py
- pytest is run from the repository root

Run:
    pytest -q tests/test_sentinel.py
"""

import pandas as pd
import pytest

from mp268.checks import sentinel_value


def test_detects_repeated_staff_sentinel():
    """Repeated 9999 values in the staff column should be reported."""
    train = pd.DataFrame(
        {
            "staff": [12, 15, 9999, 9999, 9999, 18],
            "daily_sales": [1200, 1500, 1700, 1800, 1900, 2100],
        }
    )

    assert sentinel_value(train) == {"staff": {9999: 3}}


def test_ignores_normal_zero_values():
    """Zero is a valid value in many columns and should not be flagged."""
    train = pd.DataFrame(
        {
            "staff": [0, 0, 0, 12, 15],
            "daily_sales": [0, 100, 200, 300, 400],
        }
    )

    assert sentinel_value(train) == {}


def test_ignores_sentinel_that_occurs_only_once():
    """A single sentinel-like value should not meet the repetition threshold."""
    train = pd.DataFrame(
        {
            "staff": [12, 15, 9999, 18],
            "daily_sales": [100, 200, 300, 400],
        }
    )

    assert sentinel_value(train) == {}


def test_detects_multiple_sentinel_values_and_columns():
    train = pd.DataFrame(
        {
            "staff": [9999, 9999, 10, 11],
            "temperature": [-999, -999, 24, 25],
        }
    )

    assert sentinel_value(train) == {
        "staff": {9999: 2},
        "temperature": {-999: 2},
    }


def test_ignores_text_columns():
    train = pd.DataFrame(
        {
            "staff": [10, 12, 14],
            "station_note": ["9999", "9999", "normal"],
        }
    )

    assert sentinel_value(train) == {}


def test_returns_empty_dict_when_no_sentinels_exist():
    train = pd.DataFrame(
        {
            "staff": [8, 10, 12, 14],
            "daily_sales": [1000, 1200, 1400, 1600],
        }
    )

    assert sentinel_value(train) == {}


def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="train must be a pandas DataFrame"):
        sentinel_value([1, 2, 9999])
