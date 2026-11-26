"""Tests use tiny in-memory examples; they do not load any CSV files."""
import pandas as pd
from src.data_checks import duplicate_rows, normalize_category, is_valid_staff_count, is_valid_truck_count


def test_duplicate_rows_counts_extra_copy():
    sample = pd.DataFrame({'x': [1, 2, 2], 'y': ['a', 'b', 'b']})
    assert duplicate_rows(sample) == 1


def test_normalize_category_strips_and_lowercases():
    assert normalize_category('  PRIVATE  ') == 'private'


def test_client_validation_rules():
    assert is_valid_staff_count(2)
    assert not is_valid_staff_count(1)
    assert is_valid_truck_count(4000)
    assert not is_valid_truck_count(4001)
