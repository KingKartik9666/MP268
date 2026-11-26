"""Small, reusable data-quality checks for the MP268 dataset."""
from collections.abc import Iterable


def duplicate_rows(df) -> int:
    """Return the number of extra duplicate rows (pandas duplicated count)."""
    return int(df.duplicated().sum())


def spelling_variants(df, text_columns: Iterable[str]) -> dict[str, object]:
    """Return value_counts for each supplied text column, preserving raw spellings."""
    return {column: df[column].value_counts(dropna=False) for column in text_columns}


def normalize_category(value: str) -> str:
    """Basic whitespace/case normalization for category strings."""
    return value.strip().lower()


def is_valid_staff_count(staff: float) -> bool:
    """Client rule: staff values must be at least 1.5."""
    return staff >= 1.5


def is_valid_truck_count(trucks_passing: int) -> bool:
    """Client rule: truck counts above 4000 are invalid."""
    return trucks_passing <= 4000
