"""Tests for the drift measure (PSI). Tiny generated data only."""
import numpy as np

from mp268.monitor import psi


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.uniform(0, 2000, 2000), rng.uniform(0, 2000, 2000)) < 0.02


def test_psi_is_large_when_the_range_moves():
    rng = np.random.default_rng(0)
    # like trucks_passing: 0-2000 in train, 1100-3100 two months later
    assert psi(rng.uniform(0, 2000, 1000), rng.uniform(1100, 3100, 300)) > 0.25


def test_psi_handles_whole_number_columns_with_repeated_values():
    rng = np.random.default_rng(0)
    value = psi(rng.integers(0, 11, 1000), rng.integers(0, 11, 300))   # like nearby_pumps
    assert np.isfinite(value) and value < 0.1
