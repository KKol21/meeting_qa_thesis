import pytest

from meeting_qa_chunking.uncertainty import bootstrap_mean, paired_bootstrap


def test_bootstrap_mean_is_deterministic():
    first = bootstrap_mean([1.0, 2.0, 3.0], samples=100, seed=7)
    second = bootstrap_mean([1.0, 2.0, 3.0], samples=100, seed=7)

    assert first == second
    assert first["mean"] == 2.0


def test_paired_bootstrap_uses_within_pair_differences():
    result = paired_bootstrap([2.0, 4.0], [1.0, 3.0], samples=100)

    assert result == {"mean": 1.0, "ci95": [1.0, 1.0]}


def test_paired_bootstrap_rejects_unmatched_samples():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap([1.0], [1.0, 2.0])
