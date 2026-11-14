"""Regression checks for paired lexical benchmark cutoff integrity."""
import pytest

from evalforge.compare_lexical_baselines import compare_tokenizers, main


@pytest.mark.parametrize("cutoffs", [(), (0,), (-1,), (1, 1), (1, 3, 1), (True,), (1.5,), ("1",)])
def test_paired_comparison_rejects_invalid_cutoffs_before_evaluation(cutoffs):
    with pytest.raises(ValueError, match="top-k cutoffs"):
        compare_tokenizers([], [], cutoffs)


def test_paired_cli_rejects_duplicate_cutoffs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--k", "1", "1"])
    assert exc.value.code == 2
    assert "top-k cutoffs must be unique" in capsys.readouterr().err


def test_paired_cli_rejects_zero_cutoff(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--k", "0"])
    assert exc.value.code == 2
    assert "top-k cutoffs must be nonempty positive integers" in capsys.readouterr().err
