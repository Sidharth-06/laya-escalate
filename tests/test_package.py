"""Packaging-level tests.

``test_all_exported_names_exist`` is here because ``__all__`` is documentation:
a name listed there is a promise that ``from laya_escalate import ...`` works.
``format_scorer_comparison`` was listed for a while without being imported, so
the promise was broken and nothing caught it.
"""

from __future__ import annotations

import laya_escalate


def test_all_exported_names_exist():
    missing = [name for name in laya_escalate.__all__ if not hasattr(laya_escalate, name)]
    assert not missing, f"__all__ advertises names that do not exist: {missing}"


def test_dunder_all_has_no_duplicates():
    names = list(laya_escalate.__all__)
    assert len(names) == len(set(names)), "duplicate entries in __all__"


def test_format_scorer_comparison_is_importable():
    from laya_escalate import ScorerComparison, format_scorer_comparison

    text = format_scorer_comparison(
        ScorerComparison(targets=[0.05], coverage={"a": [0.5]}, aurc={"a": 0.1})
    )
    assert "scorer comparison" in text
    assert "0.500" in text
