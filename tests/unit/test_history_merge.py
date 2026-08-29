"""Reconciling a stored analytics day with the one rebuilt from capped history.

The stored file and the rebuild disagree by construction: history is capped, so a rebuild
only sees the tasks that survived the cap, while the stored copy accumulated every task as
it completed. Which side wins for which field is the whole contract here, and getting it
wrong rewrites the user's own history rather than losing a number.
"""

from __future__ import annotations

import pytest

from modules.monitoring import history_helpers


def _day(count: int, duration: float, **categories) -> dict:
    """A day payload, with a full category breakdown unless one is overridden away."""
    day = {
        "count": count,
        "duration": duration,
        "asr": {"count": count, "duration": duration},
        "detectlang": {"count": 0, "duration": 0.0},
        "audio": {"count": 0, "duration": 0.0},
    }
    day.update(categories)
    return day


class TestUsableCategoryDetection:
    """What counts as a breakdown the merge arithmetic can actually use."""

    def test_a_full_numeric_breakdown_is_usable(self):
        """A full numeric breakdown is usable."""
        assert history_helpers._has_all_category_keys(_day(5, 10.0)) is True

    @pytest.mark.parametrize("bad", [3, "asr", [1, 2], None])
    def test_a_non_mapping_category_is_not(self, bad):
        """A hand-edited or half-migrated file put a scalar here; dict(...) then raised."""
        assert history_helpers._has_all_category_keys(_day(5, 10.0, asr=bad)) is False

    def test_a_partial_category_is_not(self):
        """`{}` passed the isinstance test, and the merge then did `["count"] += ...`."""
        assert history_helpers._has_all_category_keys(_day(5, 10.0, asr={})) is False

    def test_a_non_numeric_field_is_not(self):
        """A non numeric field is not."""
        assert history_helpers._has_all_category_keys(_day(5, 10.0, asr={"count": "5", "duration": 10.0})) is False

    def test_a_bool_is_not_a_count(self):
        """bool is an int subclass, so `isinstance(True, int)` would have accepted it."""
        assert history_helpers._has_all_category_keys(_day(5, 10.0, asr={"count": True, "duration": 10.0})) is False


class TestMergingAnOverlappingDay:
    """Totals take the larger view; the breakdown is preserved rather than overwritten."""

    def test_a_stored_breakdown_survives_a_smaller_rebuild(self):
        """The recorded defect: 40 ASR + 10 detections were reported as 50 ASR + 0.

        History was capped, so the rebuild saw fewer tasks and could not tell them apart --
        and overwriting the stored breakdown with its view silently rewrote what happened.
        """
        stored = {
            "count": 50,
            "duration": 500.0,
            "asr": {"count": 40, "duration": 400.0},
            "detectlang": {"count": 10, "duration": 100.0},
            "audio": {"count": 0, "duration": 0.0},
        }
        rebuilt = _day(20, 200.0)

        history_helpers._merge_overlapping_legacy_day(stored, rebuilt)

        assert (rebuilt["count"], rebuilt["duration"]) == (50, 500.0), "the fuller record wins the totals"
        assert rebuilt["asr"] == {"count": 40, "duration": 400.0}
        assert rebuilt["detectlang"] == {"count": 10, "duration": 100.0}

    def test_a_larger_rebuild_adds_only_the_remainder(self):
        """Keeping the stored breakdown verbatim left the categories summing to less than
        the day's own count, so the analytics page showed a total its breakdown could not
        account for."""
        stored = {
            "count": 10,
            "duration": 100.0,
            "asr": {"count": 6, "duration": 60.0},
            "detectlang": {"count": 4, "duration": 40.0},
            "audio": {"count": 0, "duration": 0.0},
        }
        rebuilt = _day(15, 150.0)

        history_helpers._merge_overlapping_legacy_day(stored, rebuilt)

        assert (rebuilt["count"], rebuilt["duration"]) == (15, 150.0)
        # The 5-task / 50s remainder is attributed to ASR, the only guess available.
        assert rebuilt["asr"] == {"count": 6 + 5, "duration": 60.0 + 50.0}
        assert rebuilt["detectlang"] == {"count": 4, "duration": 40.0}
        categories = sum(rebuilt[c]["count"] for c in ("asr", "detectlang", "audio"))
        assert categories == rebuilt["count"], "the breakdown must account for the day's total"

    def test_a_legacy_day_without_a_breakdown_still_backfills_to_asr(self):
        """No breakdown means nothing to preserve, so the original heuristic applies."""
        stored = {"count": 12, "duration": 120.0}
        rebuilt = _day(5, 50.0)

        history_helpers._merge_overlapping_legacy_day(stored, rebuilt)

        assert (rebuilt["count"], rebuilt["duration"]) == (12, 120.0)
        assert rebuilt["asr"] == {"count": 5 + 7, "duration": 50.0 + 70.0}

    def test_malformed_stored_categories_take_the_backfill_path(self):
        """The point of the usability check: route it rather than raise inside the rebuild."""
        stored = {"count": 12, "duration": 120.0, "asr": 7, "detectlang": {}, "audio": None}
        rebuilt = _day(5, 50.0)

        history_helpers._merge_overlapping_legacy_day(stored, rebuilt)

        assert rebuilt["asr"] == {"count": 12, "duration": 120.0}

    def test_agreeing_views_add_nothing(self):
        """Equal counts leave a zero remainder, which must not touch the breakdown."""
        stored = _day(8, 80.0)
        rebuilt = _day(8, 80.0)

        history_helpers._merge_overlapping_legacy_day(stored, rebuilt)

        assert rebuilt["asr"] == {"count": 8, "duration": 80.0}


class TestTheRemainderHelper:
    """A remainder is only ever added; a negative one means the kept side was already larger."""

    @pytest.mark.parametrize("count,duration", [(-3, -30.0), (0, 0.0)])
    def test_nothing_is_added_for_a_non_positive_remainder(self, count, duration):
        """Nothing is added for a non positive remainder."""
        day = _day(4, 40.0)
        history_helpers._add_uncategorised_remainder(day, count=count, duration=duration)

        assert day["asr"] == {"count": 4, "duration": 40.0}

    def test_an_absent_asr_category_is_created(self):
        """An absent asr category is created."""
        day = {"count": 2, "duration": 20.0}
        history_helpers._add_uncategorised_remainder(day, count=2, duration=20.0)

        assert day["asr"] == {"count": 2, "duration": 20.0}
