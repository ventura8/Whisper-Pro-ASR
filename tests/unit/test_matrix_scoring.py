"""How a real-audio clip's transcript is scored against the words it should contain.

The manifest chooses the comparison per clip (``"tokenizer"``), because one comparison
cannot serve every writing system: word overlap is meaningless where the model does not
put spaces where the reference does, and character bigrams are too lax where it does.
Choosing wrongly does not fail loudly -- it reports a working language as broken, or a
broken one as working -- so the strategies are pinned here rather than only exercised
through a two-hour real-engine run.
"""

from unittest import mock

import pytest

from tests.real_audio import matrix_support

YORUBA_EXPECTED = ["Kọ̀lọ̀kọ̀lọ̀", "aláwọ̀", "búráùnù", "yára", "sórí", "ajá", "ọ̀lẹ"]
YORUBA_HEARD = "kolo-kolo ala wobura nulera yara sori aja ole".split()


class TestTheFoldedStrategy:
    """`chars_folded`: for languages transcribed phonetically but not orthographically."""

    def _score(self, expected, actual):
        return matrix_support.word_overlap(expected, actual, "chars_folded")

    def test_a_transcript_without_the_diacritics_still_scores(self):
        """The case it exists for: right sounds, no tone marks, different word breaks.

        The other strategies score this recognisably-correct transcript at 0.00-0.20 and
        report a total failure that did not happen.
        """
        assert self._score(YORUBA_EXPECTED, YORUBA_HEARD) >= 0.5

    def test_it_is_laxer_than_every_other_strategy_on_that_clip(self):
        """Pins the ordering the docstring claims, so a future edit cannot quietly invert it."""
        folded = self._score(YORUBA_EXPECTED, YORUBA_HEARD)
        others = [matrix_support.word_overlap(YORUBA_EXPECTED, YORUBA_HEARD, name) for name in ("words", "chars")]
        assert folded > max(others)

    def test_unrelated_text_still_scores_low(self):
        """The load-bearing guard.

        A laxer bar is only safe while it still rejects a wrong transcript. Without this,
        `chars_folded` could be reached for to rescue any clip that is simply being
        transcribed badly, which is the opposite of what it is for.
        """
        unrelated = "the quick brown fox jumps over the lazy dog".split()
        assert self._score(YORUBA_EXPECTED, unrelated) < 0.5

    def test_folding_collapses_the_letters_yoruba_distinguishes(self):
        """Why this strategy is lax, stated as a fact rather than left to be discovered.

        Yoruba's dot-below is a combining mark like its tone marks, so `ọ` and `ẹ` fold to
        `o` and `e`: after folding, "ọlẹ" and "ole" are the same string. That is what lets a
        transcript without diacritics score at all, and it is also why this strategy cannot
        be the bar for a language whose orthography the model does get right.
        """
        assert matrix_support._fold_marks("ọ̀lẹ") == matrix_support._fold_marks("ole") == "ole"


class TestChoosingAStrategy:
    """The manifest names the comparison, and a name nothing implements must be loud."""

    def test_an_unknown_tokenizer_is_an_error_not_a_silent_fallback(self):
        """Falling back to `words` scored a CJK entry with a comparison that cannot match,
        so a typo read as a failing language rather than as a broken manifest."""
        with pytest.raises(AssertionError, match="not one of"):
            matrix_support.word_overlap(["x"], ["x"], "chars-folded")

    @pytest.mark.parametrize("name", sorted(matrix_support._OVERLAP_STRATEGIES))
    def test_a_perfect_transcript_scores_perfectly(self, name):
        """Whatever the comparison, hearing exactly the expected words is a pass.

        ``actual`` arrives already tokenized by ``words()`` -- ``word_overlap`` normalizes
        only ``expected`` -- so the perfect transcript is built the same way the service's
        would be. Passing the raw reference as ``actual`` instead scores 0.57 under the word
        tokenizer, which is a property of that asymmetry and not of any audio.
        """
        expected = "the quick brown fox jumps over the lazy dog".split()
        heard = matrix_support.words(" ".join(expected))
        assert matrix_support.word_overlap(expected, heard, name) == 1.0

    @pytest.mark.parametrize(("name", "ceiling"), [("words", 0.72), ("chars", 0.91), ("chars_folded", 0.90)])
    def test_a_perfect_yoruba_transcript_cannot_reach_one(self, name, ceiling):
        """The manifest's uniform 0.5 does not mean the same thing in every language.

        ``normalize`` replaces combining marks with spaces, so a Yoruba reference word is
        split at every tone mark -- "Kọ̀lọ̀kọ̀lọ̀" becomes four tokens -- exactly as its
        docstring records for Indic scripts, which do not include Yoruba. The effect is on
        the *reference*, so it caps the score a flawless transcript can reach. yo_mms's
        measured 0.76 is therefore 0.76 of an attainable ~0.90, not of 1.00.
        """
        heard = matrix_support.words(" ".join(YORUBA_EXPECTED))
        score = matrix_support.word_overlap(YORUBA_EXPECTED, heard, name)
        assert score < 1.0
        assert score == pytest.approx(ceiling, abs=0.02)


class TestWhichEngineADefectBelongsTo:
    """A defect can belong to an engine rather than to a clip.

    Every code-switched fixture fails on WHISPERX and none on FASTER-WHISPER, so a mark that
    ignores the engine reports six XPASSes on the default one -- the "wall of meaningless
    XPASS" the scope mechanism exists to prevent -- and forces the entries to describe the
    split in prose that nothing enforces.
    """

    ENTRY = {
        "id": "mix_en_es",
        "xfail_reason": "WHISPERX drops the English leg",
        "xfail_scope": "content",
        "xfail_engines": ["WHISPERX"],
    }

    def _marks(self, entry, running, scope="content"):
        matrix_support.service_client.forget_running_engine()
        with mock.patch.object(matrix_support.service_client, "running_engine", return_value=running):
            return matrix_support._entry_marks(entry, scope)

    def test_the_named_engine_gets_the_xfail(self):
        """The recorded defect still has to be recorded where it actually happens."""
        assert self._marks(self.ENTRY, "WHISPERX")

    def test_another_engine_holds_the_clip_strictly(self):
        """The point of the key: FASTER-WHISPER must fail if it regresses here."""
        assert not self._marks(self.ENTRY, "FASTER-WHISPER")

    def test_an_entry_naming_no_engine_still_applies_everywhere(self):
        """What every pre-existing entry means, and what they must keep meaning."""
        entry = {k: v for k, v in self.ENTRY.items() if k != "xfail_engines"}
        assert self._marks(entry, "FASTER-WHISPER")

    def test_an_unreachable_service_applies_the_mark(self):
        """Collection runs before any request. Guessing "not affected" with no service would
        turn a suite that should skip into a strict one the moment a service appeared."""
        assert self._marks(self.ENTRY, None)

    def test_the_scope_still_narrows_within_the_named_engine(self):
        """Engine and concern are independent filters; neither may swallow the other."""
        assert not self._marks(self.ENTRY, "WHISPERX", scope="identity")


class TestTheCeiling:
    """A bar is a fraction of what a flawless transcript can score, not an absolute.

    ``normalize`` turns combining marks into spaces, so in some scripts the *reference* is
    fragmented before anything is compared and identical text cannot reach 1.0. A uniform
    0.5 was therefore unreachable for Tamil and lax for English. Scaling by the ceiling
    asks every clip the same question.
    """

    def test_clean_text_has_a_ceiling_of_one(self):
        """Nothing for `normalize` to fragment, so the bar is the bar."""
        for name in ("words", "chars", "chars_folded"):
            assert matrix_support.overlap_ceiling("the quick brown fox".split(), name) == 1.0

    def test_yoruba_sits_below_one_under_every_strategy(self):
        """The case that motivated it: the same reference, three ceilings, none of them 1.0."""
        for name in ("words", "chars", "chars_folded"):
            assert matrix_support.overlap_ceiling(YORUBA_EXPECTED, name) < 1.0

    def test_the_threshold_is_the_bar_scaled_by_the_ceiling(self):
        """`content_threshold` is the one place the rule lives; both tiers read it."""
        clip = {"expect_words": YORUBA_EXPECTED, "tokenizer": "chars_folded", "min_word_overlap": 0.5}
        ceiling = matrix_support.overlap_ceiling(YORUBA_EXPECTED, "chars_folded")
        assert matrix_support.content_threshold(clip) == pytest.approx(0.5 * ceiling)
        assert matrix_support.content_threshold(clip) < 0.5

    def test_tier_a_and_combined_bars_do_not_move(self):
        """The rule is applied to every tier, so it must be a no-op where bars were already
        validated against absolute numbers: every tier-A and combined ceiling is 1.0."""
        manifest = matrix_support.load_manifest()
        entries = [e for e in manifest["clips"] if e.get("tier") == "A"] + list(manifest["combined"])
        assert entries, "manifest sections unexpectedly empty"
        for entry in entries:
            ceiling = matrix_support.overlap_ceiling(entry["expect_words"], entry.get("tokenizer", "words"))
            assert ceiling == 1.0, f"{entry['id']}: ceiling {ceiling:.2f} would change a validated bar"

    def test_some_tier_b_bars_were_unreachable_without_it(self):
        """Pins the finding, so the ceiling cannot be quietly dropped as redundant: at least one
        long-tail reference scores below its own declared bar even when transcribed perfectly."""
        manifest = matrix_support.load_manifest()
        unreachable = [
            e["id"]
            for e in manifest["clips"]
            if e.get("tier") == "B"
            and e.get("voice", "") is not None
            and matrix_support.overlap_ceiling(e["expect_words"], e.get("tokenizer", "words")) < float(e["min_word_overlap"])
        ]
        assert unreachable, "no tier-B bar is above its ceiling any more; revisit whether scaling is still needed"


class TestForeignShare:
    """The long-form fracture measure: wrong-language seconds over spoken seconds."""

    WINDOWS = [{"start": 0.0, "end": 4.0, "language": "ru"}, {"start": 10.0, "end": 14.0, "language": "ru"}]

    def test_a_transcript_in_the_spoken_language_scores_zero(self):
        """Case does not matter: the service reports what the detector gives it."""
        segments = [{"start": 0.0, "end": 4.0, "language": "ru"}, {"start": 10.0, "end": 14.0, "language": "RU"}]
        assert matrix_support.foreign_share(segments, self.WINDOWS) == 0.0

    def test_seconds_in_another_language_are_weighted_by_duration_not_counted(self):
        """One 4 s Ukrainian segment against 8 s spoken is half, however finely it is cut."""
        whole = [{"start": 0.0, "end": 4.0, "language": "uk"}, {"start": 10.0, "end": 14.0, "language": "ru"}]
        cut = [
            {"start": 0.0, "end": 1.0, "language": "uk"},
            {"start": 1.0, "end": 4.0, "language": "uk"},
            {"start": 10.0, "end": 14.0, "language": "ru"},
        ]
        assert matrix_support.foreign_share(whole, self.WINDOWS) == 0.5
        assert matrix_support.foreign_share(cut, self.WINDOWS) == 0.5

    def test_only_the_part_of_a_segment_inside_the_window_counts(self):
        """A segment spilling into silence is judged on the seconds it shares with the line."""
        segments = [{"start": 2.0, "end": 8.0, "language": "en"}, {"start": 10.0, "end": 14.0, "language": "ru"}]
        assert matrix_support.foreign_share(segments, self.WINDOWS) == pytest.approx(2.0 / 6.0)

    def test_a_segment_without_a_language_is_wrong(self):
        """The reporting path is under test too: silence about the language is a failure."""
        segments = [{"start": 0.0, "end": 4.0}, {"start": 10.0, "end": 14.0, "language": "ru"}]
        assert matrix_support.foreign_share(segments, self.WINDOWS) == 0.5

    def test_nothing_spoken_is_not_a_fracture(self):
        """No segments, or no windows: nothing was mislabelled."""
        assert matrix_support.foreign_share([], self.WINDOWS) == 0.0
        assert matrix_support.foreign_share([{"start": 0.0, "end": 4.0, "language": "uk"}], []) == 0.0


class TestRussianYo:
    """``ё`` and ``е`` are one letter for scoring: written Russian and Whisper both drop the dots."""

    def test_a_reference_with_yo_matches_a_transcript_without(self):
        """`ru_line2`: "чём" in the manifest, "чем" from the decoder -- a correct line, not a miss."""
        assert matrix_support.word_overlap(["понимаю", "чём"], matrix_support.words("Я не понимаю, о чем ты.")) == 1.0

    def test_the_fold_goes_both_ways(self):
        """A decoder that does write ё is not penalised either."""
        assert matrix_support.word_overlap(["все"], matrix_support.words("всё")) == 1.0


class TestTheRunningEngineCache:
    """Only an answer is remembered; a service that was not up yet is asked again."""

    def test_a_failed_lookup_is_not_cached(self):
        """First call during collection finds nothing; the next call, service up, gets the engine."""
        client = matrix_support.service_client
        client.forget_running_engine()
        try:
            with mock.patch.object(client, "_ask_running_engine", side_effect=[None, "WHISPERX", "FASTER-WHISPER"]) as ask:
                assert client.running_engine() is None
                assert client.running_engine() == "WHISPERX"
                assert client.running_engine() == "WHISPERX", "an answer, once given, is kept for the session"
            assert ask.call_count == 2
        finally:
            # An assertion failing above would otherwise leave "WHISPERX" cached for every
            # later test that consults the running engine.
            client.forget_running_engine()


class TestReadingTheStatusPayload:
    """Only an object naming a string engine counts; anything else is "not known"."""

    @pytest.mark.parametrize("payload", [[], "FASTER-WHISPER", None, {}, {"asr_engine": None}, {"asr_engine": 3}, {"asr_engine": ""}])
    def test_a_payload_that_does_not_name_an_engine_is_not_an_answer(self, payload):
        """A list body with a 200 attached used to raise AttributeError out of the lookup, and a
        non-string engine was stringified and cached for the session."""
        assert matrix_support.service_client._engine_in(payload) is None

    def test_a_named_engine_is_returned_as_is(self):
        """The one shape the service actually sends."""
        assert matrix_support.service_client._engine_in({"asr_engine": "WHISPERX"}) == "WHISPERX"


class TestTheEnglishMeaning:
    """What a translation is scored against: the clip's English sibling in the same pool."""

    def test_a_foreign_clip_maps_to_its_english_sibling(self):
        """`ru_line4` says what `en_line4` says; a translation of it is scored on the English."""
        assert matrix_support.english_words_for("ru_line4") == matrix_support.words("Tell me the truth.")

    def test_an_english_clip_is_its_own_meaning(self):
        """Nothing to translate: the English text is the expectation."""
        assert matrix_support.english_words_for("en_core") == matrix_support.words(
            "The quick brown fox jumps over the lazy dog. This recording verifies English speech recognition on this machine."
        )

    def test_a_clip_with_no_english_sibling_cannot_be_scored(self):
        """None, not an empty list: "nothing to compare against" must not read as "nothing expected"."""
        assert matrix_support.english_words_for("xx_nothing_like_this") is None

    def test_a_long_form_window_is_found_by_the_line_it_spoke(self):
        """A timeline window carries its line's text and language, which name the clip."""
        window = {"language": "ru", "text": "Скажи мне правду.", "start": 0.0, "end": 1.0}
        assert matrix_support.window_english_words(window) == matrix_support.words("Tell me the truth.")
        assert matrix_support.window_english_words({"language": "ru", "text": "not in the pool"}) is None

    def test_a_code_switched_clip_expects_both_legs_in_english(self):
        """The English leg's own words plus the recorded translation of the other one; the
        foreign leg's spoken words are what must be gone."""
        entry = {
            "legs": [
                {"language": "en", "text": "The meeting starts at nine."},
                {"language": "es", "text": "La reunión comienza a las nueve.", "translation": "The meeting starts at nine."},
            ]
        }
        assert matrix_support.translation_words(entry) == ["the", "meeting", "starts", "at", "nine"] * 2
        assert matrix_support.foreign_leg_words(entry) == ["la", "reunión", "comienza", "a", "las", "nueve"]

    def test_every_foreign_leg_in_the_manifest_records_its_translation(self):
        """The bar is only as good as the manifest: a leg without one would score against nothing."""
        for entry in matrix_support.load_manifest()["combined"]:
            for leg in entry["legs"]:
                assert leg["language"] == "en" or leg.get("translation"), f"{entry['id']}: {leg['language']} leg has no translation"


class TestADefectNamingSeveralConcerns:
    """``xfail_scope`` may be one concern or a list; the translation test asks for its own."""

    ENTRY = {"id": "mix", "xfail_reason": "drops a leg", "xfail_scope": ["content", "translation"]}

    def _marks(self, entry, scope):
        matrix_support.service_client.forget_running_engine()
        try:
            with mock.patch.object(matrix_support.service_client, "running_engine", return_value="FASTER-WHISPER"):
                return matrix_support._entry_marks(entry, scope)
        finally:
            matrix_support.service_client.forget_running_engine()

    def test_every_named_concern_carries_the_mark(self):
        """A leg dropped on transcription is dropped on translation too."""
        assert self._marks(self.ENTRY, "content")
        assert self._marks(self.ENTRY, "translation")

    def test_a_concern_not_named_stays_strict(self):
        """Detection still has to be right on a clip whose content is on record as broken."""
        assert not self._marks(self.ENTRY, "identity")

    def test_a_single_name_still_works_as_before(self):
        """Every pre-existing entry names one concern as a string."""
        entry = {**self.ENTRY, "xfail_scope": "translation"}
        assert self._marks(entry, "translation")
        assert not self._marks(entry, "content")


class TestTextInAWindow:
    """One rule for attributing a segment to a stretch of audio: where its midpoint sits."""

    SEGMENTS = [
        {"start": 0.0, "end": 1.9, "text": "The meeting starts at nine."},
        {"start": 1.5, "end": 3.5, "text": "straddles the boundary"},
        {"start": 2.6, "end": 4.9, "text": "La reunión comienza."},
    ]

    def test_a_segment_belongs_to_the_window_holding_its_midpoint(self):
        """The straddling segment's midpoint is 2.5, inside the second window and not the first."""
        assert matrix_support.text_in_window(self.SEGMENTS, 0.0, 1.93) == "The meeting starts at nine."
        assert matrix_support.text_in_window(self.SEGMENTS, 1.93, 4.95) == "straddles the boundary La reunión comienza."

    def test_an_empty_window_yields_no_text(self):
        """Nothing sits in the pause between the legs."""
        assert matrix_support.text_in_window(self.SEGMENTS, 5.0, 6.0) == ""
