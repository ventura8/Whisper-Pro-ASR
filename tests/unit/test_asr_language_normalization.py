"""The language parameter must mean the same thing on every engine.

Engines disagree on an unknown code: the Intel engine logs and auto-detects, while
CTranslate2 raises and the request becomes a 500. On a hybrid host the same request can
land on either unit, so this is settled before dispatch.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

from modules.api.routes import asr


def test_supported_code_is_passed_through_lowercased():
    """A supported code survives, normalised to the spelling the engines expect."""
    assert asr._normalize_language("ES") == "es"
    assert asr._normalize_language("en") == "en"


def test_surrounding_whitespace_is_tolerated():
    """Bazarr pads values; a supported code must still be recognised."""
    assert asr._normalize_language("  fr  ") == "fr"


def test_locale_and_iso_639_2_codes_normalize_to_engine_codes():
    """Clients commonly send locale or bibliographic language identifiers."""
    assert asr._normalize_language("en-US") == "en"
    assert asr._normalize_language("pt-BR") == "pt"
    assert asr._normalize_language("eng") == "en"


def test_unsupported_code_falls_back_to_auto_detection():
    """'xx' must not reach the engine, where CTranslate2 would make it fatal."""
    assert asr._normalize_language("xx") is None


def test_absent_language_is_left_alone():
    """No language means auto-detection, which must not be turned into one."""
    assert asr._normalize_language(None) is None
    assert asr._normalize_language("") == ""


def test_both_iso_639_2_forms_of_a_language_map_to_the_same_code():
    """Bibliographic and terminological codes differ for exactly these languages.

    ISO 639-2 gives some languages two three-letter codes -- a terminological (T) one and a
    bibliographic (B) one -- and a client may send either. Covering only one form of each
    leaves the other resolving to None, which quietly turns a specified language into
    auto-detection for a request that named it perfectly clearly.
    """
    assert asr._normalize_language("fra") == asr._normalize_language("fre") == "fr"
    assert asr._normalize_language("deu") == asr._normalize_language("ger") == "de"
    assert asr._normalize_language("nld") == asr._normalize_language("dut") == "nl"
    assert asr._normalize_language("zho") == asr._normalize_language("chi") == "zh"


def test_a_multi_subtag_locale_resolves_on_its_primary_subtag():
    """Only the first subtag names the language; script and region are not part of it."""
    assert asr._normalize_language("zh-Hans-CN") == "zh"


def test_a_whitespace_only_language_is_dropped():
    """Not falsy, so it takes the normalization path -- and names no language, so it goes.

    Worth its own case because it sits between the two branches already covered: `""` is
    returned unchanged as "no language given", while `"   "` reaches the lookup and must
    come back as None rather than being handed to an engine as a code.
    """
    assert asr._normalize_language("   ") is None


def test_dropping_the_callers_language_is_logged(caplog):
    """Silently ignoring a stated language changes what the request does, so it is reported."""
    with caplog.at_level("WARNING"):
        assert asr._normalize_language("klingon") is None
    assert "klingon" in caplog.text
