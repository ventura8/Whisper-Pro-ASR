"""Static language mapping used for language detection fallbacks."""

from __future__ import annotations

import json
import re

LANGUAGES = json.loads(
    '{"en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean", '
    '"fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan", '
    '"nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi", '
    '"fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", '
    '"cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", '
    '"th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin", '
    '"mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian", '
    '"lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", '
    '"et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", '
    '"ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili", '
    '"gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona", '
    '"yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", '
    '"tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", '
    '"uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", '
    '"mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog", '
    '"mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", '
    '"ba": "Bashkir", "jw": "Javanese", "su": "Sundanese"}'
)

#: ISO 639-2 (three-letter) codes to the two-letter codes the engines accept. Both the
#: bibliographic and terminological spellings appear in the wild -- Bazarr sends "fre" and
#: "fra" for the same language -- so both map here.
ISO_639_2_TO_1 = {
    "eng": "en",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "spa": "es",
    "por": "pt",
    "ita": "it",
    "nld": "nl",
    "dut": "nl",
    "rus": "ru",
    "zho": "zh",
    "chi": "zh",
    "jpn": "ja",
    "kor": "ko",
}


def supported_code(language: str) -> str | None:
    """The engine code ``language`` names, or None when unsupported. Never logs."""
    code = language.strip().lower()
    if code in LANGUAGES:
        return code
    # Split on either separator. A POSIX-style "en_US" reached the ISO 639-2 lookup whole,
    # matched nothing, and auto-detected -- while the identical "en-US" resolved to "en".
    primary_code = re.split(r"[-_]", code, maxsplit=1)[0]
    if primary_code in LANGUAGES:
        return primary_code
    mapped_code = ISO_639_2_TO_1.get(primary_code)
    return mapped_code if mapped_code in LANGUAGES else None
