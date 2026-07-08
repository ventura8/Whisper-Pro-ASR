"""
Post-processing filters for transcription outputs.
"""

import logging

from modules.core import config

logger = logging.getLogger(__name__)


def post_process_results(result, _audio_path=None):
    """Applies quality filters to the raw transcription output."""
    if not result or "segments" not in result:
        return result

    segments = result["segments"]
    if not segments:
        return result

    processed_segments = []
    repetition_count = 0
    last_text = ""

    for seg in segments:
        text = seg.get("text", "").strip()
        prob = seg.get("probability", 1.0)  # probability might not be in all engines
        text_was_filtered = False

        # 1. Silence/Low confidence filter
        if prob < config.HALLUCINATION_SILENCE_THRESHOLD:
            seg["text"] = ""
            text_was_filtered = True
            logger.debug("[Filter] Dropped segment due to low confidence (%.2f)", prob)

        # 2. Phrase filter
        elif any(phrase.lower() in text.lower() for phrase in config.HALLUCINATION_PHRASES):
            seg["text"] = ""
            text_was_filtered = True
            logger.debug("[Filter] Dropped segment containing hallucination phrase")

        # 3. Repetition filter
        elif text == last_text and text != "":
            repetition_count += 1
            if repetition_count >= config.HALLUCINATION_REPETITION_THRESHOLD:
                seg["text"] = ""
                text_was_filtered = True
                logger.debug("[Filter] Dropped repetitive segment")
        else:
            repetition_count = 0
            last_text = text

        if text_was_filtered or not seg.get("text", "").strip():
            seg.pop("words", None)

        processed_segments.append(seg)

    result["segments"] = processed_segments
    return result
