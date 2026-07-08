"""
Subtitle generation and text wrapping helper utilities.
"""

from modules.core import config


def wrap_text(text, max_line_width, max_line_count=None):
    """Wraps text to max_line_width characters per line, up to max_line_count lines."""
    if not text or not max_line_width:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        word_len = len(word)
        needed_space = word_len + (1 if current_line else 0)

        if current_len + needed_space <= max_line_width:
            current_line.append(word)
            current_len += needed_space
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = word_len

    if current_line:
        lines.append(" ".join(current_line))

    if max_line_count:
        lines = lines[:max_line_count]

    return "\n".join(lines)


def format_single_srt_block(idx, start_ts, end_ts, text, *, speaker=None, max_line_width=None, max_line_count=None):
    """Format a single subtitle segment into its SRT block representation."""
    start_fmt = format_timestamp(start_ts or 0.0)
    end_fmt = format_timestamp(end_ts or 0.0)
    clean_text = text.strip()
    if speaker:
        clean_text = f"[{speaker}]: {clean_text}"
    if max_line_width is not None:
        clean_text = wrap_text(clean_text, max_line_width, max_line_count)
    return f"{idx}\n{start_fmt} --> {end_fmt}\n{clean_text}\n\n"


def format_srt_highlighted_blocks(segment):
    """
    Generate sub-blocks for SRT highlighting where each word is shown
    in sequence with a highlight tag.
    """
    words = segment.get("words", [])
    if not words:
        return []

    sub_blocks = []
    segment_start_ts = segment.get("start", 0.0)
    if segment_start_ts is None:
        segment_start_ts = 0.0
    segment_end_ts = segment.get("end", 0.0)
    if segment_end_ts is None:
        segment_end_ts = 0.0

    for active_idx, active_word in enumerate(words):
        start_ts = active_word.get("start", segment_start_ts)
        if start_ts is None:
            start_ts = segment_start_ts
        end_ts = active_word.get("end", segment_end_ts)
        if end_ts is None:
            end_ts = segment_end_ts

        text_parts = []
        for i, w in enumerate(words):
            word_text = w.get("word", "")
            leading_spaces = len(word_text) - len(word_text.lstrip())
            space_prefix = word_text[:leading_spaces]
            clean_word = word_text[leading_spaces:]

            if i == active_idx:
                text_parts.append(f'{space_prefix}<font color="#E0E0E0">{clean_word}</font>')
            else:
                text_parts.append(f"{space_prefix}{clean_word}")

        text = "".join(text_parts).strip()
        speaker = segment.get("speaker")
        if speaker:
            text = f"[{speaker}]: {text}"

        sub_blocks.append((start_ts, end_ts, text))

    return sub_blocks


def _get_normalized_segments(result):
    """Normalize the result segments and prepend the promo segment if enabled."""

    # Fallback to empty segments if result is falsy
    original_segments = (
        result.get("segments", []) if (result and isinstance(result, dict) and "segments" in result) else []
    )

    if not original_segments:
        # Check for legacy plain-text fallback
        text_val = result.get("text", "").strip() if (result and isinstance(result, dict)) else ""
        if text_val:
            original_segments = [{"start": 0.0, "end": 5.0, "text": text_val}]
        else:
            original_segments = [{"start": 0.0, "end": 5.0, "text": "[No dialogue detected]"}]

    segments = []
    for seg in original_segments:
        # Shallow copy to avoid mutating the original dict in result
        segments.append(dict(seg))

    if config.SUBTITLE_PROMO_ENABLED and config.SUBTITLE_PROMO_TEXT:
        # Shift/adjust the start of "[No dialogue detected]" or text fallback if it starts at 0.0
        # to start at config.SUBTITLE_PROMO_DURATION.
        is_fallback = not (result and isinstance(result, dict) and "segments" in result) or not result.get("segments")
        if is_fallback and segments:
            segments[0]["start"] = config.SUBTITLE_PROMO_DURATION
            segments[0]["end"] = config.SUBTITLE_PROMO_DURATION + 5.0

        promo_seg = {"start": 0.0, "end": config.SUBTITLE_PROMO_DURATION, "text": config.SUBTITLE_PROMO_TEXT}
        segments = [promo_seg] + segments

    return segments


def generate_srt(result, max_line_width=None, max_line_count=None, highlight_words=False):
    """
    Compose industrial-standard SubRip (SRT) content from segment metadata.

    Handles time formatting, sequence indexing, and empty signal fallbacks.
    """
    segments = _get_normalized_segments(result)

    srt_lines = []
    block_idx = 1
    for segment in segments:
        words = segment.get("words", [])
        if highlight_words and words:
            sub_blocks = format_srt_highlighted_blocks(segment)
            for start_ts, end_ts, text in sub_blocks:
                start_fmt = format_timestamp(start_ts)
                end_fmt = format_timestamp(end_ts)
                if max_line_width is not None:
                    text = wrap_text(text, max_line_width, max_line_count)
                srt_lines.append(f"{block_idx}\n{start_fmt} --> {end_fmt}\n{text}\n")
                block_idx += 1
        else:
            try:
                # Try 'timestamp' tuple first, then individual 'start'/'end' keys
                if "timestamp" in segment:
                    start_ts, end_ts = segment["timestamp"]
                else:
                    start_ts = segment.get("start", 0.0)
                    end_ts = segment.get("end", 5.0)

                if start_ts is None:
                    start_ts = 0.0
                if end_ts is None:
                    end_ts = 0.0
            except (ValueError, TypeError, KeyError):
                start_ts, end_ts = 0.0, 5.0

            start_fmt = format_timestamp(start_ts)
            end_fmt = format_timestamp(end_ts)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker")
            if speaker:
                text = f"[{speaker}]: {text}"

            if max_line_width is not None:
                text = wrap_text(text, max_line_width, max_line_count)

            # Format SRT block: Index -> Timestamps -> Content
            srt_lines.append(f"{block_idx}\n{start_fmt} --> {end_fmt}\n{text}\n")
            block_idx += 1

    if not srt_lines:
        return "1\n00:00:00,000 --> 00:00:05,000\n[No dialogue detected]\n"

    return "\n".join(srt_lines)


def format_timestamp(seconds):
    """Generate the millisecond-precision timestamp required for SRT specifications."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_karaoke(words, default_start_ts):
    """Format WebVTT karaoke timestamps for words list."""
    words_formatted = []
    for w in words:
        word_start_ts = w.get("start", default_start_ts)
        if word_start_ts is None:
            word_start_ts = default_start_ts
        w_start_fmt = format_timestamp(word_start_ts).replace(",", ".")
        word_text = w.get("word", "")
        leading_spaces = len(word_text) - len(word_text.lstrip())
        space_prefix = word_text[:leading_spaces]
        clean_word = word_text[leading_spaces:]
        words_formatted.append(f"{space_prefix}<{w_start_fmt}>{clean_word}")
    return "".join(words_formatted).strip()


def generate_vtt(result, max_line_width=None, max_line_count=None, highlight_words=False):
    """
    Generate WebVTT content for web-native subtitles.
    """
    segments = _get_normalized_segments(result)

    vtt_lines = ["WEBVTT", ""]
    for idx, segment in enumerate(segments, start=1):
        words = segment.get("words", [])
        try:
            if "timestamp" in segment:
                start_ts, end_ts = segment["timestamp"]
            else:
                start_ts = segment.get("start", 0.0)
                end_ts = segment.get("end", 5.0)

            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0
        except (ValueError, TypeError, KeyError):
            start_ts, end_ts = 0.0, 5.0

        # VTT uses dot for milliseconds
        start_fmt = format_timestamp(start_ts).replace(",", ".")
        end_fmt = format_timestamp(end_ts).replace(",", ".")

        if highlight_words and words:
            # Build Karaoke-style VTT intra-cue timing tags via helper
            text = _format_vtt_karaoke(words, start_ts)
        else:
            text = segment.get("text", "").strip()

        speaker = segment.get("speaker")
        if speaker:
            text = f"[{speaker}]: {text}"

        if max_line_width is not None:
            text = wrap_text(text, max_line_width, max_line_count)

        vtt_lines.append(f"{idx}\n{start_fmt} --> {end_fmt}\n{text}\n")

    return "\n".join(vtt_lines)


def generate_txt(result):
    """
    Generate plain text transcript.
    """
    if not result:
        return ""
    segments = result.get("segments")
    if segments:
        txt_lines = []
        for segment in segments:
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker")
            if speaker:
                txt_lines.append(f"[{speaker}]: {text}")
            else:
                txt_lines.append(text)
        return "\n".join(txt_lines)
    return result.get("text", "").strip()


def generate_tsv(result):
    """
    Generate Tab-Separated Values (TSV) format.
    """
    if not result:
        return "start\tend\ttext"

    tsv_lines = ["start\tend\ttext"]
    for segment in result.get("segments", []):
        try:
            if "timestamp" in segment:
                start_ts, end_ts = segment["timestamp"]
            else:
                start_ts = segment.get("start", 0.0)
                end_ts = segment.get("end", 0.0)

            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0

            start_ms = int(start_ts * 1000)
            end_ms = int(end_ts * 1000)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker")
            if speaker:
                text = f"[{speaker}]: {text}"
            text = text.replace("\t", " ").replace("\n", " ")

            tsv_lines.append(f"{start_ms}\t{end_ms}\t{text}")
        except (TypeError, ValueError, KeyError):
            continue

    return "\n".join(tsv_lines)
