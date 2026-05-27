"""
Subtitle generation and text wrapping helper utilities.
"""


def _wrap_text(text, max_line_width, max_line_count=None):
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


def generate_srt(result, max_line_width=None, max_line_count=None):
    """
    Compose industrial-standard SubRip (SRT) content from segment metadata.

    Handles time formatting, sequence indexing, and empty signal fallbacks.
    """
    if not result:
        return "[No dialogue detected]"

    # Fallback for plain-text response format (legacy)
    if "segments" not in result:
        text = result.get("text", "").strip()
        if text:
            if max_line_width is not None:
                text = _wrap_text(text, max_line_width, max_line_count)
            return f"1\n00:00:00,000 --> 00:00:05,000\n{text}\n"
        return "[No dialogue detected]"

    srt_lines = []
    for idx, segment in enumerate(result.get("segments", []), start=1):
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
            text = _wrap_text(text, max_line_width, max_line_count)

        # Format SRT block: Index -> Timestamps -> Content
        srt_lines.append(f"{idx}\n{start_fmt} --> {end_fmt}\n{text}\n")

    if not srt_lines:
        return "[No dialogue detected]"

    return "\n".join(srt_lines)


def format_timestamp(seconds):
    """Generate the millisecond-precision timestamp required for SRT specifications."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_vtt(result, max_line_width=None, max_line_count=None):
    """
    Generate WebVTT content for web-native subtitles.
    """
    if not result:
        return "WEBVTT\n\n[No dialogue detected]"

    segments = result.get("segments", [])
    if not segments:
        # Fallback for text-only result
        text = result.get("text", "").strip()
        if text:
            if max_line_width is not None:
                text = _wrap_text(text, max_line_width, max_line_count)
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n{text}\n"
        return "WEBVTT\n\n[No dialogue detected]"

    vtt_lines = ["WEBVTT", ""]
    for idx, segment in enumerate(segments, start=1):
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
        start_fmt = format_timestamp(start_ts).replace(',', '.')
        end_fmt = format_timestamp(end_ts).replace(',', '.')
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker")
        if speaker:
            text = f"[{speaker}]: {text}"

        if max_line_width is not None:
            text = _wrap_text(text, max_line_width, max_line_count)

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
