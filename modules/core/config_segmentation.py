"""Settings for decoding by speech region rather than by fixed 30-second window.

Split out of config.py, which is at its module-length limit; grouped here because these
values are only meaningful together. A decode window is the unit of language commitment, so
the regions are what let the language follow a code-switch faster than 30 seconds: each is
detected before the decode, grouped into runs of one language with hysteresis, and each
language is then decoded in its own call, told its language outright.
"""

import os


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("true", "1", "yes")


# Measured on the 20-minute long-form fixture (RTX 3080, large-v3 float16, three identical
# runs), against the assertions the long-form test makes: `windows` misses 67/118 -> 0/118
# (budget 11), `quiet` noisy windows 11/25 -> 0/25 (budget 0), coverage 1186s -> 1202s of
# 1203s, RTF 0.044 -> 0.086.
#
# `quiet` improving is not incidental: clips exclude the silence between them, so the decoder
# is never handed a silent window to invent speech into -- the mechanism the manifest records
# for WHISPERX scoring 0/25, without WhisperX's single-language commitment.
#
# Those numbers were measured with the decoder re-detecting the language of every clip
# itself (clips with that off measured 82/118, worse than shipping). That detection is the
# same short-audio detection that mislabels one real film line in five, so the language is
# now decided before the decode -- per region, grouped into runs (SEGMENT_RUN_*) -- and each
# language is decoded in its own call, told its language. Same 0/118, and on real film the
# runs are what keep a monolingual title in one language.
#
# The clip stays one line even inside a run. Two ways of giving a window more context were
# measured on the RTX 3080 (2026-09-13) after the 8 s accuracy fixture came back "A quick
# brown fox" from its own 3-second window (beam search on that window; greedy search and the
# whole-file window say "The", log-probabilities 0.01 apart -- the model, not the pipeline):
# decoding a file in one language as the plain whole-file call lost 188, 126 and 220 of the
# three monolingual long-form fixtures' windows and invented speech in their silences; joining
# lines of one run closer than 1.0 s into one clip lost 97, 37, 86 and 118 on the film,
# film_mono, bookends and episode fixtures. Both are line-sized windows' work undone: a
# window holding several lines drops lines. Do not re-run either expecting a different answer.
ASR_SEGMENT_FIRST = _flag("ASR_SEGMENT_FIRST", "true")

# Splitting for the clip scan, deliberately separate from the decode VAD: a clip boundary
# decides where the language may change, not what gets transcribed. These are the
# values measured, not a tuned optimum -- Silero splits within an utterance rather than at its
# edges, so no region spanned a language change anywhere from 150ms to 500ms.
SEGMENT_SPLIT_MIN_SILENCE_MS = int(os.environ.get("SEGMENT_SPLIT_MIN_SILENCE_MS", 250))
SEGMENT_SPLIT_PAD_MS = int(os.environ.get("SEGMENT_SPLIT_PAD_MS", 200))

# A file must hold more than one speech region to be worth clipping, and that is the whole
# requirement -- there is deliberately no minimum duration.
#
# There was one, at 30s, on the reasoning that "below one decode window the file is already a
# single window". That describes the file and not the decode: clips are what turn one file into
# several windows, so the floor excluded precisely the case clips fix. Measured on the two
# recorded code-switched defects, both under 5 seconds:
#
#   mix_en_es   without clips: one segment, Spanish only -- the English leg dropped entirely
#               with clips:    both legs, each in its own language
#   mix_zh_en   without clips: one segment, Chinese only -- the English half dropped
#               with clips:    both halves
#
# Those are the `mix_en_es` and `mix_zh_en` entries the manifest had carried as known defects.
SEGMENT_FIRST_MIN_REGIONS = int(os.environ.get("SEGMENT_FIRST_MIN_REGIONS", 2))

# Rejoin regions separated by a pause too short to be a turn boundary. Silero splits at breaths
# inside an utterance, and the sub-second fragment that produces is where per-window language
# detection is least reliable: across nine real film excerpts the segments the decoder rendered
# in the wrong language had a 1.36s median duration against 2.40s for the correct ones, and 62%
# were under two seconds.
#
# This is real silence between regions, which is why the VAD is asked not to pad (see
# speech_clips._pad). Measured against *padded* gaps the same 0.3 merged at 0.7s of silence,
# because Silero pads each side by SEGMENT_SPLIT_PAD_MS and halves anything shorter than twice
# that. On the long-form fixture -- adjacent utterances every one a different language, never
# closer than 0.403s -- that presented the tightest boundary as a 0.016s gap and fused it:
#
#   padded gaps, 0.3   123 clips,   7 spanning a language change,   5 of 118 windows at 0.00
#   padded gaps, 0.1   213 clips,   0 spanning,                     0 of 118 windows missed
#   real gaps,   0.3   nothing merges on this fixture (min real gap 0.416s), 0 missed
#
# The stress fixture has no breaths to rejoin -- its utterances are cleanly separated -- so a
# correct threshold merging nothing there is the right answer, not a disabled feature. The
# natural fixture, which does have them, still merges 8 of its gaps at this setting.
#
# Note what this leaves the knob able to act on. The scan will not split on silence shorter
# than SEGMENT_SPLIT_MIN_SILENCE_MS (250ms), so the only gaps that can exist *and* be merged
# are those between that and this threshold -- a 250-300ms band. The merge is close to inert
# by construction, and widening it is not free: the stress fixture puts a language boundary at
# 0.403s, so anything approaching that fuses one. Whether a near-inert merge still earns its
# place is an open question, because the real-media evidence for it (wrongly-rendered segments
# having a 1.36s median against 2.40s) was gathered while the padded-gap bug was live and the
# merge was really acting at 0.7s.
SEGMENT_CLIP_MERGE_GAP_SEC = float(os.environ.get("SEGMENT_CLIP_MERGE_GAP_SEC", 0.3))

# A block of regions in a different language becomes a decode run of its own only when it
# holds this much speech, at LD_MIN_CONFIDENCE or better. Below that it is absorbed into the
# run around it. Language detection on a single ~1s line is wrong about one time in five on
# real film (median 20% of speech mislabelled across 19 monolingual films, almost always as a
# neighbouring language), and a run decoded in the wrong language is the `windows` defect
# arriving from the other direction. Every utterance on the stress fixture is over 3s, so
# each survives as its own run and 0/118 stands; a scene in another language on real film
# is many seconds long and survives too. A single foreign line under 3s does not, which is
# the trade: it cannot be told from a mislabel with the audio available.
SEGMENT_RUN_MIN_SWITCH_SEC = float(os.environ.get("SEGMENT_RUN_MIN_SWITCH_SEC", 3.0))
# ...unless the block is a fifth of everything labelled: on a five-second clip a two-second
# leg is not a slip, and the six code-switched fixtures are exactly that shape. The bar is the
# lesser of the two, so on a film the seconds decide and on a short clip the share does.
SEGMENT_RUN_MIN_SWITCH_SHARE = float(os.environ.get("SEGMENT_RUN_MIN_SWITCH_SHARE", 0.2))

# Report the language of each segment, not just of the file. No engine offers this --
# faster-whisper picks a language per decode window and never says which -- so it is measured
# per region, at one encoder pass each (~220ms on an RTX 3080, ~6% of a long-form run on a 5090).
#
# Reporting only. The per-region detection itself is no longer optional: it is what groups
# regions into language runs (SEGMENT_RUN_MIN_SWITCH_SEC above), and without it decoding by
# region fractures monolingual film. So the pass runs whenever ASR_SEGMENT_FIRST does; this
# switch decides whether its result reaches the response.
#
# Deliberately NOT gated on a whole-file "is this multilingual" signal, which was tried and
# measured useless for it. The montage vote's top-language share separates the synthetic
# fixture (0.30) from real film (0.92-1.00) cleanly -- but the genuinely code-switching films
# read 1.00 too, because the montage samples a few windows across two hours and the foreign
# passages are localised. Gating on it would label the fixture and miss every real film, which
# is optimising for the fixture. The repo's own note that a whole-file multilingual gate is the
# wrong abstraction (scripts/audio_matrix/longform.py) is now measured rather than inherited.
#
# The existing `multilingual_suspected` is worse still for this: it reports False on the
# 10-language fixture, because its lone-dissenter rule needs a language to hold two votes or a
# third of them, and nine windows spread over five languages give neither.
ASR_SEGMENT_LANGUAGES = _flag("ASR_SEGMENT_LANGUAGES", "true")

# Force transcription: one subtitle in the file's language for a film that switches. The
# runs in the file's language are transcribed; every run in another language is translated
# -- into English, the only target Whisper translates to -- so a Romanian film with Turkish
# scenes comes back as Romanian dialogue and English for the Turkish, rather than the
# Turkish decoded under the Romanian token into noise. Off by default: it changes what a
# plain transcription request returns. A transcription that names its language follows the
# switches when this is on, as a translation always does. Per request: `force_transcription`.
ASR_FORCE_TRANSCRIPTION = _flag("ASR_FORCE_TRANSCRIPTION", "false")
