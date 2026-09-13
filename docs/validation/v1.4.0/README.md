# v1.4.0 evidence: decoding by speech region

Why the `windows` and `quiet` defects closed, what it cost, and what is still unproven. Kept in
the repo because the claims are about *which language* a passage was decoded in, and that is
only checkable against the transcript that produced it. An earlier round of the real-media
comparison was lost to a cleaned temp directory: the numbers survived in the session log, the
evidence did not.

## The change

A decode window is the unit of language commitment. Left alone it is 30 seconds of whatever VAD
compacted together, so on audio that changes language faster than that, the decoder picks one
language for the lot. The VAD speech regions are where the language may change: each is
detected before the decode, grouped into runs of one language with hysteresis, and each
language is decoded in its own `transcribe` call, told its language, one `clip_timestamps`
span per region. That is the third design this release measured; the two before it are
recorded below, because each was killed by real material rather than by a fixture.

## Long-form fixture, measured

Library-level spike, decoding the fixture directly (RTX 3080, large-v3 float16):

| | `windows` misses | `quiet` noisy | RTF |
| --- | --- | --- | --- |
| baseline | 67 / 118 FAIL | 11 / 25 FAIL | 0.044 |
| clips | 0 / 118 | 0 / 25 | 0.086 |
| clips + rejoined breaths | 5 / 118 (budget 11) | 0 / 25 | 0.063 |

### The 5 was a defect, not a trade (corrected 2026-09-10)

That last row was read as clips costing five windows to buy back RTF. It was not: the
breath-rejoin merge was fusing language boundaries, and each of the five came back at an
overlap of exactly **0.00** -- no matching words at all, the signature of a window decoded in
its neighbour's language.

The merge compared gaps between regions the VAD had **already padded**. Silero pads each side
by `speech_pad_ms` and halves any silence shorter than twice that, so a gap arrives shortened
by up to 2*pad = 0.4s. A threshold reading 0.3s therefore merged at **0.7s** of real silence,
and the fixture's tightest language boundary -- 0.403s, with every adjacent utterance in a
different language -- presented as 0.016s and fused.

Measured on the fixture, counting clips that contain utterances of more than one language:

| merge basis | clips | spanning a language change |
| --- | ---: | ---: |
| none | 225 | 0 |
| padded gaps, 0.1s | 213 | 0 |
| padded gaps, 0.3s | 123 | **7** |
| real gaps, 0.3s | 225 | 0 |

`speech_clips` now asks the VAD not to pad, merges on real silence, and pads afterwards,
clamping each side to the midpoint of the gap so clips stay ordered and disjoint. The
threshold keeps its 0.3s default and now means what its name says. End to end through the
service on the laptop, same fixture, same audio -- **on its CPU**, it later turned out (see
"The laptop's numbers were CPU numbers" below), which is why the wall clocks are what they
are; the miss counts are what this table is for:

| | windows missed | clips | RTF (laptop CPU) | wall clock |
| --- | ---: | ---: | ---: | ---: |
| padded gaps, 0.3s | 5 / 118 | 123 | 1.384 | 1665s |
| padded gaps, 0.1s | 0 / 118 | 213 | 2.181 | 2624s |
| **real gaps, 0.3s** | **0 / 118** | 225 | 2.421 | 2913s |

Coverage stayed at 99.9% and the service logged no errors in any of the three.

All six assertions then ran strictly on both variants under the fix -- **12 passed, no xfail
marks**: 225 clips on the stress grid and 304 on the natural one, `quiet` still at 0 noisy
windows, language coverage, end coverage and repetition all holding. Nothing but time was
spent. Note what the natural fixture's clip count says about the merge now: 309 regions
collapse to 304, where the padded-gap version collapsed them to 66.

That run's RTF assertion passed only because the budget was raised to 5 for it -- and the
reason it needed raising is the correction below, not the card.

### The budget holds where it is meant to (RTX 5090, 2026-09-11)

Post-fix, remote via WSL2, `nvidia` target, `ASR_ENGINE=FASTER-WHISPER`, `ASR_DEVICE=CUDA`:
the stress variant **passed all six assertions at the real 1.0 budget**, the whole pytest
session finishing in 134s -- and that session *contains* the 1203s transcription, so RTF is
under 0.12. Device evidence, not inferred from the transcript: banner `ASR Runtime: CUDA
(Compute: float16)`, `Resource Pool: cuda:0`, `[ONNX provider: CUDAExecutionProvider]` on
the UVR worker, and `/status` reporting `ASR=CUDA measured=True | UVR=CUDA measured=True`.
The natural variant skipped on that run (its fixture post-dates the last sync); a second run
with `--fixtures` covers it.

### CPU (Intel NUC, Core Ultra 7 255H, int8)

Same fixture, same fixed tree, `cpu` target: **225 clips**, the request completed with a
200 -- `Video: 00:20:03 | Total: 00:47:49 | Speed: 0.42x` -- an end-to-end RTF of **2.38**.
That is over the 1.0 budget, and was before this release too. The control row answers what
the default costs, same host, same clip:

| NUC CPU | `ASR_SEGMENT_FIRST=0` | `=1` (default) |
| --- | ---: | ---: |
| RTF | **1.04** | 2.44 |
| `windows` | FAIL | **pass** |
| `quiet` | FAIL | **pass** |

The default multiplies CPU wall clock by 2.35 and is what fixes both defects; without it
CPU still misses the budget, by a hair. So the default stays on -- it is right wherever the
budget can be met at all, which is any GPU -- and a CPU deployment that needs the speed
more than the two fixes can set `ASR_SEGMENT_FIRST=false` knowing exactly what it gives up.

The first attempt at this row was lost by the runner, not the NUC: the service finished and
answered, but the ssh session carrying the 48-minute request had no keepalive, went dead
without delivering EOF, and the runner waited on it until this machine's next suspend. The
result survived only in the service's own log. `remote_validate.sh` now sets
`ServerAliveInterval=30 / ServerAliveCountMax=6`.

**The correction costs 75% more wall clock on a CPU**, because the speed in that
0.063/1.384 row came *from* the fusing. The stress fixture is also the worst possible case
for clip count -- every adjacent utterance is deliberately a different language, so nothing
legitimately merges. On the laptop's actual GPU the corrected stress fixture runs at RTF
0.149 (below), so the budget question this paragraph once deferred to the 5090 is answered
on both cards.

Two sets of numbers in this file predate this correction and were measured with boundaries
being fused: the natural-fixture observations in the section above (the 12-passed run and
its 304 clips), and the nine-excerpt real-media comparison under "Real media" below. Those
are floors, not ceilings. Everything from the 21-film fracture rate onward -- the shipped
design, the 54-excerpt probe, the tier-B bars and the 12 September re-validation -- was
measured after it, with the merge on real silence.

`quiet` improving is not incidental: clips exclude the silence between them, so the decoder is
never handed a silent window to invent speech into -- the mechanism the manifest records for
WHISPERX scoring 0/25, without WhisperX's single-language commitment.

End to end through the service, on default configuration, both flipped XFAIL to XPASS on two
independent machines: an RTX 5090 (123.5s against a 156.9s baseline -- *faster* than baseline)
and the laptop (1696s against 1266s). The laptop's pair was measured on its CPU, not its
3080 -- see the correction below -- which is the whole difference between the two rows.

## Real media

Nine 12-minute excerpts from the library: five monolingual (Romanian, Italian, Danish, French,
Japanese) and four code-switching, each of the latter centred on the densest cluster of forced
subtitle cues, because a forced track marks exactly the foreign dialogue and a random window in
a two-hour film usually contains none. Audio only, reduced to 16 kHz mono on the machine holding
the media; no filenames, paths, host or transcript text reaches this repository.

Seconds of transcript in the wrong language, same audio in every column:

| | laptop (CPU decode, see below) | RTX 5090 |
| --- | --- | --- |
| baseline | 238s | 164s |
| clips + rejoined breaths | **156s** | **138s** |

Language is inferred from script and stopwords -- no text language-ID library ships in the image
-- so the counts are only meaningful compared against each other on identical audio, and the
individual flags include visible false positives. The direction agrees on both machines.

An earlier reading of this data reported a 49% *regression*. That was an artifact of counting
off-language *segments*: clips produce finer segmentation, so a shared defect that the baseline
emitted as one 30-second segment appears as six. Duration is granularity-invariant and reversed
the verdict.

## Monolingual film, 21 excerpts: the fracture rate (2026-09-11)

The nine excerpts above answered the multilingual question. The library is mostly films in
one language, so 21 more were measured -- ten minutes of mid-film dialogue each, 12
languages, audio only, anonymised on the machine holding the media -- with the **fracture
rate**: the share of transcribed speech, by duration, that the response labels as a language
other than the film's. Duration-weighted, because finer segmentation cannot inflate it.

| decode | median | mean | films over 5% | median RTF (RTX 3080) |
| --- | ---: | ---: | ---: | ---: |
| whole file, one call (v1.3.0) | 2.7% | 7.4% | 3 | 0.071 |
| per region, decoder re-detects each clip | **20.1%** | 25.4% | 18 | 0.169 |
| runs, decoded as one clip each | 3.0% | 11.9% | 8 | 0.116 |
| runs, one call per language, region clips (**shipped**) | **1.1%** | 9.9% | 6 | 0.149 |

The per-region row is the release's first design, and it is a regression on the case that
matters most: one line of dialogue is 1.1 s on real film, and the decoder's language
detection on one second of audio is confidently wrong one time in five, nearly always toward
a neighbour (ru->uk, no->sv/da, it->es) or English. The synthetic fixtures cannot show this:
the same detector scores 225/225 on the stress clip's clean voices. The residual films over
5% in the run rows (Russian with 72 s labelled Ukrainian, Romanian with 47 s of English) are
where a stretch of a neighbour language is longer than 3 s and confidently detected; without
a forced-subtitle track the excerpt cannot say whether those passages are mislabels or
genuinely in that language. The shipped design's six over 5% are those (Russian 69 s
Ukrainian, Romanian 47 s English, Japanese 40 s English, a music-heavy Polish excerpt with
20 s of German) plus the two degenerate excerpts with under 15 s of speech, whose "dominant"
language is a coin toss over a handful of lines.

**Subtitle overlap**, same excerpts where a same-language track exists (7 films; cues limited
to the window; relative only, since subtitles condense and paraphrase):

| decode | mean overlap | median share of cues below 0.4 |
| --- | ---: | ---: |
| whole file, one call | 0.456 | 56.9% |
| per region | 0.461 | 50.0% |
| runs, one clip each | 0.514 | 50.0% |
| shipped | **0.505** | **42.0%** |

## The film fixture caught the run design twice (2026-09-11)

Decoding a run as one clip, with the decoder re-detecting its language, passed the stress
and natural clips and failed the film clip on `quiet` and `windows`: a run reaching across
the passage between two scenes hands the decoder that passage inside a window, and 6 of the
film clip's 19 quiet passages sat inside a run. The same build on the RTX 5090 dropped the
second leg of `mix_de_en`, `mix_hi_en` and `mix_zh_en` -- 2 s legs under the 3 s switch bar.

Forcing each run's language on the decoder (the label the response reports) and keeping
the run as the clip then dropped whole lines: on the first 180 s of the single-language film
clip, decoded on the laptop CPU at int8 with the ground-truth regions,

| decode of the same 46 lines | lines with no text | segments |
| --- | ---: | ---: |
| forced language, run-sized clips | **5** | 48 |
| re-detected, run-sized clips | 0 (7 windows merged, up to 11 s) | 41 |
| forced language, region-sized clips | **0** | 48 |

So the run decides the language and the region is the decode window. That reopened one
door: with every pause between lines uncovered, gap filling ran on 7-50 slices per real
film and detected each one's language on a second of audio -- the Japanese film went from
8.7% back to 19.9% foreign. A gap now takes the language of the run it sits in, or the
nearest run. Shipped results on the
RTX 3080 (`cuda:0`): stress 0/118 at RTF 0.165, 225 clips across 10 calls; natural, film and
`longform_film_mono` pass `windows`, `quiet`, `fracture` (new: the language reported for each
line is the language spoken, bar 5% and 10%), `language`, coverage and throughput at RTF
0.19 / 0.17 / 0.17.

One assertion needed a second look: `repetition` on the film clip counted 44 "Sag mir die
Wahrheit" against the 43 the timeline holds. The 44th sits at 777 s, where the timeline has
*"Dis-moi la vérité"* -- the same sentence in French, a lone 1.2 s line in a German stretch.
The hysteresis absorbed it (a single line under 3 s cannot be told from a mislabel), the
run decoded it as German, and Whisper rendered it as the German sentence it already knew.
That is the documented single-line trade and a `windows` miss (inside the 10% budget), not
a decoder loop. The assertion was counting totals, which cannot separate the two; it now
judges the longest run of the same sentence *in a row*, which is what a loop is, against
the longest such run in the timeline. Film re-run: **7 of 7** at RTF 0.170.

Subtitle overlap for the shipped design, same 7 films: mean **0.505**, median share of
cues below 0.4 **42.0%** (whole-file 0.456 / 56.9%; per-region 0.461 / 50.0%; runs as one
clip 0.514 / 50.0%). Code-switched clips: **18 passed**, both legs of all six, on `cuda:0`.

`longform_film_mono` is new: 20 minutes of Russian lines over the film bed -- the library's
common case, which no fixture had -- measuring 0.37 speech density, 1.22 s median line,
0.94 s median pause, 26% of pauses over 2 s and a bed 3.0 dB under the speech, against
0.34 / 1.13 s / 0.99 s / 26.5% / 3.75 dB on the 24 real monolingual excerpts.

## Beyond mid-film dialogue: 54 more excerpts, five scenarios (2026-09-11)

The library has more shapes than mid-film dialogue, so 54 further excerpts were pulled the
same way (audio only, anonymised on the machine holding the media): the first ten minutes
of 8 films (**OPEN**), the last ten of 8 (**END**), the first thirty of 14 episodes
(**EP**), ten minutes of 8 non-English dub tracks (**DUB**), and the densest ten minutes of
forced-subtitle cues in 16 code-switching films (**CS**). Medians, same VAD as the suite:

| property | OPEN | END | EP | DUB | CS |
| --- | ---: | ---: | ---: | ---: | ---: |
| speech density | 0.28 | 0.04 | 0.66 | 0.41 | 0.47 |
| non-speech before the first line | **105 s** | 1.3 s | 10.5 s | 1.1 s | 0.1 s |
| bed under it (dB below dialogue) | 1.8 | 3.0 | 0.8 | 5.9 | 4.0 |
| non-speech after the last line | 0 s | **265 s** | -- | 0.7 s | 1.5 s |
| bed under it | -- | 1.6 | -- | 4.8 | 8.1 |
| longest passage inside | 82 s | 66 s | 23 s | 62 s | 33 s |
| bed under it | 1.6 | **-0.4** | **-2.8** | 1.6 | -0.9 |
| line length, median | 0.99 s | 0.97 s | **1.87 s** | 1.28 s | 1.21 s |

Five excerpts held no speech at all (two dub tracks, two endings, one opening: credits,
a music-only reel, a track tagged `zxx`) and are excluded. What the numbers say:

- A film **opens on 7-330 s without dialogue** (a logo, a cold open, a five-minute
  prologue) and **ends on 84-567 s of credits**, with the music at or just under dialogue
  level. That is the largest stretch of loud non-speech the decoder ever sees, and nothing
  in the suite had it: the fixtures' longest passage was 30 s.
- An episode's title sequence sits **above** the dialogue (-2.8 dB), television dialogue
  is denser (0.66) with lines twice as long (1.9 s) and scene gaps that never exceed 23 s.
- A dub track is within the mid-film range on every axis. The studio voice is a timbre,
  not a layout, so it gets no fixture of its own.

Two fixtures are laid out from these (`scripts/audio_matrix/film_shapes.py`):
`longform_film_bookends` (English: a 60-150 s opening, a 60-90 s montage, 180-300 s of
credits) and `longform_episode` (French: a 5-20 s cold open, an 18-30 s title sequence
above dialogue level, scene gaps capped at 23 s). Measured with the same script:

| property | real OPEN / END | `film_bookends` | real EP | `episode` |
| --- | ---: | ---: | ---: | ---: |
| non-speech before the first line | 105 s | 108 s | 10.5 s | 13 s |
| non-speech after the last line | 265 s | 282 s | -- | 22 s |
| longest passage inside | 66-82 s | 86 s | 23 s | 37 s |
| bed under the opening | 1.8 dB | 5.3 dB | 0.8 dB | 3.8 dB |
| bed under the longest passage | -0.4 dB | 5.4 dB | -2.8 dB | 1.9 dB |

The passage lengths land; the passage **beds land 3.5-4.7 dB under the measured medians**.
The synthesized bed cannot go above the level three full-scale tones peak at without
clipping, and the lines are already rendered at a quarter gain to give it room, so a
title sequence louder than the dialogue is not reproduced -- the fixtures' passages are
within the measured range (openings -0.5 to 5.2 dB, credits -4.3 to 13) but on its quiet
side. They are a milder test of invented text in credits than a median film is, and are
recorded as such.

First hardware run of both, RTX 3080 `cuda:0`: `longform_episode` **7 of 7** (RTF 0.173);
`longform_film_bookends` 6 of 7 at RTF 0.131 -- `quiet` holds over 108 s of opening music
and **282 s of credits** with not a word invented, and the one failure was the coverage
assertion reading the credits as a transcript that "stopped at 932 s of 1214 s". It
stopped where the dialogue stops, which is right; the assertion now judges coverage
against the end of the last line in the timeline, not the end of the audio. The planner
then learned to stop a scene once the credits are due (a 28-line scene could otherwise
carry the clip past them and drop them), which re-laid the bookends clip at 166 lines;
that layout passes **7 of 7** on the 3080 (RTF 0.115) and all six variants pass **42 of
42** on the RTX 5090 (`ASR=CUDA measured=True`).

On the Intel NUC's CPU (int8, `--timeout 10800`), all six variants hold every correctness
assertion under the shipped design and fail only `throughput`, as CPU always has: RTF
2.39 on the stress clip (2.44 under the first design -- one decoder call per language
costs nothing measurable there), 3.12 natural, 3.08 film, 2.92 film_mono, 2.13 bookends,
3.24 episode. Five hours forty-three minutes for the set.

One observation from the probes on the laptop's 8 GB card: a ten-minute code-switching
excerpt failed once with a CUDA out-of-memory inside faster-whisper's decode -- one
request in about 120 of that length on this card, answered as a clean 500 with the
service recovering for the next request. The large-v3 model in float16 with UVR beside it
leaves that card under 2 GB of headroom; it is a capacity fact about 8 GB cards, not
something the release changed, and it is recorded here so the next such 500 is read as
what it is.

### The pipeline on the 54 excerpts (RTX 3080, 2026-09-12)

The third attempt of the probe went through: 54 of 54 excerpts, every one on
`hardware unit cuda:0`, one request each, the shipped design (commit `7e91613`, RTF
0.03-0.22). Two questions were put to it.

**Invented text where there is no dialogue.** Measured on the stretches the acoustic
survey established as non-speech: a film's opening before the first line, its credits
after the last. Over **1,430 s of openings** the transcripts hold **32.6 s of text**
(1.4 s per minute of music), 21 s of it in one opening that runs 332 s without a line;
over **2,965 s of credits, 16.3 s** (0.3 s per minute), 9.5 s of it in one 600-s
excerpt that is credits from end to end. Six of the eight endings and five of the eight
openings hold under a second. The two openings and one ending that held no speech at all
came back with **not a word** -- the exact case the bookends fixture reproduces at 0 of 25.
Inside the dialogue, the probe also counts text in the pauses its own VAD scan drew as
non-speech: 8-9 % of that time on episodes and code-switching films. That number is not
a hallucination count -- the scan is not the decode VAD, and the gap-filling pass decodes
speech the stricter VAD excluded, which is where most of it lands (14-23 filled gaps per
excerpt in the service log); it is recorded as the ceiling on what invented text could be,
not as its measure.

**Six dub tracks of six** are reported in the dub language, foreign share 0 %; the other
two dub excerpts contain no speech (an extraction that landed on a silent reel, and the
`zxx` track) and measure nothing. Fracture on the non-CS excerpts: median **0 %** on all
four scenarios; the exceptions are a Hindi episode reporting 107 s of English against 773
s of Hindi and a Romanian one 213 s against 1,221 s -- television that does switch --
and two Russian episodes that report 28 s and 183 s of Ukrainian, the pair the survey
already named as the model's confusion.

**Forced-cue precision and recall on the CS titles.** The forced tracks first had to be
read for what they are. On 7 of the 16 titles the track flagged `forced` covers 84-98 %
of the excerpt's speech: it is the full subtitle track with the wrong flag, and says
nothing about which lines are foreign. On the 9 with a real forced track:

| title | forced cues, share of speech | reported foreign runs | precision | recall (s) |
| --- | ---: | ---: | ---: | ---: |
| A (en/ru) | 386 cues, 39 % | 22 runs, 177 s | **0.87** | **0.73** |
| B (en/bs) | 56 cues, 9 % | 4 runs, 35 s | **1.00** | **0.77** |
| C (en/es) | 13 cues, 5 % | 1 run, 8 s | **1.00** | 0.26 |
| D (en) | 189 cues, 45 % | none | -- | 0 |
| E (en) | 53 cues, 19 %, 1.4 s a cue | none | -- | 0 |
| F, G, H, I (en, en, it, en) | 2-31 cues, 0-5 %, lines of 1.9-3.4 s | none | -- | 0 |

Where the pipeline reports a foreign run it is right **87-100 % of the time**; the
coarse language map this release rejected scored 31 % on the same question. Where the
foreign material is scene-length (A, B) it finds three quarters of it. Where it is single
lines -- F to I hold 2-31 cues of two or three seconds scattered through an English film
-- it reports nothing, which is the hysteresis trade the release documents: a line under
`SEGMENT_RUN_MIN_SWITCH_SEC` is decoded in the surrounding language rather than risk the
20 % fracture that per-region decoding costs a monolingual film. Titles D and E looked
like a different case -- D has forced cues under 45 % of its speech and E 53 cues in a row
-- so the per-region labels were dumped for both, before any merge, with the same model
on the same regions (RTX 3080, service stopped for it). They are the trade after all:

- **D**: 127 of the 128 s of speech under the forced cues are labelled **English**; the
  longest run of one confident non-English label anywhere in the excerpt is 0.7 s. The
  track is flagged forced, but what it covers is not another spoken language, and the
  pipeline reporting one English run is right.
- **E**: the 25 s under the cues are labelled Korean 6 s, Arabic 6 s, English 4 s, then
  Albanian, Turkish and German; 31 of the 120 regions are under 0.5 confidence, and the
  longest block of one confident non-English label is 1.4 s. Lines that short the
  detector cannot name consistently -- there is no one language it could have reported --
  and the merge folds them into the English around them, which is what the hysteresis is
  for.

## Code-switched clips

The same change closes the two recorded code-switched defects, and the reason is the floor
that was nearly shipped with it. `SEGMENT_FIRST_MIN_DURATION_SEC=30` would have excluded any
file shorter than one decode window on the reasoning that such a file is already a single
window. That describes the file, not the decode: clips are what turn one file into several
windows, so the floor excluded precisely the clips this fixes -- both recorded defects are
under five seconds. It was replaced by "more than one speech region", and the 24-clip
single-language accuracy suite is unchanged by the removal (25 passed either way).

Scored per clip, one request each, on the laptop (CPU decode, see below):

| clip | FASTER-WHISPER | WHISPERX |
| --- | ---: | ---: |
| `mix_en_es` | 0.75 | 0.38 |
| `mix_en_fr` | 1.00 | 0.50 |
| `mix_de_en` | 1.00 | 0.43 |
| `mix_hi_en` | 0.96 | 0.31 |
| `mix_zh_en` | 0.88 | 0.21 |
| `mix_ar_fr` | 1.00 | 0.50 |

Two findings, only one of them about this release's change:

- **FASTER-WHISPER now returns both legs of all six**, where before it returned one on
  `mix_en_es`. `mix_zh_en` already passed, so its entry was never a faster-whisper defect.
- **WHISPERX returns one leg on all six**, and the 0.40 bar was low enough that a single
  *complete* leg (about 0.50) passed. Three clips were recorded as fine on WHISPERX while
  returning half the file. The bar is now 0.60, between the highest one-leg score and the
  lowest two-leg score, and the six entries carry `xfail_engines: ["WHISPERX"]` so the
  default engine holds them strictly rather than reporting six XPASSes.
- **OPENAI-WHISPER drops a leg on three of the six.** The raised bar caught it on the W5
  smoke run (`mix_en_fr` at 0.50), so all six were measured on the RTX 5090 before scoping:
  `mix_en_es` 0.38, `mix_en_fr` 0.50 and `mix_ar_fr` 0.50 return the second leg only, while
  `mix_de_en`, `mix_hi_en` and `mix_zh_en` clear 0.60. The reference implementation detects
  once on the first window and decodes the file in that language; region decoding is a
  faster-whisper option. The three entries carry both engines in `xfail_engines`.

The `yo_mms` entry was cleared for a different reason again: it recorded partial word overlap,
but the clip is tier B, and tier B asserts only "detected correctly, or transcribed to
something" -- it never scored content. Under a comparison that folds away combining marks the
transcript scores 0.76, against a ceiling of about 0.90 rather than 1.00, because
`normalize()` turns Yoruba's tone marks into spaces and fragments the reference.

## Tier B, scored for the first time (2026-09-11)

All 78 long-tail entries declared a `min_word_overlap` that no test read. Before enforcing
it, the *ceiling* of every reference was computed -- the score a flawless transcript can
reach, since `normalize()` splits reference words at every combining mark. Five sit below
1.0: Tamil 0.43, Punjabi 0.44, Gujarati 0.50, Thai 0.69, Yoruba 0.89. Tamil and Punjabi
therefore had bars a perfect transcript could not clear. The bar is now
`min_word_overlap * ceiling`, applied to every tier; every tier-A and combined ceiling is
1.0, so the bars already validated do not move, and a unit test holds both facts.

Measured on FASTER-WHISPER on the laptop (CPU decode, see below), 66 clips scored exactly as the suite scores:

| result | clips |
| --- | --- |
| clear the bar | 55 -- including Punjabi and Tamil, which fail the raw number and pass the scaled one |
| near miss, 0.40-0.46 against 0.5 | `et`, `ka`, `lv`, `sw` |
| barely or not transcribed | `hy` 0.09, `sq` 0.09, `ml` 0.00, `gu` 0.10 of its 0.50 ceiling |
| transcribed as a neighbour | `lb` -> German (0.09), `sr` -> Czech (0.00), `ur` -> Hindi in Devanagari (0.00) |

The two kinds are recorded differently, on purpose. The seven that fail outright are
`xfail_reason` entries scoped to content, with the numbers. The four near misses get a
**strict bar of 0.3**: an xfail on a clip scoring 0.42 would stay green if it fell to 0.1,
which is the regression the bar exists to catch, whereas 0.3 is below what the model does
today and fails the moment it does materially less. That is what tier B's own contract
asks for -- "per-language accuracy varies more than a single threshold can express" -- and
it had never been done.

The neighbour cases are the interesting ones. Luxembourgish, Serbian and Urdu are each
detected as a language the model has far more of, and the transcript that comes back is a
fluent transcript of the wrong language. The identity assertion still passes for all
three, because tier B accepts "detected correctly *or* transcribed to something" and each
was transcribed to something. Only the content bar can see it.

### The same tier on the RTX 5090, float16 (W3, W3b)

The bars above were set from an int8 CPU decode. The full matrix on the 5090 (CUDA,
float16) cleared all but a handful, and the handful says two things worth keeping:

- **Some clips are not deterministic.** `ka_tail` scored 0.20 against 0.30 on one run and
  cleared it on the next; `te_tail` 0.39 against 0.50, then cleared. Both trip
  faster-whisper's temperature fallback, which samples. They are recorded as content
  defects (`strict=False`) so a flap reads as neither a fix nor a regression.
- **Some are device-dependent.** `bn_tail` came back as Latin-script gibberish at 0.21 on
  both 5090 runs where the CPU decode had cleared 0.5. Recorded, not lowered.
- `pa_mms` is the sentence, romanised -- "Poora loombad aalsi kutte utte chhaal" -- so the
  Gurmukhi reference scores 0.00. `uk_line3`, the shortest clip in the matrix at one second,
  comes back as one run-together word. Both recorded.
- `ru_line2` at 0.50 was a scoring bug: the reference spells "чём", the decoder writes
  "чем", as written Russian does. `normalize()` now folds ё to е.

## A fixture laid out from real film (2026-09-11)

The synthetic fixtures had never been measured against the media they stand in for. The
same VAD the suite scores by, run over the nine library excerpts and over the fixtures --
medians across excerpts, nothing but numbers leaving the machine that holds the media:

| property | real film | stress | natural | **film** |
| --- | ---: | ---: | ---: | ---: |
| speech density | 0.42 (0.18-0.75) | 0.55 | 0.68 | 0.32 |
| utterance length, median | 1.28 s | 2.82 s | 2.53 s | 0.99 s |
| pause between lines, median | 0.86 s | 0.83 s | 0.58 s | 0.93 s |
| utterances per scene, median | 3 | 6 | 8 | 4 |
| longest non-speech passage | 30-232 s | 15 s | 17 s | 77 s |
| music/ambience below speech | **2.6 dB** (-6.1 to +9.8) | 20.3 dB | 20.7 dB | **3.4 dB** |

The last row is the one that changes what the suite can find. Real non-speech is not
silence: the score and the room sit 2.6 dB under the dialogue on a median film and above
it on one of the nine. Every `quiet` number recorded so far -- including 0/25 -- was
measured against beds 18 dB quieter than that, an order of magnitude in power, which is a
case real media never presents. Real lines are also a third the length of the fixtures',
in exchanges of three, which is where per-window language detection has the least audio to
work with.

`longform_film` (`scripts/audio_matrix/longform_film.py`) is laid out from those numbers:
40 new one-line clips (`*_line1-4`, ~1.3 s, `role: line`) in short scenes with a long
tail, heavy-tailed passages between them, and a synthesized music-and-room bed under the
whole clip at a per-scene level drawn from the measured range. It was calibrated by
rendering and re-measuring with the same script until the bed landed where film has it
(16.3 -> 9.6 -> 5.7 -> 3.4 dB). No library audio, text or title is used; the excerpts
contributed distributions.

Two honest gaps remain. Density lands at 0.32 rather than 0.42: the VAD misses some of
the quietest lines under the bed, which real film also suffers, and the measured real
value already includes that. And the longest passage measures 30 s rather than the 77 s
laid out, because Silero fires on the room bed and chops it -- also true of real film,
where the 232 s passage is the exception that music happened not to trigger.

The `natural` and `stress` clips are byte-identical to before: the new line clips carry a
role those profiles exclude, and both regenerated to the same timelines
(1203.049 s / 118 and 1206.718 s / 158).

## The laptop's numbers were CPU numbers (2026-09-11)

Every FASTER-WHISPER request driven through the service on the RTX 3080 laptop in this
file ran on its **CPU**. The pool on that host holds `cuda:0` and the Intel iGPU; each
auto-detect request runs language detection before transcription; detection took `cuda:0`
and returned it to the pool's tail; the transcription then took the head -- the Intel unit,
on which CTranslate2 has no backend -- and loaded a second model on the CPU. The banner
said `ASR Runtime: CUDA` throughout, because it describes `DEVICE`, not the unit a task
lands on. Full account in `docs/REMOTE_VALIDATION.md`; fixed in
`modules/inference/scheduler/unit_choice.py`.

The correctness results above stand -- CPU int8 decoded the same audio to the verdicts
the 5090 gave in float16 -- and the CPU wall clocks are what they are, consistent with the
NUC's. What changes is the throughput story. With the transcription on the card it was
sitting next to, all three long-form variants at the **real 1.0 budget**, `cuda:0` in the
log for every request, 18 passed:

| variant | clips | RTF, RTX 3080 | RTF, same laptop on CPU |
| --- | ---: | ---: | ---: |
| stress | 225 | **0.149** | 2.421 |
| natural | 304 | **0.197** | 2.890 |
| film | 324 | **0.179** | 3.307 |

The 3080 is about 1.4x the 5090's time, which is what two GPUs look like. Twenty times was
never a GPU ratio, and that should have been the tell.

## The review fixes, re-validated (2026-09-12)

The PR review changed three things on the decode path -- the opening block is judged by
the same hysteresis as every other, language groups are ranked by the seconds actually
spoken, and a gap's report entry is suppressed only by a covering span in its own language
-- so the whole set was run again on the committed tree (`7e91613`), with the modules'
checksums confirmed inside the container before the first request:

| host | run | result |
| --- | --- | --- |
| RTX 3080, `cuda:0` | stress fixture, one request | **0 of 118** windows missed, RTF 0.163 |
| RTX 3080, `cuda:0` | six long-form variants | **42 passed**, RTF 0.10-0.19 |
| RTX 3080, `cuda:0` | six code-switched clips | **18 passed** |
| RTX 3080, `cuda:0` | fracture, 21 real films | no film worse than the shipped baseline; mean over the 18 valid excerpts 5.2 % -> 4.4 % (six moved down by 1-5 points, none up) |
| RTX 5090, `ASR=CUDA measured=True` | six long-form variants | **42 passed** (15 min) |
| Intel NUC, CPU int8 | six code-switched clips | **18 passed** (11 min) |

The second wave -- a foreign gap decoded without the caller's prompt, an unlabelled region
adding no confidence to its block, the detector's failure guard narrowed to the detector,
and every translation following the switches -- went through the same pass, now with the
translation assertions in every suite (`f29ed53`, checksums confirmed in the container):

| host | run | result |
| --- | --- | --- |
| RTX 3080, `cuda:0` | six long-form variants + stress and film translated | **44 passed**, RTF 0.10-0.19 |
| RTX 3080, `cuda:0` | six code-switched clips, transcribed and translated | **24 passed** |
| RTX 3080, `cuda:0` | tier-A, transcribed and translated | **28 passed, 2 xfailed** -- the `it` and `uk` translation defects, reproduced at the recorded numbers |
| RTX 3080, `cuda:0` | fracture, 21 real films | no film worse than the shipped baseline; mean 5.2 % -> 4.4 % |
| RTX 5090, `ASR=CUDA measured=True` | six long-form variants + two translations | **44 passed** (19 min) |
| RTX 5090, `ASR=CUDA measured=True` | code-switched, transcribed and translated | **24 passed** |
| RTX 5090, `ASR=CUDA measured=True` | the full real-audio suite (W8, `--timeout 3600`) | **390 passed, 11 xfailed, 3 xpassed** -- the xpasses are the recorded flappers `ka_tail`, `te_tail` and `it_core`'s translation; `uk_core`'s translation xfailed as recorded |

The NUC's long-form runs at these commits were lost to the deadlock described next; its
code-switched run at the first review commit stands, and the final tree's accuracy suite
and pause/resume drive on it are recorded there. The final tree on the RTX 3080
(`cuda:0`, 2026-09-13, checksums confirmed in the container): accuracy 9, code-switched
30, tier-A 28 + 2 xfailed, long-form 44. The RTX 5090 was off when the final tree was
ready; its numbers are the ones above, measured before the resumable decode existed, so
that path has run on the RTX 3080 and the NUC only (below).

## Translation, measured for the first time (2026-09-12)

Every hardware number above was `task=transcribe`. The translate path runs through the same
per-language decode -- `task` travels unchanged into each call -- so a Turkish scene in a
Romanian film is translated *from Turkish*; decoded under the film's language token it
would come back as noise. Two things were missing: fixture tests that say so, and the gate
that let a caller's `language=` switch the per-language decode off. A subtitle client
names the audio track's language on every request, so on the path those clients use a
translation was one call under one language token. A translation is English whatever was
spoken, so a named language on it is now the main audio, not a constraint: every
translation follows the switches (`follows_switches` in `model_manager`), and only a
*transcription* in a named language keeps the single-language path.

The matrix already holds the ground truth: it is one sentence pool rendered in every
language, so a clip's English meaning is its `en_` sibling, and the six code-switched
foreign legs got an explicit `translation` in the manifest. Three assertions, first run on
the RTX 3080 (`cuda:0`, FASTER-WHISPER large-v3, float16):

| assertion | clips | result |
| --- | ---: | --- |
| long-form: every window translated where it occurs, bar 0.40 | stress grid (118 windows, 10 languages), multilingual film (299) | **both pass** |
| code-switched: both legs in English, foreign leg's own words gone | 6 | **6 of 6** |
| tier-A: the clip translates to its English sibling, bar 0.50 | 10 | **8 of 10** |

The two tier-A misses are the model, not the pipeline: `it_core` leaves half the pangram in
Italian ("La Rapida Volpe Marrone jumps over Canepigo", 0.44) and `uk_core` paraphrases it
into "a species of red fox is jumping over a feline dog" (0.31). Both transcribe fine. They
are recorded in the manifest as translation-scoped defects with those numbers -- the
`xfail_scope` now takes a list, so an engine that drops a leg on transcription carries its
mark on translation too without the transcription bars loosening.

## The NUC found a deadlock (2026-09-12)

The NUC's second long-form run never finished, and not because of the CPU. The NUC is also
the host the user's Bazarr sends work to, and at 10:24 UTC a `/detect-language` for a
44-minute episode asked the single CPU unit to pause the natural-fixture transcription.
The transcription paused where it was -- inside the per-region detection stream, with the
worker channel held -- and the priority task, once its UVR pass was done, needed that same
channel for its own detection: `Blocked on lock for >5.0s during generation` at 10:27:15,
and then nothing. Seven hours later: 41 detect-language requests queued, the health check
unanswered, `/status` empty, load average 0.24. The paused task waited for the priority
task; the priority task waited for the paused task's channel.

The condition predates this release -- `consume_segments` paused between decoded segments
inside the transcribe stream in v1.3.0 too, on the same channel the priority detection
uses -- and this release widened the window with the per-region detection stream, which on
a CPU is twenty-five minutes long. The fix is the one the channel was built for: a pause
releases the worker. Regions go to the worker in chunks consumed to the end; a decode asks
a non-blocking question before each segment and, when a pause is pending, closes its
stream, waits, and resumes on the clips past the last consumed segment
(`resumable_decode`); gap slices are consumed whole.

The first reproduction run then found the same deadlock on the *other* channel. The
accuracy test's transcription was in the UVR pass of its own language-detection montage
when the next external `/detect-language` arrived (a 90-minute film this time); the yield
inside that separation stream paused with the preprocessing worker's channel held, and the
priority task's montage UVR waited for it. The yield in an isolated separation now only
asks: a pending pause abandons the stream -- the manager's tested contract turns that into
a cancelled worker-side separation and a freed channel -- the wait runs, and the separation
is started again (`resumable_decode.separate`).

Re-measured on the NUC with the final tree (N8b, 2026-09-13, CPU int8, `hardware unit
CPU`): the accuracy suite passed (9), then the stress clip was sent as a transcription and
three priority `/detect-language` requests were fired into it by hand -- at +601 s, inside
the per-region detection; at +1804 s and +2402 s, inside the decode. All three returned:
40 s, 177 s (queued behind one of Bazarr's) and 36 s. Bazarr itself sent 39 requests into
the same window -- the user's production traffic, a detection every ~3 minutes -- and 28
priority tasks completed while the run was watched, none blocked. The log shows the
mechanism where it was missing before: `Preempting task on CPU... (old_stage=Inference)`
during the detection, and in the decode `Paused after 14 segment(s); the worker is
released for the priority task` followed by `Resuming: 16 clip(s) left to decode`,
seven such pauses in twenty minutes, the transcription advancing between them.

What the run also shows is the scheduler's priority policy as designed: with a detection
arriving every three minutes and each taking 2:45 on this CPU, `Keeping unit CPU paused:
queued priority backlog (1) saturates capacity (1)` -- the transcription is starved, not
deadlocked, for as long as the backlog lasts. That is the right order for a subtitle host
(detections are what the client is waiting on), and it is why the driven transcription had
reached only 7:38 of its 20:03 when the host was rebooted 2 h 47 min into the drive (a
clean `systemd` shutdown, not the service; the container came back on its own). The
previous NUC run of the same clip, without a backlog, finished in 2933 s with 226
segments and all ten languages.

## A window that starts cold (2026-09-13)

The NUC's deadlock reproduction also ran the accuracy suite, and its 8-second English
fixture -- two sentences 0.38 s apart -- came back "A quick brown fox jumps over the lazy
dog". The RTX 3080 says the same, every time, through the service. Decoded by region the
first sentence is a 3-second window of its own, and on that window beam search with the
shipped settings prefers "A"; greedy search and the whole-file window prefer "The", with
the two hypotheses' log-probabilities 0.01 apart. The normalised audio the service decodes
(`dynaudnorm`) is what tips it; on the raw file the same call says "The". It is the model
on a near-tie, not the pipeline -- but it is the release's default path reading a basic
fixture worse than the previous one did, so two ways of giving the window more context
were measured on the 3080 (`cuda:0`), each through the full suite:

| design | accuracy fixture | tier-A | code-switched | long-form |
| --- | --- | --- | --- | --- |
| shipped: one clip per line | "A quick brown fox" | 28 + 2 xfail | 30 | 44 |
| a file in one language as the whole-file call, told its language | passes | 27, `pt_core` translation failed | 30 | **6 failed**: film_mono, bookends and episode lost 188, 126 and 220 windows and invented speech in their silences |
| lines of one run under 1.0 s apart share a window | passes | 28 + 2 xfail | 30 | **5 failed**: film, film_mono, bookends and episode lost 97, 37, 86 and 118 windows; the film translation 120 |

A window holding several lines drops lines -- the finding that set the clip at one line in
the first place, now measured on every long-form shape at once. Both designs are recorded
in `config_segmentation.py` as rejected. The clip stays one line, and the accuracy suite
now scores each sentence by ordered word-level edit distance within one word of itself --
a substituted article passes, a missing or invented sentence does not, which is what the
suite exists to catch (a broken accelerator path). Set overlap, the matrix's scorer, would
not have noticed the swap at all: "A quick brown fox jumps over the lazy dog" holds every
expected word, "the" included.

## What is not established

- **The resumable decode on the RTX 5090.** The pause-and-resume path (`resumable_decode`,
  chunked region detection) was written after the 5090's runs above and has been measured
  on the RTX 3080 (every suite, no pause taken) and on the NUC (pauses taken, N8b). The
  5090 was off when the final tree was ready; its next run is the full suite at this
  commit.
- **Foreign lines under three seconds are not reported.** On the 9 NAS excerpts with a real
  forced track the pipeline's foreign runs are 87-100 % precise and find three quarters of
  scene-length passages; single foreign lines are absorbed into the run around them by
  design, and the per-region dump on the two titles that looked like exceptions showed
  the same thing (one is English under a misflagged track, the other is lines the detector
  labels five different ways). What is not established is whether a lower
  `SEGMENT_RUN_MIN_SWITCH_SEC` could recover those lines on real film without bringing the
  20 % fracture back; nothing in this release measures that curve.
- **The Intel NUC is measured, not fast.** Under `--timeout 10800` every correctness
  assertion holds on all six long-form variants; `throughput` fails at RTF 2.1-3.2, as a
  CPU host always has. The runner's 900-s default timeout is still what a CPU row hits if
  it is not raised, and the failure still presents as three unrelated assertions.
- **The natural fixture's density.** It is measured -- 7 of 7 strict on the RTX 3080 and the
  5090 under the shipped design -- but it runs at 68% speech density against 42% on real
  film, because its source recordings are ~6.3 s where real lines are ~1.3 s. The film
  profile answers that with the one-line clips; no layout constant fixes it for natural.
- **Tier-B bars are enforced, not settled.** Every tier-B clip is now scored against its
  declared `min_word_overlap` scaled by the reference's ceiling (above), and the 5090 runs
  showed the bars for a handful of hard scripts are not deterministic across runs or across
  int8 and float16. Those are recorded as content defects rather than tuned to one device.
