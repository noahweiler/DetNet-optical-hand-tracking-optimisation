# BBT Design Decisions — Implementation Reference

This document is a complete recap of every design choice made for the gamified
Box and Block Test (BBT) harness built on top of this repo's DetNet and
MediaPipe trackers. It is intended as a self-contained companion to the thesis
chapter / section on the interactive evaluation tool — every numbered decision
below maps to a justification in the main text plus a paragraph in the
methodology appendix.

The complete bibliography of cited works is given at the end. Inline cites use
first-author-and-year form (e.g. *Mathiowetz 1985*) and link to the
corresponding entry in [§ References](#references).

## Context

The clinical Box and Block Test (*Mathiowetz 1985*) is a 60-second timed task
for assessing gross manual dexterity: a subject moves as many wooden blocks as
possible, one at a time, from one compartment of a partitioned box into the
other. The score is the number of blocks transferred. The test is a standard
clinical outcome measure in upper-extremity rehabilitation.

This repurposes the BBT as an **evaluation harness for vision-based
hand-tracking models**. The physical box is replaced by an on-screen game
driven by webcam hand tracking; the subject pinches to grab a virtual block,
drags it across a centre partition, and releases to drop it on the target
side. The user task and timing are identical to the clinical version, so the
difficulty floor is anchored to a well-established benchmark; the *variable*
is the hand-tracking backend that resolves the pinch + cursor signal. Four
models are supported via a pluggable interface — **MediaPipe Hands**
(*Zhang 2020*), **DetNet baseline** (this repo's epoch-71 FP32 checkpoint),
**DetNet pruned + fine-tuned** (any of the L1 / Taylor variants from
[`analysis/pruning_decisions.md`](pruning_decisions.md)), and **DetNet
quantized** (M2 / M3 from [`quant/README.md`](../quant/README.md)).

All design decisions below are made so that the **only variable** across
model-backend comparisons is the choice of hand-pose model. The frame
capture pipeline, mirroring strategy, pinch state machine, physics, scoring
rules, video / CSV recording, and HUD are all fixed identically and shared
through one tracker-agnostic game loop.

---

## 1. Architecture & Code Layout

### D1. Additive-only — no edits to existing repo code

**Decision.** The entire BBT system is implemented as new files in a new
top-level `bbt/` folder and a single new file
[`analysis/bbt_decisions.md`](bbt_decisions.md). No file outside `bbt/` (with
the exception of this document) is modified, refactored, renamed, or deleted.
Existing functionality is consumed by **import**, not by edit — in particular
this repo's [`webcam_detnet.py`](../webcam_detnet.py) is imported for its
`load_model`, `load_quantized`, `get_hand_detection`, `preprocess`,
`_draw_skeleton_at`, and `HEATMAP_SIZE` symbols.

**Rationale.** The pruning + quantization pipelines and the trained
checkpoints they produce are scientific artefacts whose value depends on the
exact code that produced them being immutable. Adding the BBT as a
non-invasive layer keeps those artefacts trivially reproducible and audit-able:
nothing in the model training / pruning / quantization paths can be
accidentally perturbed by the interactive-tool work. The constraint also
forces the BBT to share its inference path with the existing
`webcam_detnet.py` demo, which is independently verified.

**Code.** All new files live under [`bbt/`](../bbt/); cross-references to
inherited code use absolute imports against the repo root (e.g.
`from webcam_detnet import load_model`).

### D2. Three-file layout: shared tracking + two runnable scripts

**Decision.** The `bbt/` folder contains exactly:

- [`bbt/hand_tracking.py`](../bbt/hand_tracking.py) — shared abstraction
  layer: `HandResult` dataclass, `HandTracker` base, `MediaPipeTracker`,
  `DetNetTracker(variant=...)`, `make_tracker(...)` factory,
  `pinch_distance` helper, `PinchState` state machine.
- [`bbt/cursor_control.py`](../bbt/cursor_control.py) — runnable script that
  drives the OS mouse via PyAutoGUI (pinch = click-and-hold).
- [`bbt/box_block_test.py`](../bbt/box_block_test.py) — the BBT game itself.

**Rationale.** Both runnable scripts need identical model loading + pinch
logic. Folding the tracking layer into a shared module guarantees that
`cursor_control.py` and `box_block_test.py` cannot diverge in how a model is
constructed or how pinch state is computed — a non-trivial source of
comparison invalidation if it were duplicated. Game internals
(Block class, physics, scene, HUD, recorder, CSV writer) stay inline in
`box_block_test.py` because they are not shared with `cursor_control.py` and
splitting them would add structure for its own sake.

### D3. OpenCV-only rendering (no pygame / no GL)

**Decision.** All compositing — webcam frame, partition, blocks, hand
skeleton, HUD, countdown, end-screen — is done with `cv2` primitives
(`cv2.rectangle`, `cv2.line`, `cv2.circle`, `cv2.putText`,
`cv2.addWeighted`) into a single numpy `uint8` array, which is then written
to both `cv2.imshow` and `cv2.VideoWriter`.

**Rationale.** Three reasons. (a) The existing capture loop in
`webcam_detnet.py` is already OpenCV, so the BBT reuses it without dragging
in a new graphics framework; (b) the loop is **frame-driven by the camera**,
not event-driven, so pygame's event/input system buys nothing — the only
"input" is the hand; (c) the composite we record **is** a numpy frame, so
`cv2.VideoWriter.write(frame)` is direct — with pygame we'd render to a
surface then convert back to numpy every frame for the writer, which is pure
overhead. Since the system measures model speed, minimising per-frame
rendering / conversion overhead keeps the reported FPS and `inference_ms`
numbers clean.

**Trade-off.** OpenCV's text / HUD drawing is cruder than pygame's. For a
title + score + timer + blocks this is acceptable; we use
`cv2.putText` with a shadow pass for legibility.

---

## 2. Model Abstraction Layer

### D4. `HandResult` + `HandTracker` interface, four backends behind it

**Decision.** All backends implement
`HandTracker.detect(frame_bgr) -> HandResult | None`, where `HandResult`
carries (i) `landmarks_px : (21, 2) int32` pixel coordinates in the input
frame, (ii) `landmarks_norm : (21, 2 or 3)` raw model-space coordinates
([row, col] heatmap for DetNet, [x, y, z] normalised for MediaPipe), and
(iii) `inference_ms : float`. A `make_tracker(model_name, ...)` factory
resolves the four CLI choices `mediapipe | detnet-baseline | detnet-pruned |
detnet-quantized` to a concrete tracker.

**Rationale.** The downstream game and cursor scripts are written *once*
against the abstract `HandTracker` interface. Adding a new backend (e.g. a
future quantization-aware DetNet) means writing one adapter class — the
rest of the system is untouched. The `inference_ms` field is part of the
contract so that benchmarking is uniform across backends.

**Code.** [`bbt/hand_tracking.py:38-58`](../bbt/hand_tracking.py).

### D5. DetNet variants reuse MediaPipe Hands for the bbox crop

**Decision.** Every `DetNetTracker.detect()` call internally runs MediaPipe
Hands first to produce a square hand bbox; that bbox is passed to
`preprocess()` to crop the frame to 128×128 before the DetNet forward pass.
This is the *exact* pipeline used by [`webcam_detnet.py:118-160 +
163-188`](../webcam_detnet.py); the BBT adapters import those helpers
verbatim.

**Rationale.** DetNet was trained on 128×128 hand-centred crops with the
keypoint loss applied only inside the crop. It is **not** translation- or
scale-invariant at the full-frame level; without an upstream hand
localiser, its predictions are noise. The deployed inference pipeline in
this repo already pairs MediaPipe (bbox) with DetNet (keypoints) for this
reason; replicating the choice in the BBT keeps the comparison consistent
with the demo and with the evaluation pipeline.

**Methodological caveat.** The "DetNet variants" evaluated in the BBT are
strictly *MediaPipe (bbox) → DetNet (keypoints)* pipelines, not pure
DetNet. The L1-vs-Taylor / baseline-vs-quantized comparisons remain fair
because all four DetNet variants share the same MediaPipe front-end — only
the DetNet stage differs — but the thesis methodology section should state
this dependency explicitly.

**Code.** [`bbt/hand_tracking.py:192-228`](../bbt/hand_tracking.py).

### D6. `inference_ms` measures only the DetNet forward pass

**Decision.** For DetNet variants the reported `inference_ms` brackets only
`module(input_tensor)`; the MediaPipe bbox call, cropping, and normalisation
are excluded. For the MediaPipe-only variant the field brackets the entire
`HandLandmarker.detect()` call.

**Rationale.** The metric of interest is the *DetNet variant under test*.
Bundling the MediaPipe overhead into every DetNet number would mask the
pruning / quantization speedups, since MediaPipe latency is invariant
across DetNet variants. Effective end-to-end frame-rate is still recorded
via the `fps` column in the CSV (see [D27](#d27-per-frame-csv-with-all-21-landmarks-and-game-state)).

**Code.** [`bbt/hand_tracking.py:212-216`](../bbt/hand_tracking.py).

### D7. Quantized variant is re-quantised in-process, not loaded from disk

**Decision.** `--model detnet-quantized` calls
`webcam_detnet.load_quantized(tag)` (where `tag ∈ {qmm, qmse}`), which
re-runs the deterministic quantization pipeline in
[`quant/qquant.py`](../quant/qquant.py) at startup (~3 min). The INT8
state-dict is not persisted as a standalone `.pth`.

**Rationale.** This is inherited from the existing demo and is forced by
PyTorch's eager-mode INT8 quantization: whole-model INT8 pickles do not
round-trip cleanly through `torch.load`, but the pipeline (calibrate →
convert → bias-correct) is deterministic given the fixed fork weights and
`quant/calib_tensors.pt`, so re-quantising on launch produces the same
model that `quant/method3.py` saved. CPU-only (FBGEMM x86 backend).

**Code.** [`bbt/hand_tracking.py:202-205`](../bbt/hand_tracking.py),
[`webcam_detnet.py:96-115`](../webcam_detnet.py).

---

## 3. Frame Pipeline

### D8. Detect on the *un-mirrored* frame; mirror landmarks for display

**Decision.** The main loop captures the raw camera frame, calls
`tracker.detect(raw)` on it, *then* flips both the frame
(`cv2.flip(raw, 1)`) and the landmarks (`landmarks_px[:, 0] = (W-1) -
landmarks_px[:, 0]`) for the display composite. The display, recording, and
all downstream pinch / physics logic see consistent mirrored-frame
coordinates; the model sees un-mirrored input.

**Rationale.** This repo's DetNet variants (baseline, every pruned variant,
both quantized variants) are not robust to horizontal flipping of the
input. Feeding DetNet a `cv2.flip(frame, 1)`-mirrored frame puts the thumb
on the wrong side of the bbox crop and the model **inverts its finger
labels**: joint 8 (which should be the index tip) gets predicted at the
pinky tip position; joint 12 (middle) at the ring tip; joint 16 (ring) at
the middle tip; joint 20 (pinky) at the index tip. Joints 0–4 (wrist + thumb
chain) stay correct because the thumb is detected positionally as the
short, off-axis chain. The MediaPipe `HandLandmarker` is handedness-aware
(*Zhang 2020*) and is robust to mirroring, but the DetNet stage in the
DetNet variants is not, so the failure manifests as soon as a DetNet backend
is selected. This was confirmed empirically by switching the same test
between mirrored and un-mirrored detection on the same hand: un-mirrored
predictions are anatomically correct on the user's natural right-hand
control hand.

**Implementation note.** The post-detection mirror of `landmarks_px`
mutates the array in place; nothing downstream of detection sees the
un-mirrored coordinates, which simplifies the cursor mapping and the
skeleton draw — both can treat the landmarks as being in display-frame
coordinates without an extra flip.

**Code.** [`bbt/cursor_control.py:153-164`](../bbt/cursor_control.py),
[`bbt/box_block_test.py:417-430`](../bbt/box_block_test.py). A persistent
project memory at
[`memory/project_detnet_mirror.md`](../memory/project_detnet_mirror.md)
records this quirk for future sessions.

### D9. Predicted hand position = index fingertip (joint 8)

**Decision.** The "control point" the BBT records as `hand_x, hand_y` in the
CSV, uses for cursor mapping, and uses to test block grab is the index
fingertip — landmark index 8 under both MediaPipe's and this repo's SNAP
joint orderings.

**Rationale.** Index-fingertip control is the natural human pointer
analogue (the same convention the existing `webcam_detnet.py` overlay
uses), and matches the camera-based BBT reference (*Ko 2024,
ScienceDirect S1434841123002364*). Joint 8 lives at the same array index
in both backends because MediaPipe Hands and the repo's SNAP joint scheme
share the same numbering (wrist=0, thumb 1-4, index 5-8, middle 9-12, ring
13-16, pinky 17-20) — verified at
[`config.py:52-74`](../config.py).

**Code.** [`bbt/cursor_control.py:174-177`](../bbt/cursor_control.py)
(cursor source), [`bbt/box_block_test.py:495-503`](../bbt/box_block_test.py)
(block hit-test + held-block follow), [`bbt/hand_tracking.py:64-68`](../bbt/hand_tracking.py)
(`pinch_distance` joint indices 4 and 8).

---

## 4. Pinch Gesture

### D10. Pinch signal = Euclidean distance(thumb_tip, index_tip), per active model

**Decision.** The pinch distance is `||landmarks_px[4] -
landmarks_px[8]||` in pixels, computed from the *active model's*
landmarks (DetNet output for DetNet modes; MediaPipe output for
`--model mediapipe`). The same numbers go to both the click state machine
and the CSV.

**Rationale.** Using the active model's landmarks is what makes the BBT a
real end-to-end evaluation of that model: pinch quality and pinch latency
are part of the user-perceived performance and are recorded in the data.
Routing the pinch through a separate always-on MediaPipe instance would
mask the DetNet variants' real interaction quality and defeat the purpose
of the harness.

**Code.** [`bbt/hand_tracking.py:60-68`](../bbt/hand_tracking.py).

### D11. Asymmetric pinch thresholds: engage at 30 px, release at 20 px

**Decision.** The pinch state machine engages when smoothed `d < 30 px` and
releases when smoothed `d > 20 px`. Both thresholds are CLI overrideable
(`--pinch-threshold`, `--pinch-release-threshold`).

**Rationale.** Single-threshold pinch detection is the simplest form
specified in the design plan (§1.2), but with smoothed distance values
sitting near the boundary during drag motion, a single threshold produces
chatter. Using *inverted hysteresis* (release threshold **lower** than
engage threshold) gives:

- a "loose pinch" engagement at d=29 px (low effort to start a click), and
- a hold band of d ≤ 20 px (must stay firm to stay clicked).

Once engaged at, say, d=28, the release timer immediately starts ticking
because 28 > 20; the user has the duration of `release_debounce` (see
[D12](#d12-asymmetric-pinch-debounce-cursor-control-vs-game)) to bring d
below 20 to fully secure the hold. This matches the user-stated preference
for click semantics that engage easily but only stay engaged when the
pinch is genuinely tight. The 30 / 20 pairing was chosen by direct user
calibration during interactive testing.

**Code.** [`bbt/hand_tracking.py:71-91`](../bbt/hand_tracking.py)
(`PinchState`), [`bbt/cursor_control.py:46-57`](../bbt/cursor_control.py)
(cursor defaults), [`bbt/box_block_test.py:35-39`](../bbt/box_block_test.py)
(game defaults).

### D12. Asymmetric pinch debounce: cursor control vs game

**Decision.** Engagement requires `engage_debounce` consecutive frames
below `engage_threshold`; release requires `release_debounce` consecutive
frames above `release_threshold`. Defaults differ by use case:

| Script | `engage_debounce` | `release_debounce` |
|---|---|---|
| `cursor_control.py` | 3 frames | 10 frames |
| `box_block_test.py` | 2 frames | 2 frames |

**Rationale.** Cursor control needs sticky click-and-hold for drawing /
selection use cases: a 10-frame (~333 ms at 30 FPS) release debounce
absorbs landmark-noise spikes during fast drag motion that would otherwise
fragment a single stroke. The BBT game does not need sticky semantics —
block-follow is robust to brief release events (the block falls under
gravity for a few frames and is re-grabbed if the pinch reasserts), and a
shorter release debounce keeps the game feeling responsive. Both
parameters are CLI-overrideable.

**Code.** [`bbt/hand_tracking.py:71-128`](../bbt/hand_tracking.py).

### D13. EMA smoothing on the pinch distance signal

**Decision.** Before the threshold check, the per-frame raw distance is
EMA-smoothed: `smoothed_d = α · smoothed_d_prev + (1-α) · raw_d`, with
α = 0.5.

**Rationale.** Frame-to-frame landmark noise on the thumb and index tips
typically gives the raw distance a ~5-10 px standard deviation even when
fingers are physically still. A bare-threshold check on the raw signal
crosses 30 px on noise alone during slow open-hand motion. EMA on the
distance pulls transient spikes back into the "still pinched" band before
the threshold sees them, which suppresses spurious release events without
needing larger debounce values.

**Code.** [`bbt/cursor_control.py:50-54`](../bbt/cursor_control.py)
(constant), [`bbt/cursor_control.py:195-201`](../bbt/cursor_control.py)
(application).

### D14. Hand-loss tolerance — short MediaPipe dropouts do not release the click

**Decision.** When `tracker.detect()` returns `None` (no hand found),
`cursor_control.py` increments a `missed_frames` counter but does **not**
release the click or move the cursor. Only after `missed_frames` reaches
`HAND_LOSS_TOLERANCE_FRAMES = 10` is the click force-released and the
position EMA reset. The BBT game uses an equivalent counter integrated with
the block-follow logic.

**Rationale.** MediaPipe Hands intermittently fails to detect during
high-acceleration drag motion (motion blur on the thumb tip is the most
common cause). Without tolerance, every dropped frame ends the click and
restarts a fresh one on the next detection — producing the "many short
strokes" failure mode confirmed empirically against MS Paint. 10 frames
≈ 333 ms at 30 FPS, comfortably longer than typical MediaPipe dropouts.

**Code.** [`bbt/cursor_control.py:42-46`](../bbt/cursor_control.py)
(constant), [`bbt/cursor_control.py:230-247`](../bbt/cursor_control.py)
(application).

---

## 5. Cursor Control (`cursor_control.py`)

### D15. Click-and-hold semantics via PyAutoGUI, not rapid clicks

**Decision.** Engagement fires one `pyautogui.mouseDown()`; release fires
one `pyautogui.mouseUp()`. Between engage and release, only `moveTo()`
calls are issued. This is *real* OS click-and-hold (drag).

**Rationale.** Click-and-hold supports the full set of common OS
interactions — drag-and-drop, window drag, slider drag, drawing strokes,
rubber-band selection — whereas a rapid-fire single-click scheme (one
complete `mouseDown` + `mouseUp` per frame) only produces sensible output
in apps that interpret each click as a discrete brush deposit. Plan §1.2
explicitly specified click-and-hold; preserving it keeps the cursor tool
useful as a general accessibility / interaction harness, not only a Paint
demo.

**Code.** [`bbt/cursor_control.py:182-193`](../bbt/cursor_control.py).

### D16. EMA smoothing on the screen-mapped cursor position

**Decision.** Before each `pyautogui.moveTo`, the index fingertip position
is EMA-smoothed: `smoothed_idx = α · smoothed_idx_prev + (1-α) ·
raw_idx`, with α = 0.6 (= ~2-3 frame settling time on a step input).

**Rationale.** A 640×480 frame mapped to a 1920×1080 screen amplifies every
pixel of landmark jitter by ~3×; uncorrected, the cursor wobbles
distractingly. α = 0.6 attenuates jitter without adding noticeable lag at
MediaPipe's ~30 FPS. The EMA state is reset when the hand-loss tolerance
window is exceeded so a re-acquisition in a new position does not pull the
cursor through a long slow drift.

**Code.** [`bbt/cursor_control.py:37-40`](../bbt/cursor_control.py)
(constant), [`bbt/cursor_control.py:181-187`](../bbt/cursor_control.py)
(application).

### D17. Inter-frame cursor interpolation (12 short hops per frame)

**Decision.** Each frame's `moveTo` is split into up to 12 intermediate
`moveTo` calls along a straight line from the previous frame's cursor
position to the current target. Step count is bounded by the screen-pixel
distance (one hop per ~25 px).

**Rationale.** `pyautogui.moveTo(target, duration=0)` issues a single
absolute-position OS event; the cursor teleports to the target and the
in-between path is never seen by the OS or by user-space applications.
Drawing apps with a click-and-hold drag then render a *straight line*
between consecutive frame positions, which at low FPS produces visibly
piecewise-linear strokes for any curved hand motion. Interpolating into
many short hops generates one `WM_MOUSEMOVE` per hop, which Paint and
similar tools render as a chain of short straight segments approximating
the actual hand path. Cost is ~1 ms / frame in the worst case
(12 × ~80 µs / `moveTo`).

**Code.** [`bbt/cursor_control.py:97-129`](../bbt/cursor_control.py).

---

## 6. BBT Game (`box_block_test.py`)

### D18. 60-second active play (configurable)

**Decision.** A 3-second countdown precedes 60 seconds of play, followed by
a 5-second end-screen showing the final score. `--duration` overrides the
60.

**Rationale.** 60 seconds matches the clinical BBT (*Mathiowetz 1985*),
preserving direct comparability with published normative scores. The
countdown and end-screen sit outside the timed window so subjects begin
and end at well-defined moments.

**Code.** [`bbt/box_block_test.py:30-32`](../bbt/box_block_test.py)
(constants), [`bbt/box_block_test.py:447-451 + 510-517`](../bbt/box_block_test.py)
(phase machine).

### D19. Full-screen, mirrored display

**Decision.** Default is full-screen (`cv2.WINDOW_FULLSCREEN`) with the
webcam frame horizontally flipped (`cv2.flip(raw, 1)`). `--windowed`
disables fullscreen; `--window-scale` (default 2.0) sets the windowed
display size.

**Rationale.** Mirroring is the standard webcam UX (selfie-style), making
hand-to-cursor mapping intuitive ("hand right → cursor right"). Full-screen
maximises the playing area, which keeps block sizes generous in absolute
pixels and matches the immersive feel of the reference camera-based BBT
(*Ko 2024*). The mirror is applied **after** model inference (see
[D8](#d8-detect-on-the-un-mirrored-frame-mirror-landmarks-for-display)) so
DetNet still sees its training distribution.

**Code.** [`bbt/box_block_test.py:425`](../bbt/box_block_test.py) (mirror),
[`bbt/box_block_test.py:386-389`](../bbt/box_block_test.py) (window mode).

### D20. Frame edges are hard walls; blocks cannot leave the frame

**Decision.** The top, bottom, left, and right edges of the mirrored
webcam frame are collision boundaries. A block's AABB is clamped to stay
inside `[0, W-1] × [0, H-1]` after every physics step. The bottom edge
acts as the floor (blocks settle and stack against it).

**Rationale.** Anchoring the simulated box to the visible frame keeps
blocks always on-screen and always interactable. Sub-windowed layouts
(a virtual box with internal walls smaller than the camera frame) would
either waste screen real-estate or risk clipping fast-moving blocks
off-screen.

**Code.** [`bbt/box_block_test.py:120-141`](../bbt/box_block_test.py)
(physics).

### D21. Layout is fractional — adapts to any camera resolution

**Decision.** All in-game pixel quantities (partition position, partition
width, block size, gravity, terminal velocity, pinch thresholds) are
expressed as **fractions of the camera frame width or height** and
resolved to pixels at startup once the camera resolution is known.

**Rationale.** Camera resolutions vary across machines (640×480, 1280×720,
etc.) and the BBT must look and play the same on each. Fractional layout
guarantees the partition sits at the centre regardless of width, blocks
fit the same number per side regardless of aspect, and the physics feels
the same regardless of resolution (gravity scales with height).

**Code.** [`bbt/box_block_test.py:40-58 + 156-184`](../bbt/box_block_test.py).

### D22. 8 blocks, hand-rolled AABB physics

**Decision.** 8 axis-aligned square blocks, palette-coloured, spawn in
left-compartment stacks. Per-frame integration: `vy += g`; integrate;
resolve collisions against floor, ceiling, side walls, partition, and
other non-held blocks. Two passes of pairwise AABB resolution stabilise
stacks.

**Rationale.** With only 8 blocks and pure axis-aligned squares the
collision search is trivially `O(n²)` (28 pair tests) — orders of
magnitude under the model inference cost — and deterministic per frame,
which keeps the recording reproducible across replays.
[`pymunk`](https://www.pymunk.org) was considered and rejected:
introducing a physics dependency for 28 box-box tests is unjustified, and
the deterministic per-frame stepping is harder to guarantee in a
general-purpose engine.

**Code.** [`bbt/box_block_test.py:82-149`](../bbt/box_block_test.py).

### D23. Held block follows the index fingertip; partition passes through

**Decision.** While a block is held (`block.held = True`), its centre is
slaved to the index-fingertip position each frame and it bypasses gravity
and collision resolution. The full-height partition acts as a wall for
non-held blocks; held blocks pass through it because they skip the partition
collision branch.

**Rationale.** This is the on-screen analogue of physically *lifting* a
block over the box partition: in the clinical BBT, you grasp and lift,
not push. Allowing the held block to pass through other blocks (rather
than displacing them) is a minor visual liberty taken for simplicity —
displacing other blocks while held could destabilise existing stacks
mid-game and discourage carry strategies.

**Code.** [`bbt/box_block_test.py:484-509`](../bbt/box_block_test.py).

### D24. Score = released past partition; re-arm on carry-back

**Decision.** When a held block is released and its centre is on the
**target** side of the partition (cx > partition_x), the block's
`scored` flag is set and the score increments by 1. The flag prevents
double-counting: a block already counted does not score again on subsequent
releases unless its `cx` returns to the source side, which clears the flag.

**Rationale.** The score must match the clinical BBT semantics: count of
blocks successfully transferred from source to target during the timed
window. The re-arm-on-carry-back rule means a subject who picks up a
delivered block, carries it back to source, and re-delivers it gets one
new score increment — exactly mirroring the clinical convention of
counting transfers, not absolute final positions.

**Code.** [`bbt/box_block_test.py:471-509`](../bbt/box_block_test.py).

---

## 7. Recording & Logging

### D25. Composite video recording — what the player saw

**Decision.** The MP4 written to `BBT recordings/` is the **composited
display frame** (mirrored webcam + partition + blocks + skeleton +
fingertip dot + HUD), one frame per main-loop iteration, nominal 30 FPS.
Filename format: `{timestamp}_{model}_score{N}.mp4`.

**Rationale.** Recording the composite (rather than the raw webcam)
captures the player's exact perception of the system: what they saw, what
the HUD reported as the model name + score + FPS, where the skeleton
landed on their hand, and where the held block was at each moment.
Reviewing a session for failure-mode analysis (e.g. "why did the score
plateau at 4?") is far easier with the overlays baked in. The composite
is already a numpy `uint8` array, so `cv2.VideoWriter.write(frame)` is
direct.

**Trade-off.** Writer FPS is hard-set to 30 (`VIDEO_FPS_NOMINAL`); the
true effective FPS is whatever the loop achieved, and is recorded in the
per-frame CSV (see [D27](#d27-per-frame-csv-with-all-21-landmarks-and-game-state))
as ground truth for timing. Playback at the nominal rate is therefore an
approximate speed indicator only — analytical pipelines should use the CSV
`t_seconds` column instead.

**Code.** [`bbt/box_block_test.py:283-322`](../bbt/box_block_test.py)
(Recorder).

### D26. Output folder name: `BBT recordings/` (with the space)

**Decision.** Per the design plan, the literal folder name is
`BBT recordings/` (capital B, space, lowercase r), created at the repo
root on first run.

**Rationale.** A non-technical user-facing folder name was specified in
the plan; the BBT system should not litter the repo with developer-style
snake-case folder names for output the user is expected to browse.

**Code.** [`bbt/box_block_test.py:75`](../bbt/box_block_test.py).

### D27. Per-frame CSV with all 21 landmarks and game state

**Decision.** One CSV row per main-loop iteration, columns:

| Column | Meaning |
|---|---|
| `frame` | frame index |
| `t_seconds` | seconds since play started (countdown excluded) |
| `model` | active backend name |
| `hand_detected` | 0/1 |
| `hand_x`, `hand_y` | index-fingertip pixel position in mirrored frame |
| `landmarks_px` | all 21 (x, y) as JSON-encoded list-of-lists |
| `pinch_distance` | smoothed thumb-index distance (px) |
| `grabbing` | 0/1 — pinch state machine output |
| `held_block_id` | block id (0-7) if carrying, else -1 |
| `score` | running score |
| `fps` | EMA-smoothed loop FPS |
| `inference_ms` | DetNet forward-pass time for that frame (or MediaPipe for `--model mediapipe`) |

Filename matches the video: `{timestamp}_{model}_score{N}.csv`.

**Rationale.** All 21 landmarks per frame is the headline data the BBT
records: it is what enables downstream comparison of *which model
predicted what* per frame, not just an aggregate FPS / score. JSON-encoded
into one column (rather than spread across 42 columns) keeps the schema
narrow and stable as the joint count is fixed at 21 by the SNAP scheme
(see [D9](#d9-predicted-hand-position--index-fingertip-joint-8)). Writes
are flushed every 30 frames so a mid-session crash leaves usable data.

**Code.** [`bbt/box_block_test.py:325-381`](../bbt/box_block_test.py)
(FrameLogger).

### D28. Session-mean FPS reported at exit

**Decision.** On loop exit (q / ESC / `--duration` expiry), the terminal
prints a Session Statistics block including `Mean FPS = total_frames /
session_seconds`. Session = full loop time, including 3-s countdown + 60-s
play + 5-s end-screen.

**Rationale.** Matches the FPS-reporting precedent in
[`webcam_detnet.py:436-446`](../webcam_detnet.py); a single
session-wide number is the cleanest apples-to-apples metric across model
runs, and the per-frame `fps` column in the CSV is the ground truth for
finer-grained analysis.

**Code.** [`bbt/box_block_test.py:586-597`](../bbt/box_block_test.py).

---

## 8. Summary Table

| # | Decision | Key code | Anchor |
|--:|----------|----------|--------|
| D1  | Additive-only — no edits to existing repo | All new files in [`bbt/`](../bbt/) | (Reproducibility) |
| D2  | Three-file layout (shared tracking + 2 scripts) | [`bbt/`](../bbt/) | (Architecture) |
| D3  | OpenCV-only rendering (no pygame) | [`bbt/box_block_test.py`](../bbt/box_block_test.py) | (Plan §7) |
| D4  | `HandResult` + `HandTracker` interface | [`bbt/hand_tracking.py:38-58`](../bbt/hand_tracking.py) | (Plan §3) |
| D5  | DetNet variants reuse MediaPipe for bbox | [`bbt/hand_tracking.py:192-228`](../bbt/hand_tracking.py) | webcam_detnet.py |
| D6  | `inference_ms` measures only DetNet forward pass | [`bbt/hand_tracking.py:212-216`](../bbt/hand_tracking.py) | (Methodology) |
| D7  | Quantized variant re-quantised in-process | [`bbt/hand_tracking.py:202-205`](../bbt/hand_tracking.py) | [quant/README.md](../quant/README.md) |
| D8  | Detect on un-mirrored frame; mirror landmarks | [`bbt/cursor_control.py:153-164`](../bbt/cursor_control.py), [`bbt/box_block_test.py:417-430`](../bbt/box_block_test.py) | (DetNet quirk) |
| D9  | Control point = index fingertip (joint 8) | [`bbt/hand_tracking.py:60-68`](../bbt/hand_tracking.py) | (Plan §1.3) |
| D10 | Pinch = Euclidean(thumb_tip, index_tip), active model | [`bbt/hand_tracking.py:60-68`](../bbt/hand_tracking.py) | (Plan §1.2) |
| D11 | Asymmetric thresholds: engage 30 / release 20 | [`bbt/hand_tracking.py:71-91`](../bbt/hand_tracking.py) | (User calibration) |
| D12 | Asymmetric debounce: cursor 3/10, game 2/2 | [`bbt/hand_tracking.py:71-128`](../bbt/hand_tracking.py) | (Use-case split) |
| D13 | EMA on pinch distance (α = 0.5) | [`bbt/cursor_control.py:50-54`](../bbt/cursor_control.py) | (Noise mitigation) |
| D14 | Hand-loss tolerance — 10 frames | [`bbt/cursor_control.py:42-46`](../bbt/cursor_control.py) | (MediaPipe dropouts) |
| D15 | PyAutoGUI click-and-hold (not rapid-click) | [`bbt/cursor_control.py:182-193`](../bbt/cursor_control.py) | (Plan §1.2) |
| D16 | Cursor position EMA (α = 0.6) | [`bbt/cursor_control.py:37-40`](../bbt/cursor_control.py) | (Jitter mitigation) |
| D17 | Inter-frame cursor interpolation (≤12 hops) | [`bbt/cursor_control.py:97-129`](../bbt/cursor_control.py) | (Drag-stroke quality) |
| D18 | 60-second active play (configurable) | [`bbt/box_block_test.py:30-32`](../bbt/box_block_test.py) | *Mathiowetz 1985* |
| D19 | Full-screen mirrored display | [`bbt/box_block_test.py:425`](../bbt/box_block_test.py) | (Plan §1.5) |
| D20 | Frame edges = hard walls | [`bbt/box_block_test.py:120-141`](../bbt/box_block_test.py) | (Plan §1.4) |
| D21 | Fractional layout — adapts to frame size | [`bbt/box_block_test.py:40-58 + 156-184`](../bbt/box_block_test.py) | (Resolution-agnostic) |
| D22 | 8 blocks, hand-rolled AABB physics | [`bbt/box_block_test.py:82-149`](../bbt/box_block_test.py) | (Determinism) |
| D23 | Held block follows fingertip; partition passes through | [`bbt/box_block_test.py:484-509`](../bbt/box_block_test.py) | (Lifting analogue) |
| D24 | Score = released past partition; re-arm on carry-back | [`bbt/box_block_test.py:471-509`](../bbt/box_block_test.py) | *Mathiowetz 1985* |
| D25 | Composite video recording (what player saw) | [`bbt/box_block_test.py:283-322`](../bbt/box_block_test.py) | (Plan §5) |
| D26 | Output folder: `BBT recordings/` | [`bbt/box_block_test.py:75`](../bbt/box_block_test.py) | (Plan §5) |
| D27 | Per-frame CSV with all 21 landmarks (JSON) | [`bbt/box_block_test.py:325-381`](../bbt/box_block_test.py) | (Plan §5) |
| D28 | Session-mean FPS reported at exit | [`bbt/box_block_test.py:586-597`](../bbt/box_block_test.py) | webcam_detnet.py:436 |

---

## References

Citations are listed alphabetically by first author. For each, the
bibtex-ready fields are given.

#### Ko 2024

S. Ko, K. R. Kim, J. Lee, H. Kim, J. Kim.
*A vision-based Box and Block Test using a single camera for upper-limb
function assessment.*
Computer Methods and Programs in Biomedicine 2024.
DOI / ScienceDirect article S1434841123002364.
**Used for:** [§ Context](#context),
[D9](#d9-predicted-hand-position--index-fingertip-joint-8) (camera-based
BBT precedent and reference visual style),
[D19](#d19-full-screen-mirrored-display) (immersive on-screen layout).

#### Mathiowetz 1985

V. Mathiowetz, G. Volland, N. Kashman, K. Weber.
*Adult norms for the Box and Block Test of manual dexterity.*
American Journal of Occupational Therapy, 39(6):386-391, 1985.
**Used for:** [§ Context](#context),
[D18](#d18-60-second-active-play-configurable) (60-second timing),
[D24](#d24-score--released-past-partition-re-arm-on-carry-back) (count-of-transfers
scoring), [D27](#d27-per-frame-csv-with-all-21-landmarks-and-game-state)
(per-trial recording is the natural extension of the per-trial count
recorded in clinical practice).

#### Zhang 2020

F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C. Chang,
M. Grundmann.
*MediaPipe Hands: On-device Real-time Hand Tracking.*
CVPR 2020 Workshop on Computer Vision for AR/VR. arXiv:2006.10214.
**Used for:** [D5](#d5-detnet-variants-reuse-mediapipe-hands-for-the-bbox-crop)
(hand-detector + bbox source for the DetNet pipeline),
[D8](#d8-detect-on-the-un-mirrored-frame-mirror-landmarks-for-display)
(handedness-aware landmark prediction — MediaPipe is robust to the mirror
artefact that breaks DetNet variants).

---

## Cross-references with other thesis-supporting documents

- Pruned-DetNet variants consumed via `--model detnet-pruned` are produced
  by the pipeline documented in
  [`analysis/pruning_decisions.md`](pruning_decisions.md). Specifically the
  fine-tuned checkpoints at
  `checkpoints_finetuned/{l1,taylor}_{r}pct/ft_*_rhdbest.pth` and their
  matching architecture shells at
  `pruned_architectures/detnet_{l1,taylor}_{r}pct_ep71_noft.pth`.
- Quantized variants consumed via `--model detnet-quantized` are produced
  by the pipeline documented in [`quant/README.md`](../quant/README.md).
  Specifically the M2 (MinMax) and M3 (Histogram-MSE) configs re-quantised
  in-process from the fixed fork weights + `quant/calib_tensors.pt`.
- Both DetNet inference paths (FP32 baseline / pruned / quantized) share
  the load + crop + preprocess code with
  [`webcam_detnet.py`](../webcam_detnet.py) (imported as
  `load_model`, `load_quantized`, `get_hand_detection`, `preprocess`,
  `_draw_skeleton_at`), keeping the BBT's measurements directly
  comparable to the standalone real-time demo.
- The DetNet input-mirror quirk discovered while building this harness
  ([D8](#d8-detect-on-the-un-mirrored-frame-mirror-landmarks-for-display))
  is documented as a project-scope memory at
  [`memory/project_detnet_mirror.md`](../memory/project_detnet_mirror.md)
  for future sessions.