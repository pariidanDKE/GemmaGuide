# Tasks: SpatialSense — Blind Navigation Assistant

**Input**: Design documents from `specs/001-spatialsense-blind-nav/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/tool-schemas.md ✓, quickstart.md ✓

**Tests**: Excluded for now. Can be added later when validating against live models.

**Organization**: Tasks grouped by user story. Each story is independently implementable and testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on in-progress tasks)
- **[Story]**: User story label (US1–US4) from spec.md

## Path Conventions

Single-project layout per plan.md: `pipeline/`, `server/`, `tests/` at repository root.

---

## Today Checklist (2026-05-04)

Purpose: prioritize user-story prerequisite validation gates today; treat all non-gate work as secondary unless a gate is blocked by missing observability or setup.

### Primary Completion Set (Must Be Done Today)

- [ ] G1 US1 gate: verify system answer is correct with respect to TIPS true value output on test examples
- [ ] G2 US1 gate: verify TIPS depth is accurate enough for navigation target range
- [ ] G3 US1 gate (multi-instance): validate behavior when two same-class objects exist (nearest instance + secondary instance with distinct bearing)
- [ ] G4 US2 gate: run 2-3 out-of-vocabulary tests and confirm expected tool flow: `search_seg_classes` miss -> `call_encoder_zero_shot` -> box/measure flow
- [ ] G5 US3 gate: confirm scene description is rich and explicitly calls out near-forward obstacle risk (around 0 degrees and close distance)

### Secondary Set (Only If Time Remains)

- [ ] S1 T026 quickstart validation end-to-end on current setup (`bash scripts/start_gemma4.sh`, `python app.py`), and update `quickstart.md` Known Issues for any breakage
- [ ] S2 T031 lightweight observability: metadata-only logs for timestamp, latency, model/tool status, confidence, and error class
- [ ] S3 T022 latency benchmark over 5 queries and document per-step timings against SC-001 in `quickstart.md`

### Blocker Experiment Set (Run if gates are ambiguous/blocked)

- [ ] E1 TIPS depth backbone comparison (b14 vs l14): run the same G1/G2/G3 samples with `google/tipsv2-b14-dpt` and `google/tipsv2-l14-dpt`; log distance error, confidence, and multi-instance behavior deltas
- [ ] E2 TIPS short-side sweep: run selected gate samples at `{896, 1024, 1152}` short-side; log VRAM headroom, latency, and measurable G1/G2 impact
- [ ] E3 Gemma precision sweep: run gate prompts under BF16 baseline vs FP8 (`--quantization fp8 --kv-cache-dtype fp8`) and log quality/latency trade-offs

### Recommended Execution Model (Gate-First)

- [ ] A1 Lane 1: US1 gate sweep (G1 + G2 + G3)
- [ ] A2 Lane 2: US2 out-of-vocabulary flow checks (G4)
- [ ] A3 Lane 3: US3 safety-richness scene checks (G5)
- [ ] A4 Lane 4: if a gate is blocked, run S2 observability + E1/E2/E3, then rerun blocked gate checks
- [ ] A5 Lane 5: if all gates pass, run S1/S3 and update `quickstart.md`

### Defer-If-Needed Items (not mandatory today)

- [ ] D1 T019 colormap comparison (low priority)
- [ ] D2 T020 segmentation-threshold experiment unless mask noise is blocking quality
- [ ] D3 T025 stale reference cleanup if hard-priority set slips
- [ ] D4 T027 demo video recording + technical write-up (explicitly not for today)
- [ ] D5 US4 demo-polish checks (mic/image/TTS UX aesthetics pass) if schedule allows later

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and directory structure

- [X] T001 Create directory structure: `pipeline/`, `server/`, `tests/unit/`, `tests/integration/`
- [X] T002 [P] Create Python package init files: `pipeline/__init__.py`, `server/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- [X] T003 [P] Create `requirements.txt` with all dependencies: `vllm[audio]`, `transformers`, `torch`, `torchvision`, `gradio`, `gtts`, `numpy`, `pillow`, `matplotlib`, `openai`, `pytest`

**Checkpoint**: Structure ready — foundational implementation can begin

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before any user story work can begin

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 [P] Implement `pipeline/intrinsics.py` — `CameraIntrinsics` dataclass with fields `fx, fy, cx, cy: float` and `source: str` (`"exif"` or `"assumed"`); `extract_intrinsics(image: PIL.Image) -> CameraIntrinsics`: read `FocalLength` and `FocalLengthIn35mmFilm` tags from image EXIF via `PIL.Image._getexif()`, compute `crop_factor = focal_35mm / focal_mm`, `sensor_width = 36 / crop_factor`, `fx = (focal_mm / sensor_width) * image.width`, `fy = (focal_mm / sensor_width) * image.height`, `cx = image.width / 2`, `cy = image.height / 2`; if EXIF absent or image is PNG, fall back to `fx = image.width / (2 * tan(radians(32.5)))`, `cx = image.width / 2`, `cy = image.height / 2`, `source = "assumed"`; validate using `data/media/images/original_highres.jpeg`
- [X] T005 [P] Implement `pipeline/session.py` — `Session` dataclass with fields: `session_id: str` (UUID), `image: PIL.Image`, `question: str | bytes`, `intrinsics: CameraIntrinsics`, `depth_tensor: torch.Tensor | None`, `seg_mask: torch.Tensor | None`, `depth_colormap: PIL.Image | None`, `spatial_report: str | None`; `depth_tensor` and `seg_mask` initialized to `None`; `intrinsics` populated at creation via `extract_intrinsics(image)`; include `release()` method that sets `depth_tensor` and `seg_mask` to `None`
- [X] T006 [P] Implement `pipeline/tips_runner.py` — define `ADE20K_CLASSES: list[str]` (150 entries); `load_model()` loads `google/tipsv2-b14-dpt` from HuggingFace; `run_dpt_inference(image: PIL.Image) -> tuple[Tensor, Tensor, PIL.Image]` returns `(depth_tensor (1,1,H,W) float32 meters, seg_mask (H,W) int ADE20K class index, depth_colormap PIL.Image)` where `depth_colormap` is a demo display artifact only; `get_detected_classes(seg_mask: Tensor) -> list[str]` returns unique ADE20K class names present in mask; `TIPSV2_SHORT_SIDE = 896` constant. Reference model repos for config/modeling files: `https://huggingface.co/google/tipsv2-b14/tree/main` and `https://huggingface.co/google/tipsv2-b14-dpt`
- [X] T007 [P] Implement `server/schemas.py` — `TOOL_SCHEMAS: list[dict]` containing all 4 JSON tool schemas matching `contracts/tool-schemas.md` exactly: `search_seg_classes`, `call_dpt_head`, `call_encoder_zero_shot`, `measure_object`

**Checkpoint**: Foundation ready — user story implementation can begin

---

## Phase 3: User Story 1 — Specific Object Distance Query (Priority: P1) 🎯 MVP

**Goal**: User speaks a targeted question ("how far is the chair?") → system returns metric distance and bearing in degrees

**Independent Test**: Upload a photo with a clearly visible object, send a typed question, confirm the response contains a distance in meters and a bearing in degrees

- [X] T008 [US1] Implement `search_seg_classes(query: str) -> dict` in `pipeline/tools.py` — case-insensitive substring match over `ADE20K_CLASSES` imported from `pipeline/tips_runner.py`; returns `{"matches": list[str]}` or `{"matches": []}` if no match
- [X] T009 [US1] Implement `call_dpt_head(image_ref: str, session: Session) -> dict` in `pipeline/tools.py` — calls `tips_runner.run_dpt_inference(session.image)`; stores `depth_tensor`, `seg_mask`, and `depth_colormap` (demo display artifact, never passed to Gemma) in `session`; returns `{"status": "ready", "detected_classes": list[str]}` (depends on T008)
- [X] T010 [US1] Implement `measure_object(class_name: str, box_2d: list[int], session: Session) -> dict` in `pipeline/tools.py` — validates `session.depth_tensor` and `session.seg_mask` not `None`; finds `class_idx` in `ADE20K_CLASSES`; computes `mask = (seg_mask == class_idx) & pixels_inside_box_2d` where `box_2d = [ymin, xmin, ymax, xmax]`; computes `distance_m` as median depth over mask pixels; computes pixel centroid `(px, py)` of mask; computes `bearing_deg = degrees(atan2((px - session.intrinsics.cx) / session.intrinsics.fx, 1))` (negative = left, positive = right); returns `{"distance_m": float, "bearing_deg": float, "confidence": "high" | "medium" | "low"}`; confidence `"low"` when pixel count < 50, or `distance_m` is outside 0.5–8.0 m, or sampled depth is at/near model bounds (1e-3 or 10.0) (depends on T009)
- [X] T011 [US1] Implement `server/agent.py` — `build_messages(image: PIL.Image, question: str | bytes) -> list[dict]` constructs OpenAI-format messages with image as `image_url` content block and question as text or audio; `run_agent_loop(session: Session) -> str` runs OpenAI client against `http://localhost:8000/v1` with `TOOL_SCHEMAS`, dispatches tool calls to `pipeline/tools.py`, returns final Gemma text response; system prompt instructs Gemma it is a navigation assistant for a blind user, must call `call_dpt_head` before any spatial measurement, and must report `bearing_deg` as natural language degrees ("X degrees to your left/right", "directly ahead" within ±5°). When parsing model-produced box outputs for tool calls, follow the object-detection JSON-extraction approach shown in `notebooks/Gemma4_(E2B)-Multimodal.ipynb` (adapted to vLLM responses) (depends on T007, T008, T009, T010)

**Checkpoint**: US1 fully functional — core use case demonstrable end-to-end:

- [ ] System gives correct answer w.r.t the true value report by TIPS. 
- [ ] The TIPS model gives an accurate true distance. 
- [ ] The model can discern between two objects of same type (e.g. there is a chair at 2m, but also one at 3m at 30 degrees from your POV)
**Prerequisites** :  I think both 1 and 2 and achieved, very likelu to already befully functional. The third point would require a bit more system engineering. If Gemma is super stupid and cannot do that well, what we can do is perhaps use TIPS to create segments which are not continous, consider discontinuous chair segments as different charis then report back in call_dpt_head the # of chairs as well.
---

## Phase 4: User Story 2 — Zero-Shot Out-of-Vocabulary Objects (Priority: P2)

**Goal**: User asks about a non-standard object ("is there a shopping cart?") → system attempts zero-shot recognition and reports presence, distance, and bearing

**Independent Test**: Provide a photo with an out-of-vocabulary object, ask for it by name, confirm the response either reports distance + bearing or clearly states it could not be confirmed

- [x] T012 [US2] Extend `pipeline/tips_runner.py` — add `run_zero_shot_inference(image: PIL.Image, class_list: list[str]) -> tuple[Tensor, dict[str, Tensor], list[str]]` using TIPSv2 base encoder and text encoder: compute cosine similarity map between image patch embeddings and each class text embedding; return `(depth_tensor, {class_name: cosine_sim_map}, detected_classes)` (depends on T006)
- [X] T013 [US2] Implement `call_encoder_zero_shot(image_ref: str, class_list: list[str], session: Session) -> dict` in `pipeline/tools.py` — calls `tips_runner.run_zero_shot_inference`; stores `depth_tensor` and cosine similarity maps in session; returns `{"status": "ready", "detected_classes": list[str]}` (depends on T012)
- [X] T014 [US2] Extend `measure_object` in `pipeline/tools.py` to handle zero-shot cosine similarity mask — detect zero-shot session state; use cosine similarity map for bounding box intersection; bearing computation uses same `session.intrinsics` path; set `confidence: "medium"` for all zero-shot results (depends on T013)

**Checkpoint**: US1 and US2 independently functional
- [ ] US2: Test with 2-3 out of vocab objects, make sure full flow runs:
	1. Gemma calls search_seg, does not find the classes
	2. Gemma call zero_shot tool with appropraite class list
	3. Everything else goes as before (bonuding box, interesect etc)
**Prerequisites** : This should already be fully achieved, should wait for testing to confirm
---

## Phase 5: User Story 3 — General Scene Description (Priority: P3)

**Goal**: User asks "what's around me?" → system identifies up to 5 prominent objects with distances and bearings, ordered closest to farthest

**Independent Test**: Upload a diverse scene photo, ask "what do you see?", confirm response names multiple objects with distances and bearings in closest-to-farthest order

- [X] T015 [US3] Refine system prompt in `server/agent.py` for scene description — prompt variant instructs Gemma to call `call_dpt_head`, skip non-navigation classes (wall, floor, ceiling, sky), call `measure_object` for each remaining detected class, report at most 5 objects ordered closest-to-farthest, report `bearing_deg` per object as natural language degrees, and cap Gemma response generation to 256 output tokens (depends on T011)

**Checkpoint**: US1, US2, and US3 independently functional

- [ ] The model gives a rich description of surrounding, it also does not fail to mention if there is an object directly in the way of the person (at ~0 degrees) and relatively close.
**Prequisites:** Thus far I got it to describe 3-4 objects in the scene and their distance, or just a general description. Perhaps needs some more work on the prompt. Should experiment with how complex the System Prompt can be made (likely the better option is to Finetune on data like this).

---

## Phase 6: User Story 4 — Hackathon Demo Interface (Priority: P4)

**Goal**: Non-technical user uploads a photo, speaks a question, receives a spoken answer within 30 seconds via web browser

**Independent Test**: Give the interface to someone unfamiliar with it; they complete a full interaction (upload → speak → hear answer) within 30 seconds without instructions

- [X] T016 [P] [US4] Implement `pipeline/tts.py` — `synthesize(text: str) -> bytes`: convert response text to audio bytes using `gTTS`; document `AUDIO_FALLBACK=whisper` env var path in module docstring referencing `quickstart.md`
- [X] T017 [US4] Implement `app.py` — Gradio demo with: `gr.Image(type="pil")` upload, `gr.Audio(sources=["microphone"], type="numpy")` mic input, `gr.Textbox()` optional text question, `gr.Textbox()` response text output, `gr.Audio(type="numpy")` spoken audio output, `gr.Image(type="pil")` depth map display (session.depth_colormap demo artifact); submit validates image present and at least one question provided; all exceptions caught and returned as natural language fallback; calls `server/agent.py` and `pipeline/tts.py` (depends on T011, T016)
- [X] T018 [US4] Implement edge case handlers in `server/agent.py` — `"not visible in current view"` when queried object absent from `detected_classes`; cautious language when `confidence: "low"`; for missing/invalid/inaudible audio return spoken + text retry prompt and skip spatial inference for that turn; low-light/blurry image fallback; note when `intrinsics.source == "assumed"` and bearing is approximate (depends on T017)

**Checkpoint**: All four user stories functional — demo ready for validation

US4: 
- [ ] Launch the app, test the mic, image input and tts of course
- [ ] Be satisfied with the aesthetics of the implementation

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validation, benchmarking, documentation corrections, and hackathon submission

- [ ] T019 [P] Empirically compare depth-map colormaps in `pipeline/tips_runner.py` — test plasma, viridis, grayscale on 3 diverse images for demo display quality; set confirmed winner as `DEFAULT_COLORMAP` constant (resolves research.md Decision 3 open item) - **Low Prio**
- [ ] T020 [P] Validate segmentation threshold in `pipeline/tips_runner.py` — compare argmax default vs. soft threshold at P=0.5 softmax confidence on 5 test images; set approach as `SEGMENTATION_THRESHOLD` constant (resolves research.md Decision 4 open item) - **Not sure what this means, is this at what threshold a pixel is considered to be 100% part of a class fpr the dpt head?**

- [ ] T021 Run Phase 1.5 TIPSv2 output contract validation — 5 diverse images through full pipeline at 896px short side; confirm VRAM ≤ 16GB via `nvidia-smi`; if OOM, reduce to 672px and update `TIPSV2_SHORT_SIDE` in `pipeline/tips_runner.py`; document in `quickstart.md` Known Issues - Currently we are mostly working with an RTX 3090, perhaps it would be cool to run the demo on the T4, but we can view that as stretch.

- [ ] T022 Run Phase 1.7 latency benchmark — measure per-step times (TIPSv2 inference, Gemma agent loop, TTS) across 5 queries; document breakdown against SC-001 (≤ 30s) in `quickstart.md` - **Medium Prio**
- [ ] T023 Run SC-002 distance accuracy validation — 5 test scenes with measured ground-truth distances in 0.5–8m range; assert ±20% accuracy for all 5; document in `quickstart.md` - **Medium Prio, altough does that mean I should find GT of my images.**
- [ ] T024 Run Phase 3.2 regression against all 12 success criteria SC-001 through SC-012; document pass/fail for each in `quickstart.md` - **What does it mean phase 3.2 regression?**
- [ ] T025 [P] Correct stale tool references in `specs/001-spatialsense-blind-nav/research.md` — Decision 6: replace old tools with current (`search_seg_classes`, `call_dpt_head`, `call_encoder_zero_shot`, `measure_object`); Decision 7: replace "depth-colored overlay image" with "original RGB image" and note bearing is now computed, not qualitative; Decision 8: replace `sample_depth` with `measure_object` - **Low Prio**
- [ ] T026 [P] Validate `quickstart.md` end-to-end on clean environment — follow all steps, confirm `bash scripts/start_gemma4.sh` starts the server and `python app.py` launches; update Known Issues if anything fails - **Hard Prio**
- [ ] T027 Demo video recording and technical write-up for hackathon submission (plan.md Phase 3.0 and 3.1) - **Hard Prio**
- [ ] T031 [P] Implement lightweight debug observability in `server/agent.py` (or logging module) — emit metadata-only logs (`timestamp`, latency, model/tool status, confidence, error class) when enabled, and ensure no raw image/audio/question-text persistence - **Medium Prio, but we should not do this as T31, we should do it much earlier so we have observability over our experiments**
- [X] T032 [P] Add bounding-box debug utilities in `pipeline/` or `tests/` inspired by `notebooks/Gemma4_(E2B)-Multimodal.ipynb` object-detection section — include robust JSON extraction for model box outputs and optional Pascal VOC box drawing on sample images for local validation (debug only; not required in end-user UI)

---

## Phase 8: Phone Access Stretch Goals (Post-v1, Optional)

**Purpose**: Enable mobile-browser camera capture and phone access to the local demo server

- [ ] T028 [P] [Stretch] Add mobile browser image capture option in `app.py` — support camera capture via web-compatible input path (mobile browser file capture / camera source) while preserving desktop upload flow
- [ ] T029 [Stretch] Add LAN-access run mode documentation in `quickstart.md` — bind server to `0.0.0.0`, document how to open from phone using host LAN IP and port, and include microphone/camera permission notes
- [ ] T030 [Stretch] Validate phone-to-local-server flow on at least one Android or iOS device — capture image on phone, ask spoken question, receive spoken response; document known browser limitations in `quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 — this is the MVP
- **Phase 4 (US2)**: Depends on Phase 3 (agent.py must exist for tool dispatch)
- **Phase 5 (US3)**: Depends on Phase 3 (refines agent.py system prompt)
- **Phase 6 (US4)**: Depends on Phase 3; T016 (tts.py) is parallelizable with Phase 3
- **Phase 7 (Polish)**: Depends on all desired user stories being complete
- **Phase 8 (Phone Stretch Goals)**: Optional; depends on Phase 6 and should run after demo-critical Phase 7 items

### Parallel Opportunities

```
# Phase 2 — all four foundational files independent:
T004: pipeline/intrinsics.py
T005: pipeline/session.py    ← depends on T004 (imports CameraIntrinsics), run after T004
T006: pipeline/tips_runner.py
T007: server/schemas.py

# Phase 6 — tts.py independent of agent work:
T016: pipeline/tts.py  ← can start as soon as Phase 2 is done
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T007) — CRITICAL, blocks everything
3. Complete Phase 3: User Story 1 (T008–T011)
4. **STOP and VALIDATE**: test US1 end-to-end with typed questions and real images
5. Demo: metric distance + bearing in degrees for named objects

### Incremental Delivery

1. Phase 1 + Phase 2 → Foundation ready
2. Phase 3 (US1) → MVP → validate SC-001, SC-002, SC-003
3. Phase 4 (US2) → Zero-shot objects → validate SC-006
4. Phase 5 (US3) → General scene description
5. Phase 6 (US4) → Full demo UI with audio I/O → validate SC-004, SC-005, SC-007, SC-008, SC-011
6. Phase 7 → Polish, benchmarks, lightweight debug observability, submission
7. Phase 8 (optional) → Mobile browser camera capture + phone LAN access

### Hackathon Deadline

Submission deadline: 2026-05-18. US1 + demo UI is the minimum submittable demo. US2 (zero-shot) is the differentiating capability for the Digital Equity track — prioritize after US1.

---

## Notes

- `[P]` tasks target different files with no in-flight dependencies
- T005 (session.py) imports `CameraIntrinsics` from T004 (intrinsics.py) — complete T004 first
- `depth_colormap` in tasks is a demo display artifact shown in Gradio UI and never sent to Gemma
- `bearing_deg` in `measure_object`: depth cancels in `atan2((px-cx)/fx, 1)` — bearing is depth-independent and reliable even when distance confidence is low
- `measure_object` confidence: `"low"` when intersecting pixels < 50, distance falls outside 0.5–8.0 m, or sampled depth is at/near model bounds (1e-3 or 10.0); `"medium"` for zero-shot path; `"high"` for clean argmax segmentation in reliable range
- When `intrinsics.source == "assumed"`, bearing accuracy is ±5° typical; Gemma should express this as approximate in its response
- The object-detection section in `notebooks/Gemma4_(E2B)-Multimodal.ipynb` is a reference for parsing and visualizing bounding boxes; keep inference-path integration aligned with vLLM/OpenAI-compatible message formats
