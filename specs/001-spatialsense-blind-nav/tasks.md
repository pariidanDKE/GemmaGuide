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

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and directory structure

- [ ] T001 Create directory structure: `pipeline/`, `server/`, `tests/unit/`, `tests/integration/`
- [ ] T002 [P] Create Python package init files: `pipeline/__init__.py`, `server/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- [ ] T003 [P] Create `requirements.txt` with all dependencies: `vllm[audio]`, `transformers`, `torch`, `torchvision`, `gradio`, `gtts`, `numpy`, `pillow`, `matplotlib`, `openai`, `pytest`

**Checkpoint**: Structure ready — foundational implementation can begin

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before any user story work can begin

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 [P] Implement `pipeline/intrinsics.py` — `CameraIntrinsics` dataclass with fields `fx, fy, cx, cy: float` and `source: str` (`"exif"` or `"assumed"`); `extract_intrinsics(image: PIL.Image) -> CameraIntrinsics`: read `FocalLength` and `FocalLengthIn35mmFilm` tags from image EXIF via `PIL.Image._getexif()`, compute `crop_factor = focal_35mm / focal_mm`, `sensor_width = 36 / crop_factor`, `fx = (focal_mm / sensor_width) * image.width`, `fy = (focal_mm / sensor_width) * image.height`, `cx = image.width / 2`, `cy = image.height / 2`; if EXIF absent or image is PNG, fall back to `fx = image.width / (2 * tan(radians(32.5)))`, `cx = image.width / 2`, `cy = image.height / 2`, `source = "assumed"`
- [ ] T005 [P] Implement `pipeline/session.py` — `Session` dataclass with fields: `session_id: str` (UUID), `image: PIL.Image`, `question: str | bytes`, `intrinsics: CameraIntrinsics`, `depth_tensor: torch.Tensor | None`, `seg_mask: torch.Tensor | None`, `depth_colormap: PIL.Image | None`, `overlay: PIL.Image | None`, `spatial_report: str | None`; `depth_tensor` and `seg_mask` initialized to `None`; `intrinsics` populated at creation via `extract_intrinsics(image)`; include `release()` method that sets `depth_tensor` and `seg_mask` to `None`
- [ ] T006 [P] Implement `pipeline/tips_runner.py` — define `ADE20K_CLASSES: list[str]` (150 entries); `load_model()` loads `google/tipsv2-b14-dpt` from HuggingFace; `run_dpt_inference(image: PIL.Image) -> tuple[Tensor, Tensor, PIL.Image, PIL.Image]` returns `(depth_tensor (1,1,H,W) float32 meters, seg_mask (H,W) int ADE20K class index, depth_colormap PIL.Image, overlay PIL.Image)` where `depth_colormap` and `overlay` are demo display artifacts only; `get_detected_classes(seg_mask: Tensor) -> list[str]` returns unique ADE20K class names present in mask; `TIPSV2_SHORT_SIDE = 896` constant
- [ ] T007 [P] Implement `server/schemas.py` — `TOOL_SCHEMAS: list[dict]` containing all 4 JSON tool schemas matching `contracts/tool-schemas.md` exactly: `search_seg_classes`, `call_dpt_head`, `call_encoder_zero_shot`, `measure_object`

**Checkpoint**: Foundation ready — user story implementation can begin

---

## Phase 3: User Story 1 — Specific Object Distance Query (Priority: P1) 🎯 MVP

**Goal**: User speaks a targeted question ("how far is the chair?") → system returns metric distance and bearing in degrees

**Independent Test**: Upload a photo with a clearly visible object, send a typed question, confirm the response contains a distance in meters and a bearing in degrees

- [ ] T008 [US1] Implement `search_seg_classes(query: str) -> dict` in `pipeline/tools.py` — case-insensitive substring match over `ADE20K_CLASSES` imported from `pipeline/tips_runner.py`; returns `{"matches": list[str]}` or `{"matches": []}` if no match
- [ ] T009 [US1] Implement `call_dpt_head(image_ref: str, session: Session) -> dict` in `pipeline/tools.py` — calls `tips_runner.run_dpt_inference(session.image)`; stores `depth_tensor`, `seg_mask`, `depth_colormap`, and `overlay` (demo display artifact, never passed to Gemma) in `session`; returns `{"status": "ready", "detected_classes": list[str]}` (depends on T008)
- [ ] T010 [US1] Implement `measure_object(class_name: str, box_2d: list[int], session: Session) -> dict` in `pipeline/tools.py` — validates `session.depth_tensor` and `session.seg_mask` not `None`; finds `class_idx` in `ADE20K_CLASSES`; computes `mask = (seg_mask == class_idx) & pixels_inside_box_2d` where `box_2d = [ymin, xmin, ymax, xmax]`; computes `distance_m` as median depth over mask pixels; computes pixel centroid `(px, py)` of mask; computes `bearing_deg = degrees(atan2((px - session.intrinsics.cx) / session.intrinsics.fx, 1))` (negative = left, positive = right); returns `{"distance_m": float, "bearing_deg": float, "confidence": "high" | "medium" | "low"}`; confidence `"low"` when pixel count < 50, or `distance_m` is outside 0.5–8.0 m, or sampled depth is at/near model bounds (1e-3 or 10.0) (depends on T009)
- [ ] T011 [US1] Implement `server/agent.py` — `build_messages(image: PIL.Image, question: str | bytes) -> list[dict]` constructs OpenAI-format messages with image as `image_url` content block and question as text or audio; `run_agent_loop(session: Session) -> str` runs OpenAI client against `http://localhost:8000/v1` with `TOOL_SCHEMAS`, dispatches tool calls to `pipeline/tools.py`, returns final Gemma text response; system prompt instructs Gemma it is a navigation assistant for a blind user, must call `call_dpt_head` before any spatial measurement, and must report `bearing_deg` as natural language degrees ("X degrees to your left/right", "directly ahead" within ±5°) (depends on T007, T008, T009, T010)

**Checkpoint**: US1 fully functional — core use case demonstrable end-to-end

---

## Phase 4: User Story 2 — Zero-Shot Out-of-Vocabulary Objects (Priority: P2)

**Goal**: User asks about a non-standard object ("is there a shopping cart?") → system attempts zero-shot recognition and reports presence, distance, and bearing

**Independent Test**: Provide a photo with an out-of-vocabulary object, ask for it by name, confirm the response either reports distance + bearing or clearly states it could not be confirmed

- [ ] T012 [US2] Extend `pipeline/tips_runner.py` — add `run_zero_shot_inference(image: PIL.Image, class_list: list[str]) -> tuple[Tensor, dict[str, Tensor], list[str]]` using TIPSv2 base encoder and text encoder: compute cosine similarity map between image patch embeddings and each class text embedding; return `(depth_tensor, {class_name: cosine_sim_map}, detected_classes)` (depends on T006)
- [ ] T013 [US2] Implement `call_encoder_zero_shot(image_ref: str, class_list: list[str], session: Session) -> dict` in `pipeline/tools.py` — calls `tips_runner.run_zero_shot_inference`; stores `depth_tensor` and cosine similarity maps in session; returns `{"status": "ready", "detected_classes": list[str]}` (depends on T012)
- [ ] T014 [US2] Extend `measure_object` in `pipeline/tools.py` to handle zero-shot cosine similarity mask — detect zero-shot session state; use cosine similarity map for bounding box intersection; bearing computation uses same `session.intrinsics` path; set `confidence: "medium"` for all zero-shot results (depends on T013)

**Checkpoint**: US1 and US2 independently functional

---

## Phase 5: User Story 3 — General Scene Description (Priority: P3)

**Goal**: User asks "what's around me?" → system identifies up to 5 prominent objects with distances and bearings, ordered closest to farthest

**Independent Test**: Upload a diverse scene photo, ask "what do you see?", confirm response names multiple objects with distances and bearings in closest-to-farthest order

- [ ] T015 [US3] Refine system prompt in `server/agent.py` for scene description — prompt variant instructs Gemma to call `call_dpt_head`, skip non-navigation classes (wall, floor, ceiling, sky), call `measure_object` for each remaining detected class, report at most 5 objects ordered closest-to-farthest, report `bearing_deg` per object as natural language degrees, and cap Gemma response generation to 256 output tokens (depends on T011)

**Checkpoint**: US1, US2, and US3 independently functional

---

## Phase 6: User Story 4 — Hackathon Demo Interface (Priority: P4)

**Goal**: Non-technical user uploads a photo, speaks a question, receives a spoken answer within 30 seconds via web browser

**Independent Test**: Give the interface to someone unfamiliar with it; they complete a full interaction (upload → speak → hear answer) within 30 seconds without instructions

- [ ] T016 [P] [US4] Implement `pipeline/tts.py` — `synthesize(text: str) -> bytes`: convert response text to audio bytes using `gTTS`; document `AUDIO_FALLBACK=whisper` env var path in module docstring referencing `quickstart.md`
- [ ] T017 [US4] Implement `app.py` — Gradio demo with: `gr.Image(type="pil")` upload, `gr.Audio(sources=["microphone"], type="numpy")` mic input, `gr.Textbox()` optional text question, `gr.Textbox()` response text output, `gr.Audio(type="numpy")` spoken audio output, `gr.Image(type="pil")` overlay display (session.overlay demo artifact); submit validates image present and at least one question provided; all exceptions caught and returned as natural language fallback; calls `server/agent.py` and `pipeline/tts.py` (depends on T011, T016)
- [ ] T018 [US4] Implement edge case handlers in `server/agent.py` — `"not visible in current view"` when queried object absent from `detected_classes`; cautious language when `confidence: "low"`; for missing/invalid/inaudible audio return spoken + text retry prompt and skip spatial inference for that turn; low-light/blurry image fallback; note when `intrinsics.source == "assumed"` and bearing is approximate (depends on T017)

**Checkpoint**: All four user stories functional — demo ready for validation

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validation, benchmarking, documentation corrections, and hackathon submission

- [ ] T019 [P] Empirically compare overlay colormaps in `pipeline/tips_runner.py` — test plasma, viridis, grayscale on 3 diverse images for demo display quality; set confirmed winner as `DEFAULT_COLORMAP` constant (resolves research.md Decision 3 open item)
- [ ] T020 [P] Validate segmentation threshold in `pipeline/tips_runner.py` — compare argmax default vs. soft threshold at P=0.5 softmax confidence on 5 test images; set approach as `SEGMENTATION_THRESHOLD` constant (resolves research.md Decision 4 open item)
- [ ] T021 Run Phase 1.5 TIPSv2 output contract validation — 5 diverse images through full pipeline at 896px short side; confirm VRAM ≤ 16GB via `nvidia-smi`; if OOM, reduce to 672px and update `TIPSV2_SHORT_SIDE` in `pipeline/tips_runner.py`; document in `quickstart.md` Known Issues
- [ ] T022 Run Phase 1.7 latency benchmark — measure per-step times (TIPSv2 inference, Gemma agent loop, TTS) across 5 queries; document breakdown against SC-001 (≤ 30s) in `quickstart.md`
- [ ] T023 Run SC-002 distance accuracy validation — 5 test scenes with measured ground-truth distances in 0.5–8m range; assert ±20% accuracy for all 5; document in `quickstart.md`
- [ ] T024 Run Phase 3.2 regression against all 12 success criteria SC-001 through SC-012; document pass/fail for each in `quickstart.md`
- [ ] T025 [P] Correct stale tool references in `specs/001-spatialsense-blind-nav/research.md` — Decision 6: replace old tools with current (`search_seg_classes`, `call_dpt_head`, `call_encoder_zero_shot`, `measure_object`); Decision 7: replace "depth-colored overlay image" with "original RGB image" and note bearing is now computed, not qualitative; Decision 8: replace `sample_depth` with `measure_object`
- [ ] T026 [P] Validate `quickstart.md` end-to-end on clean environment — follow all steps, confirm `bash scripts/start_gemma4.sh` starts the server and `python app.py` launches; update Known Issues if anything fails
- [ ] T027 Demo video recording and technical write-up for hackathon submission (plan.md Phase 3.0 and 3.1)
- [ ] T031 [P] Implement lightweight debug observability in `server/agent.py` (or logging module) — emit metadata-only logs (`timestamp`, latency, model/tool status, confidence, error class) when enabled, and ensure no raw image/audio/question-text persistence

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
- `overlay` in tasks refers to the depth-colormap alpha-blended onto RGB — demo display artifact in `session.overlay`, shown in Gradio UI, never sent to Gemma
- `bearing_deg` in `measure_object`: depth cancels in `atan2((px-cx)/fx, 1)` — bearing is depth-independent and reliable even when distance confidence is low
- `measure_object` confidence: `"low"` when intersecting pixels < 50, distance falls outside 0.5–8.0 m, or sampled depth is at/near model bounds (1e-3 or 10.0); `"medium"` for zero-shot path; `"high"` for clean argmax segmentation in reliable range
- When `intrinsics.source == "assumed"`, bearing accuracy is ±5° typical; Gemma should express this as approximate in its response
