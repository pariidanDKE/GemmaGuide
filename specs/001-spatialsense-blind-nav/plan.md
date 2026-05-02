# Implementation Plan: SpatialSense — Blind Navigation Assistant

**Branch**: `001-spatialsense-blind-nav` | **Date**: 2026-04-30 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-spatialsense-blind-nav/spec.md`

## Summary

SpatialSense combines TIPSv2 b14-dpt (metric depth + ADE20K segmentation from a single camera frame) with Gemma 4 4B-IT (multimodal reasoning + native function calling). Gemma receives the original RGB image and the user's question, then acts as an agent: it calls tools to run TIPSv2 inference (caching the depth tensor and segmentation mask), draws a bounding box around the specific object instance it wants to measure, and calls `measure_object` to get the metric distance. Horizontal bearing to each object is computed arithmetically from the pixel centroid of the segmented intersection and camera intrinsics extracted from image EXIF (with a 65° FOV fallback when EXIF is absent). The response is read aloud via TTS. All six Phase 0 feasibility checks passed on 2026-04-30; Phase 1 targets vLLM serving with an OpenAI-compatible client.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: TIPSv2 b14-dpt (`transformers` + `torch`), Gemma 4 4B-IT via `vllm` (OpenAI-compatible client), `gradio`, `gTTS` (TTS baseline), `numpy`, `Pillow`, `matplotlib`  
**Storage**: Raw user media is in-memory only per session; optional debug metadata logs only (no raw media/text)  
**Testing**: `pytest`; integration tests run against live model instances (no mocks — mock/live divergence is a confirmed risk)  
**Target Platform**: Linux server with GPU — Colab T4 16GB (baseline); Vast.ai / local RTX 3090 (fallback)  
**Project Type**: AI inference pipeline + web-service demo  
**Performance Goals**: End-to-end latency ≤ 30 seconds (query submission → spoken answer) on T4 16GB baseline  
**Constraints**: Combined VRAM ≤ 16GB; Gemma 4B + TIPSv2 must coexist (confirmed ~13GB combined on RTX 3090 in Phase 0)  
**Scale/Scope**: Single-user hackathon demo; no concurrent user requirement

## Constitution Check

No project constitution has been defined (template is blank). Applying default engineering principles:
- Keep it simple: minimal abstraction layers, no framework overhead beyond what is needed
- Validate with real model integration tests: mock behavior diverges from live model behavior
- No premature optimization: prioritize working demo over performance until latency benchmarks are done

**Gate: PASS** — No constitution violations to track.

## Project Structure

### Documentation (this feature)

```text
specs/001-spatialsense-blind-nav/
├── plan.md              # This file
├── research.md          # Phase 0 research decisions
├── data-model.md        # Entity model for the pipeline
├── quickstart.md        # Setup and run instructions
├── contracts/
│   └── tool-schemas.md  # Tool call schemas for Gemma function calling + Gradio UI contract
└── tasks.md             # Phase 2 output generated via /speckit-tasks
```

### Source Code (repository root)

```text
pipeline/
├── tips_runner.py       # TIPSv2 model loading, forward pass, depth tensor + seg mask caching, overlay generation (demo artifact)
├── tools.py             # Tool implementations: search_seg_classes, call_dpt_head, call_encoder_zero_shot, measure_object
├── intrinsics.py        # Camera intrinsics: EXIF extraction (focal length → fx/fy/cx/cy) with 65° FOV fallback
├── session.py           # Per-query session state: intrinsics, depth tensor, seg mask, image store, session lifecycle
└── tts.py               # Text-to-speech: converts response text → audio bytes (gTTS baseline)

server/
├── agent.py             # Gemma tool-calling agent loop (OpenAI client → vLLM, tool dispatch)
└── schemas.py           # JSON tool schemas passed to Gemma via the tools parameter

app.py                   # Gradio demo UI: image upload + mic input + text + response display + audio out

scripts/
└── start_gemma4.sh      # Start vLLM server with Gemma 4 E4B-IT (tool calling + vision + audio)

tests/
├── integration/         # End-to-end tests: real TIPSv2 + real Gemma (requires GPU + vLLM server)
└── unit/                # Offline tests: measure_object mask intersection logic, session lifecycle
```

**Structure Decision**: Single-project layout with two internal layers (`pipeline/` for inference, `server/` for agent orchestration). No backend/frontend split — Gradio runs from `app.py` at the root and imports directly from `pipeline/` and `server/`. This keeps the codebase flat enough for a hackathon prototype while separating model concerns from serving concerns.

## Complexity Tracking

No constitution defined; no violations to track.

---

## Build Phases

### Phase 1: Core Pipeline (Priority — must complete before UI)

Tracks against `myspecs/plan/PHASE_1_CHECKLIST.md`. See `research.md` for all architecture decisions.

**1.0 — vLLM Server Setup**
- Install vLLM with audio extras; serve `google/gemma-4-4b-it` on GPU
- Validate: OpenAI client reaches server; image input works; audio input works; tool calling works

**1.1 — Pipeline Application**
- Accept image + audio/text input → pass RGB + question to Gemma agent → Gemma calls tools as needed → return natural language answer + TTS audio
- Overlay is generated as a demo display artifact but is NOT passed to Gemma
- For scene summaries, cap object reporting to 5 items (closest-to-farthest) and cap Gemma generation to 256 output tokens

**1.2 — Tool Implementations** (see `contracts/tool-schemas.md` for schemas)
- `search_seg_classes(query)`, `call_dpt_head(image_ref)`, `call_encoder_zero_shot(image_ref, class_list)`, `measure_object(class_name, box_2d)` → returns `{distance_m, bearing_deg, confidence}`

**1.3 — Gemma Tool Calling**
- Register tool schemas; validate Gemma selects tools correctly; confirm Gemma draws bounding boxes and calls `measure_object` for metric depth and bearing; confirm `bearing_deg` is computed from pixel centroid + camera intrinsics and reported by Gemma as natural language degrees

**1.4 — Multimodal Approach Validation** (Phase 0 deferred)
- Test image + text and image + audio combinations; compare prompt structures; document best performer

**1.5 — TIPSv2 Output Contract Validation** (Phase 0 deferred)
- 5 diverse test images through full pipeline; confirm VRAM holds at 896px short side

**1.6 — TIPSv2 → Gemma Reasoning Handoff Validation** (Phase 0 deferred)
- 5-query regression set; confirm Gemma draws bounding boxes and calls `measure_object` correctly; confirm `measure_object` returns accurate metric values; confirm aim corrections in Gemma's responses are spatially consistent with object positions in the RGB

**1.7 — Latency Benchmarking** (Phase 0 deferred)
- Measure end-to-end latency; per-step breakdown; document against 30-second acceptance threshold

**1.8 — Open Questions** (resolve during build)
- Segmentation threshold: argmax vs. soft threshold for the segmentation mask
- System prompt wording for Gemma's navigation role and bounding box instructions

### Phase 2: STT/TTS + Demo UI

**2.0 — TTS Integration**
- Integrate gTTS; synthesize response → audio bytes → Gradio audio output

**2.1 — Gradio UI**
- Image upload + microphone input + optional typed question → response text + audio output + overlay image display
- Target: interaction completable in < 30 seconds without instructions

**2.2 — Edge Case Handling**
- Low-light / blurry image → fallback response
- Object not in frame → explicit "not visible" response
- Missing/invalid/inaudible audio → spoken + text retry prompt; skip spatial inference for that turn
- Depth at extremes (< 0.5m or > 8m) → cautious language in response

**2.3 — Observability Policy**
- Keep observability lightweight (debug-focused metadata only: timestamp, latency, model/tool status, confidence, error class)
- No raw image/audio/question-text logging

### Phase 3: Polish (Week 3, pre-submission)

**3.0** — Demo video recording  
**3.1** — Technical write-up for hackathon submission  
**3.2** — Regression test run against all 12 success criteria from spec

---

*Detailed research decisions → `research.md`*  
*Entity definitions → `data-model.md`*  
*Tool schemas and interface contracts → `contracts/tool-schemas.md`*  
*Setup instructions → `quickstart.md`*
