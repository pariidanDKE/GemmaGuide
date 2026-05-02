# Phase 1 Checklist: Core Pipeline Build

Phase 0 signed off (GO). This checklist tracks the core pipeline build and closes the validation items deferred from Phase 0.

## Rules

- Mark each item as: `[ ]` Not started, `[~]` In progress, `[x]` Done, `[-]` Dropped.
- Validation items carried over from Phase 0 are marked with `[Phase 0 deferred]`.
- Pipeline must be working end-to-end before moving to Phase 2 (STT/TTS, Gradio UI).

---

## 1.0 vLLM Server Setup (Requirement)

Gemma 4 4B-IT will be served via a local vLLM instance accessed through OpenAI-compatible endpoints. All Phase 1 pipeline work targets this serving path — not Unsloth (used in Phase 0 experiments only).

Docs: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html

### Checklist

- [ ] Install vLLM with audio extras (`vllm[audio]`)
- [ ] Confirm `google/gemma-4-4b-it` loads and serves on RTX 3090
- [ ] Confirm OpenAI client (`base_url=http://localhost:8000/v1`, `api_key="EMPTY"`) reaches the server
- [ ] Confirm image input works: `image_url` in message content → correct scene description
- [ ] Confirm audio input works: `audio_url` in message content → correct transcription/answer (re-validates Phase 0.3 on the vLLM path)
- [ ] Confirm tool calling works end-to-end: tool schema in `tools` param → Gemma selects tool → pipeline executes → result injected → final answer returned

---

## 1.1 Pipeline Application

Build the core pipeline that all Phase 0 deferred validation runs through.

- [ ] Accept image + audio/text input
- [ ] Run TIPSv2 DPT → depth tensor + segmentation
- [ ] Build depth-colored overlay image (ADE20K segments colored by per-segment mean depth, alpha-blended onto RGB)
- [ ] Cache depth tensor server-side for `sample_depth` calls
- [ ] Pass RGB + depth colormap + overlay images to Gemma via tool call
- [ ] Return natural language answer

---

## 1.2 Tool Implementation

- [ ] `get_seg_classes()` — returns list of 150 ADE20K class names
- [ ] `call_dpt_head(image)` — runs TIPSv2 DPT, returns [depth_colormap_image, overlay_image], caches depth tensor
- [ ] `call_encoder_zero_shot(image, class_list)` — runs TIPSv2 base encoder + text encoder, returns overlay image, caches depth tensor
- [ ] `sample_depth(x, y)` — queries cached depth tensor at pixel coordinates, returns metric float
- [ ] `sample_depth_region(x1, y1, x2, y2)` — returns mean depth over a pixel region

---

## 1.3 Gemma Tool Calling Setup

- [ ] Define tool schemas (function signatures + descriptions) for all 4 tools
- [ ] Confirm Gemma correctly selects tools based on question type (depth query vs. scene description)
- [ ] Confirm Gemma uses `sample_depth` after locating an object visually rather than reading colorbar

---

## 1.4 Multimodal Approach Validation [Phase 0 deferred]

Test the full multimodal input combinations through the pipeline and settle on what works best. No isolated text-only or image-only benchmarks — evaluate everything end-to-end.

- [ ] Test image + text input: RGB + overlay → navigation answer quality
- [ ] Test image + audio input: RGB + spoken question → navigation answer quality
- [ ] Compare prompt structures (e.g., system prompt wording, how images are ordered in the message) and pick the best performer
- [ ] Record failure modes and pick the combination that goes into the main pipeline

---

## 1.5 Deferred Validation: TIPSv2 Output Contract [Phase 0 deferred]

- [ ] Run at least 5 diverse test images through the full pipeline — confirm depth + segmentation outputs match expected schema without manual fixes
- [ ] Confirm VRAM does not OOM at 896px short side during joint TIPSv2 + Gemma run

---

## 1.6 Deferred Validation: TIPSv2 -> Gemma Reasoning Handoff [Phase 0 deferred]

- [ ] Build regression set of at least 5 diverse question/expected-answer pairs
- [ ] Confirm depth-colored overlay fixes the colorbar-reading failure mode from Phase 0.5
- [ ] Confirm `sample_depth` returns correct metric values and Gemma uses them accurately
- [ ] Test low-confidence depth input handling — clipped depth values (near 0.001m or 10m) should trigger cautious language from Gemma

---

## 1.7 Deferred Validation: Latency [Phase 0 deferred]

- [ ] Measure end-to-end latency: voice question → answer
- [ ] Measure per-query latency breakdown: TIPSv2 inference / overlay rendering / Gemma tool calls
- [ ] Document median latency against prototype acceptability threshold

---

## 1.8 Open Questions to Resolve During Build

These were left open at Phase 0 and should be answered during pipeline development:

- [ ] Which colormap works best for Gemma depth interpretation — plasma, viridis, grayscale?
- [ ] Optimal segmentation probability threshold for clean masks?
- [ ] Optimal prompt structure when composite overlay image is provided?
