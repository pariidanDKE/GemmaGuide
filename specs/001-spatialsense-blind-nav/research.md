# Research: SpatialSense — Blind Navigation Assistant

**Phase**: 0 (Plan Research)  
**Date**: 2026-04-30  
**Status**: Complete — all decisions resolved; three open items carry forward to Phase 1 build

---

## Decision 1: LLM Serving Backend

**Decision**: vLLM with OpenAI-compatible API (`base_url=http://localhost:8000/v1`, `api_key="EMPTY"`)  
**Rationale**: Unsloth was sufficient for Phase 0 notebook experimentation but does not expose a stable OpenAI-compatible server. vLLM provides production-quality serving, correct multimodal input handling (image + audio in OpenAI message format), and native tool/function calling support. The Gemma 4 4B-IT vLLM recipe is documented at `docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html`.  
**Alternatives considered**:
- Unsloth: Phase 0 only. No server mode. Dropped for Phase 1.
- Ollama: Does not support Gemma 4 tool calling as of 2026-04-30.
- HuggingFace `generate()` directly: Requires manual tool call parsing; vLLM handles this natively.

---

## Decision 2: Text-to-Speech Provider

**Decision**: `gTTS` (Google Text-to-Speech, no API key) as the baseline. ElevenLabs as an optional quality upgrade if build time permits in Phase 2.  
**Rationale**: gTTS is free, has no setup friction, and is sufficient for hackathon demo quality. The spec requires spoken output but does not require studio-quality voice. The TTS component is isolated in `pipeline/tts.py` so the provider can be swapped without changing anything else.  
**Alternatives considered**:
- Coqui TTS: Open-source, higher quality than gTTS, but requires local model download and additional VRAM. T4 16GB constraint makes this risky.
- ElevenLabs: Best voice quality; requires API key and adds network latency. Consider as upgrade only.
- pyttsx3: Offline, but audio quality is poor; not suitable for a demo.

---

## Decision 3: Depth Colormap for Overlay

**Decision**: Evaluate plasma, viridis, and grayscale during Phase 1 build (item 1.8). Default to `plasma` for initial testing — highest perceptual contrast across its range, makes object depth boundaries visually salient.  
**Rationale**: The colormap affects how well Gemma distinguishes depth zones in the overlay image. This cannot be decided analytically — it requires empirical comparison of Gemma's response quality. The depth-colored overlay (not a labeled colorbar) is what matters: depth is baked onto object segments in RGB space, so Gemma reasons spatially rather than reading a legend label.  
**Phase 0 context**: Phase 0.5 confirmed that a labeled colorbar caused Gemma to shortcut by reading the min/max label rather than spatially reasoning. The overlay architecture eliminates this failure mode regardless of colormap choice.  
**Resolution**: Set during Phase 1 build item 1.8.

---

## Decision 4: Segmentation Threshold

**Decision**: Use argmax segmentation (each pixel assigned to its highest-confidence class) as the default. Evaluate soft thresholding (discard pixels below P=0.5 softmax confidence) if mask edges are too noisy during Phase 1 validation.  
**Rationale**: Argmax gives the cleanest class assignment and requires no hyperparameter tuning. If masks are too coarse or contain noise at object boundaries, a confidence threshold can be applied to the softmax output before argmax.  
**Resolution**: Set during Phase 1 build item 1.8.

---

## Decision 5: Audio Ingestion Path

**Decision**: Direct audio to Gemma via vLLM's audio support (no separate Whisper STT front-end).  
**Rationale**: Confirmed working in Phase 0.3 via the Unsloth path. Must be re-validated on the vLLM path in Phase 1 item 1.0. Keeping audio direct to Gemma is a key hackathon differentiator — it demonstrates Gemma 4's native multimodal capability (image + audio simultaneously).  
**Fallback**: If direct audio fails on vLLM: run `openai-whisper` base model on the audio file → pass transcript text as the question. This adds ~2 seconds latency and loses the native multimodal demo, but keeps the demo functional.  
**Alternatives considered**: Whisper-only from the start. Rejected because losing the audio-direct path removes demonstrated Gemma 4 multimodal capability.

---

## Decision 6: Tool Architecture

**Decision**: Five tools exposed to Gemma: `get_seg_classes()`, `call_dpt_head(image)`, `call_encoder_zero_shot(image, class_list)`, `sample_depth(x, y)`, `sample_depth_region(x1, y1, x2, y2)`.  
**Rationale**: Validated through Phase 0.5 failure analysis. Tools return images (not structured dicts) — Gemma's visual reasoning over images is sufficient and eliminates a brittle parsing layer. The depth tensor is cached server-side to bridge visual localization and metric readout.  
**Key design insight**: `sample_depth` exists to separate two concerns — Gemma is good at visual localization (identifying where in an image an object is) but unreliable at reading metric values from color legends. The tool lets Gemma do the visual work, then directly queries the tensor for the exact number.  
**Alternatives considered**:
- Return depth as a structured dict (pixel → depth value): Requires Gemma to process a large mapping. Tools-as-images + sample_depth probe is simpler.
- Single unified tool: Loses the ability to decouple the segmentation path (DPT head vs. zero-shot encoder) from depth sampling.

---

## Decision 7: Aim-Correction Mechanism

**Decision**: Aim correction is derived from Gemma's visual reasoning over the depth-colored overlay image. Gemma identifies where in the frame (left/center/right of center axis) an object sits and expresses this as a camera-relative aim correction. No separate centroid algorithm.  
**Rationale**: The camera is a directional probe (~60-80° FOV); everything in the frame is already in front of the user. Gemma identifies horizontal position directly from the RGB/overlay images — the same visual act used to localize the object for depth probing. A post-processing centroid step would add complexity without benefit, and would fail for multi-instance classes (ADE20K merges all instances of a class into one mask).  
**Language constraint**: Responses MUST use camera-relative language only ("directly where you're pointing", "bear slightly left", "bear slightly right"). Room-level directional language ("to your left", "to your right") is prohibited — the system has no 360° awareness.  
**Known limitation**: ADE20K merges all instances of a class into one mask. Two chairs produce one merged segment. Gemma is prompted to report per-instance aim correction hints based on visual separation in the overlay (e.g., "one directly ahead, another slightly to the right").

---

## Decision 8: System Prompt Structure

**Decision**: To be finalized during Phase 1 build (item 1.8). Initial approach: system prompt instructs Gemma that it is a navigation assistant for a blind user, that it must call `call_dpt_head` before answering any spatial question, and that it must call `sample_depth` after visually localizing an object rather than reading color scales.  
**Open**: Exact wording and image ordering in the message (RGB first vs. overlay first) requires empirical testing.

---

## Decision 9: Target Hardware and Input Resolution

**Decision**: T4 16GB as the baseline deployment target; RTX 3090 (24GB) as confirmed fallback.  
**Rationale**: Phase 0.6 confirmed both models fit simultaneously at ~13GB combined VRAM on RTX 3090 with TIPSv2 at 896px short side. The T4 has 16GB total; 13GB combined leaves 3GB headroom for overlay rendering and Gradio.  
**Resolution for T4 VRAM pressure**: If OOM occurs during Phase 1 validation, reduce TIPSv2 input from 896px to 672px or 448px short side. This trades depth map resolution for memory headroom.

---

## Open Items Carrying Forward to Phase 1 Build

| Item | What to decide | Where resolved |
|------|----------------|----------------|
| Colormap | plasma vs. viridis vs. grayscale | Phase 1 item 1.8 |
| Segmentation threshold | argmax default vs. soft threshold | Phase 1 item 1.8 |
| System prompt wording | Exact prompt for Gemma navigation role + image ordering | Phase 1 item 1.8 |
| TTS provider upgrade | gTTS baseline vs. ElevenLabs | Phase 2 if time permits |
| Audio on vLLM | Re-validate direct audio path on vLLM serving | Phase 1 item 1.0 |
