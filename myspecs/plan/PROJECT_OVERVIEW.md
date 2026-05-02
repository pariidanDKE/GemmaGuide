# SpatialSense: Blind Navigation Assistant
### Gemma 4 Good Hackathon — Project Overview

---

## The Problem

Blind and visually impaired people navigate the world without access to real-time spatial information. Existing assistive tools (white canes, guide dogs) are physical and limited. AI-powered spatial understanding has never been practically accessible in a conversational, voice-driven form.

The question this project answers: **can a blind person ask "how far is the table from me?" and get a useful, accurate answer from their phone camera?**

---

## The Hackathon

**Gemma 4 Good Hackathon** — Kaggle × Google DeepMind  
- Prize pool: $200,000  
- Deadline: May 18, 2026  
- Target tracks: **Digital Equity & Inclusivity** ($10k) + **Main Track** ($50k)  
- Requirement: must use at least one Gemma 4 model  

The project fits the hackathon brief because it uses Gemma 4 for multimodal reasoning over spatial data — not just as a chatbot backend — and addresses a real accessibility gap.

---

## The Models

### TIPSv2 (Google DeepMind, CVPR 2026)
**Text-Image Pretraining with Spatial awareness v2**  
Paper: [arxiv 2604.12012](https://arxiv.org/abs/2604.12012)  
Weights: [huggingface.co/google/tipsv2-b14-dpt](https://huggingface.co/google/tipsv2-b14-dpt)

TIPSv2 is a vision-language encoder that achieves state-of-the-art patch-text alignment through a novel training objective (iBOT++). The key insight of the paper is that supervising all patch tokens — not just masked ones — dramatically improves dense spatial understanding.

The `-dpt` variant attaches a Dense Prediction Transformer head on top of the frozen encoder, trained on NYU Depth V2, providing three outputs from a single forward pass:

| Output | Shape | Description |
|--------|-------|-------------|
| `outputs.depth` | `(1, 1, 448, 448)` | Per-pixel depth in **real meters** (confirmed: NYUv2 ground truth is metric) |
| `outputs.normals` | `(1, 3, 448, 448)` | Per-pixel surface normal vectors |
| `outputs.segmentation` | `(1, 150, 448, 448)` | Per-pixel class logits, 150 ADE20K classes |

**Confirmed in experiments:**
- Depth range on a real indoor kitchen scene: 2.27m – 6.82m ✓
- All three output shapes confirmed on T4 Colab ✓

**What TIPSv2 cannot do out of the box:**
- Instance segmentation (two chairs = one "chair" class)
- Zero-shot segmentation for classes outside the 150 ADE20K classes (requires the base encoder + text encoder)
- Natural language descriptions

### Gemma 4 4B-IT (Google DeepMind)
**Instruction-tuned, multimodal, function calling support**  
Weights: `google/gemma-4-4b-it`

Gemma 4 is chosen for:
- Native function/tool calling support
- Multimodal input (can receive multiple images in one call)
- Small enough (4B) to be feasible on-device in the future
- Apache 2.0 license — no restrictions
- Required by the hackathon

---

## The Architecture

```
User speaks
    │
    ▼
Gemma 4 4B-IT  ◄──── Camera image (live or uploaded)
  (audio direct)       + audio question (direct — no STT front-end needed)
    │
    │   Gemma acts as agent: decides which tools to call based on the question
    │
    ├── Tool: get_seg_classes()
    │       Returns list of 150 ADE20K class names
    │       Gemma checks if the object the user asked about is covered
    │
    ├── Tool: call_dpt_head(image)
    │       Calls TIPSv2 b14-dpt
    │       Post-processes depth + segmentation into two images:
    │         - Depth colormap image (metric scale bar baked in)
    │         - Depth-colored overlay image (ADE20K class segments
    │           colored by their per-segment mean depth, alpha-blended onto RGB)
    │       Returns: [depth_colormap_image, overlay_image]
    │       Side-effect: depth tensor cached server-side for sample_depth calls
    │
    ├── Tool: call_encoder_zero_shot(image, class_list)
    │       For objects NOT in the 150 ADE20K classes
    │       Uses TIPSv2 base encoder + text encoder
    │       Cosine similarity: patch embeddings vs text embeddings of custom class names
    │       Returns: overlay image with zero-shot segments colored by depth
    │       Side-effect: depth tensor cached server-side for sample_depth calls
    │
    └── Tool: sample_depth(x, y)  /  sample_depth_region(x1, y1, x2, y2)
            Available after any TIPSv2 call (call_dpt_head or call_encoder_zero_shot)
            Queries the cached depth tensor directly at pixel coordinates
            Returns: metric depth value in meters (float)
            Purpose: precise metric readout without relying on Gemma to interpret
            color legends — Gemma localizes the object visually, then probes
            the exact depth at that location
    │
    ▼
Gemma 4 reasons over:
    - Original RGB image
    - Depth colormap image and/or overlay image (from tool calls)
    - Exact metric depth values (from sample_depth)
    - User's original question
    │
    ▼
Natural language navigation description
    │
    ▼
Text-to-Speech (TTS)
    │
    ▼
User hears: "There is a chair 1.2 meters to your left. 
             The door is 4.1 meters ahead of you. 
             The counter is 5.5 meters to your right."
```

---

## Phase 0: Feasibility Gate (Dead-on-Arrival Risk Check)

Before building the full product pipeline, we run a strict **Phase 0** to validate the core assumptions. If any critical test fails, we pivot architecture immediately.

### Goal
Prove that Gemma + TIPSv2 can exchange useful multimodal information for navigation tasks, and that audio-first interaction is technically viable.

### Phase 0 Test Matrix

| Test | What we verify | Pass criteria | Failure implication |
|------|----------------|---------------|---------------------|
| Gemma 4B text I/O | Basic instruction following and response stability | Consistent, grounded responses across repeated prompts | Prompting layer too unstable for assistive UX |
| Gemma 4B image/video understanding | Gemma can interpret single frames and short visual context inputs | Correct object/location summaries on controlled examples | Vision path cannot be trusted for navigation prompts |
| Gemma audio input path | Whether Gemma can consume audio directly, or requires STT front-end | Clear, reproducible ingestion path documented and tested | Voice UX blocked unless STT bridge is mandatory |
| TIPSv2 output semantics | Exact output types/ranges for depth, normals, segmentation logits | Output schema documented with real sample tensors and units | Downstream parser/prompt contract is undefined |
| TIPSv2 -> Gemma handoff | Whether Gemma reasons correctly over structured TIPSv2 outputs | Accurate answers to distance/position questions from injected structured data | Core two-model architecture fails |
| Joint runtime feasibility | Gemma 4B + TIPSv2 coexistence on target hardware (T4 baseline) | Successful inference without OOM at acceptable latency | Deployment target must change or architecture simplified |

### Mandatory Phase 0 Deliverables

- Working interface notes for Gemma <-> TIPS exchanges (living contract during Phase 0)
- A reproducible notebook/script showing Gemma answers from TIPSv2-derived structured inputs
- A decision note on audio path: direct-to-Gemma vs Whisper->text->Gemma
- A go/no-go checklist signed off before UI or demo polish work (decision recorded in checklist)
- Execution checklist: `specs/plan/PHASE_0_CHECKLIST.md`

### Exit Criteria (Go/No-Go)

- Go if all critical items pass: Gemma/TIPS interface assumptions are validated, Gemma understands structured prompt inputs, and audio interaction path is operational.
- No-Go if any critical item fails without workaround: especially audio ingestion path or Gemma reasoning over TIPSv2 outputs.

---

## Why This Architecture

**TIPSv2 alone can't describe a scene.** It outputs tensors — depth values and class indices. No language.

**Gemma alone can't measure distances.** It can describe what it sees but has no metric depth understanding from a single RGB image.

**Together they do something neither can do solo.** TIPSv2 handles the spatial heavy lifting, Gemma handles the reasoning and language generation. This is a genuine demonstration of Gemma 4's multimodal + function calling capabilities, which is exactly what the hackathon is looking for.

---

## Key Technical Decisions

**Why TIPSv2 over SAM3 + separate depth model?**  
Single model, single forward pass gives depth + segmentation simultaneously. SAM3 requires text prompts and a separate depth model. TIPSv2-dpt is simpler and faster for a real-time navigation use case.

**Why images from tools, not a raw heatmap handed directly to Gemma?**  
Raw heatmaps let Gemma shortcut by reading the colorbar label rather than spatially localizing objects (confirmed failure mode in Phase 0.5). The depth-colored overlay bakes depth information directly onto object regions in the RGB, forcing genuine spatial reasoning. Tools return these pre-processed images rather than structured dicts — Gemma's multimodal reasoning over images is sufficient and eliminates a brittle parsing layer.

**Why 4B and not a larger Gemma model?**  
Future-proofing for on-device inference. Gemma 4 E2B/E4B variants are designed for mobile. Starting at 4B keeps the architecture compatible with eventual on-device deployment.

**Why the zero-shot fallback tool?**  
150 ADE20K classes miss navigation-critical objects (steps, curbs, shopping carts, puddles). The fallback uses TIPSv2's base encoder with the text encoder to do zero-shot segmentation for arbitrary class names. This makes the system extensible without retraining.

**Why sample_depth as a separate tool?**  
Both `call_dpt_head` and `call_encoder_zero_shot` return images — Gemma visualizes where objects are, but color-to-metric translation is unreliable (confirmed: Gemma read a colorbar label rather than spatially reasoning in Phase 0.5). `sample_depth(x, y)` acts as an eye dropper: Gemma identifies the pixel location of an object from the RGB/overlay images, then probes the cached depth tensor directly for the exact metric value. This separates visual localization (Gemma's strength) from metric readout (raw tensor lookup), and works regardless of which segmentation path was used.

---

## Known Limitations & Honest Assessment

| Limitation | Severity | Notes |
|------------|----------|-------|
| No instance segmentation | Medium | Two chairs → one "chair" reading. Mitigated by left/center/right spatial zones |
| 150 class ceiling for DPT head | Medium | Zero-shot fallback covers this but is coarser |
| Latency | High | Multiple model calls per user question. Not real-time yet |
| Depth is indoor-optimized | Medium | NYUv2 is indoor scenes. Outdoor performance untested |
| No continuous video tracking | High | Single frame, not a continuous stream |
| Gemma function calling overhead | Low | Adds ~1-2 round trips per question |

---

## What Has Been Confirmed Working

- [x] TIPSv2 b14-dpt loads on Colab T4
- [x] Depth output confirmed metric: 2.27m – 6.82m on indoor kitchen image
- [x] All three output heads confirmed: depth, normals, segmentation
- [x] Depth visualization with matplotlib + colorbar in meters
- [x] `torch.no_grad()` inference working correctly

## What Needs Testing Next

- [ ] **Phase 0.1:** Gemma 4B text-only reliability baseline (prompt consistency + safety wording)
- [ ] **Phase 0.2:** Gemma 4B visual reasoning check (single image and short video/frame-sequence behavior)
- [ ] **Phase 0.3:** Audio ingestion decision test (direct audio into Gemma vs Whisper front-end)
- [ ] **Phase 0.4:** TIPSv2 output contract validation (depth/normals/segmentation schema and units)
- [ ] **Phase 0.5:** TIPSv2-to-Gemma reasoning test (structured dict -> accurate navigation answers)
- [ ] **Phase 0.6:** Joint memory/latency feasibility on T4 16GB (or explicit fallback hardware plan)
- [ ] **Phase 1:** Build end-to-end assistant loop after Phase 0 sign-off

---

## Stack

| Component | Technology |
|-----------|-----------|
| Depth + Segmentation | TIPSv2 b14-dpt (Google DeepMind) |
| Language + Reasoning | Gemma 4 4b-it (Google DeepMind) |
| LLM Serving | vLLM |
| STT | Direct audio to Gemma (Whisper not required — confirmed Phase 0.3) |
| TTS | TBD (gTTS / Coqui / ElevenLabs) |
| Demo UI | Gradio |
| Compute | Colab T4 / Vast.ai 3090 / Local RTX 3090 |

---

## Timeline

**Phase 0 (Week 0 / Immediate)** — Feasibility gate
- Validate Gemma text + vision behavior for navigation prompts
- Determine audio path (direct or STT bridge)
- Validate TIPSv2 output contract and Gemma reasoning on structured spatial data
- Confirm hardware feasibility (T4 baseline) and latency envelope

**Week 1** — Core pipeline build (only if Phase 0 passes)
- Finalize depth + segmentation combination logic
- Gemma tool calling setup with stable schema contract
- First end-to-end test with voice input/output path selected in Phase 0

**Week 2** — Full pipeline integration
- STT + TTS
- Gradio UI with image/webcam input
- Prompt tuning

**Week 3** — Polish
- Demo video recording
- Technical write-up
- Edge cases

---

## References

- TIPSv2 paper: Zhou et al., CVPR 2026. [arxiv 2604.12012](https://arxiv.org/abs/2604.12012)
- NYU Depth V2 dataset: Silberman et al., ECCV 2012
- ADE20K dataset: Zhou et al., CVPR 2017
- DPT: Ranftl et al., ICCV 2021
- Gemma 4: Google DeepMind, 2026
- Kaggle Gemma 4 Good Hackathon: [kaggle.com/competitions/gemma-4-good-hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
