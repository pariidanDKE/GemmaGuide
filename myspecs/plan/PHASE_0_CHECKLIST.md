# Phase 0 Checklist: Feasibility Gate

Use this checklist to decide whether the project is viable before full pipeline implementation.

## Rules

- Do not start Phase 1 until all Critical checks pass, or explicit fallback decisions are documented.
- Record reproducible evidence for every item (notebook cell output, logs, metrics, screenshots).
- Mark each item as: `[ ]` Not started, `[~]` In progress, `[x]` Passed, `[-]` Failed.

## Success Criteria Summary

- Critical checks passed:
  - Gemma can reason over TIPSv2-derived visual spatial input (depth colormap + segmentation image)
  - Audio interaction path is confirmed and reproducible
  - Runtime is feasible on baseline hardware or fallback hardware is approved
- Required artifacts produced:
  - Provisional interface notes (what is passed between TIPSv2 and Gemma during tests)
  - Reproducible test notebook/script
  - Go/No-Go decision recorded in this checklist

---

## 0.1 Gemma 4B Text Reliability (Critical)

Status: [x]

- [x] Confirm model loads and responds deterministically enough for assistive prompts
- [x] Verify response format follows required structure (distance, direction, confidence wording)

Evidence:
- Notebook/script: notebooks/Gemma4_(E2B)_Unsloth.ipynb
- Logs: Model loaded on RTX 3090 (23.566 GB) via Unsloth 16-bit LoRA, no errors. Structured format `Task | Answer | Confidence | Reason` followed in cells 18 & 19.
- Notes: Format compliance demonstrated on 2 prompts. Cell 19 produced a hallucinated distance (250m for an indoor table) when audio + images were combined — indicates prompt sensitivity, not a format failure. Full reliability and safety phrasing validation deferred to Phase 1 pipeline runs.

---

## 0.2 Gemma 4B Image/Video Understanding (Critical)

Status: [x]

- [x] Test single-image object-location summarization on known examples
- [x] Compare responses against expected object presence and coarse position
- [x] Record failure modes (missed objects, wrong side, overconfident language)

Evidence:
- Dataset/sample set: data/media/images/ (original.png, depth.png, depth_visualization.png)
- Notebook/script: notebooks/Gemma4_(E2B)_Unsloth.ipynb cells 10, 12, 18, 19
- Metrics: Qualitative validation passed. Cell 10 correctly described scene and spatial positions ("to the left of the monitors"). Cell 18 estimated chair at 4.5m with medium confidence (reasonable). Cell 19 hallucinated 250m for indoor table — failure mode confirmed and recorded.
- Notes: Failure mode confirmed — audio + multi-image input can cause metric hallucination. Medium confidence used correctly in cell 18. Formal pass-rate and frame sequence behavior deferred to Phase 1 pipeline runs.

---

## 0.3 Audio Input Path Decision (Critical)

Status: [x]

- [x] Verify whether Gemma serving stack accepts direct audio input
- [x] If direct audio is not available/reliable, validate Whisper -> text -> Gemma path
- [x] Document chosen architecture and rationale

Decision:
- [x] Direct audio to Gemma
- [ ] Whisper front-end then text to Gemma

Evidence:
- Integration notes: notebooks/Gemma4_(E2B)_Unsloth.ipynb cells 14–16. Gemma received audio file (simple_request.mp3) and correctly identified and answered the spoken question ("The table is four meters away."). Unsloth FastModel + transformers 5.5.0 audio path confirmed working on RTX 3090.
- Final decision note: Direct audio to Gemma is viable and reproducible. Whisper path not needed. End-to-end latency measurement deferred to Phase 1.

---

## 0.4 TIPSv2 Output Contract Validation (Critical)

Status: [x]

- [x] Confirm output tensors and shapes for depth/normals/segmentation
- [x] Confirm depth units and expected numeric range on real samples
- [x] Define post-processing for per-object distance and position bins
- [x] Define confidence/quality flags (e.g., low-seg confidence, depth outliers)

Interface notes artifact:
- Path: see "Working Interface Notes" section below

Evidence:
- Source: google/tipsv2-b14-dpt (modeling_dpt.py, dpt_head.py, config.json)
- `depth`: (B, 1, H, W), metric meters, range [0.001, 10.0]. 256-bin classification → weighted expected value, bilinearly upsampled to input resolution.
- `segmentation`: (B, 150, H, W) logits, ADE20K 150 classes. Post-process: `.argmax(dim=1)` → (B, H, W) per-pixel class.
- `normals`: (B, 3, H, W) — not used in current pipeline.
- Depth trained on NYU Depth V2 (indoor). Reliability above 10m is not guaranteed — outdoor scenes beyond ~10m should be flagged as low-confidence.
- Confidence/quality flag: depth values near max_depth (10.0m) or min_depth (0.001m) clipped → treat as uncertain. Segmentation softmax entropy can serve as a per-patch confidence signal.
- Post-processing: render depth tensor as colormap image; render segmentation argmax as depth-colored class overlay alpha-blended onto RGB. Both fed as images alongside RGB — no structured dict needed.
- Notes: Contract derived from source code and confirmed on real samples. Multi-image diversity validation deferred to Phase 1 pipeline runs.

---

## 0.5 TIPSv2 -> Gemma Reasoning Handoff (Critical)

Status: [x]

- [x] Feed Gemma RGB + TIPSv2 depth colormap as images
- [x] Identify failure modes in the handoff

Evidence:
- Regression set: 1 sample (desk scene, audio question "how far away is the table?")
- Evaluation notebook/script: notebooks/TIPSV2_Gemma4_Combined.ipynb
- Metrics: 1/1 correct answer (1.40m reported, plausible for desk at ~1.4–2m). Format compliance 1/1.
- Notes: **Failure mode identified and architectural fix designed** — Gemma answered 1.40m by reading the colorbar label ("Depth [1.40 m – 6.02 m]") rather than spatially localizing the table in the depth map. Fix: depth-colored overlay paints depth directly onto object regions in the RGB so Gemma cannot shortcut via the colorbar. Additional fix: `sample_depth(x, y)` tool lets Gemma probe the depth tensor at pixel coordinates directly, eliminating color interpretation entirely. Formal regression validation deferred to Phase 1 pipeline runs.

---

## 0.6 Joint Runtime Feasibility (Critical)

Status: [x]

- [x] Run Gemma + TIPSv2 in target baseline environment (T4 16GB)
- [x] Test at least one fallback environment (e.g., 3090) if T4 is insufficient
- [x] Document minimal viable deployment target

Evidence:
- Hardware configs tested: Tesla T4 (14.56 GB) — TIPSv2 DPT + Gemma E2B both loaded and ran inference end-to-end in TIPSV2_Gemma4_Combined.ipynb. No OOM. Baseline VRAM ~13 GB allocated. ~1.5 GB headroom remaining — monitor for OOM if input resolution is increased above 896px short side.
- RTX 3090 (23.566 GB) confirmed as fallback with ample headroom.
- Final deployment recommendation: T4 (16 GB) confirmed as viable baseline deployment target. Per-query latency measurement deferred to Phase 1.

---

## Required Artifacts

- [x] Working Interface Notes section completed in this file
- [x] TIPSV2_Gemma4_Combined.ipynb — experimentation notebook confirming handoff and failure mode
- [x] Go/No-Go section completed in this file

---

## Working Interface Notes

- Current direction: TIPSv2 DPT (google/tipsv2-b14-dpt) renders depth + segmentation as images → fed alongside RGB to Gemma 4 E2B via Unsloth FastModel.
- What goes into Gemma: RGB image + depth heatmap image (with scale bar baked in) + depth-colored class overlay image (class segments from DPT full-res segmentation, colored by their depth value, alpha-blended onto RGB) + text question. Audio optionally included as an additional input modality.
- What TIPSv2 DPT returns:
  - `depth`: (B, 1, H, W) float tensor, meters, [0.001, 10.0]
  - `segmentation`: (B, 150, H, W) full-resolution ADE20K logits — user class names mapped to ADE20K class indices, threshold per-pixel probability → precise binary mask → colored by depth value, alpha-blended onto RGB. Full-resolution (H, W) masks, no upsampling artefacts.
  - Input must be [0, 1] range, multiples of 14px recommended (standard 448px, tested up to 896px short side)
- Class coverage: two paths depending on whether the class is in ADE20K:
  - **In ADE20K (150 classes):** use `tipsv2-b14-dpt` → full-resolution logits from the DPT seg head directly.
  - **Not in ADE20K (custom classes):** use base encoder (`tipsv2-b14`, no DPT head) + text encoder → cosine similarity between patch embeddings and class text embeddings → bilinearly upsample to full pixel resolution → argmax per pixel.
- Depth tensor cached server-side after any TIPSv2 call → available to `sample_depth(x, y)` tool for direct metric readout.
- Confidence/quality flags:
  - Depth at or near 10m boundary → flag as uncertain (outdoor or very far)
  - Low segmentation probability at threshold → low-confidence detection, skip or warn
- Open questions (deferred to Phase 1): optimal colormap for Gemma depth interpretation; optimal segmentation probability threshold; optimal prompt structure with composite overlay; VRAM peak at 896px during joint run.

---

## Go/No-Go Decision

Date: 2026-04-30
Owner: Dan Parii

Critical checks:
- [x] 0.1 Passed
- [x] 0.2 Passed
- [x] 0.3 Passed
- [x] 0.4 Passed
- [x] 0.5 Passed
- [x] 0.6 Passed

Decision:
- [x] GO to Phase 1
- [ ] NO-GO (pivot required)
