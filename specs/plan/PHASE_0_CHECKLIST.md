# Phase 0 Checklist: Feasibility Gate

Use this checklist to decide whether the project is viable before full pipeline implementation.

## Rules

- Do not start Phase 1 until all Critical checks pass, or explicit fallback decisions are documented.
- Record reproducible evidence for every item (notebook cell output, logs, metrics, screenshots).
- Mark each item as: `[ ]` Not started, `[~]` In progress, `[x]` Passed, `[-]` Failed.

## Success Criteria Summary

- Critical checks passed:
  - Gemma can reason over TIPSv2-derived structured spatial input
  - Audio interaction path is confirmed and reproducible
  - Runtime is feasible on baseline hardware or fallback hardware is approved
- Required artifacts produced:
  - Provisional interface notes (what is passed between Gemma and TIPS during tests)
  - Reproducible test notebook/script
  - Go/No-Go decision recorded in this checklist

---

## 0.1 Gemma 4B Text Reliability (Critical)

Status: [ ]

- [ ] Confirm model loads and responds deterministically enough for assistive prompts
- [ ] Run at least 10 repeated prompts with fixed settings
- [ ] Verify response format follows required structure (distance, direction, confidence wording)
- [ ] Capture unsafe/uncertain cases and required safety phrasing

Pass if:
- Response format compliance >= 90% on repeated prompts
- No hallucinated metric claims when no metric input is provided

Evidence:
- Notebook/script:
- Logs:
- Notes:

---

## 0.2 Gemma 4B Image/Video Understanding (Critical)

Status: [ ]

- [ ] Test single-image object-location summarization on known examples
- [ ] Test short frame sequence behavior (video-like context) if supported in your serving path
- [ ] Compare responses against expected object presence and coarse position
- [ ] Record failure modes (missed objects, wrong side, overconfident language)

Pass if:
- Correct coarse localization (left/center/right) in >= 80% of test cases
- Model uses uncertainty language when visibility is low/ambiguous

Evidence:
- Dataset/sample set:
- Notebook/script:
- Metrics:
- Notes:

---

## 0.3 Audio Input Path Decision (Critical)

Status: [ ]

- [ ] Verify whether Gemma serving stack accepts direct audio input
- [ ] If direct audio is not available/reliable, validate Whisper -> text -> Gemma path
- [ ] Measure end-to-end latency for voice question -> spoken answer (or text answer in this phase)
- [ ] Document chosen architecture and rationale

Pass if:
- At least one stable audio interaction path is reproducible
- Latency is acceptable for prototype use (target and measured values documented)

Decision:
- [ ] Direct audio to Gemma
- [ ] Whisper front-end then text to Gemma

Evidence:
- Integration notes:
- Latency measurements:
- Final decision note:

---

## 0.4 TIPSv2 Output Contract Validation (Critical)

Status: [ ]

- [ ] Confirm output tensors and shapes for depth/normals/segmentation
- [ ] Confirm depth units and expected numeric range on real samples
- [ ] Define post-processing for per-object distance and position bins
- [ ] Define confidence/quality flags (e.g., low-seg confidence, depth outliers)

Pass if:
- Provisional interface is documented and reproducible
- At least 5 diverse test images produce valid schema outputs without manual fixes

Interface notes artifact:
- Path: add section below in this file under "Working Interface Notes"

Evidence:
- Notebook/script:
- Sample outputs:
- Notes:

---

## 0.5 TIPSv2 -> Gemma Reasoning Handoff (Critical)

Status: [ ]

- [ ] Feed Gemma only structured spatial dict + user question
- [ ] Evaluate if answers match structured facts (distance and position)
- [ ] Test contradictory input handling and uncertainty responses
- [ ] Create a small regression set of question/expected answer pairs

Pass if:
- Fact consistency >= 90% on regression set
- Contradictory/low-confidence inputs trigger cautious language

Evidence:
- Regression set:
- Evaluation notebook/script:
- Metrics:
- Notes:

---

## 0.6 Joint Runtime Feasibility (Critical)

Status: [ ]

- [ ] Run Gemma + TIPSv2 in target baseline environment (T4 16GB)
- [ ] Measure memory usage and per-query latency
- [ ] Test at least one fallback environment (e.g., 3090) if T4 is insufficient
- [ ] Document minimal viable deployment target

Pass if:
- No OOM in chosen target setup
- Median latency meets declared prototype threshold

Evidence:
- Hardware configs tested:
- Memory/latency table:
- Final deployment recommendation:

---

## Required Artifacts Before Go/No-Go

- [ ] Working Interface Notes section completed in this file
- [ ] notebooks or scripts reproducing 0.1-0.6 results
- [ ] Go/No-Go section completed in this file

---

## Working Interface Notes (Living)

Use this section as a living record while the interface is still unknown.

- Current direction(s) tested:
- What Gemma sends to tools:
- What TIPS-derived outputs return to Gemma:
- Open questions:
- Next experiment to reduce uncertainty:

---

## Go/No-Go Decision

Date:
Owner:

Critical checks:
- [ ] 0.1 Passed
- [ ] 0.2 Passed
- [ ] 0.3 Passed
- [ ] 0.4 Passed
- [ ] 0.5 Passed
- [ ] 0.6 Passed

Decision:
- [ ] GO to Phase 1
- [ ] NO-GO (pivot required)

If NO-GO, required pivot summary:
- Blocker:
- Proposed workaround/architecture change:
- Re-test scope:
- New decision date:
