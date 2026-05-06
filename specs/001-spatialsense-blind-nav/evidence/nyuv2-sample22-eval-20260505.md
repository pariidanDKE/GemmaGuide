# NYUv2 Sample 22 — Distance Accuracy Evaluation

- **Date:** 2026-05-05
- **Image:** `data/nyuv2_samples/images/0022.png` (NYUv2 val split, index 22)
- **GT depth:** `data/nyuv2_samples/depth/0022.npy` (Kinect, meters)
- **GT depth vis:** `data/media/debug_boxes/nyuv2_0022_gt_depth.png`
- **Model:** `gemma-4-e4b-it` via vLLM, fp8
- **TIPS:** `google/tipsv2-l14-dpt`, short-side 896px
- **Detected classes:** wall, floor, cabinet, table, chair, painting, desk, swivel, poster, screen

---

## Results

| # | Question | Object | TIPS (m) | GT median (m) | Error (m) | Abs Err % | Bearing | Notes |
|---|----------|--------|----------|---------------|-----------|-----------|---------|-------|
| 1 | "How far is the red swivel?" | swivel | 1.83 | 2.69 | −0.86 | 32.0% | −0.19° → straight ahead | ⚠️ TIPS underestimates chair depth |
| 2 | "How far is the white board?" | wall (proxy) | 3.10 | 3.08 | +0.02 | 0.6% | −18.2° left | ✅ OOV — whiteboard not in ADE20K, Gemma correctly proxied via wall |

---

## Agent Responses

**Q1 — red swivel chair**
> "I see the red swivel chair right in front of you. It is about 1.8 meters straight ahead."

Rerun:
`python scripts/run_agent_loop.py --nyuv2-sample 22 --question "How far is the red swivel?" --print-tool-io`

**Q2 — white board**
> "The white board is about 3.1 meters away, slightly to your left. It looks like it's on the wall. Would you like me to tell you about the desk or the chairs?"

Rerun:
`python scripts/run_agent_loop.py --nyuv2-sample 22 --question "How far is the white board?" --print-tool-io`

---

## Latency

| Run | Total (s) | Rounds |
|-----|-----------|--------|
| Q1 swivel | 7.01 | 3 |
| Q2 whiteboard | 11.22 | 3 |

---

## Observations

- **Bearing fix (Q1):** −0.19° correctly reported as "straight ahead" — the 10° dead zone is working.
- **Swivel depth error (Q1):** −0.86m (32%). Consistent with sample 4 bookcase — TIPS systematically underestimates depth on chairs and soft-textured objects. May be related to the chair seat absorbing IR in the Kinect GT, or TIPS treating the chair back as closer than its geometric center.
- **OOV whiteboard (Q2):** "whiteboard" is not an ADE20K class and not in detected_classes. Gemma correctly identified it as wall-mounted and called `measure_object` with `class_name: "wall"`. The wall proxy gave +0.02m error — essentially perfect. This is a clean example of the fallback working for US2-style OOV queries without needing `call_encoder_zero_shot`.
- **Whiteboard latency (Q2):** 11.2s vs 7.0s for Q1. The extra 5.5s in round 1 LLM latency suggests Gemma spent more time reasoning about the OOV class before deciding on the wall proxy.

## Open Questions

- Is the swivel/chair underestimate systematic? Worth testing on more chair samples to confirm.
- Should OOV objects that are clearly wall-mounted (whiteboard, painting, poster) be guided to use wall as a proxy in the system prompt, or is Gemma's emergent behaviour reliable enough?
