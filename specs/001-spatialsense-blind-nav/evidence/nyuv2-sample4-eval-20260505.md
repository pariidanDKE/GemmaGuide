# NYUv2 Sample 4 — Distance Accuracy Evaluation

- **Date:** 2026-05-05
- **Image:** `data/nyuv2_samples/images/0004.png` (NYUv2 val split, index 4)
- **GT depth:** `data/nyuv2_samples/depth/0004.npy` (Kinect, meters)
- **GT depth vis:** `data/media/debug_boxes/nyuv2_0004_gt_depth.png`
- **Model:** `gemma-4-e4b-it` via vLLM, fp8
- **TIPS:** `google/tipsv2-l14-dpt`, short-side 896px
- **Intrinsics:** NYUv2 Kinect (fx=230.4, fy=311.7, cx=195.3, cy=152.2 at 384×288)

---

## Results

| # | Question | Object | TIPS (m) | GT median (m) | Error (m) | Abs Err % | Bearing | Tool flow | Notes |
|---|----------|--------|----------|---------------|-----------|-----------|---------|-----------|-------|
| 1 | "How far is the door?" | door | 2.18 | 2.13 | +0.05 | 2.3% | +19.8° right | dpt_head → measure | ✅ Clean |
| 2 | "How far is the bookshelf?" | bookcase | 3.63 | 4.48 | −0.85 | 19.0% | −28.4° left | dpt_head → measure | ⚠️ Underestimate |
| 3 | "How far is the painting?" | painting | 2.61 | 4.45 | −1.83 | 41.1% | +8.6° right | dpt_head → measure | ❌ Large error — small object (1009px), GT box dominated by wall depth |

---

## Agent Responses

**Q1 — door**
> "The door is about 2.2 meters away, positioned about 20 degrees to your right. Be careful as you move toward it."

Rerun:
`python scripts/run_agent_loop.py --nyuv2-sample 4 --question "How far is the door?" --print-tool-io`

**Q2 — bookshelf**
> "The bookshelf is about 3.6 meters away and is to your left, angled at about 28 degrees."

Rerun:
`python scripts/run_agent_loop.py --nyuv2-sample 4 --question "How far is the bookshelf?" --print-tool-io`

**Q3 — painting**
> "The painting is visible on the right wall, about 2.6 meters away, slightly to your right."

Rerun:
`python scripts/run_agent_loop.py --nyuv2-sample 4 --question "How far is the painting?" --print-tool-io`

---

## Latency

| Run | Total (s) | Rounds |
|-----|-----------|--------|
| Q1 door | 8.26 | 3 |
| Q2 bookshelf | 7.28 | 3 |
| Q3 painting | 7.54 | 3 |

---

## Observations

- **G2 gate (door):** TIPS depth error of +0.05m (2.3%) on the door — excellent. Door is a large, flat, well-lit surface which is exactly the favourable case for monocular depth.
- **G2 gate (bookcase):** 19% error. Bookcase is a textured, recessed object — TIPS tends to underestimate depth on cluttered surfaces.
- **Painting accuracy (Q3):** 41% error. The painting has very few TIPS-segmented pixels (1009), and the GT box median is dominated by the surrounding wall at ~4.5m. This is a known failure mode for small objects: the bounding box that Gemma draws is much larger than the object, so the GT median picks up background depth rather than the painting itself. The TIPS measurement has the same problem.
- **Bearing:** Consistent and directionally correct across all runs. Door correctly identified as right (+20°), bookcase as left (−28°).

## Open Questions

- Is the painting GT error a TIPS accuracy issue or a GT alignment issue (small object, bounding box includes wall)?
- Should `measure_object` use only class-segmented pixels for GT comparison (not the full bounding box)?
