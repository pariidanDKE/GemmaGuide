# On-Device Deployment: Edge Gallery Exploration, Blockers, and Roadmap

## Motivation

The server-based demo validates the end-to-end pipeline, but the original vision for this project is on-device inference. Three problems make local deployment specifically compelling for an assistive tool aimed at visually impaired users.

**Cost.** Frontier model APIs carry subscription costs that, for a disabled user who depends on the tool as a functional necessity rather than a productivity enhancement, represent a real access barrier — not a minor inconvenience.

**Privacy.** Assistive vision applications operate inside users' homes and in deeply personal contexts. Many users will reasonably object to streaming that imagery to third-party cloud providers. A locally-hosted model eliminates this concern entirely.

**Sufficiency.** The LLM's role in this architecture is deliberately well-scoped: understand audio input, reason over a compact structured scene description, and guide the user spatially. It does not need long-horizon reasoning or encyclopedic knowledge. This is a task that small, edge-optimized models can perform today — Gemma 4 E2B and E4B being prime examples.

These three threads converge on Google AI Edge Gallery as the most natural deployment target: it runs Gemma 4 fully on-device, supports custom agent skills with tool calling, and is available on both Android and iOS.

---

## What We Explored

### Agent Skills architecture

Skills in Edge Gallery are defined by a `SKILL.md` file containing LLM instructions, optionally paired with a JavaScript execution environment (`scripts/index.html`) running inside a hidden mobile webview. The JS environment has full access to the web ecosystem: `fetch()` calls to external servers, CDN libraries, WebAssembly, and standard Web APIs. Skills are loaded into the app via URL or local file — no app store submission required.

When a skill is active, its tool definitions are appended to Gemma's system prompt. When Gemma decides to invoke a tool, it emits a structured function call which the app routes to the skill's `ai_edge_gallery_get_result(data)` function. The skill executes, returns a JSON result, and Gemma continues reasoning over it.

This is a capable, well-designed system for text-grounded tool use.

### Validating multimodal input in skills mode

A key unknown was whether Gemma could see an attached image while in Agent Skills mode. The two features — Ask Image and Agent Skills — are presented as separate tiles in the app, raising the concern that they were mutually exclusive contexts.

Testing confirmed this is **not** the case. An image attached in the Agent Skills chat is visible to Gemma. In a test, attaching a screenshot of a Wikipedia article and asking Gemma to "search for this guy" caused it to extract the person's name from the image and correctly invoke the Wikipedia skill with that name as the search parameter. Gemma successfully used visual information to parameterize a tool call.

This was the critical validation. It means Gemma can, in principle, look at a camera image, identify an object instance, draw a bounding box around it, and pass those pixel coordinates as a tool call parameter — exactly what `measure_object` requires.

### Proposed skill mapping

The four tools in the current server-based architecture map naturally to four skills:

| Tool | Skill behavior |
|---|---|
| `search_seg_classes` | Pure text lookup. Receives a query string, returns matching ADE20K class names. Can be a hardcoded JS lookup with no server call. |
| `call_dpt_head` | Captures camera frame, POSTs to TIPSv2 server, returns detected classes with cached depth/segmentation. |
| `call_encoder_zero_shot` | Same as above, additionally receives a `class_list` from Gemma for zero-shot object classes outside the standard vocabulary. |
| `measure_object` | Receives `class_name` + `box_2d` from Gemma. POSTs to TIPSv2 which uses the server-side cached depth/segmentation. No image capture needed at this step. |

---

## The Hard Blocker

### Image context is not shared with the skill

The architecture depends on a precise chain:

```
Gemma sees image → Gemma draws bounding box → box passed to measure_object → TIPSv2 uses cached depth
```

The bounding box is how the system disambiguates between multiple instances of the same class — two chairs at different distances, for example. Without it, depth measurement is undefined. Removing it is not a simplification; it breaks the core function.

The blocker is this: **the image Gemma is analyzing is not accessible inside the skill's JS execution context.**

The skill's `ai_edge_gallery_get_result(data)` function receives only the parameters Gemma explicitly emits in the tool call — strings, numbers, arrays. Gemma can pass a bounding box, but it cannot pass the image it derived that box from. The skill's webview has no reference to the chat image.

An alternative — having the skill capture its own camera frame via `getUserMedia()` — does not solve this. That produces a different image from a different moment in time. Gemma's bounding box coordinates, derived from the chat-attached image, are invalid against a separately-captured frame. The pixel coordinates simply do not correspond.

### Why simpler workarounds fail

**Pre-computing all instances server-side.** Running TIPSv2 before Gemma and returning all detected instances with distances reintroduces the disambiguation problem. If there are two chairs, which distance does Gemma report? Without visual grounding there is no principled answer.

**Passing image as base64 in the tool call.** Gemma would need to emit the full image as a parameter, which is architecturally unsound — the image already exists in the chat context, and LLMs do not serialize their visual inputs into tool parameters.

**Splitting into two turns.** Having the user send the image separately to a dedicated endpoint before asking the question introduces friction that is unacceptable for a blind user relying on voice interaction.

---

## What Would Fix It

The missing capability is narrow and well-defined:

> **Edge Gallery should pass the current chat image into the skill's JS execution context alongside the tool call parameters.**

Concretely, this means the `ai_edge_gallery_get_result(data)` function receiving an additional argument — or a field within `data` — containing the image as a base64-encoded string when an image is present in the current chat turn.

The app already has the image in memory at the point the skill is invoked. The change is a matter of threading it throughh to te webview. Gemma's bounding box and the skill's image would then refer to the same frame, and `measure_object` works correctly.

This is not an edge case specific to this project. Any skill that requires visual grounding to parameterize a tool call hits this same wall. Assistive navigation simply makes the gap unusually visible because the bounding box is not optional — it is the mechanism.

---

## Roadmap

### Phase 1 — Current state (complete)
Full pipeline running on a PC server. Gemma 4 4B-IT receives the RGB image and audio question, calls four tools backed by TIPSv2, returns metric distances and bearings as spoken navigation guidance. The architecture is validated end-to-end.

### Phase 2 — Partial on-device (unblocked today)
Gemma 4 E2B or E4B running fully on-device inside Edge Gallery. TIPSv2 remains as a remote service. This resolves the cost and partial privacy concerns — Gemma inference is local and the user's conversational context never leaves the device. Only the camera frame is sent to TIPSv2 for depth processing, which is a much narrower data exposure than sending everything to a frontier model API.

This phase is unblocked pending the Edge Gallery feature described above. A contribution to the gallery's open-source repository to pass the chat image into skill context would enable this immediately.

### Phase 3 — Full on-device (future work)
TIPSv2 ported to run on-device via LiteRT (TensorFlow Lite). TIPSv2 is already a lightweight model by design — this is a feasible but non-trivial engineering task. At this stage the pipeline runs entirely offline with no data leaving the device, fulfilling the original privacy and accessibility vision completely.

---

## Summary

Edge Gallery is the right deployment target for this project. The infrastructure — on-device Gemma 4, agent skills, tool calling, multimodal input — is all present and functional. The single blocking gap is that the chat image is not forwarded into the skill's JS context at tool call time. Filing this as a feature request, or contributing the change directly to the open-source gallery repository, is the most direct path to a fully on-device assistive navigation tool.
