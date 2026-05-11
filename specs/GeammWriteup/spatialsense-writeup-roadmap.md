# SpatialSense Writeup Roadmap

## Purpose

This is not the final writeup.

This is the working roadmap for producing the final hackathon writeup, demo story, and the last week of polish. It is grounded in:

- `specs/001-spatialsense-blind-nav/spec.md`
- `specs/001-spatialsense-blind-nav/tasks.md`
- `specs/GeammWriteup/Gemma Write up Ideas.md`
- `specs/GeammWriteup/edge_gallery_exploration.md`
- the current implementation in `app_blind.py`, `server/agent.py`, `pipeline/tools.py`, and related files

The main goal is to answer:

1. What should the final submission say?
2. What proof does the current code already give us?
3. What is still missing before the claim is submission-ready?

---

## Core Submission Thesis

This should be the center of the writeup.

### One-line version

SpatialSense uses Gemma 4 as a multimodal navigation agent: it takes spoken questions, sees the scene, localizes the relevant object, calls spatial tools, and returns grounded navigation guidance with metric distance, relative direction, and obstacle-aware cues.

### Slightly longer version

Most vision assistants can describe a room, but they cannot reliably answer the navigation question a blind user actually needs answered:

> what is in front of me, how far away is it, and how do I move safely?

SpatialSense combines:

- Gemma 4 for audio understanding, image reasoning, tool calling, and bounding-box localization
- TIPSv2 for depth and segmentation
- a blind-first audio-led interface for practical use

The result is not generic scene description, but grounded assistive navigation.

---

## What The Writeup Must Make Obvious

If the final submission works, judges should remember these points:

### 1. This is not just another vision chatbot

- It does not stop at scene description
- It measures distance
- It computes relative direction
- It can reason about obstacles in the path
- It explicitly handles uncertainty and missing visibility

### 2. Gemma 4 is central, not incidental

You should be explicit that Gemma 4 is doing real work here:

- spoken audio input goes directly into Gemma 4
- RGB image goes directly into Gemma 4
- Gemma 4 decides when to use tools
- Gemma 4 localizes the relevant object instance
- Gemma 4 synthesizes the final navigation response

### 3. The app is blind-first in interaction design

This is important because you already have more of this than a generic Gradio demo:

- audio-led interaction
- simplified action model
- large buttons
- mobile-style custom blind-first surface
- low-visual-complexity flow in `app_blind.py`

### 4. The technical challenge was real

Your strongest engineering story is not “I used a model.”

It is:

- plain VLMs are weak at grounded metric navigation
- depth alone is not enough
- language alone is not enough
- instance localization is the hard bridge between the two
- bounding-box-guided measurement is the core mechanism

---

## Recommended Final Writeup Structure

This is the structure I would use for the final submission.

## 1. Motivation

### What this section needs to say

- Blind users need more than visual description
- They need spatially actionable information
- Existing AI vision tools are often either:
  - descriptive but ungrounded
  - cloud-dependent
  - not designed around accessibility-first interaction

### Main bullets to hit

- Modern multimodal models are now good enough at audio + image understanding to support assistive perception
- But precise depth and distance estimation still require dedicated spatial models
- This creates an architectural opportunity: use Gemma 4 as the multimodal agent, and a spatial model as the grounding tool
- This is especially relevant for blind navigation, where vague answers are not enough

### Existing source material

- `Gemma Write up Ideas.md` already contains the cost/privacy/sufficiency framing
- `spec.md` already defines the navigation problem very tightly

### Gaps to close

- rewrite motivation so it is less “local AI in general” and more “why grounded navigation matters specifically”
- tighten the claim so it foregrounds the blind user need, not just the system architecture

---

## 2. Why Gemma 4

### What this section needs to say

This section should make clear that your project is an example of Gemma 4’s multimodal and agentic strengths working together.

### Main bullets to hit

- Gemma 4 accepts spoken audio questions directly
- Gemma 4 sees the original RGB scene directly
- Gemma 4 can call tools
- Gemma 4 has box-grounding / object-localization behavior that can be used to identify the instance to measure
- Gemma 4 turns grounded tool results back into natural language guidance

### The key phrasing

Use wording like:

> Gemma 4 handles multimodal understanding, tool orchestration, and instance localization, while the spatial toolchain supplies grounded metric measurement.

Avoid saying:

> Gemma 4 does depth estimation

because that blurs the architecture and weakens the technical clarity.

### Existing proof in code

- `server/agent.py`
  - multimodal messages
  - audio path
  - image path
  - tool schemas
  - tool loop
- `app_blind.py`
  - spoken interaction path
- notebook and debug work imply box-grounded interaction

### Gaps to close

- collect 1-2 screenshots or logs showing the actual tool-call chain
- prepare one simple diagram for this section

---

## 3. Problem Definition

### What this section needs to say

You need a short section that proves you scoped the problem correctly.

### Main bullets to hit

- The camera is a directional probe, not a 360-degree world model
- The system only speaks about what is in the current frame
- The goal is not room-level navigation or compass directions
- The goal is object distance, relative bearing, and scene-level navigation assistance

### Why this matters

This makes the project sound rigorous rather than hand-wavy.

### Existing proof in spec

- `spec.md` is very strong here
- especially:
  - frame-bounded claims
  - no room-level layout claims
  - uncertainty rules

### Gaps to close

- compress the spec language into 4-6 writeup bullets

---

## 4. Solution Overview

### What this section needs to say

This should explain the architecture simply, not like a paper.

### Recommended bullets

- User provides an image and a spoken question
- Gemma 4 receives both modalities
- Gemma determines whether a spatial tool is needed
- TIPSv2 provides depth and segmentation
- Gemma localizes the target object instance with a bounding box
- `measure_object` intersects that box with the relevant object pixels
- The system computes median distance and horizontal bearing
- Gemma turns those grounded values into a concise spoken navigation answer

### Existing proof in code

- `server/agent.py`
- `pipeline/tools.py`
- `pipeline/session.py`
- `pipeline/intrinsics.py`

### Gaps to close

- produce one clean architecture figure
- produce one end-to-end flow graphic

---

## 5. Key Technical Innovations

This section should probably be a numbered list.

## 5.1 Bounding-box-guided depth measurement

### What to say

- The hard problem is not just detecting an object class
- The hard problem is measuring the correct instance
- Your system uses Gemma’s localization ability to identify the instance, then computes depth only from intersected object pixels

### Why it matters

- avoids background contamination
- avoids naïve “class-level average distance”
- handles multiple same-class objects more honestly

### Code support

- `measure_object` in `pipeline/tools.py`

## 5.2 Bearing from image geometry, not vague language

### What to say

- Relative left/right direction is computed from centroid position and camera intrinsics
- EXIF intrinsics are used when available
- a 65° FOV fallback is used when absent

### Why it matters

- direction is grounded, not merely qualitative
- remains useful even when depth confidence drops

## 5.3 Zero-shot object path

### What to say

- The system is not limited to the closed ADE20K label set
- If the user names an out-of-vocabulary object, Gemma can take the zero-shot path

### Why it matters

- this is important for real navigation
- shopping carts, ladders, curbs, steps, etc. matter

## 5.4 Obstacle-aware navigation output

### What to say

- The system is not just answering “where is the object?”
- It can also identify intervening obstacles along the bearing corridor

### Why it matters

- much closer to a navigation assistant than a descriptive assistant

### Code support

- `_obstacles_in_corridor` in `pipeline/tools.py`

## 5.5 Audio-native multimodal loop

### What to say

- Spoken question goes directly to Gemma 4
- no mandatory separate STT pipeline
- this is part of the multimodal submission story

---

## 6. Development Journey / Challenges

This should be one of the strongest sections. It makes the project feel real.

Unlike the previous winner’s writeup, your challenge list should focus on the hard parts of grounded multimodal navigation.

## Recommended subsections

### 6.1 Getting reliable tool behavior from Gemma

- When should Gemma call `call_dpt_head`?
- How do we stop it from speaking ungrounded distances?
- How do we force measurement before claims?

### 6.2 Turning visual localization into real measurement

- Bounding boxes are the bridge
- They must map correctly into the resized TIPS tensor space
- They must isolate the correct object pixels

### 6.3 Handling multiple same-class objects

- Two chairs are not one chair
- Per-instance measurement matters
- Need to avoid collapsing to one blended answer

### 6.4 Avoiding hallucinated object substitutions

- If Gemma asks for something weakly grounded, the system must not confidently fabricate a measurement
- Your fallback/substitution path is powerful but needs careful honesty

### 6.5 Making direction robust

- Need real intrinsics when possible
- Need principled fallback when EXIF is absent
- Need natural language phrasing that remains useful

### 6.6 Building a blind-first interface instead of a developer demo

- Large actions
- audio-first flow
- reduced visible clutter
- custom HTML surface

### 6.7 On-device ambition vs. shipping reality

- Edge Gallery exploration is valuable
- image-context blocker is real
- current server path is the demo reality

### Existing source material

- `edge_gallery_exploration.md`
- `tasks.md`
- your current implementation

### Gaps to close

- convert these into short challenge/solution mini-sections
- avoid turning this into a diary
- keep each challenge tied to a concrete engineering decision

---

## 7. Blind-First UX

This section should exist because it stops judges from reading the project as “backend only.”

### What to say

- The interaction model is intentionally simplified
- Audio is the primary mode
- The UI avoids unnecessary visual complexity
- Large actions and reduced control count improve accessibility
- The goal is to reduce friction for a blind or low-vision user

### Existing proof

- `app_blind.py`
- `blind_first_phone_v2.html`

### What to show

- 2-3 screenshots
- one short explanation of the main interaction flow

### Gaps to close

- decide whether this is the primary demo surface
- if yes, ensure the writeup and video both use it consistently

---

## 8. Validation and Results

This section is essential because your spec is unusually measurable.

### What to include

- end-to-end scenarios validated
- gates already passed from `tasks.md`
- examples of:
  - direct object distance
  - multi-instance handling
  - OOV object detection
  - scene summary
  - obstacle-aware response

### Strongest proof already available

- G1–G5 are marked complete in `tasks.md`
- this is good writeup evidence even if the formal benchmark section is still incomplete

### Missing but high-value additions

- one compact latency table
- one compact qualitative eval table
- 3-5 screenshot/case study examples

### Important note

You do not need a giant benchmark section.

You need a compact, believable validation section that says:

- we tested the important behaviors
- the system is grounded
- the demo cases are not cherry-picked nonsense

---

## 9. Future Work

This section should be short and disciplined.

### Recommended bullets

- partial on-device deployment with Gemma on device and TIPS remote
- full on-device deployment if TIPS can be ported
- more embodied phone/camera capture flow
- broader outdoor validation
- stronger scene-to-stepwise navigation guidance

### Existing source material

- `edge_gallery_exploration.md`

### Caution

Do not let future work overshadow the current accomplishment.

---

## 10. Closing Positioning

End with something sharp.

### Recommended closing idea

Most multimodal assistants can describe what a camera sees.

SpatialSense is built to answer the navigation question that matters in practice:

> What is in front of me, how far away is it, and how do I move safely?

By combining Gemma 4’s multimodal and agentic capabilities with grounded spatial tools, SpatialSense moves from scene description toward true assistive navigation.

---

## Writeup Asset Checklist

These are the artifacts you should gather while polishing.

### Screenshots

- blind-first main screen
- spoken-question interaction
- grounded answer with distance + direction
- obstacle-aware answer
- zero-shot example

### Diagrams

- one architecture diagram
- one request flow diagram

### Evidence snippets

- one tool-call example
- one bounding-box measurement example
- one multi-instance example

### Demo examples

- “How far is the chair?”
- “What is around me?”
- “How do I get to the door?”
- one OOV example like shopping cart

---

## Submission Gaps Still Open

This is the concrete gap list after reading spec, tasks, and code.

## High priority

- Decide the primary demo surface:
  - `app_blind.py` should likely be the main demo path
- Finalize the writeup thesis around Gemma 4 multimodality + grounded navigation
- Produce asset-quality screenshots and diagrams
- Run a compact latency check and case-study validation set
- Finish aesthetics/demo polish task from `tasks.md`
- Prepare the video script around your strongest scenarios

## Medium priority

- Clean up stale wording in docs/research where the architecture changed
- add one compact observability/evidence story for debugging credibility
- ensure all claims in writeup match actual current behavior

## Lower priority

- exhaustive colormap experiments
- deep segmentation-threshold analysis
- broad T4 optimization work unless it is demo-critical

---

## One-Week Execution Roadmap

This is the practical roadmap for using the writeup as the driver of final polish.

## Day 1: Lock the narrative

### Output

- final section headings
- final thesis sentence
- final list of 4 demo scenarios

### Tasks

- write the opening paragraph
- write the “why Gemma 4” section in bullet form
- write the “what makes this different” section in bullet form
- decide the exact proof points for each claim

## Day 2: Lock the demo surface

### Output

- one primary interface path for screenshots and video

### Tasks

- commit to `app_blind.py` or equivalent blind-first flow as the hero surface
- remove or hide non-essential UI clutter
- tighten voice-first text and labels
- make the interaction path consistent

## Day 3: Build evidence assets

### Output

- screenshots
- diagrams
- tool-call examples

### Tasks

- capture 3-5 representative examples
- produce one architecture diagram
- produce one flow diagram
- collect one example of measurement grounding

## Day 4: Validation pass

### Output

- compact metrics/evidence section

### Tasks

- run demo scenarios repeatedly
- log latency roughly
- log failures and fixes
- write short case-study bullets

## Day 5: Draft full submission

### Output

- first complete writeup draft

### Tasks

- convert this roadmap into prose
- keep the submission skimmable
- avoid giant dense blocks
- use short subsections and bullets where useful

## Day 6: Demo and video script polish

### Output

- final demo script
- final video shot list

### Tasks

- align writeup order with video order
- ensure each demo scene proves a distinct claim
- cut anything redundant

## Day 7: Final consistency pass

### Output

- submission-ready materials

### Tasks

- verify claims against code
- verify screenshots against actual current UI
- verify demo against current reliability
- tighten language one last time

---

## Immediate Next Step

The next best move is not more freeform polishing.

It is to draft the actual final writeup skeleton using the structure above, then fill each section only with claims you can already prove from code, assets, or tests.

That will tell you exactly what remaining implementation polish is worth doing this week.
