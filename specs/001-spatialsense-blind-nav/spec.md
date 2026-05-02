# Feature Specification: SpatialSense — Blind Navigation Assistant

**Feature Branch**: `001-spatialsense-blind-nav`
**Created**: 2026-04-30
**Status**: Draft
**Hackathon**: Gemma 4 Good — Kaggle × Google DeepMind (Deadline: 2026-05-18)
**Target Tracks**: Digital Equity & Inclusivity + Main Track

---

## Overview

SpatialSense is a voice-driven assistive tool that lets blind and visually impaired people ask natural language questions about their immediate environment and receive spoken answers containing metric distances and aim-correction hints. A user provides a camera image of their environment, asks "how far is the table from me?" via voice, and hears "The table is approximately 1.4 meters away, 5 degrees to your right. There's also a chair at about 0.9 meters — 22 degrees to your left."

The camera is a directional probe, not a 360° sensor. Everything in the frame is already in front of the user. The system's spatial output tells the user how far something is and the horizontal bearing in degrees to each identified object — it does not report compass directions or room-level layout.

The system combines a spatial understanding model (TIPSv2, which produces metric depth and semantic segmentation from a single camera frame) with a conversational AI (Gemma 4 4B-IT, which orchestrates spatial queries and generates natural language descriptions). Neither model alone can solve the problem — TIPSv2 provides metric depth but no language; Gemma provides language but cannot measure distances from a single image without spatial grounding.

Phase 0 feasibility work is complete and signed off (GO). All six Phase 0 checks passed on 2026-04-30. This specification covers the full project from Phase 1 build through hackathon submission.

---

## Clarifications

### Session 2026-05-02

- Q: What is the v1 data retention policy for user image/audio inputs? → A: Do not persist raw user image/audio; process in-memory per session and release immediately after response generation.
- Q: What observability data should be retained in v1? → A: Keep observability lightweight in v1; optional debug metadata may be emitted, but never store raw image/audio or question text.
- Q: How should v1 handle missing, invalid, or inaudible microphone input? → A: Return a spoken + text retry prompt and skip spatial inference for that turn.
- Q: What should be the v1 scene-summary verbosity policy? → A: Include up to 5 objects and cap model response output to 256 tokens.
- Q: How explicit should v1 be about phone-to-local-server access requirements? → A: Keep phone/LAN access as an informal stretch only; no formal v1 acceptance criteria.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Ask About a Specific Object's Distance (Priority: P1)

A blind user provides a camera image of their environment and speaks a question about a specific object: "How far away is the door?" or "Is there anything in front of me?" The system analyzes the camera frame, identifies the object, measures its metric distance, and tells the user the horizontal bearing in degrees to the object — then reads the response aloud.

**Why this priority**: This is the core use case — the single interaction that defines whether the product is useful for navigation. Everything else is secondary to this working reliably.

**Independent Test**: Can be tested by uploading a photo of a known scene and speaking a question via microphone — the system must accept and process the audio clip and return both a text transcript and a spoken response.

**Acceptance Scenarios**:

1. **Given** a camera frame of an indoor scene with a clearly visible table, **When** the user asks "how far is the table?", **Then** the system responds with the table's approximate distance in meters and a horizontal bearing in degrees (e.g., "directly ahead", "22 degrees to your left", "8 degrees to your right").
2. **Given** a query about an object that is present in the frame, **When** the system responds, **Then** the response uses plain language a non-technical person can understand (no raw numbers without units, no technical jargon).
3. **Given** a query about an object NOT visible in the frame, **When** the system processes the image, **Then** the system explicitly says the object is not visible in the current view rather than fabricating a position.
4. **Given** two instances of the same object class (e.g., two chairs), **When** the system responds, **Then** it reports the nearest instance's distance and aim correction, and notes that another instance exists with its own bearing (e.g., "there's also one 20 degrees to your right").

---

### User Story 2 — Ask About Objects Not in Standard Categories (Priority: P2)

A blind user asks about a navigation-critical object that may not be in the standard 150-object recognition vocabulary — for example, a shopping cart, a step, a curb, or a puddle. The system attempts zero-shot recognition using descriptive language and reports the object's presence and distance if found.

**Why this priority**: Navigating public spaces requires detecting objects that are not common indoor furniture. The zero-shot capability is what separates this from a system trained only on standard categories.

**Independent Test**: Can be tested by providing a photo containing an out-of-vocabulary object and speaking its name as part of the question. Delivers value by showing extensibility without retraining.

**Acceptance Scenarios**:

1. **Given** a frame containing a shopping cart (not in standard 150-class vocabulary), **When** the user asks "is there a shopping cart in front of me?", **Then** the system correctly identifies its presence or absence and reports an approximate distance if found.
2. **Given** an object name the system cannot confidently locate, **When** the system responds, **Then** it expresses low confidence or says it cannot confirm the object's presence — it does not hallucinate a confident answer.

---

### User Story 3 — Get a General Scene Description (Priority: P3)

A blind user wants situational awareness about their surroundings without asking about a specific object: "What's around me?" or "Describe what's in front of me." The system identifies the most prominent objects, their approximate distances, and their spatial positions, then delivers a structured spoken summary.

**Why this priority**: Specific object queries require the user to already know what to ask about. A general scene description gives situational awareness in unfamiliar environments.

**Independent Test**: Can be tested by uploading a photo and speaking "What do you see?" via microphone — the system must process both the image and audio clip together and return a multi-object spoken summary.

**Acceptance Scenarios**:

1. **Given** a scene with several distinct objects (furniture, doors, walls), **When** the user asks "what's around me?", **Then** the system names up to 5 prominent objects with distances and horizontal bearings in degrees relative to camera center.
2. **Given** a scene description response, **When** the user listens to it, **Then** the description is ordered closest to farthest and makes clear that all described objects are within the current camera view, not room-wide positions.

---

### User Story 4 — Hackathon Demo: Photo Upload + Spoken Question (Priority: P4)

A hackathon judge or demo viewer uploads a photo from their computer, speaks a question via microphone, and receives both a displayed text response and a spoken audio answer. The demo interface is simple enough that a non-technical person can complete an interaction within 30 seconds without instructions.

**Why this priority**: Demo success directly affects hackathon placement and prize eligibility. The audio-in + audio-out flow must be demonstrable end-to-end and showcases Gemma 4's full multimodal capability (image + audio simultaneously).

**Independent Test**: Can be tested by giving the interface to a person who has never seen it and measuring whether they upload a photo, speak a question, and receive a spoken answer — all without help.

**Acceptance Scenarios**:

1. **Given** a web demo interface, **When** a user uploads a photo and speaks a question via microphone, **Then** the system processes both the image and the audio clip together and returns a text + spoken response with metric distances within 30 seconds.
2. **Given** a spoken question that is ambiguous or partially unclear, **When** the system processes the audio, **Then** it responds with its best interpretation and expresses any uncertainty in the answer rather than failing silently.

---

### Edge Cases

- What happens when the camera image is too dark, blurry, or low-resolution for reliable spatial analysis?
- How does the system handle outdoor scenes where depth accuracy degrades beyond 8–10 meters?
- What if the user's audio question is inaudible or has significant background noise?
- What happens when the queried object is extremely close (under 0.5 meters) or at the edge of the depth model's reliable range?
- How does the system behave when multiple objects overlap spatially (e.g., a bag on a table)?
- What if the user asks about an abstract concept ("is it safe to walk here?") that cannot be answered with depth + segmentation alone?
- What if the user asks "where is the exit?" expecting a room-level compass answer — the system can only report what is visible in the current frame and must communicate this limitation clearly.

---

## Requirements *(mandatory)*

### Functional Requirements

**Input Handling**

- **FR-001**: System MUST accept an uploaded camera image as the primary spatial input for every query. Live camera capture is an optional stretch capability and is not required for v1.
- **FR-002**: System MUST accept the user's question as spoken audio. Typed text input MUST also be supported as a secondary path for testing and development purposes.
- **FR-003**: System MUST support still-image input (single frame); continuous video streaming is out of scope for v1.

**Spatial Analysis**

- **FR-004**: System MUST identify named objects in the scene by semantic category and report their approximate metric distance from the camera in meters.
- **FR-005**: System MUST compute a horizontal bearing angle for each identified object and include it in every spatial response. The pixel centroid of the segmented object within the bounding box, combined with camera intrinsics, yields a signed bearing in degrees — negative = left of camera center, positive = right. Gemma reports this as natural language (e.g., "22 degrees to your left", "8 degrees to your right", "directly ahead" when within ±5°). Camera intrinsics (focal length, principal point) are extracted from image EXIF at session creation; when EXIF is absent or stripped (e.g., PNG input), the system falls back to assuming a 65° horizontal field of view. The intrinsics source (`"exif"` or `"assumed"`) is recorded in the session. Bearing is depth-independent — it depends only on pixel position and focal length, so it remains reliable even when depth confidence is low.
- **FR-006**: System MUST handle objects within the 150-class standard vocabulary using the primary spatial analysis path.
- **FR-007**: System MUST attempt to identify objects outside the 150-class vocabulary using a zero-shot recognition path when the user explicitly names an out-of-vocabulary object.
- **FR-008**: System MUST cache the depth map and segmentation mask from each spatial analysis run. When the conversational AI calls the measurement tool with a bounding box identifying a specific object instance, the system intersects the cached segmentation mask with that bounding box so that depth is sampled only from the target object's own pixels — not from background or other objects inside the box. This separation — AI identifies the instance via bounding box, system computes depth from clean object pixels — produces reliable metric distances without the AI misreading color scales. The same pixel intersection is used to compute the object centroid, which combined with session-cached camera intrinsics yields the horizontal bearing angle.

**Response Generation**

- **FR-009**: System MUST respond in natural language that a non-technical person can understand without domain knowledge.
- **FR-010**: System MUST include metric distance in meters and a horizontal bearing in degrees in every spatial response. The response MUST use camera-relative, degree-qualified phrasing (e.g., "22 degrees to your left", "8 degrees to your right", "directly ahead" when bearing is within ±5°) and MUST NOT use compass directions or room-level layout claims.
- **FR-011**: System MUST express uncertainty when depth confidence is low — specifically when estimated distance is outside the reliable navigation range (0.5–8.0 m), when depth values are at or near the model bounds (1e-3 to 10.0 m), or when segmentation confidence is low.
- **FR-012**: System MUST acknowledge when a queried object is not visible in the frame rather than fabricating a position.
- **FR-013**: System MUST provide a spoken audio output of the response for every query (not text-only).
- **FR-017**: If microphone audio input is missing, invalid, or inaudible, the system MUST return a spoken and text fallback prompt asking the user to retry, and MUST NOT run spatial inference for that turn.
- **FR-018**: For general scene-description responses, the system MUST report up to 5 objects ordered from closest to farthest and constrain Gemma response generation to a maximum of 256 output tokens.

**Demo Interface**

- **FR-014**: System MUST provide a web-accessible demo interface that supports photo upload, spoken audio question input (microphone), and response display with both text and audio output.
- **FR-015**: System MUST process user-provided image and audio inputs in-memory for the active session only and MUST release raw media buffers immediately after response generation. Raw user image/audio MUST NOT be persisted to disk or long-term storage in v1.
- **FR-016**: System MAY emit lightweight, non-sensitive debug metadata for observability (`timestamp`, latency, model/tool status, confidence, error class). The system MUST NOT persist raw image/audio or raw question text in logs.

### Key Entities

- **Scene**: A single camera frame representing the user's immediate physical environment at query time.
- **Query**: A spoken audio question paired with a Scene; the primary unit of user interaction. Typed text is also accepted as a secondary testing path.
- **Spatial Report**: The system's structured natural language response containing identified objects, their metric distances, and horizontal bearing angles relative to camera center.
- **Object Observation**: An entry within a Spatial Report identifying one object: its semantic label, approximate distance in meters, horizontal bearing in degrees (signed: negative = left, positive = right), and confidence level.
- **Depth Map**: The per-pixel metric depth representation derived from a Scene; cached for the duration of a query session to enable precise metric lookups.
- **Camera Intrinsics**: The focal length and principal point of the camera that captured the scene, derived from image EXIF metadata (35mm equivalent focal length) or estimated from a 65° horizontal FOV fallback when EXIF is absent. Extracted once at session creation. Used to convert pixel centroids into bearing angles.
- **Session**: A single query lifecycle — one image + one question → one Spatial Report; state does not persist between sessions in v1.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can receive a spoken spatial answer (object + distance + bearing in degrees) within 30 seconds of submitting a photo and a spoken audio question.
- **SC-002**: Reported distances for objects within the 0.5–8 meter range are within ±20% of ground-truth distance on at least 5 diverse test scenes.
- **SC-003**: The system correctly identifies and reports on the queried object in at least 8 out of 10 test queries where the object is visibly present in the frame.
- **SC-004**: A person with no prior knowledge of the system can successfully complete a question-and-answer interaction via the demo interface within 30 seconds and without assistance.
- **SC-005**: The system never exposes a raw error, stack trace, or technical message to the user — every failure produces a natural language fallback response.
- **SC-006**: The system correctly handles at least 3 zero-shot object queries (objects outside the 150-class standard vocabulary) during the demo regression test.
- **SC-007**: The system runs end-to-end on the T4 16GB baseline hardware target without running out of memory.
- **SC-008**: The demo interface is accessible from a standard web browser and supports photo upload and microphone audio input without installing a native app.
- **SC-009**: In v1, raw user image/audio artifacts are not retained after completion of each query session (in-memory only, no persistent storage).
- **SC-010**: If observability logs are enabled in v1, they contain metadata only (no raw image/audio/question text).
- **SC-011**: For missing, invalid, or inaudible microphone input, the system always returns a retry fallback response (text + spoken) and produces no object-distance claims for that turn.
- **SC-012**: For general scene-description queries, responses include no more than 5 objects and remain within a 256-token model output limit.

---

## Assumptions

- The primary deployment context for v1 is indoors (homes, offices, supermarkets). Outdoor performance is expected to degrade beyond 8–10 meters and will be disclosed as a known limitation.
- A single still frame is sufficient for v1 navigation queries; real-time continuous video tracking is a future enhancement.
- Compute target is a cloud-hosted GPU (T4 16GB baseline; RTX 3090 as confirmed fallback). On-device mobile inference is a future goal, not a v1 requirement.
- Audio input is delivered directly to the language model without a separate speech-to-text preprocessing step (confirmed working in Phase 0.3).
- The demo UI runs as a Gradio web application accessible from a standard desktop/laptop browser. Mobile browser camera capture and phone-to-local-server usage are informal stretch goals for post-v1 and are not part of formal v1 acceptance criteria; v1 requires image file upload.
- Text-to-speech for the spoken response is a required feature for v1 but the specific TTS provider (cloud-hosted or local) may be selected during implementation based on latency and quality evaluation.
- Instance-level discrimination (distinguishing Chair A from Chair B) is not required; per-instance bearing angles (e.g., "one 5 degrees to your right, another 22 degrees to your left") are the v1 approach for handling multiple instances of the same class.
- The system does not provide room-level or compass navigation. All spatial references are relative to the camera's current pointing direction. Users must repoint the camera and re-query to explore different parts of their environment.
- The system is a hackathon prototype. Production-grade authentication and rate limiting are out of scope. For v1 privacy posture, raw user image/audio is handled in-memory only and not persisted; observability is lightweight and metadata-only when enabled.
- All six Phase 0 feasibility checks passed as of 2026-04-30; Phase 1 build begins immediately.
