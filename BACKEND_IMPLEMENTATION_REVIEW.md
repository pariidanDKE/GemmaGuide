# Backend Implementation Review

## Scope

This review covers the non-frontend implementation in:

- `app_blind.py`
- `app.py` as the legacy/alternate Gradio entrypoint
- `server/agent.py`
- `server/schemas.py`
- `pipeline/session.py`
- `pipeline/intrinsics.py`
- `pipeline/tips_runner.py`
- `pipeline/tools.py`
- `pipeline/tts.py`
- `scripts/start_gemma4.sh`
- `scripts/run_agent_loop.py`
- `scripts/run_bbox_only.py`
- `scripts/fetch_nyuv2_samples.py`

This is intentionally implementation-focused, not product- or UX-focused.

## High-Level Backend Shape

There are effectively two server entrypoints:

1. `app_blind.py`
   - Main FastAPI app serving the custom blind-first HTML UI and `/api/query`.
   - Implements the current production-style orchestration path:
     - ingest image/audio
     - create a `Session`
     - run `Scout`
     - optionally run `Mapper`
     - optionally render annotations
     - run `Navigator`
     - synthesize TTS
     - persist lightweight session state in `_sessions`

2. `app.py`
   - Older Gradio entrypoint.
   - Uses a simpler single-loop `run_agent_loop()` flow without Scout/Mapper/Navigator separation.

The core computation splits into three backend layers:

- `server/agent.py`
  - LLM prompting, multi-agent routing, tool-call dispatch, message construction, and vLLM API calls.
- `pipeline/tips_runner.py`
  - TIPSv2 model loading and raw inference for depth/segmentation/zero-shot maps.
- `pipeline/tools.py`
  - Tool implementations that transform TIPS outputs into model-callable operations such as `call_dpt_head`, `measure_object`, and `call_encoder_zero_shot`.

`pipeline/session.py` is the shared mutable state object passed across those layers.

## Main Request Flow

### `app_blind.py`

#### Global state

- `_sessions: dict[str, dict]`
  - In-memory session store for:
    - shared conversation history
    - previous measurements
    - last image
    - aggregated metrics
  - This is the only persistence layer for conversational continuity.

#### `_image_to_jpeg_b64(image, resize_to=None)`

- Converts a PIL image to RGB JPEG and returns base64.
- Used for debug payloads and returning depth/annotated images to the browser.

#### `_history_for_debug(history)`

- Converts mixed multimodal history into preview-friendly debug records.
- Rewrites content blocks into readable placeholders such as `[image]` and `[audio]`.

#### `_strip_markdown(text)`

- Removes lightweight markdown before TTS.
- Duplicated with nearly identical logic in `app.py`.

#### `_summarize_metrics(summary, counts, limit=10)`

- Ranks metric stages by total time and returns the top stages.
- Pure response-shaping helper.

#### `_merge_metrics(existing, turn_metrics)`

- Aggregates current turn metrics into session-level totals and keeps per-turn snapshots.
- Behavior is correct but the structure is fully ad hoc dict-based.

#### `blind_ui()`

- Reads `designs/gemma_guide.html` and returns it.
- Thin static page endpoint.

#### `api_query(image, audio, session_id)`

This is the real backend controller and the most important function in the app.

Responsibilities:

- Decode uploaded image.
- Convert uploaded audio into mono 16-bit WAV bytes.
- Recover prior session state from `_sessions`.
- Determine whether the current turn should include the image in LLM context.
- Run the Scout route classifier.
- If Scout selects spatial reasoning:
  - create a second `Session`
  - run `Mapper`
  - render annotations
  - run `Navigator`
- Otherwise answer from Scout directly.
- Merge timings into session metrics.
- Persist stripped session state back into `_sessions`.
- Run TTS and return JSON response with:
  - `response`
  - `route`
  - `audio_b64`
  - `depth_b64`
  - `metrics`
  - `debug`

Behavioral detail:

- The default question is `"What do you see around me?"` when no audio is present.
- Spatial requests without an active image are forced back to a direct response path.
- Measurement tensors are stripped before session persistence by dropping `mask_dpt`.

Implementation critique:

- This function is too large and too stateful.
- It mixes transport concerns, session persistence, route control, metrics, TTS, and debug payload generation.
- It uses `StopIteration` as control flow, which is a strong smell for application code.
- There is duplicated turn-finalization logic between the normal and forced-direct branches.

#### `clear_session(session_id)`

- Removes a session from `_sessions`.

#### `_patch_httpx_localhost_verify()`

- Monkey-patches `httpx.get` to disable TLS verification for localhost-like hosts.
- Duplicated in `app.py`.

#### `_env_flag(name, default=False)`

- Small environment boolean parser.

#### `_run_server()`

- Reads host, port, SSL env vars.
- Falls back to hard-coded Tailscale certificate paths unless `APP_DISABLE_SSL` is set.
- Starts `uvicorn`.

### `app.py` (legacy Gradio path)

This file is simpler but still useful to understand because it reuses the same backend core.

#### `_numpy_to_wav_bytes(audio_np)`

- Converts Gradio numpy audio input into WAV bytes.

#### `_strip_markdown(text)`

- Same TTS-oriented markdown stripping as `app_blind.py`.

#### `_tts_to_numpy(text)`

- Calls `pipeline.tts.synthesize()`.
- Decodes MP3 bytes using `pydub` and returns `(sample_rate, samples)` for Gradio playback.

#### `_image_hash(image)`

- MD5 hash of raw image bytes.
- Used to detect whether a new image was supplied between turns.

#### `process(image, audio_input, text_question, history, last_image_hash)`

- Validates input.
- Rejects very dark images.
- Chooses audio bytes over text if available.
- Detects whether the image changed and whether to resend it to the LLM.
- Creates a `Session` and runs the older `run_agent_loop()` path.
- Returns response text, synthesized speech, depth display, updated history, and image hash.

#### `process_gen(...)`

- Generator wrapper to emit an immediate “thinking” placeholder before actual work completes.

## Shared Session Layer

### `pipeline/session.py`

#### `Session` dataclass

Fields:

- `session_id`
- `image`
- `question`
- `intrinsics`
- `depth_tensor`
- `seg_mask`
- `depth_colormap`
- `spatial_report`
- `measurements`
- `metrics`

This is the mutable working state object shared across the whole pipeline.

#### `Session.add_timing(stage, seconds, **meta)`

- Adds detailed timing entries and also maintains per-stage totals and counts.

#### `Session.export_metrics()`

- Returns shallow copies of metric collections for response emission.

#### `Session.release()`

- Clears `depth_tensor` and `seg_mask`.
- Does not clear all heavy fields such as `depth_colormap` or dynamically attached fields like `_zero_shot_maps`.

#### `create_session(image, question, intrinsics=None, metrics=None)`

- Extracts camera intrinsics from EXIF if not supplied.
- Applies EXIF orientation.
- Converts image to RGB.
- Creates a `Session`.

Implementation critique:

- `Session` is the right abstraction, but it is under-typed.
- Important ephemeral fields are added outside the dataclass, for example `_zero_shot_maps`.
- That weakens invariants and makes lifecycle management harder.

## Camera Model Layer

### `pipeline/intrinsics.py`

#### `CameraIntrinsics`

- Simple dataclass for `fx`, `fy`, `cx`, `cy`, and `source`.

#### `nyuv2_intrinsics(width, height)`

- Returns scaled Kinect intrinsics for NYUv2 evaluation runs.

#### `extract_intrinsics(image)`

- Attempts to compute intrinsics from EXIF focal length tags:
  - `37386` focal length
  - `41989` focal length in 35mm film terms
- If unavailable, falls back to an assumed 65 degree horizontal FOV.

Implementation critique:

- Pragmatic and easy to follow.
- Broad exception swallowing hides why EXIF extraction failed.
- The fallback is reasonable for demos but should be surfaced more explicitly in downstream measurement metadata, because distance confidence depends on it.

## TIPS Inference Layer

### `pipeline/tips_runner.py`

#### Module globals

- `TIPSV2_SHORT_SIDE`
- `TIPSV2_DPT_MODEL_ID`
- `TIPSV2_BASE_MODEL_ID`
- `DEFAULT_COLORMAP`
- `SEGMENTATION_THRESHOLD`
- `ADE20K_CLASSES`
- `_model`
- `_transform`

This module is effectively a singleton model host.

#### `_load_ade20k_classes()`

- Reads `data/ade20k/objectInfo150.csv`.
- Extracts the first semicolon-separated label as the primary ADE name.
- Asserts that there are exactly 150 classes.

#### `load_model()`

- Lazily loads the TIPSv2 DPT model using `AutoModel.from_pretrained(..., trust_remote_code=True)`.
- Moves it to CUDA if available.
- Stores it in module-global `_model`.

Implementation critique:

- Simple and effective.
- Not thread-safe.
- Bakes all model choice into process globals.
- `TIPSV2_BASE_MODEL_ID` is declared but unused.

#### `_resize_for_tips(image, short_side)`

- Preserves aspect ratio while snapping dimensions to multiples of 14 for ViT patching.

#### `run_dpt_inference(image)`

- Loads model.
- Resizes input.
- Converts to tensor.
- Runs model forward pass.
- Returns:
  - `depth_tensor`
  - `seg_mask`
  - `depth_colormap`

#### `_make_colormap(depth_np)`

- Normalizes depth to `[0,1]`.
- Applies matplotlib colormap.

#### `get_detected_classes(seg_mask, top_k=None)`

- Converts segmentation class indices into ADE class names sorted by pixel coverage descending.

#### `run_zero_shot_inference(image, class_list)`

- Reuses TIPS backbone embeddings for zero-shot patch-text similarity.
- Generates per-class probability maps.
- Returns:
  - `depth_tensor`
  - `cosine_maps`
  - `detected`

Implementation critique:

- The idea is good: reuse the same model rather than loading another detector.
- The implementation is optimistic to the point of being misleading:
  - every requested class is appended to `detected` regardless of confidence
  - failures in the backbone path are swallowed and still return a mostly valid-looking structure
- This inflates the apparent reliability of zero-shot detection.

## Tool Layer

### `server/schemas.py`

Defines the LLM-callable tool schemas for:

- `search_seg_classes`
- `call_dpt_head`
- `call_encoder_zero_shot`
- `measure_object`

This module is clean and focused. It is mostly declarative and one of the better-structured pieces in the repo.

### `pipeline/tools.py`

This module turns raw model outputs into navigation-relevant operations and debugging artifacts.

#### Internal helpers

##### `_safe_slug(value)`

- Sanitizes strings for filenames.

##### `_session_debug_dir(session)`

- Returns a per-session debug output directory and creates it if needed.

##### `_normalize_box_to_original_coords(box_2d, image_w, image_h)`

- Assumes model boxes are normalized in a 0..1000 range.
- Converts them into original image pixel coordinates.

Important detail:

- `measure_object()` assumes all boxes are normalized, not native pixels.
- This must stay aligned with the model prompt contract.

##### `_top_classes_in_box(seg_mask, ymin, xmin, ymax, xmax, top_k=5)`

- Counts dominant segmentation classes within a box.
- Used for ADE fallback and debugging.

##### `_draw_text_badge(...)`

- Draws readable label badges on images.

##### `_add_depth_legend_panel(image, d_min, d_max)`

- Adds a depth legend panel to a debug image.

##### `_save_depth_colormap_with_matplotlib(depth_2d, out_path)`

- Writes a depth map visualization with a standard matplotlib colorbar.

##### `_save_depth_and_seg_debug_images(session)`

- Saves:
  - depth colormap with legend
  - segmentation overlay

##### `_save_measurement_debug_overlay(...)`

- Produces a rich DPT-space debugging visualization for a single measurement.
- Shows:
  - requested/used class
  - box in DPT and original coordinates
  - class region
  - selected pixels
  - legend panel

Implementation critique:

- Very useful during model-debugging.
- It significantly bloats `pipeline/tools.py`.
- The debug-rendering concerns are tightly coupled to core measurement logic.
- This file is doing too many jobs at once:
  - tool semantics
  - geometry
  - image rendering
  - debug persistence

#### `search_seg_classes(query)`

- Case-insensitive substring search over `ADE20K_CLASSES`.
- Pure helper tool for vocabulary discovery.

#### `call_dpt_head(session)`

- Runs DPT inference.
- Stores depth, segmentation, and depth colormap on the `Session`.
- Saves debug images.
- Returns the top detected ADE classes after filtering certain environment/background labels.

Important detail:

- Detected classes are filtered by `_TOP_K_EXCLUDE`.
- Output is capped by `_DETECTED_CLASSES_TOP_K`.

#### `measure_object(class_name, box_2d, session)`

This is the most important tool in the stack.

Behavior:

- Converts normalized `box_2d` into original-image coordinates.
- Requires depth and segmentation state to already exist.
- Resolves the requested class either through:
  - ADE class index
  - session zero-shot cosine maps
- Rescales the box into DPT tensor coordinates.
- Builds a mask for the target object.
- Two paths:
  - zero-shot thresholded cosine mask
  - ADE segmentation mask
- If the requested ADE class is weak in the box, it falls back to the dominant segmented class in the box.
- Computes:
  - median depth in meters
  - horizontal bearing
  - spoken direction string
- Appends a measurement record into `session.measurements`.

Measurement record fields:

- `class_name`
- `requested_class_name`
- `used_class_name`
- `class_substituted`
- `box_original`
- `tips_distance_m`
- `direction`
- `mask_dpt`

Implementation critique:

- The median-depth-on-class-mask approach is sensible and much better than measuring full bounding boxes.
- The fallback behavior is risky:
  - if zero-shot fails or is weak, the function can silently substitute the dominant ADE class in the box
  - that means a request for an OOV object can return a real distance for the wrong object
- The function is also doing too much:
  - coordinate normalization
  - class resolution
  - zero-shot thresholding
  - ADE fallback selection
  - bearing math
  - debug rendering
  - session mutation
  - response formatting

This is a classic “smart utility” that should be split.

#### `render_annotated_image(session)`

- Draws numbered boxes from `session.measurements` on the original image.
- Saves the annotated image into the debug directory.

#### `call_encoder_zero_shot(class_list, session)`

- Runs zero-shot inference.
- Stores `depth_tensor`.
- Dynamically attaches `_zero_shot_maps` to the session.
- Creates a dummy zero segmentation mask if missing.

Implementation critique:

- Functional, but the session monkey-patching is a smell.
- Reusing `seg_mask` as a dummy sentinel to satisfy precondition checks is another smell.
- The code is compensating for a too-weak state model.

## LLM Orchestration Layer

### `server/agent.py`

This module is the orchestration brain.

It contains:

- long prompt definitions
- vLLM/OpenAI client payload creation
- multimodal message shaping
- tool dispatch
- three distinct agent loops:
  - `Scout`
  - `Mapper`
  - `Navigator`
- legacy single-loop `run_agent_loop()`

### Prompt constants

- `SYSTEM_PROMPT`
  - legacy single-agent tool-using navigation prompt
- `SCOUT_SYSTEM_PROMPT`
  - route classifier / direct-answer prompt
- `SCOUT_RESPONSE_SCHEMA`
  - strict JSON schema for Scout
- `MAPPER_SYSTEM_PROMPT`
  - tool-using scene construction prompt
- `NAVIGATOR_SYSTEM_PROMPT`
  - answer-only prompt that consumes measured scene summaries

Implementation critique:

- The prompt design is thoughtful and safety-oriented.
- But `server/agent.py` is overloaded with prompt text and execution logic in the same file.
- Prompt changes and control-flow changes are tightly coupled, which makes review and testing harder.

### Helper functions

#### `_looks_like_raw_tool_protocol(text)`

- Detects malformed model output where raw tool syntax leaks into content.

#### `_maybe_dump_request(round_idx, payload)`

- Appends request payloads to JSONL when configured.

#### `_maybe_dump_response(round_idx, response)`

- Appends response summaries and tool calls to JSONL when configured.

#### `_active_tools()`

- Returns either:
  - only `call_dpt_head` in `dpt_only` mode
  - full `TOOL_SCHEMAS` otherwise

#### `_resize_for_gemma(image, multiple)`

- Resizes images so width and height are multiples of `GEMMA_IMAGE_MULTIPLE`.

#### `_image_to_data_url(image)`

- Encodes image as JPEG data URL for vLLM multimodal input.

#### `_build_question_content(question)`

- Returns either text content blocks or input-audio content blocks.

#### `build_turn_user_content(image, question, send_image=True)`

- Builds multimodal user turn content by combining image blocks and question blocks.

#### `build_messages(image, question)`

- Legacy single-agent message constructor.

#### `_dispatch_tool(name, args, session)`

- Executes one tool by name.
- Measures latency.
- Returns JSON-serialized tool output.

Implementation critique:

- Works, but string-dispatch `if/elif` over tool names is brittle.
- It would be cleaner as a static registry of tool name -> callable.

#### `_extract_json_object(text)`

- Tries to salvage a JSON object from a malformed Scout response.

#### `_parse_scout_response(text)`

- Parses Scout JSON and normalizes to `(route, text, reason)`.
- Routes:
  - `"navigator"`
  - `"restart"`
  - `"direct"`

#### `_dispatch_tool_calls(tool_calls, session)`

- Parses tool call arguments.
- Runs them sequentially unless every call is `measure_object`, in which case it parallelizes them via `ThreadPoolExecutor`.

Implementation critique:

- The intent is good.
- Parallel appends into `session.measurements` are not synchronized, so ordering and state mutation are not robust.

### Agent loops

#### `run_agent_loop(session, history=None, send_image=True)`

- Legacy single-agent loop.
- Builds messages.
- Iterates up to `MAX_TOOL_ROUNDS`.
- Calls vLLM with tool choice enabled.
- Executes tool calls.
- Stops on final assistant text.

This is effectively the original all-in-one agent design.

#### `run_scout_loop(session, history=None, send_image=True, has_active_image=True)`

- Runs the route-selection agent.
- Enforces JSON schema output.
- Either:
  - answers directly
  - asks for restart
  - hands off to navigator path

#### `run_mapper_loop(session, history=None, prior_measurements=None, prior_turn_count=0, fresh_image_attached=False)`

- Runs the tool-using scene-construction agent.
- Uses tool calls to populate `session.measurements`.
- Retries if the model answers in prose without first measuring the scene.
- Discards final prose; only the mutated session state matters.

#### `_build_prior_measurements_context(prior_measurements)`

- Converts prior measurement state into prompt text for follow-up turns.

#### `build_mapper_messages(...)`

- Builds the mapper-specific prompt framing.
- Changes framing between:
  - first-turn “scan the full scene”
  - follow-up “reuse prior measurements only as context”

#### `_build_scene_summary(session)`

- Converts `session.measurements` into a numbered plain-text summary for Navigator.

#### `run_navigator_loop(session, annotated_image, history=None, send_image=True)`

- Single non-tool LLM call.
- Supplies:
  - annotated image
  - scene summary
  - user question
- Returns the spoken natural-language answer.

Implementation critique:

- The Scout/Mapper/Navigator decomposition is the strongest architectural move in the project.
- It reduces one big prompt into purpose-specific stages.
- The downside is duplication: each loop rebuilds similar payloads and manages similar response handling.
- A generic `LLMRunner` or `AgentRunner` abstraction could remove repetition while keeping the three-agent design.

## Utility Scripts

### `scripts/start_gemma4.sh`

- Starts vLLM with Gemma 4 and tool-calling enabled.
- Controls:
  - model
  - served name
  - port
  - max model length
  - GPU memory utilization
  - tensor parallelism
  - max sequences
  - vision token budget
  - quantization

Important detail:

- Uses:
  - `--tool-call-parser gemma4`
  - `--enable-auto-tool-choice`
  - `--reasoning-parser gemma4`
  - `--async-scheduling`

Implementation critique:

- Good operator-facing comments.
- Hard-coded environment assumptions remain:
  - specific pyenv virtualenv path
  - default quantization
  - no runtime validation of model/quantization compatibility

### `scripts/run_agent_loop.py`

- CLI for running a single backend turn against the local vLLM server.
- Supports:
  - normal image input
  - NYUv2 sample input
  - text question
  - audio question
  - request/response dumps
  - debug box reporting
  - tool I/O printing

#### `parse_args()`

- Defines CLI flags.

#### `set_env_if(name, value)`

- Helper for optional env overrides.

#### `_suppress_noisy_third_party_logs()`

- Lowers verbosity from HTTP and ML libraries.

#### `_compute_depth_metrics(session, sample_idx)`

- Compares measured depth against NYUv2 ground-truth depth using the exact measurement mask when available.

#### `_save_nyuv2_gt_depth(sample_idx, debug_dir)`

- Saves GT depth visualization for sample inspection.

#### `main()`

- Loads image/audio.
- Creates session.
- Calls `run_agent_loop(session)`.
- Prints response and optionally new debug files.

Implementation critique:

- There is a real correctness bug here:
  - `run_agent_loop()` returns `(text, history)`
  - `main()` stores it in `response_text`
  - that means the script prints a tuple, not the response string
- This script is out of sync with the current server API.

### `scripts/run_bbox_only.py`

- Minimal debugging harness for model-generated bounding boxes.
- Useful for isolating multimodal box behavior without TIPS or tool orchestration.

### `scripts/fetch_nyuv2_samples.py`

- Dataset bootstrap helper for downloading and inspecting NYUv2 samples.
- Well-scoped utility script.

## Findings

### 1. `scripts/run_agent_loop.py` is currently incorrect

Location:

- `scripts/run_agent_loop.py:272`

Problem:

- `run_agent_loop(session)` returns a tuple `(final_text, updated_history)`.
- The script assigns that tuple to `response_text` and prints it as if it were a string.

Why it matters:

- The debugging CLI is lying about the real API contract.
- Anyone using the script as a backend harness gets misleading output.

Behavior-preserving fix:

- Change to:
  - `response_text, _history = run_agent_loop(session)`

### 2. Zero-shot measurement can collapse into measuring the wrong object

Locations:

- `pipeline/tools.py:449-480`
- `pipeline/tools.py:548-568`
- `pipeline/tips_runner.py:178-183`

Problem:

- `run_zero_shot_inference()` reports requested classes as “detected” even when confidence is weak.
- Then `measure_object()` can fall back from the requested zero-shot class to the dominant ADE segmentation class in the box.
- That can produce a valid distance for a different object than the one the user asked for.

Why it matters:

- This directly conflicts with the prompt-level promise not to invent or substitute unrelated objects.
- It is the most serious backend reasoning risk in the current implementation.

Behavior-preserving fix:

- Keep fallback internally for debugging, but do not emit a successful distance for the requested class when fallback substitution crosses vocabularies or class families.
- Return a structured “ambiguous measurement” result instead.
- Let Navigator/Mapper ask for a tighter box or different class choice.

### 3. Parallel `measure_object()` calls mutate shared session state unsafely

Location:

- `server/agent.py:507-540`
- `pipeline/tools.py:558-567`

Problem:

- `_dispatch_tool_calls()` parallelizes `measure_object()` calls with threads.
- `measure_object()` appends to `session.measurements`.
- There is no lock and no stable ordering guarantee.

Why it matters:

- Measurement ordering can drift from tool-call ordering.
- Navigator summaries and annotation numbering depend on list order.
- This is not catastrophic under CPython, but it is sloppy shared-state concurrency.

Behavior-preserving fix:

- Make `measure_object()` pure and return its measurement record.
- Append to `session.measurements` in `_dispatch_tool_calls()` after futures complete, in original tool-call order.

### 4. `app_blind.py:api_query()` is a controller blob

Location:

- `app_blind.py:129-380`

Problem:

- One function owns transport parsing, route control, session persistence, metrics aggregation, TTS, debug payload assembly, and exceptional control flow.

Why it matters:

- It is hard to reason about, test, or modify safely.
- The `StopIteration` branch is particularly fragile.

Behavior-preserving fix:

- Split into:
  - `load_turn_inputs()`
  - `resolve_session_state()`
  - `execute_turn()`
  - `persist_session_state()`
  - `build_response_payload()`

### 5. `_sessions` is an unbounded mutable global with no lifecycle management

Location:

- `app_blind.py:33`

Problem:

- All conversation state lives in a process-global dictionary.
- No TTL, size cap, eviction, persistence, or locking.

Why it matters:

- Memory growth is unbounded.
- Multi-worker deployment semantics become undefined.
- Concurrency assumptions are weak in an async web app.

Behavior-preserving fix:

- Wrap the store behind a `SessionStore` interface.
- Keep the first implementation in-memory if you want, but add:
  - max age
  - max sessions
  - explicit serialization schema

### 6. Session state is under-modeled and uses monkey-patched fields

Location:

- `pipeline/session.py`
- `pipeline/tools.py:640-645`

Problem:

- `Session` does not explicitly define all state it carries.
- `_zero_shot_maps` is attached dynamically.
- `seg_mask` is sometimes a true segmentation mask and sometimes a dummy zero tensor.

Why it matters:

- State invariants become implicit and brittle.
- Tool preconditions are faked rather than modeled.

Behavior-preserving fix:

- Add explicit optional fields to `Session`:
  - `zero_shot_maps`
  - `mode` or `measurement_backend`
  - `has_dense_segmentation`

### 7. Utility duplication is already starting to spread

Locations:

- `app.py:37-43`
- `app_blind.py:85-90`
- `app.py:206-213`
- `app_blind.py:389-399`

Problem:

- `_strip_markdown()` is duplicated.
- localhost `httpx` TLS monkey-patch is duplicated.

Why it matters:

- Small duplication becomes policy drift later.

Behavior-preserving fix:

- Move these into a tiny backend utility module and import them.

### 8. Prompt text and execution logic are too tightly coupled

Location:

- `server/agent.py`

Problem:

- The file mixes:
  - 300+ lines of prompt text
  - tool dispatch
  - message shaping
  - loop execution
  - JSON parsing

Why it matters:

- Hard to review, hard to diff, hard to test.

Behavior-preserving fix:

- Split into:
  - `server/prompts.py`
  - `server/message_builders.py`
  - `server/tool_dispatch.py`
  - `server/loops.py`

### 9. Debug rendering is over-coupled to measurement logic

Location:

- `pipeline/tools.py`

Problem:

- Measurement computation and debug artifact generation are interleaved.

Why it matters:

- Core logic is harder to test.
- Debug overhead is harder to disable cleanly.

Behavior-preserving fix:

- Move all debug image creation into `pipeline/debug_render.py`.
- Keep tool functions responsible only for producing data, not presentation.

### 10. Test surface is effectively absent

Location:

- `tests/unit/__init__.py`
- `tests/integration/__init__.py`

Problem:

- There are test directories, but no actual test coverage in the repo snapshot reviewed here.

Why it matters:

- The backend depends on fragile prompt contracts and geometry conversions.
- Those are exactly the kinds of things that need regression tests.

Behavior-preserving fix:

- Add fast tests for:
  - box normalization
  - DPT-space scaling
  - direction phrase generation
  - Scout JSON parsing fallbacks
  - metric aggregation
  - session persistence behavior

## Refactor Plan That Preserves Functionality

If the goal is to keep behavior the same while cleaning up implementation quality, the refactor order should be:

1. Fix contract mismatches and unsafe state mutation.
   - `scripts/run_agent_loop.py` tuple bug
   - ordered collection of parallel measurement results
   - explicit zero-shot ambiguity handling

2. Strengthen the data model.
   - formalize `Session`
   - formalize persisted session payloads
   - replace monkey-patched session fields

3. Split orchestration concerns.
   - extract prompt definitions
   - extract response/finalization helpers from `api_query()`
   - extract tool registry/dispatch

4. Split debug concerns from core logic.
   - move image rendering/debug persistence out of `pipeline/tools.py`

5. Add regression tests around the geometry and routing contracts.

## Bottom-Line Assessment

What is strong:

- The Scout/Mapper/Navigator decomposition is a genuinely good architectural direction.
- The `measure_object()` depth-on-mask strategy is much better than naive box depth.
- The backend has practical operator tooling for dumps, debug images, and NYUv2 evaluation.

What is weak:

- State modeling is loose.
- Core functions are too large.
- There are real correctness risks around zero-shot fallback and threaded measurement mutation.
- Debugging code and production logic are overly interwoven.

The backend is not fundamentally over-engineered in architecture. The main issue is different: the architecture is decent, but too much logic is packed into a few highly stateful modules, and the implementation relies on implicit contracts instead of explicit types and boundaries.
