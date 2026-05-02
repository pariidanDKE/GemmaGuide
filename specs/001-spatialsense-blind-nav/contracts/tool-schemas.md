# Tool Schemas: SpatialSense Agent

**Phase**: 1  
**Date**: 2026-04-30

These are the function schemas passed to Gemma 4 4B-IT via the `tools` parameter in the OpenAI-compatible API. Gemma receives only the original RGB image and the audio question. No overlay or depth colormap is passed to Gemma — all depth reasoning is delegated to the tools.

---

## Tool 1: search_seg_classes

Substring search over the 150 ADE20K class names. Returns only matches, not the full list.

```json
{
  "type": "function",
  "function": {
    "name": "search_seg_classes",
    "description": "Searches the 150 object class names that the spatial analysis model knows. Pass the name of the object you are looking for. Returns matching class names. If the result is empty, the object is not in the standard vocabulary — use call_encoder_zero_shot instead.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Object name to search for. Case-insensitive substring match (e.g. 'chair' matches 'chair' and 'armchair')."
        }
      },
      "required": ["query"]
    }
  }
}
```

**Return value**: `{"matches": ["chair", "armchair"]}` or `{"matches": []}` if not found.

---

## Tool 2: call_dpt_head

Runs TIPSv2 DPT inference. Caches the depth tensor and segmentation mask server-side for the session. Returns a list of detected class names so Gemma knows what is measurable without receiving any images.

```json
{
  "type": "function",
  "function": {
    "name": "call_dpt_head",
    "description": "Runs the spatial analysis model on the camera image. Caches the depth map and object segmentation for this session. Returns the list of object classes detected in the scene. Call this before measure_object.",
    "parameters": {
      "type": "object",
      "properties": {
        "image_ref": {
          "type": "string",
          "description": "Reference to the input camera image. Use 'input_image'."
        }
      },
      "required": ["image_ref"]
    }
  }
}
```

**Return value**: `{"status": "ready", "detected_classes": ["wall", "floor", "chair", "table", "door"]}`

**Side effects**:
- Depth tensor `(1,1,H,W)` float32 meters cached in session
- Segmentation mask `(H,W)` int cached in session (ADE20K class index per pixel)
- Depth-colored overlay generated and stored for demo display — not sent to Gemma

---

## Tool 3: call_encoder_zero_shot

Zero-shot segmentation for objects not in the ADE20K vocabulary. Same caching contract as `call_dpt_head`.

```json
{
  "type": "function",
  "function": {
    "name": "call_encoder_zero_shot",
    "description": "Runs zero-shot spatial analysis for object classes not in the standard vocabulary (e.g. 'shopping cart', 'step', 'curb', 'puddle'). Use only when search_seg_classes returns no matches. Returns which of the requested classes were detected.",
    "parameters": {
      "type": "object",
      "properties": {
        "image_ref": {
          "type": "string",
          "description": "Reference to the input camera image. Use 'input_image'."
        },
        "class_list": {
          "type": "array",
          "items": {"type": "string"},
          "description": "1–5 object class names to search for using zero-shot recognition."
        }
      },
      "required": ["image_ref", "class_list"]
    }
  }
}
```

**Return value**: `{"status": "ready", "detected_classes": ["shopping cart"]}` or `{"status": "ready", "detected_classes": []}`

**Side effects**: Cosine similarity map cached as the object mask for subsequent `measure_object` calls. Depth tensor also cached (same contract as `call_dpt_head`).

---

## Tool 4: measure_object

Returns the metric distance to a specific object instance. Gemma provides a bounding box identifying which instance it means — the tool intersects the segmentation mask with that box so only the target object's pixels contribute to the depth reading.

```json
{
  "type": "function",
  "function": {
    "name": "measure_object",
    "description": "Returns the metric distance in meters to a specific object in the scene. You must have called call_dpt_head or call_encoder_zero_shot first. Look at the original image and draw a bounding box around the specific object instance you want to measure, then pass those coordinates here. The box is used to select which instance you mean — the depth is computed only from the object's own pixels within that box, not from background or other objects inside the box.",
    "parameters": {
      "type": "object",
      "properties": {
        "class_name": {
          "type": "string",
          "description": "The object class name as returned by search_seg_classes or call_encoder_zero_shot."
        },
        "box_2d": {
          "type": "array",
          "items": {"type": "integer"},
          "description": "Bounding box of the target object instance: [ymin, xmin, ymax, xmax] in image pixel coordinates (origin at top-left)."
        }
      },
      "required": ["class_name", "box_2d"]
    }
  }
}
```

**Return value**: `{"distance_m": 1.42, "bearing_deg": -22.3, "confidence": "high"}`  
- `bearing_deg`: signed horizontal bearing — negative = left of camera center, positive = right; Gemma reports "directly ahead" when within ±5°  
- `confidence: "low"` when depth values are near 0.001m or 10.0m (TIPSv2 clip extremes), or when fewer than 50 pixels of the target class fall within the bounding box  
- `confidence: "low"` also signals to Gemma to express uncertainty in the distance (bearing remains reliable regardless of depth confidence)

**Note on bearing**: `bearing_deg` is computed arithmetically from the pixel centroid of the segmented intersection and session-cached camera intrinsics. Bearing is depth-independent — it depends only on pixel position and focal length, so it is reliable even when distance confidence is low. Gemma reports it as natural language: "22 degrees to your left", "8 degrees to your right", or "directly ahead".

---

## Agent Interaction Pattern

```
User submits: [RGB image, audio question]
Gemma receives: RGB image + audio only

── "How far is the chair?" ──────────────────────────────────────────

Gemma calls search_seg_classes("chair")
← {"matches": ["chair", "armchair"]}

Gemma calls call_dpt_head("input_image")
← {"status": "ready", "detected_classes": ["floor", "chair", "table", "door"]}

Gemma looks at RGB → draws bounding box around the specific chair instance
Gemma calls measure_object("chair", [120, 80, 340, 260])
← {"distance_m": 1.42, "confidence": "high"}

← {"distance_m": 1.42, "bearing_deg": -3.2, "confidence": "high"}

Gemma: "The chair is approximately 1.4 meters away, directly ahead."

── "What's around me?" ──────────────────────────────────────────────

Gemma calls call_dpt_head("input_image")
← {"status": "ready", "detected_classes": ["floor", "chair", "table", "door", "lamp"]}

Gemma decides: floor is not navigation-relevant → measures chair, table, door, lamp
For each: draws box on RGB → calls measure_object(class, box)
← distances + confidences per object

Gemma orders closest to farthest, reports bearing_deg per object
Gemma: "Closest to you is a chair at about 1.4 meters, directly ahead. A table at 2.1 meters,
        8 degrees to your right. The door is at 3.8 meters, 3 degrees to your left."

── "Is there a shopping cart?" ──────────────────────────────────────

Gemma calls search_seg_classes("shopping cart")
← {"matches": []}

Gemma calls call_dpt_head("input_image")  ← caches depth tensor
← {"status": "ready", "detected_classes": [...]}

Gemma calls call_encoder_zero_shot("input_image", ["shopping cart"])
← {"status": "ready", "detected_classes": ["shopping cart"]}

Gemma draws box around detected region → calls measure_object("shopping cart", box)
← {"distance_m": 2.3, "confidence": "medium"}

← {"distance_m": 2.3, "bearing_deg": -15.1, "confidence": "medium"}

Gemma: "There appears to be a shopping cart about 2.3 meters ahead, 15 degrees to your left."
```

---

## Gradio Interface Contract

### Inputs

| Component | Gradio Type | Description |
|-----------|-------------|-------------|
| Camera frame | `gr.Image(type="pil")` | Photo upload (PNG/JPEG) |
| Audio question | `gr.Audio(sources=["microphone"], type="numpy")` | Microphone recording; primary path |
| Text question | `gr.Textbox()` | Optional typed question; testing/dev path only |

### Outputs

| Component | Gradio Type | Description |
|-----------|-------------|-------------|
| Response text | `gr.Textbox()` | Natural language navigation response from Gemma |
| Audio response | `gr.Audio(type="numpy")` | TTS-synthesized audio of the response |
| Overlay image | `gr.Image(type="pil")` | Depth-colored overlay for demo display — generated by pipeline, never seen by Gemma |

### Behavior

- Submit requires an image and at least one of audio or text question
- All errors caught and returned as natural language (no stack traces exposed)
- Text and audio response returned together
