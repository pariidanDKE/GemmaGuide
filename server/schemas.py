TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_seg_classes",
            "description": (
                "Searches the 150 object class names that the spatial analysis model knows. "
                "Pass the name of the object you are looking for. Returns matching class names. "
                "If the result is empty, the object is not in the standard vocabulary — "
                "use call_encoder_zero_shot instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Object name to search for. "
                            "Case-insensitive substring match (e.g. 'chair' matches 'chair' and 'armchair')."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_dpt_head",
            "description": (
                "Runs the spatial analysis model on the camera image. "
                "Caches the depth map and object segmentation for this session. "
                "Returns the list of object classes detected in the scene. "
                "Call this before measure_object."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_encoder_zero_shot",
            "description": (
                "Runs zero-shot spatial analysis for object classes not in the standard vocabulary "
                "(e.g. 'shopping cart', 'step', 'curb', 'puddle'). "
                "Use only when search_seg_classes returns no matches. "
                "Returns which of the requested classes were detected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1–5 object class names to search for using zero-shot recognition.",
                    },
                },
                "required": ["class_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_object",
            "description": (
                "Returns the metric distance in meters to a specific object in the scene. "
                "You must have called call_dpt_head or call_encoder_zero_shot first. "
                "Look at the original image and draw a bounding box around the specific object instance "
                "you want to measure, then pass those coordinates here. "
                "The box is used to select which instance you mean — the depth is computed only from "
                "the object's own pixels within that box, not from background or other objects inside the box."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": (
                            "The object class name as returned by search_seg_classes or call_encoder_zero_shot."
                        ),
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Bounding box of the target object instance: "
                            "[ymin, xmin, ymax, xmax] in image pixel coordinates (origin at top-left)."
                        ),
                    },
                },
                "required": ["class_name", "box_2d"],
            },
        },
    },
]
