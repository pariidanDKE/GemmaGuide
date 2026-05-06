TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_seg_classes",
            "description": (
                "Searches the 150 object class names that the spatial analysis model knows. "
                "Pass the name of the object you are looking for. Returns matching class names. "
                "If the result is empty, the object is likely not in the standard vocabulary — "
                "use call_encoder_zero_shot. "
                "This function does not return distance."
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
                "Use this detected_classes list to choose class_name for measure_object. "
                "Use this for in-vocabulary/ADE classes. "
                "This function does not return numeric distance by itself; call measure_object afterwards to get distance."
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
                "Use when search_seg_classes returns no useful ADE match, or when the requested object is not in call_dpt_head detected_classes and no close ADE label is reasonable. "
                "Returns which of the requested classes were detected. "
                "This function does not return numeric distance by itself; call measure_object afterwards to get distance."
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
                "For any numeric distance answer, this call is required. "
                "For ADE/DPT path, class_name should be chosen from call_dpt_head.detected_classes whenever possible. "
                "Look at the original image and draw a bounding box around the specific object instance "
                "you want to measure, then pass those coordinates here. "
                "The box is used to select which instance you mean — the depth is computed only from "
                "the object's own pixels within that box, not from background or other objects inside the box. "
                "If you receive error with class_pixels_total=0 or selected_pixels=0, retry once with a closer class label from detected_classes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": (
                            "Object class label. Prefer a label present in call_dpt_head.detected_classes for ADE/DPT measurements; "
                            "otherwise use a zero-shot class from call_encoder_zero_shot output."
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
