### Motivation

For a blind user, the important question is not just what is in front of me, but how far away it is and how I should move safely. That is the gap between scene description and real navigation assistance. A useful system must do more than describe a scene in natural language; it must produce grounded spatial answers that guide movement in the real world.

I followed the previous competition and was really moved by the story behind **Gemma Vision**. Their success highlighted that for this community, a "blind-first" UX is the only way to build. I designed my system around those same core principles of extreme simplicity and audio-driven interaction, which allowed me to focus the technical heart of the project on **spatial grounding.**

Language models are not reliable depth sensors, but with **Gemma 4**, they can act as an agent that identifies an object, calls specialized spatial tools, and turns grounded distance estimates into practical guidance. **Gemma Guide** is built to turn that idea into a reality.

### The Solution


Gemma Guide combines Gemma 4 and TIPSv2 into a grounded navigation system. Rather than a standard chatbot, the solution is designed as a "grounded navigator" where **the user interface completely hides the complexity of the architecture.**

- **The Intelligence (Gemma 4)**: Acts as the multimodal orchestrator. Leveraging its native audio-visual understanding, it interprets spoken intent and visual scenes simultaneously, localizing objects with high precision and deciding when to invoke the spatial tool-stack.

- **The Grounding (TIPSv2)**: A vision-language encoder with a DPT head that provides the dense metric depth and semantic segmentation that traditional LLMs lack.

- **The Interface**: The UI is built around spatial muscle memory, using a simplified two-zone layout and a "tap-anywhere" shutter. Distinct audio soundscapes and TTS guidance bridge the gap during model reasoning, ensuring a continuous, accessible user loop.

### Why Gemma 4

Gemma 4 is a particularly strong fit for this project because it combines several capabilities that are rarely available together in a model this compact: image understanding, audio understanding, native function calling, and advanced reasoning. That combination matters directly for an assistive navigation system. The interaction is voice-driven, the scene must be interpreted visually, and the model must decide when a conversational answer is sufficient and when grounded spatial measurement is required. Its compactness is also important in practice: models in this class create a more realistic path toward mobile and edge deployment, reducing dependence on continuous internet access and making privacy-sensitive assistive use more feasible. Additionally, the model card lists core capabilities such as pointing and interleaved interaction, which are directly relevant for this use case. Pointing supports grounded object selection, while interleaved multimodal interaction helps me support follow-up audio and visual inputs in a practical assistive workflow.


### Why TIPSv2

TIPSv2 is a strong fit for this project because it provides the dense spatial perception that a language model lacks on its own. It is a vision-language encoder with strong text-aligned patch representations, which makes it especially useful for the three capabilities my system depends on most: semantic segmentation, metric depth estimation, and grounded zero-shot matching. That is exactly the combination needed when navigation depends on measuring specific regions of the scene rather than producing a single global description.

TIPSv2 is especially valuable because it provides both pieces my system needs most. Its DPT heads produce metric depth and semantic segmentation, which are the core signals needed for grounded object-level measurement. Its text-aligned encoder is a strong complementary capability, enabling open-vocabulary zero-shot matching through spatially rich embeddings when the user refers to objects outside the fixed segmentation label set.

### How It Works

Gemma Guide runs as a routed multi-agent pipeline. A lightweight Gemma-based Scout first decides whether a question can be answered directly through general visual understanding, such as scene description or reading, or whether it requires grounded spatial reasoning. If spatial analysis is needed, the request is handed off to the navigation pipeline.

That spatial pipeline begins with a Gemma-based `Mapper`, which receives the user's spoken question and the scene image, determines which objects matter for the request, and localizes those objects in the image. It then calls a TIPSv2-based spatial tool stack to obtain the grounded scene information needed for measurement. On the first pass, the Mapper can also measure several navigation-relevant objects in parallel, allowing the system to build a broader grounded scene state instead of reasoning about a single queried object in isolation.

The key step is object-level measurement. The system does not treat Gemma's localization box as the final answer. Instead, it intersects that region with an appropriate object mask from TIPSv2, which also provides segmentation outputs, so depth is measured only over the most relevant pixels rather than over an entire coarse bounding box or a full-scene depth map. This produces more reliable grounded measurements for individual objects, especially in cluttered scenes. From the selected object region, the system computes both metric distance and horizontal direction, allowing it to answer not only what is present, but where it is relative to the user.

This design also supports open-vocabulary grounding. For objects that map cleanly to the fixed label set, the system uses the TIPSv2 DPT heads to produce semantic segmentation and metric depth. For objects outside that vocabulary, the Mapper can route the request through the TIPSv2 backbone, which produces zero-shot text-aligned similarity maps over candidate class names. The resulting matched region is then passed into the same downstream measurement pipeline for distance and direction estimation.

After measurement, the system packages the grounded scene into an annotated image and a compact structured summary. A third Gemma-based agent, the `Navigator`, receives this cleaned representation and generates the final user-facing guidance. I use this split because the two stages place different demands on the model: the Mapper must manage tool calls and spatial grounding, while the Navigator is more reliable when reasoning over a simplified measured world model rather than raw intermediate tool outputs.

### Challenges

This project only became reliable after several architectural changes. The final pipeline was shaped directly by the failure modes we encountered in early versions of the system.

- **Distance alone was not enough for navigation.** Reporting that an object was a few meters away was not sufficient for safe guidance; users also needed to know where that object was relative to the camera. I addressed this by estimating horizontal bearing from the selected object region and camera intrinsics, so the system can report both distance and direction.

- **Whole-scene depth reasoning was too ambiguous.** Early attempts asked the model to infer object distance from the original image and a depth visualization, but this was too unreliable for object-level navigation. I instead made Gemma localize the relevant object first and used that localization as structured input to the measurement pipeline.

- **Bounding boxes were too coarse on their own.** Measuring depth across an entire box often mixed the target object with background pixels or overlapping objects. To make measurement more precise, I combined Gemma's selected region with TIPSv2 segmentation masks when the object was in-vocabulary, and with zero-shot similarity maps when it was not.

- **A fixed segmentation vocabulary was not enough for real user queries.** Many practical navigation questions refer to objects outside a closed label set. I addressed this by routing those cases through the TIPSv2 backbone for open-vocabulary zero-shot matching, then feeding the result into the same measurement pipeline.

- **A single-agent design became overloaded** . When one model handled scene description, tool orchestration, and final navigation reasoning, the output became unreliable and the agent became confused. I made the pipeline more reliable by splitting those roles across `Scout`, `Mapper`, and `Navigator`.


### Toward On-Device Deployment

A major next step, and what would turn this from a demo towards something practical, is on-device deployment. I explored Google AI Edge Gallery as a promising path because it already supports on-device Gemma, multimodal interaction, and tool-calling skills, making it a natural option for a partial on-device version with Gemma running locally while the TIPS stack remains remote. The main blocker for that approach is that the image Gemma sees in chat is not forwarded into the skill execution context, which breaks grounded measurement because Gemma’s bounding box must refer to the same image the spatial tool receives. A standalone mobile app may therefore be the stronger long-term path, since it would give tighter control over the camera, voice interaction, and accessibility experience while also creating a clearer route toward fully offline use, hosting both models on the same mobile device. This is especially plausible because the vision model is compact enough that both it and the language model may fit on edge hardware.
