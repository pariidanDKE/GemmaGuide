

### Motivation

Current vision accessibility apps built around LLMs are already useful, but many still depend on remote APIs. That creates latency, unreliable connectivity, recurring cost, and privacy exposure in exactly the situations where an assistive tool needs to be dependable. For blind and visually impaired users, this is not a minor inconvenience. It directly affects whether the system can be trusted in everyday use.

This is why local multimodal models matter. Last year's winning project, `gemma-vision`, showed that a local Gemma-based accessibility app can be more than a technical demo if it is designed as a genuinely blind-first product. Its writeup is valuable because it treats latency, privacy, device practicality, and usability as core requirements.

But local deployment alone is not enough. Neither small local models nor frontier models are reliable depth sensors, and navigation depends on grounded spatial answers, not just plausible description. A useful system must answer questions like what is in front of me, how far away it is, and how I should move safely.

This is where the agentic shift matters. Modern multimodal models can now understand images, handle spoken interaction, and use tools. With the right tools, a small model can outperform a larger frontier model on narrow tasks that require grounding and measurement rather than broad world knowledge.

### Solution Approach

This is where Gemma 4 becomes interesting. In Gemma Guide, Gemma 4 is the multimodal navigation agent: it takes the user's spoken question, sees the original scene, identifies the relevant object instance, decides when to call tools, and turns the resulting measurements into clear spoken guidance.

Instead of asking one model to do everything, Gemma Guide uses Gemma 4 for intent understanding, multimodal reasoning, tool orchestration, and object localization, while a dedicated spatial toolchain handles depth, segmentation, and metric measurement. That division of labor is the core idea: the model does not merely describe the world, it guides the user through it.

That architecture is especially important at the edge. A small local model with the right tools can be more useful for a narrow assistive task than a larger general model without grounding. The goal of Gemma Guide is therefore not to build a general vision assistant. It is to show that a local, tool-using Gemma 4 agent can move beyond description and provide spatially actionable navigation guidance.
