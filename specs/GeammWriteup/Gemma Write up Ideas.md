
### **The Problem: Local AI for Visually Impaired Users**

Modern multimodal models have reached a point where spatial reasoning and audio understanding are no longer exclusive to large, cloud-hosted systems. Small models capable of running directly on consumer hardware are emerging with remarkable capability in these areas — Gemma 4 being one of the clearest examples of this shift.

This matters enormously for visually impaired users, who stand to benefit more than most from AI-assisted perception. Yet the current landscape creates three compounding barriers.

#### **The Cost Problem**

As compute becomes more scarce and the demand for local AI grows, the economics of cloud-based assistive tools will only worsen. Frontier models like Opus-4.7 or GPT5.5 already come with subscription costs that can be prohibitive. For disabled users who depend on these tools not as a productivity enhancement but as a functional necessity, that pricing structure is not a minor inconvenience — it is a real access problem.

#### **The Privacy Problem**

Assistive vision applications operate in deeply personal spaces: inside users' homes, in private moments, in contexts where the user may have limited ability to audit what is being captured and sent. Many users will reasonably object to streaming that data to third-party cloud providers. A locally-hosted model eliminates this concern entirely.


#### **The Technical Problem — and Why Local AI Is Actually Sufficient**

Frontier LLMs, despite their general strength, are notably poor at precise depth and distance estimation. This is not a gap that scale alone closes — it requires purpose-built models trained specifically for spatial perception tasks. Rather than treating this as a limitation, we can treat it as an architectural opportunity: offload depth and segmentation to a dedicated tool, and expose it to the LLM via a tool call. 

In this paradigm, the LLM's role becomes well-scoped. It does not need long-horizon reasoning, encyclopedic knowledge, or frontier-scale parameters. It needs to understand audio input, reason over a compact scene description, and guide the user clearly. That is a task local — and even edge — models can perform effectively today. 

### Motivation Conclusion

Given the previous two problems of Cost and Privacy, and the fact that well-scoped capabilities can be sufficient for such a tool, this creates a perfect opportunity for a Local Ai model to shine. Specifically the smallest models in the Gemma 4 line are tiny, and can fit on a phone. 

- The issue with this is that you still need a TIPS model to actually get depth, that does not fit on a phone. Ideally that is an MCP server that the model can access, but at least the privacy angle kind of goes out of the window. Of course, what we do now is just a demo, and TIPS is a tiny model that can run very lmited hardware as well.
