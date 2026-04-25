# 🎬 CodeRL++ Demo Strategy & Script

This document summarizes the 2-minute demo video for the Meta x OpenEv Hackathon.

## Scene 1: The Problem (0:00 - 0:30)
- **Visual**: Show a complex code diff with a subtle SQL injection.
- **Narrative**: "Standard LLMs often generate code but struggle to review it with precision. They miss edge cases and repeat historical mistakes."
- **Action**: Run `inference.py` with an untrained baseline. Show it missing the bug.

## Scene 2: The Solution (0:30 - 1:15)
- **Visual**: Switch to the CodeRL++ Gradio UI running on HuggingFace Spaces.
- **Narrative**: "Introducing CodeRL++: An agentic RL environment where agents detect, fix, and verify bugs through multi-phase reasoning."
- **Action**: Show the agent calling `get_call_graph()` and `inspect_function()`. Explain the 'Detect → Fix → Verify' loop.
- **Visual**: Zoom in on `AgentMemory` in the UI. "Agents learn across episodes, tracking their own failure patterns."

## Scene 3: The Results (1:15 - 1:45)
- **Visual**: Display the W&B reward curve and the 'Before vs. After' table.
- **Narrative**: "After 3,000 steps of GRPO fine-tuning, we see a 138% increase in vulnerability recall and a 57% reduction in false positives."
- **Action**: Run the trained agent on the same SQL injection task. Show it catching the bug, proposing a parameterized fix, and verifying it.

## Scene 4: OpenEnv Compliance (1:45 - 2:00)
- **Visual**: Show the `openenv.yaml` file and the `/health` endpoint.
- **Narrative**: "Fully OpenEnv compliant and docker-ready. CodeRL++ is a step toward closing the gap between raw LLM capability and genuine engineering intelligence."
- **Call to Action**: "Try it now on HuggingFace Spaces."

---

## Technical Setup for Recording
- **Environment**: HuggingFace Spaces (A100 Large)
- **Model**: Qwen2.5-7B-Instruct (Fine-tuned)
- **Capture Tool**: OBS Studio at 1080p, 60fps
- **Editor**: DaVinci Resolve
