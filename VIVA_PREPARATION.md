# Pocket-Agent: Viva & Defense Preparation Guide

This document contains a detailed breakdown of the Pocket-Agent project. It is structured to help you answer questions confidently during your viva or project presentation.

---

## 1. Executive Summary
**Objective:** Build a tiny, on-device (CPU-only) language model capable of structured tool calling (Weather, Calendar, Convert, Currency, SQL) and polite refusals, while aggressively minimizing model size and maximizing inference speed.

**Final Solution:** 
- **Base Model:** `SmolLM2-360M-Instruct` (360 million parameters)
- **Fine-Tuning:** Native FP16 LoRA (Low-Rank Adaptation)
- **Quantization:** GGUF `Q4_K_M` (4-bit quantization)
- **Final File Size:** ~200 MB
- **Inference Stack:** `llama-cpp-python`

---

## 2. Key Architectural Decisions

### Why SmolLM2-360M?
- **Size Constraint:** The hackathon required `<500MB` to pass, but offered a massive **+10 point bonus** for `<250MB`. 
- **Efficiency:** SmolLM2 is exceptionally capable for its tiny size (360M params). By selecting it over Qwen2.5-0.5B, we guaranteed the 250MB bonus point threshold.

### Why LoRA (instead of Full Fine-Tuning)?
- **Resource Constraints:** Full fine-tuning updates 100% of the weights, which requires massive VRAM and takes hours.
- **LoRA Efficiency:** LoRA freezes the original weights and only trains small "adapter" matrices. We trained just 2.34% of the network (8.6M parameters), allowing us to train on a single free Colab T4 GPU in under 5 minutes without running out of memory.

### Why fp16 over QLoRA (4-bit)?
- **Hardware/Software Conflicts:** Google Colab uses CUDA 12.x, but the `bitsandbytes` library required for QLoRA frequently forces CUDA 13.x dependencies, causing runtime crashes (`libnvJitLink.so.13 missing` or `torchvision::nms does not exist`).
- **Resource Math:** Because SmolLM2 is only ~720MB in native FP16 (16-bit precision), it easily fits entirely into a 16GB T4 GPU. Therefore, we bypassed QLoRA entirely, trained in pure fp16, and avoided all the complex dependency hell of 4-bit training.

### Why Q4_K_M Quantization?
- `Q4` means 4-bit integer quantization.
- `K_M` refers to a specific "medium" K-quantization mix. It keeps critical tensors (like embeddings) at higher precision (e.g., 6-bit) while compressing regular weights to 4-bit.
- **Result:** It provides the best balance between preserving the model's intelligence and shrinking the file size (down to ~200MB).

---

## 3. Data Engineering Strategy

The data generation pipeline was critical to our success. LLMs learn *what* to do based on their architecture, but *how* to behave based on data.

* **Total Dataset Size:** ~255 carefully curated examples.
* **Format:** `<tool_call>...JSON...</tool_call>`. It was essential to train the model to output valid JSON inside specific XML tags so the local script (`inference.py`) could parse it deterministically.

### The Refusal Strategy (Avoiding Penalities)
- **The Problem:** The grader subtracts **-0.5 points** for "false positive" tool calls (calling a tool when it shouldn't).
- **The Solution:** We deliberately made **~18%** of the dataset "Refusal" examples (approx. 46 examples).
- **Tricky Refusals:** We specifically trained the model on adversarial traps. For example, *"Book a flight to Paris"* or *"Send 500 dollars"*. In earlier iterations, the model saw "Paris" and automatically triggered the Weather tool. We fed it data teaching it to ignore location names unless the context explicitly asked for weather.

### Achieving Bonus Points
- **Multi-turn Context (+5 points):** We included 12 multi-turn examples where the user references previous turns (e.g., User: "Weather in London" -> Assistant: (tool) -> User: "What about Paris?").
- **Adversarial Input (+5 points):** Included 28 examples containing typos ("Wats the weathr"), mixed-languages/Roman Urdu ("Lahore mein mausam kya hai?"), and Spanish/Turkish.

---

## 4. Technical Challenges Faced

**Challenge 1: Google Colab Package Incompatibilities**
- *Issue:* Conflicting torch/torchvision/bitsandbytes packages breaking `import transformers` due to CUDA mismatches.
- *Fix:* Stripped the environment down. Removed QLoRA dependency. Explicitly uninstalled the broken `torchvision` state immediately before GGUF conversion to fix the broken `LlamaConfig` import pipeline.

**Challenge 2: The Hallucinated Tool Calls**
- *Issue:* Tiny models are eager to please. If you mention a number, they'll try to convert it.
- *Fix:* Iterative data refinement. Built a custom subset of "Tricky Refusals" exactly matching the edge cases causing false-positives.

---

## 5. Potential Viva Questions & Answers

**Q: Why didn't you use OpenAI or an external API for the tools?**
*A:* "The hackathon strictly mandated zero-network-dependency processing. The model itself must process the text and generate a structured JSON output representing the API request. The execution of that API is handled by the grading system, not our code."

**Q: What is GGUF and why use it?**
*A:* "GGUF is a binary file format created by the llama.cpp team. It is highly optimized for fast loading and CPU-based inference. We used it so our model could run entirely on-device (like a mobile phone CPU) without requiring a discrete Nvidia GPU."

**Q: How did you prevent catastrophic forgetting when fine-tuning?**
*A:* "Using LoRA is a great defense against catastrophic forgetting because the underlying foundational weights of SmolLM2 are frozen. The LLM retains its general language understanding while the tiny LoRA adapters 'steer' it toward the specific JSON tool-call format."

**Q: If you had more time, what would you improve?**
*A:* "I would implement larger synthetic dataset generation using a frontier model (like Claude 3.5 Sonnet or Gemini 1.5 Pro) to generate thousands of obscure multi-turn adversarial edge cases. I'd also experiment with system-prompt optimization."
