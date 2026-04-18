# Pocket-Agent: On-Device Tool-Calling Assistant

A fine-tuned, quantized language model that performs structured tool calling entirely on-device (CPU only, no network). Built for the Pocket-Agent Hackathon.

## 🎯 What It Does

The model receives natural language prompts and either:
- **Calls a tool** with structured JSON arguments via `<tool_call>{...}</tool_call>` tags
- **Refuses politely** when no tool matches the request

### Supported Tools

| Tool | Description | Example |
|------|-------------|---------|
| `weather` | Get weather for a city | "What's the weather in Tokyo?" |
| `calendar` | List/create calendar events | "Schedule 'Meeting' on 2025-05-01" |
| `convert` | Unit conversion | "Convert 5 miles to kilometers" |
| `currency` | Currency exchange | "How much is 100 USD in EUR?" |
| `sql` | Natural language to SQL | "Show me all users" |

## 📁 Project Structure

```
pocket-agent/
├── models/
│   └── pocket-agent.gguf        # Quantized model (Q4_K_M, ~350MB)
├── data/
│   ├── generate_templates.py    # Template-based data generator
│   └── generate_data.py         # Gemini-powered data generator (optional)
├── train/
│   ├── train_lora.py            # QLoRA fine-tuning script
│   └── merge_and_quantize.py    # Merge adapter + GGUF conversion
├── starter/
│   ├── tool_schemas.json        # Tool definitions
│   ├── public_test.jsonl        # 40 dev-set examples
│   └── teacher_examples.jsonl   # 20 seed examples
├── inference.py                 # Grader-compatible inference (CPU-only)
├── eval.py                      # Local evaluation harness
├── demo.py                      # Gradio chatbot demo
├── colab_cell1.py               # Colab training notebook (cell 1)
├── colab_cell2.py               # Colab training notebook (cell 2)
├── colab_small_cell1.py         # SmolLM2-360M training (cell 1)
├── colab_small_cell2.py         # SmolLM2-360M training (cell 2)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🏗️ Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-0.5B-Instruct |
| **Fine-tuning** | QLoRA (r=16, alpha=32, 1.75% trainable) |
| **Training Data** | ~255 template-generated examples |
| **Quantization** | GGUF Q4_K_M |
| **Model Size** | ~350 MB (passes ≤500MB gate) |
| **Inference** | llama-cpp-python, CPU-only |

### Alternative Small Model

| Property | Value |
|----------|-------|
| **Base Model** | SmolLM2-360M-Instruct |
| **Model Size** | ~200 MB (qualifies for ≤250MB bonus) |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo.py
```

### 3. Run Evaluation
```bash
python eval.py
```

## 🔧 Training Pipeline

### Option A: Train on Google Colab (Recommended)

1. Open a new Colab notebook with T4 GPU
2. **Cell 1**: Paste contents of `colab_cell1.py` → Run → Wait for restart
3. **Cell 2**: Paste contents of `colab_cell2.py` → Run → Download GGUF

### Option B: Train Locally
```bash
python data/generate_templates.py    # Generate training data
python train/train_lora.py           # Fine-tune with QLoRA
python train/merge_and_quantize.py   # Merge + quantize to GGUF
```

## 📊 Training Data Distribution

| Category | Count | Description |
|----------|-------|-------------|
| Weather | 66 | Celsius + Fahrenheit queries |
| Calendar | 32 | List + create events |
| Convert | 41 | Unit conversions |
| Currency | 44 | Exchange rate queries |
| SQL | 26 | Natural language to SQL |
| **Refusal** | **46** | **Standard + tricky refusals** |
| **Total** | **255** | |

### Data Quality Strategy
- **15-18% refusal examples** to minimize false-positive tool calls (-0.5 penalty)
- **Tricky refusals**: prompts mentioning cities/amounts that should NOT trigger tools (e.g., "Book a flight to Paris", "Find a hotel in Dubai")
- **Adversarial**: Code-switched (Urdu/Hindi/Spanish/Turkish), typos, misspellings
- **Multi-turn**: Context-dependent follow-up queries

## 🔒 Hard Gates

| Gate | Status | Details |
|------|--------|---------|
| Model ≤ 500MB | ✅ PASS | ~350MB (Q4_K_M) |
| Latency ≤ 30s/prompt | ✅ PASS | ~2-5s on CPU |
| No network imports | ✅ PASS | AST-verified, CPU-only |
| Zero training/test overlap | ✅ PASS | Template-generated data |
| Parameters ≤ 2B | ✅ PASS | 494M parameters |

## 🎁 Bonus Points

| Bonus | Points | Status |
|-------|--------|--------|
| Model ≤ 250MB | +10 | ✅ SmolLM2-360M variant (~200MB) |
| Multi-turn context | +5 | ✅ 12 multi-turn training examples |
| Adversarial robustness | +5 | ✅ 28 code-switched/typo examples |

## ⚙️ Inference API

```python
import inference

# Single turn
response = inference.run("What's the weather in Tokyo?", [])

# Multi-turn
history = [
    {"role": "user", "content": "Weather in London?"},
    {"role": "assistant", "content": '<tool_call>{"tool":"weather","args":{"location":"London","unit":"C"}}</tool_call>'}
]
response = inference.run("What about Paris?", history)
```

## 📝 Design Decisions

1. **Qwen2.5-0.5B**: Chosen for strong instruction-following and JSON output at small size
2. **Q4_K_M quantization**: Best accuracy-to-size ratio at ~350MB
3. **Template data**: Ensures consistent tool-call format and high signal-to-noise
4. **18% refusal ratio**: Prevents costly false-positive tool calls (-0.5 each)
5. **ChatML format**: Native to Qwen, no format conversion needed

## 🐛 Error Analysis

| Error Type | Mitigation |
|------------|------------|
| False positive tool calls | Tricky refusal training (cities in non-tool contexts) |
| Missing args (e.g., unit) | Explicit schemas in system prompt |
| Wrong tool selection | Per-tool training with clear boundaries |
| Code-switched inputs | Adversarial training (Urdu, Spanish, Turkish) |
