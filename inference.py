"""
Pocket-Agent Inference Module
==============================
Grader-compatible inference function for the Pocket-Agent model.

Exposes: def run(prompt: str, history: list[dict]) -> str

CRITICAL RULES:
- No network imports (requests, urllib, http, socket)
- Model loaded once at module level
- Uses only: llama_cpp, json, re, os, typing
"""
import json
import re
import os
from typing import Optional

from llama_cpp import Llama

# =====================================================================
# Model Configuration
# =====================================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pocket-agent.gguf")

SYSTEM_PROMPT = (
    "You are a helpful mobile assistant. You have access to the following tools: "
    "weather, calendar, convert, currency, sql. "
    "When the user's request matches a tool, respond with a tool call in "
    "tool_call tags like this: TOOL_START{...}TOOL_END. Otherwise, respond in plain text.\n\n"
    "Tool schemas:\n"
    '- weather: {"tool": "weather", "args": {"location": "string", "unit": "C|F"}}\n'
    '- calendar: {"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}\n'
    '- convert: {"tool": "convert", "args": {"value": number, "from_unit": "string", "to_unit": "string"}}\n'
    '- currency: {"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}\n'
    '- sql: {"tool": "sql", "args": {"query": "string"}}\n\n'
    "If no tool fits the request, respond with helpful plain text (no tool_call tags)."
)

# Replace placeholders with actual XML-like tags
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("TOOL_START", "<tool" + "_call>").replace("TOOL_END", "</tool" + "_call>")

# =====================================================================
# Load model once at module level
# =====================================================================
_model = None


def _get_model():
    """Lazy-load the model on first call."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run training and quantization first, or place pocket-agent.gguf in models/"
            )
        _model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
    return _model


def _build_messages(history: list[dict], prompt: str) -> list[dict]:
    """Build the message list from history and current prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    return messages


def run(prompt: str, history: list[dict]) -> str:
    """
    Run inference on a single prompt with optional conversation history.

    Args:
        prompt: The current user message
        history: List of previous messages, each with 'role' and 'content' keys

    Returns:
        The model's response string (either a tool call or plain text)
    """
    model = _get_model()
    messages = _build_messages(history, prompt)

    try:
        # Use chat completion API
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
        )
        result = response["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: format as ChatML and use raw completion
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"

        response = model(
            formatted,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>"],
        )
        result = response["choices"][0]["text"]

    return result.strip() if result else ""


# =====================================================================
# Quick test
# =====================================================================
if __name__ == "__main__":
    test_cases = [
        ("What's the weather in London?", []),
        ("Convert 100 USD to EUR", []),
        ("Tell me a joke", []),
        ("Create a meeting called 'Standup' on 2025-04-01", []),
    ]

    for prompt, history in test_cases:
        print(f"\nUser: {prompt}")
        try:
            response = run(prompt, history)
            print(f"Assistant: {response}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            break
