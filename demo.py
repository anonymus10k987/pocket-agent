"""
Pocket-Agent Demo — Gradio Chatbot UI
======================================
Clean, professional chatbot showing model's tool-calling capabilities.
Run: python demo.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

TOOL_ICONS = {
    "weather": "🌤️", "calendar": "📅", "convert": "🔄",
    "currency": "💱", "sql": "🗄️",
}

def format_response(response: str) -> str:
    """Parse tool calls and format them nicely for the UI."""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            tc = json.loads(match.group(1))
            tool = tc.get("tool", "unknown")
            args = tc.get("args", {})
            icon = TOOL_ICONS.get(tool, "🔧")
            args_formatted = "\n".join(f"  • **{k}**: `{v}`" for k, v in args.items())
            return (
                f"{icon} **Tool Call: `{tool}`**\n\n"
                f"{args_formatted}\n\n"
                f"```json\n{json.dumps(tc, indent=2)}\n```"
            )
        except json.JSONDecodeError:
            pass
    return response


def chat(message: str, history: list) -> str:
    """Process user message and return formatted response."""
    import inference

    # Convert Gradio history format to our format
    formatted_history = []
    for turn in history:
        if isinstance(turn, dict):
            formatted_history.append(turn)
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            formatted_history.append({"role": "user", "content": turn[0]})
            if turn[1]:
                formatted_history.append({"role": "assistant", "content": turn[1]})

    try:
        raw = inference.run(message, formatted_history)
        return format_response(raw)
    except FileNotFoundError:
        return (
            "⚠️ **Model not found!**\n\n"
            "Place `pocket-agent.gguf` in the `models/` folder.\n\n"
            "```\nmodels/pocket-agent.gguf\n```"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


EXAMPLES = [
    ["What's the weather in Tokyo?"],
    ["Convert 100 USD to EUR"],
    ["Schedule 'Team Meeting' on 2025-05-01"],
    ["How many kilometers is 5 miles?"],
    ["Show me all users"],
    ["Lahore mein mausam kya hai?"],
    ["Tell me a joke"],
    ["Book me a flight to Paris"],
]

CSS = """
.gradio-container { max-width: 800px !important; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Pocket-Agent") as demo:
    gr.Markdown(
        """
        # 🤖 Pocket-Agent
        ### On-Device Tool-Calling Assistant

        A fine-tuned language model that runs **100% locally on CPU** (no internet needed).
        It can call 5 tools: **weather**, **calendar**, **convert**, **currency**, and **sql**.

        | Tool | Example |
        |------|---------|
        | 🌤️ Weather | "What's the weather in Tokyo?" |
        | 📅 Calendar | "Schedule 'Meeting' on 2025-05-01" |
        | 🔄 Convert | "Convert 5 miles to kilometers" |
        | 💱 Currency | "How much is 100 USD in EUR?" |
        | 🗄️ SQL | "Show me all users" |

        ---
        """
    )

    chatbot = gr.ChatInterface(
        fn=chat,
        examples=EXAMPLES,
        retry_btn="🔄 Retry",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )

    gr.Markdown(
        """
        ---
        *Pocket-Agent • SmolLM2-360M fine-tuned with LoRA • GGUF Q4_K_M quantized • CPU inference via llama.cpp*
        """
    )

demo.launch(share=True, server_name="0.0.0.0")
