"""
Merge LoRA Adapter + Convert to GGUF + Quantize
=================================================
Merges the trained LoRA adapter into the base model, converts to GGUF format,
and quantizes for efficient CPU inference.

Requirements:
    pip install transformers peft torch
    
    For GGUF conversion, you need llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && pip install -r requirements.txt

Usage:
    python train/merge_and_quantize.py
"""
import os
import sys
import shutil
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# =====================================================================
# Configuration
# =====================================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lora_adapter")
MERGED_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "merged_model")
GGUF_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "models", "pocket-agent.gguf")
QUANT_TYPE = "Q4_K_M"  # Good balance of size and quality. ~350MB for 0.5B model


def merge_lora():
    """Merge LoRA adapter into base model."""
    print("=" * 60)
    print("Step 1: Merging LoRA adapter into base model")
    print("=" * 60)
    
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA adapter: {LORA_PATH}")
    
    # Load base model
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Load and merge LoRA
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("  Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    os.makedirs(MERGED_PATH, exist_ok=True)
    print(f"  Saving merged model to {MERGED_PATH}...")
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    
    print("  ✓ Merge complete!")
    return MERGED_PATH


def convert_to_gguf(merged_path):
    """Convert merged model to GGUF format using llama.cpp."""
    print("\n" + "=" * 60)
    print("Step 2: Converting to GGUF format")
    print("=" * 60)
    
    # Try to find llama.cpp convert script
    llama_cpp_paths = [
        os.path.join(os.path.dirname(__file__), "..", "llama.cpp"),
        os.path.expanduser("~/llama.cpp"),
        "/content/llama.cpp",  # Colab
    ]
    
    convert_script = None
    for base in llama_cpp_paths:
        candidate = os.path.join(base, "convert_hf_to_gguf.py")
        if os.path.exists(candidate):
            convert_script = candidate
            break
    
    if convert_script is None:
        print("  llama.cpp not found. Cloning...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp"],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            check=True,
        )
        convert_script = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "convert_hf_to_gguf.py")
        
        # Install deps
        req_file = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "requirements.txt")
        if os.path.exists(req_file):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)
    
    # Convert to f16 GGUF first
    f16_gguf = GGUF_OUTPUT.replace(".gguf", "-f16.gguf")
    print(f"  Converting to f16 GGUF...")
    subprocess.run(
        [sys.executable, convert_script, merged_path, "--outfile", f16_gguf, "--outtype", "f16"],
        check=True,
    )
    
    print("  ✓ f16 GGUF created!")
    return f16_gguf


def quantize_gguf(f16_path):
    """Quantize the GGUF model."""
    print("\n" + "=" * 60)
    print(f"Step 3: Quantizing to {QUANT_TYPE}")
    print("=" * 60)
    
    # Find llama-quantize binary
    llama_cpp_base = os.path.join(os.path.dirname(__file__), "..", "llama.cpp")
    quantize_bin = None
    
    for candidate in [
        os.path.join(llama_cpp_base, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_base, "build", "bin", "llama-quantize.exe"),
        os.path.join(llama_cpp_base, "llama-quantize"),
        shutil.which("llama-quantize"),
    ]:
        if candidate and os.path.exists(candidate):
            quantize_bin = candidate
            break
    
    if quantize_bin is None:
        # Try building llama.cpp
        print("  Building llama.cpp quantize tool...")
        build_dir = os.path.join(llama_cpp_base, "build")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--target", "llama-quantize", "-j"], cwd=build_dir, check=True)
        
        for candidate in [
            os.path.join(build_dir, "bin", "llama-quantize"),
            os.path.join(build_dir, "bin", "llama-quantize.exe"),
        ]:
            if os.path.exists(candidate):
                quantize_bin = candidate
                break
    
    if quantize_bin is None:
        print("  ⚠ Could not find/build llama-quantize.")
        print("  Using Python-based quantization via llama-cpp-python instead...")
        # Alternative: use the convert script with quantization
        llama_cpp_base = os.path.join(os.path.dirname(__file__), "..", "llama.cpp")
        convert_script = os.path.join(llama_cpp_base, "convert_hf_to_gguf.py")
        
        merged_path = os.path.join(os.path.dirname(__file__), "..", "models", "merged_model")
        subprocess.run(
            [sys.executable, convert_script, merged_path, 
             "--outfile", GGUF_OUTPUT, "--outtype", "q4_k_m" if QUANT_TYPE == "Q4_K_M" else QUANT_TYPE.lower()],
            check=True,
        )
        
        if os.path.exists(f16_path):
            os.remove(f16_path)
        
        size_mb = os.path.getsize(GGUF_OUTPUT) / (1024 * 1024)
        print(f"  ✓ Quantized model: {GGUF_OUTPUT} ({size_mb:.1f} MB)")
        return GGUF_OUTPUT
    
    print(f"  Quantizing {f16_path} → {GGUF_OUTPUT}")
    subprocess.run(
        [quantize_bin, f16_path, GGUF_OUTPUT, QUANT_TYPE],
        check=True,
    )
    
    # Clean up f16
    if os.path.exists(f16_path):
        os.remove(f16_path)
    
    size_mb = os.path.getsize(GGUF_OUTPUT) / (1024 * 1024)
    print(f"  ✓ Quantized model: {GGUF_OUTPUT} ({size_mb:.1f} MB)")
    
    # Gate check
    if size_mb <= 500:
        print(f"  ✅ PASSES ≤500 MB gate!")
    else:
        print(f"  ⚠ FAILS ≤500 MB gate! Consider using Q3_K_S or Q2_K")
    
    if size_mb <= 250:
        print(f"  🏆 QUALIFIES for ≤250 MB bonus!")
    
    return GGUF_OUTPUT


def main():
    # Step 1: Merge LoRA
    merged_path = merge_lora()
    
    # Step 2: Convert to GGUF
    f16_path = convert_to_gguf(merged_path)
    
    # Step 3: Quantize
    gguf_path = quantize_gguf(f16_path)
    
    # Clean up merged model
    print(f"\n🧹 Cleaning up merged model directory...")
    if os.path.exists(MERGED_PATH):
        shutil.rmtree(MERGED_PATH)
    
    print(f"\n✅ Done! Quantized model ready at: {gguf_path}")
    print(f"   Next step: python eval.py")


if __name__ == "__main__":
    main()
