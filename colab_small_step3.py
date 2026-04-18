# ============================================================
# STEP 3 ONLY: Convert and Quantize (NO RETRAINING)
# Paste this in your Colab cell and run it!
# ============================================================

import os, sys, subprocess

# UNINSTALL the broken torchvision that's causing the LlamaConfig import to fail!
print("Uninstalling broken torchvision to fix LlamaConfig import error...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True)

if not os.path.exists("llama.cpp"):
    subprocess.run(["git","clone","--depth=1","https://github.com/ggerganov/llama.cpp"], check=True)

req_file = "llama.cpp/requirements.txt"
if os.path.exists(req_file):
    subprocess.check_call([sys.executable,"-m","pip","install","-q","-r",req_file])

print("\nConverting to GGUF...")
res = subprocess.run([sys.executable,"llama.cpp/convert_hf_to_gguf.py","models/merged_model_small",
    "--outfile","models/pocket-agent-small-f16.gguf","--outtype","f16"], capture_output=True, text=True)

if res.returncode != 0:
    print("🚨 CONVERSION FAILED! 🚨")
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
else:
    print("✅ GGUF Conversion successful!")
    try:
        os.makedirs("llama.cpp/build", exist_ok=True)
        print("Building llama-quantize...")
        subprocess.run(["cmake",".."], cwd="llama.cpp/build", check=True, capture_output=True)
        subprocess.run(["cmake","--build",".","--target","llama-quantize","-j4"],
                       cwd="llama.cpp/build", check=True, capture_output=True)
        print("Quantizing to Q4_K_M (this takes ~1 min)...")
        subprocess.run(["llama.cpp/build/bin/llama-quantize","models/pocket-agent-small-f16.gguf",
                        "models/pocket-agent-small.gguf","Q4_K_M"], check=True)
        os.remove("models/pocket-agent-small-f16.gguf")
        
        size_mb = os.path.getsize("models/pocket-agent-small.gguf") / (1024*1024)
        print(f"\n🎉 Quantization complete! Final GGUF Size: {size_mb:.1f} MB 🎉")
        
        try:
            from google.colab import files
            print("Triggering download...")
            files.download("models/pocket-agent-small.gguf")
        except: pass
    except Exception as e:
        print(f"Quantization failed ({e})")
