# ============================================================
# CELL 1 (SMALL MODEL): Install deps + restart
# DO NOT touch torch/torchvision — Colab's versions are fine
# FIRST: Runtime → "Disconnect and delete runtime"
# ============================================================
import subprocess
import sys

# ONLY upgrade pyarrow (for datasets) and install ML packages
# DO NOT upgrade torch or torchvision!
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--upgrade", "pyarrow>=17.0.0"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0",
    "peft>=0.13.0",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "sentencepiece>=0.2.0",
])

print("All installed! Restarting...")
import os
os.kill(os.getpid(), 9)
