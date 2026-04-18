# ============================================================
# CELL 1: Install dependencies + fix pyarrow + restart runtime
# Run this cell FIRST. It will restart the runtime automatically.
# Then run Cell 2.
# ============================================================
import subprocess
import sys

# Fix pyarrow FIRST (must be before other installs)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pyarrow>=17.0.0"])

# Install all packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0",
    "peft>=0.13.0",
    "trl>=0.12.0",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.44.0",
    "sentencepiece>=0.2.0",
])

print("All dependencies installed! Restarting runtime...")

# Auto-restart runtime to pick up new pyarrow binary
import os
os.kill(os.getpid(), 9)
