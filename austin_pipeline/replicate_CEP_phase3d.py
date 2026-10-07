#!/usr/bin/env python3
"""One-command replication for the Phase 3D CEP package."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for script in ["replicate_CEP.py", "phase3d_analysis.py"]:
    subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)
print("CEP Phase 3D replication complete.")
