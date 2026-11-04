import sys
from pathlib import Path

# Tests import `src` and `eval` as packages, so the repo root must be on the
# path regardless of where pytest was invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
