#!/usr/bin/env python
"""Standalone entry point for the Knowledge Palace eval harness.

Usage:
    .venv/bin/python evals/run_eval.py --gate --show-misses
    .venv/bin/python evals/run_eval.py --domains cpp,rust
    .venv/bin/python evals/run_eval.py --gate --baseline evals/baselines/baseline-2026-10-26.json

Prefer ``kp eval`` — this wrapper exists for direct invocation without the CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_palace.evalharness import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
