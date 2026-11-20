#!/usr/bin/env python3
"""Backward-compatible source-checkout entry point for SciFact benchmarks."""
import sys
from pathlib import Path

# Support running this script directly from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalforge.scifact_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
