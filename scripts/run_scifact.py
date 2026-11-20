#!/usr/bin/env python3
"""Backward-compatible source-checkout entry point for SciFact benchmarks."""
from evalforge.scifact_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
