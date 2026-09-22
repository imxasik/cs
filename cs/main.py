#!/usr/bin/env python3
"""
Cyclone Track & Cone Plotter — entry point.

Interactive wizard (Pydroid 3 / phone friendly):

    python main.py

One-shot CLI (laptop / Termux / scripts):

    python main.py data/2025/ditwah.txt --buffer 2.5 --ucr 0.25
    python main.py --list

All behaviour (toggles, map extent, DPI, footer, ...) lives in config.ini.
"""
from cyclone.cli import main

if __name__ == "__main__":
    main()
