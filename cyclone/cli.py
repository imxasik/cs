"""
Command-line interface for the Cyclone Track & Cone Plotter.

Two ways to run
---------------
1. Wizard (default, phone-friendly)::

       python main.py

   Shows a numbered list of data files (newest first) and asks a few
   questions — pressing Enter at any prompt accepts the config.ini value.

2. One-shot CLI (laptop / Termux / automation)::

       python main.py data/2025/ditwah.txt
       python main.py data/95B.txt --buffer 2.5 --ucr 0.25 --no-features
       python main.py --list

   With a file (or any option) given, no questions are asked: every
   setting comes from config.ini unless overridden on the command line.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))  # make `assets` / `features` importable from anywhere

from cyclone.data_loader import process_combined_cyclone_data
from cyclone.feature_manager import run_all_features
from cyclone import config as cfg
from cyclone import plotting

# --- ANSI colors (safe on Pydroid / Termux / most desktop terminals) ---
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"


# --------------------------------------------------------------------------
# Small prompt helpers (wizard mode)
# --------------------------------------------------------------------------
def _ask(prompt: str) -> str:
    """input() that survives EOF (piped input / no console) -> returns ''."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)


def ask_float_with_default(prompt: str, default_val: float) -> float:
    raw = _ask(f"{prompt} [current: {default_val}] {GREEN}(Enter for same){RESET}: ")
    if raw == "":
        return default_val
    try:
        return float(raw)
    except ValueError:
        print(f"{YELLOW}Invalid number. Keeping default {default_val}.{RESET}")
        return default_val


def ask_bool_with_default(prompt: str, default_val: bool) -> bool:
    default_str = "yes" if default_val else "no"
    raw = _ask(f"{prompt} [current: {default_str}] {GREEN}(yes/no){RESET}: ").lower()
    if raw == "":
        return default_val
    if raw in ("1", "y", "yes", "true", "on"):
        return True
    if raw in ("0", "n", "no", "false", "off"):
        return False
    print(f"{YELLOW}Invalid input. Keeping current: {default_str}.{RESET}")
    return default_val


# --------------------------------------------------------------------------
# Data-file discovery
# --------------------------------------------------------------------------
def find_data_files(data_dir: Path | None = None) -> list[Path]:
    """All .txt track files under data/ (sub-folders included), newest first."""
    data_dir = data_dir or (THIS_DIR / "data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    files = [p for p in data_dir.rglob("*.txt") if p.is_file()]
    files += [p for p in data_dir.rglob("*.TXT") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def print_file_list(files: list[Path]) -> None:
    print(f"{BOLD}{BLUE}Available data files (newest first):{RESET}")
    for idx, fpath in enumerate(files, start=1):
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        try:
            shown = fpath.relative_to(THIS_DIR)
        except ValueError:
            shown = fpath
        print(f"  {YELLOW}{idx:2d}{RESET}. {shown}  {CYAN}({mtime}){RESET}")
    print()


def choose_data_file_interactive(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            # Allow passing just a name, e.g. `--file ditwah`
            for cand in find_data_files():
                if cand.stem.lower() == p.stem.lower():
                    return cand
            raise FileNotFoundError(f"Data file not found: {explicit}")
        return p

    files = find_data_files()
    if not files:
        raise FileNotFoundError(
            "No .txt files found in data/ — add a track file first (see data/README.md)."
        )
    print_file_list(files)
    while True:
        raw = _ask(f"{GREEN}Select a file [1-{len(files)}, Enter=1 (newest)]: {RESET}")
        if raw == "":
            idx = 1
        elif raw.isdigit() and 1 <= int(raw) <= len(files):
            idx = int(raw)
        else:
            print(f"{YELLOW}Please enter a number from 1 to {len(files)}.{RESET}")
            continue
        return files[idx - 1]


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------
def process(cyclone_name, track_obs, track_for, is_invest,
            out_root: Path, run_features: bool = True) -> Path:
    """Generate the plot + run feature plugins. Returns the PNG path."""
    map_file = THIS_DIR / "assets" / "Map.png"
    if not map_file.exists():
        print(f"{YELLOW}WARNING: Map image not found at {map_file}. "
              f"Plot will use a blank background.{RESET}")

    plots_dir = out_root / "plots"
    files_dir = out_root / "files"
    plots_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    if cfg.ORGANIZE_BY_YEAR:
        if not track_obs.empty:
            year = str(track_obs["tnd"].iloc[-1].year)
        else:
            year = str(datetime.now().year)
        plots_dir = plots_dir / year
        plots_dir.mkdir(parents=True, exist_ok=True)

    output_path = plots_dir / f"{cyclone_name}_Track.png"

    print(f"{BOLD}{BLUE}Generating track & cone plot...{RESET}\n")
    plotting.plot_cyclone(
        cyclone_name=cyclone_name,
        track_data_obs=track_obs,
        track_data_for=track_for,
        is_invest=is_invest,
        map_image_path=str(map_file),
        output_path=str(output_path),
    )
    print(f"{GREEN}✓ Plot saved to:{RESET} {BOLD}{output_path}{RESET}")

    if run_features:
        context = {
            "cyclone_name": cyclone_name,
            "track_obs": track_obs,
            "track_for": track_for,
            "is_invest": is_invest,
            "output_dir": str(files_dir),
            "map_file": str(map_file),
            "output_image": str(output_path),
            "plots_dir": str(plots_dir),
            "files_dir": str(files_dir),
            "outputs_dir": str(out_root),
        }
        run_all_features(context)
        print(f"{GREEN}✓ Feature outputs in:{RESET} {BOLD}{files_dir}{RESET}")

    return output_path


def load_track(data_file: Path):
    print(f"\n{GREEN}Using data file:{RESET} {BOLD}{data_file}{RESET}")
    return process_combined_cyclone_data(str(data_file))


def print_header():
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}   TROPICAL CYCLONE TRACK & CONE PLOTTER{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cyclone-tracker",
        description="NHC-style cyclone track & uncertainty cone plotter.",
        epilog="Run without arguments for the interactive wizard.",
    )
    p.add_argument("file", nargs="?", help="track .txt file (path under data/, or bare name)")
    p.add_argument("-l", "--list", action="store_true", help="list data files and exit")
    p.add_argument("--buffer", type=float, default=None, help="map padding around track (default: config)")
    p.add_argument("--ucr", type=float, default=None, help="cone growth rate (default: config)")
    p.add_argument("--dpi", type=int, default=None, help="output PNG DPI (default: config)")
    p.add_argument("--show-ai", action="store_true", help="force AI overlays ON (position + BC track)")
    p.add_argument("--no-ai", action="store_true", help="force AI overlays OFF")
    p.add_argument("--no-features", action="store_true", help="skip plugins in features/")
    p.add_argument("-o", "--outdir", default=None, help="output root folder (default: output/)")
    p.add_argument("-y", "--yes", action="store_true", help="never prompt; accept config defaults")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print_header()

    if args.list:
        files = find_data_files()
        if not files:
            print(f"{YELLOW}No .txt files found in data/.{RESET}")
            return 1
        print_file_list(files)
        return 0

    # ---- Resolve the data file ----
    interactive = args.file is None and not args.yes and sys.stdin.isatty()
    data_file = choose_data_file_interactive(args.file)

    cyclone_name, track_obs, track_for, is_invest = load_track(data_file)

    # ---- Apply settings: config.ini defaults, then overrides ----
    if interactive:
        print(f"{BOLD}{BLUE}Plot configuration (from config.ini):{RESET}")
        print(f"  BUFFER: {MAGENTA}{cfg.BUFFER}{RESET} | "
              f"UCR: {MAGENTA}{cfg.UCR}{RESET} | "
              f"DPI: {MAGENTA}{cfg.OUTPUT_DPI}{RESET}\n")
        if args.buffer is None:
            args.buffer = ask_float_with_default(f"{BOLD}BUFFER{RESET}", cfg.BUFFER)
        if args.ucr is None:
            args.ucr = ask_float_with_default(f"{BOLD}UCR{RESET}", cfg.UCR)
        if args.show_ai is False and not args.no_ai:
            change_ai = _ask(
                f"{BOLD}Change AI overlays?{RESET} [current: "
                f"{'yes' if cfg.SHOW_AI_POSITION or cfg.SHOW_BIAS_TRACK else 'no'}] "
                f"{GREEN}(Enter=no){RESET}: "
            ).lower()
            if change_ai in ("1", "y", "yes", "true", "on"):
                args.show_ai = ask_bool_with_default(
                    f"{BOLD}Show AI overlays (position + BC track){RESET}", True
                )
                args.no_ai = not args.show_ai

    if args.buffer is not None:
        plotting.BUFFER = cfg.BUFFER = args.buffer
    if args.ucr is not None:
        plotting.UCR = cfg.UCR = args.ucr
    if args.dpi is not None:
        plotting.OUTPUT_DPI = args.dpi
    if args.show_ai:
        plotting.SHOW_AI_POSITION = cfg.SHOW_AI_POSITION = True
        plotting.SHOW_BIAS_TRACK = cfg.SHOW_BIAS_TRACK = True
    if args.no_ai:
        plotting.SHOW_AI_POSITION = cfg.SHOW_AI_POSITION = False
        plotting.SHOW_BIAS_TRACK = cfg.SHOW_BIAS_TRACK = False

    out_root = Path(args.outdir).resolve() if args.outdir else (THIS_DIR / "output")

    try:
        output_path = process(
            cyclone_name, track_obs, track_for, is_invest,
            out_root=out_root, run_features=not args.no_features,
        )
    except Exception as e:
        print(f"{RED}ERROR: {e}{RESET}")
        print(f"{YELLOW}If this looks like a bug, please open an issue with the data "
              f"file you used.{RESET}")
        raise

    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}Done.{RESET} Open: {BOLD}{output_path}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Cancelled.{RESET}")
        sys.exit(130)
