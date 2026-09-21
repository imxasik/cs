import os
from pathlib import Path

from cyclone.data_loader import process_combined_cyclone_data
from cyclone.feature_manager import run_all_features
from cyclone import plotting          # we'll override plotting.BUFFER and plotting.UCR
from cyclone import config as cfg     # to read defaults from config.ini


# ANSI escape codes for colorful terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
ASSETS_DIR = THIS_DIR / "assets"

DEFAULT_MAP_FILE = ASSETS_DIR / "Map.png"


def print_header():
    """Nice colorful header."""
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}   TROPICAL CYCLONE TRACK & CONE PLOTTER{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")
    print()


def choose_data_file():
    """List all .txt files in the data folder and let the user choose one."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    files = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    if not files:
        raise FileNotFoundError(f"No .txt files found in data folder: {DATA_DIR}")

    print(f"{BOLD}{BLUE}Available data files:{RESET}")
    for idx, fpath in enumerate(files, start=1):
        print(f"  {YELLOW}{idx:2d}{RESET}. {fpath.name}")
    print()

    while True:
        choice = input(
            f"{GREEN}Select a file by number (1-{len(files)}): {RESET}"
        ).strip()
        if not choice:
            print(f"{YELLOW}Please enter a number from 1 to {len(files)}.{RESET}")
            continue
        if not choice.isdigit():
            print(f"{YELLOW}Invalid input. Please enter a number.{RESET}")
            continue
        idx = int(choice)
        if 1 <= idx <= len(files):
            return files[idx - 1]
        print(f"{YELLOW}Number out of range. Try again.{RESET}")


def ask_float_with_default(prompt: str, default_val: float) -> float:
    """
    Ask user for a float value.
    - If user presses Enter => keep default.
    - If user types something invalid => keep default and warn.
    """
    full_prompt = (
        f"{prompt} "
        f"[current: {default_val}] "
        f"{GREEN}(Enter for same){RESET}: "
    )
    raw = input(full_prompt).strip()

    if raw == "":
        # Use default
        print(f"{CYAN}Using current value: {default_val}{RESET}\n")
        return default_val

    try:
        val = float(raw)
        print(f"{CYAN}Using custom value: {val}{RESET}\n")
        return val
    except ValueError:
        print(f"{YELLOW}Invalid number. Keeping default {default_val}.{RESET}\n")
        return default_val


def ask_bool_with_default(prompt: str, default_val: bool) -> bool:
    """
    Ask user for a yes/no (boolean) value.
    - Accepts: yes/no, y/n, 1/0, true/false, on/off (case-insensitive).
    - Enter => keep default.
    """
    default_str = "yes" if default_val else "no"
    full_prompt = (
        f"{prompt} "
        f"[current: {default_str}] "
        f"{GREEN}(yes/no){RESET}: "
    )
    raw = input(full_prompt).strip().lower()

    if raw == "":
        print(f"{CYAN}Using current value: {default_str}{RESET}\n")
        return default_val

    if raw in ("1", "y", "yes", "true", "on"):
        print(f"{CYAN}Using: yes{RESET}\n")
        return True
    if raw in ("0", "n", "no", "false", "off"):
        print(f"{CYAN}Using: no{RESET}\n")
        return False

    print(f"{YELLOW}Invalid input. Keeping current: {default_str}.{RESET}\n")
    return default_val


def main():
    print_header()

    # ---------------- Select data file ----------------
    data_file = choose_data_file()
    print(f"\n{GREEN}Using data file:{RESET} {BOLD}{data_file.name}{RESET}\n")

    # ---------------- Load map ----------------
    map_file = DEFAULT_MAP_FILE
    if not map_file.exists():
        print(f"{YELLOW}WARNING: Map image not found at {map_file}. "
              f"Plot will use blank background.{RESET}")

    # ---------------- Load cyclone data ----------------
    cyclone_name, track_obs, track_for, is_invest = process_combined_cyclone_data(str(data_file))

    # ---------------- Ask for BUFFER & UCR overrides ----------------
    print(f"{BOLD}{BLUE}Plot configuration (from config.ini):{RESET}")
    print(f"  Current BUFFER: {MAGENTA}{cfg.BUFFER}{RESET}")
    print(f"  Current UCR: {MAGENTA}{cfg.UCR}{RESET}")
    print(f"  Current AI Position visible: {MAGENTA}{cfg.SHOW_AI_POSITION}{RESET}")
    print(f"  Current AI Track visible:  {MAGENTA}{cfg.SHOW_BIAS_TRACK}{RESET}\n")

    # Ask user if they want to override BUFFER & UCR, with Enter = keep default
    new_buffer = ask_float_with_default(
        f"{BOLD}Enter BUFFER{RESET}", cfg.BUFFER
    )
    new_ucr = ask_float_with_default(
        f"{BOLD}Enter UCR{RESET}", cfg.UCR
    )

    # Apply overrides to the plotting module (used internally by plot_cyclone)
    plotting.BUFFER = new_buffer
    plotting.UCR = new_ucr

    # Also override in config module (optional, for consistency if used elsewhere)
    cfg.BUFFER = new_buffer
    cfg.UCR = new_ucr

    # ---------------- Ask for AI visibility overrides ----------------
    # Top-level question: default should be NO, and Enter = NO
    change_ai_raw = input(
        f"{BOLD}Change AI Visibility?{RESET} "
        f"[default: {MAGENTA}no{RESET}] "
        f"{GREEN}(yes/no){RESET}: "
    ).strip().lower()

    if change_ai_raw in ("1", "y", "yes", "true", "on"):
        print(f"{CYAN}AI visibility override enabled for this run.{RESET}\n")

        # Ask for AI Position marker visibility (default = current config)
        new_ai_pos = ask_bool_with_default(
            f"{BOLD}Show AI Position marker?{RESET}", cfg.SHOW_AI_POSITION
        )

        # Ask for Bias-Corrected Track visibility (default = current config)
        new_bias_track = ask_bool_with_default(
            f"{BOLD}Show AI-BC Track?{RESET}", cfg.SHOW_BIAS_TRACK
        )

        # Override in both modules (like BUFFER/UCR)
        plotting.SHOW_AI_POSITION = new_ai_pos
        plotting.SHOW_BIAS_TRACK = new_bias_track
        cfg.SHOW_AI_POSITION = new_ai_pos
        cfg.SHOW_BIAS_TRACK = new_bias_track

    else:
        print(f"{CYAN}AI visibility settings unchanged (using config.ini).{RESET}\n")

    print()
    print(f"{CYAN}Final settings for this run:{RESET}")
    print(f"  BUFFER = {GREEN}{new_buffer}{RESET}")
    print(f"  UCR    = {GREEN}{new_ucr}{RESET}")
    print(f"  AI Position visible: {GREEN}{cfg.SHOW_AI_POSITION}{RESET}")
    print(f"  AI Track visible:  {GREEN}{cfg.SHOW_BIAS_TRACK}{RESET}\n")

    # ---------------- Outputs folders ----------------
    outputs_dir = THIS_DIR / "output"
    plots_dir = outputs_dir / "plots"
    files_dir = outputs_dir / "files"

    plots_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Plot output path ----------------
    output_name = f"{cyclone_name}_Track.png"
    output_path = plots_dir / output_name

    print(f"{BOLD}{BLUE}Generating track & cone plot...{RESET}\n")

    # Call plotting function
    plotting.plot_cyclone(
        cyclone_name=cyclone_name,
        track_data_obs=track_obs,
        track_data_for=track_for,
        is_invest=is_invest,
        map_image_path=str(map_file),
        output_path=str(output_path)
    )

    # ---------------- Run pluggable features ----------------
    context = {
        "cyclone_name": cyclone_name,
        "track_obs": track_obs,
        "track_for": track_for,
        "is_invest": is_invest,
        # Summary / text features should write here:
        "output_dir": str(files_dir),
        "map_file": str(map_file),
        "output_image": str(output_path),
        # Extra info for future features:
        "plots_dir": str(plots_dir),
        "files_dir": str(files_dir),
        "outputs_dir": str(outputs_dir),
    }
    run_all_features(context)

    # ---------------- Final messages ----------------
    print(f"{GREEN}✓ Plot saved to:{RESET} {BOLD}output/{output_name}{RESET}")
    print(f"{GREEN}✓ Feature saved in:{RESET} {BOLD}output/files{RESET}")
    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}Done.{RESET} You can now open the PNG from the plots folder.")
    print(f"{CYAN}{'=' * 60}{RESET}")


if __name__ == "__main__":
    main()
