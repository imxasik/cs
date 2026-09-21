import importlib
import pkgutil
import sys
from pathlib import Path

def run_all_features(context: dict):
    """
    Auto-discover and run all feature modules under the root-level `features` package.

    Each module that wants to participate must define:

        def run_feature(context: dict):
            ...

    The `context` dict typically contains:
        - cyclone_name
        - track_obs
        - track_for
        - is_invest
        - output_dir
        - map_file
        - output_image
    """
    root_dir = Path(__file__).resolve().parent.parent
    features_dir = root_dir / "features"

    if not features_dir.exists():
        return

    # Ensure root_dir is in sys.path so `features` is importable
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    try:
        import features  # type: ignore
    except ImportError as e:
        print(f"[FEATURE] Could not import features package: {e}")
        return

    for module_info in pkgutil.iter_modules(features.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        full_name = f"features.{name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception as e:
            print(f"[FEATURE] Failed to import {full_name}: {e}")
            continue

        fn = getattr(mod, "run_feature", None)
        if callable(fn):
            try:
                print(f"[FEATURE] Running {full_name}.run_feature()")
                fn(context)
            except Exception as e:
                print(f"[FEATURE] Error in {full_name}.run_feature: {e}")
