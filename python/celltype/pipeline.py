"""Cell-type pipeline: Suite2p -> features -> label / train / predict.

Run from the Calcium_2p_preprocessing\\python folder:
    python -m celltype.pipeline --config celltype\\config.yaml --mode auto

Modes
    extract   Suite2p on each session (run_suite2p_on_MC.py), skips finished ones
    features  extract + per-ROI feature tables (cached per session)
    label     extract + features + napari labeling GUI
    train     fit models on everything in the label store (+ CV report)
    predict   extract + features + apply the saved model -> <session>\\celltype\\labels.*
    auto      model exists -> predict; else labels exist -> train + predict;
              else label -> train -> predict
"""
import argparse
from pathlib import Path

from . import extraction
from . import features as ft
from . import io_utils as io
from . import models


def _enough_labels(cfg):
    lab = io.load_labels(cfg["label_store"])
    spn = lab["stage1"] == "SPN"
    return ((spn & (lab["stage2"] == "red+")).sum() >= 5
            and (spn & (lab["stage2"] == "red-")).sum() >= 5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="auto",
                    choices=["extract", "features", "label", "train", "predict", "auto"])
    ap.add_argument("--sessions", nargs="*", help="override inputs in the config")
    ap.add_argument("--skip-extract", action="store_true",
                    help="do not call Suite2p (outputs must already exist)")
    ap.add_argument("--refresh-features", action="store_true",
                    help="recompute cached feature tables")
    a = ap.parse_args(argv)

    cfg = io.load_config(a.config)
    if a.sessions:
        cfg["inputs"], cfg["inputs_file"] = a.sessions, None

    if a.mode == "train":
        models.train(cfg)
        return

    sessions = io.resolve_sessions(cfg)
    if not sessions:
        raise SystemExit("no sessions: set inputs / inputs_file in the config or use --sessions")
    print(f"{len(sessions)} session(s)")

    if not a.skip_extract:
        sessions = extraction.run_extraction(sessions, cfg)
    else:
        sessions = [s for s in sessions if (io.plane_dir(s) / "stat.npy").is_file()]
    if a.mode == "extract":
        return

    feats = ft.extract_features(sessions, cfg, force=a.refresh_features)
    sessions = [Path(s) for s in feats]
    if a.mode == "features":
        return

    if a.mode == "label":
        from .gui import label_gui
        label_gui(sessions, cfg)
        return

    if a.mode == "predict":
        models.predict(sessions, cfg)
        return

    # auto
    bundle = models.load_model(cfg)
    if bundle is None:
        if not _enough_labels(cfg):
            print("no model and not enough labels -> opening the labeling GUI")
            from .gui import label_gui
            label_gui(sessions, cfg)
            if not _enough_labels(cfg):
                print("still not enough labels (need >= 5 red+ and 5 red- SPNs); stopping")
                return
        bundle = models.train(cfg)
    models.predict(sessions, cfg, bundle)


if __name__ == "__main__":
    main()
