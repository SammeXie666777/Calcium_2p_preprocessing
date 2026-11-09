"""Cell-type pipeline: features -> label / train / predict.

Run from the Calcium_2p_preprocessing\\python folder:
    python -m celltype.pipeline --config celltype\\config.yaml --mode auto

Modes
    features  per-ROI feature tables (cached per session)
    label     features + napari labeling GUI
    train     fit models on everything in the label store (+ CV report)
    predict   features + apply the saved model -> <session>\\celltype\\celltype_model_<id>.*
              (sessions this model has already predicted are skipped; --force redoes them;
              many sessions: see batch_celltype.py)
    auto      model exists -> predict; else labels exist -> train + predict;
              else label -> train -> predict
"""
import argparse
from pathlib import Path

from . import features as ft
from . import io_utils as io
from . import models


def _enough_labels(cfg):
    lab = io.training_labels(cfg)
    spn = lab["stage1"] == "SPN"
    return ((spn & (lab["stage2"] == "red+")).sum() >= 5
            and (spn & (lab["stage2"] == "red-")).sum() >= 5)


def expand_sessions(cfg):
    """The sessions of the config (or --sessions): a path may be a session, its suite2p* output
    folder, plane0, or a folder above sessions; one reference per output folder found."""
    expanded = []
    for s in io.resolve_sessions(cfg):
        found = io.find_sessions(s)
        if not found:
            print(f"  ! no <output>\\plane0\\stat.npy in or below {s}: skipped")
        expanded += found
    return expanded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="auto",
                    choices=["features", "label", "train", "predict", "auto"])
    ap.add_argument("--sessions", nargs="*", help="session folder(s); overrides inputs in the config")
    ap.add_argument("--refresh-features", action="store_true",
                    help="recompute the cached feature tables of --mode features / label (train and predict "
                         "always compute them for their model)")
    ap.add_argument("--model-id", default=None,
                    help="use this saved model (e.g. 20261021-1432) for predict / auto / the labeling GUI; "
                         "default: the latest model (newest id)")
    ap.add_argument("--force", action="store_true",
                    help="predict again even if this model already predicted the session")
    a = ap.parse_args(argv)

    cfg = io.load_config(a.config)
    if a.sessions:
        cfg["inputs"], cfg["inputs_file"] = a.sessions, None

    if a.mode == "train":
        models.train(cfg)
        return

    sessions = expand_sessions(cfg)
    if not sessions and a.mode == "label":
        # no sessions given: open the GUI empty and use its 'open session...' button
        from .gui import label_gui
        label_gui([], cfg, model_id=a.model_id)
        return
    if not sessions:
        raise SystemExit("no sessions: set inputs / inputs_file in the config or use --sessions")
    print(f"{len(sessions)} session(s)")

    if a.mode in ("features", "label"):          # predict / auto compute the features of the sessions they predict, for the model
        feats = ft.extract_features(sessions, cfg, force=a.refresh_features)
        sessions = [Path(s) for s in feats]
    if a.mode == "features":
        return

    if a.mode == "label":
        from .gui import label_gui
        label_gui(sessions, cfg, model_id=a.model_id)
        return

    if a.mode == "predict":
        models.predict(sessions, cfg, force=a.force, model_id=a.model_id)
        return

    # auto
    bundle = models.load_model(cfg, a.model_id)
    if bundle is None:
        if not _enough_labels(cfg):
            print("no model and not enough labels -> opening the labeling GUI")
            from .gui import label_gui
            label_gui(sessions, cfg, model_id=a.model_id)
            if not _enough_labels(cfg):
                print("still not enough labels (need >= 5 red+ and 5 red- SPNs); stopping")
                return
        bundle = models.train(cfg)
    models.predict(sessions, cfg, bundle, force=a.force)


if __name__ == "__main__":
    main()
