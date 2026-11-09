"""Batch cell-type prediction and review queue over many sessions (details: README.md, section
'Batch prediction and review queue').

Run from the Calcium_2p_preprocessing/python folder:

    python -m celltype.batch_celltype predict --config celltype/config.yaml
    python -m celltype.batch_celltype review  --config celltype/config.yaml
    python -m celltype.batch_celltype status  --config celltype/config.yaml

    predict   predict every session with the saved model and write celltype_model_<id>.csv/.mat
              (no traces) into <session>/celltype/. Sessions this model has already predicted for
              their current Suite2p run are skipped; a newer model predicts them again and the older
              files stay. --force predicts again anyway.
    review    open the GUI on the sessions that have a prediction of the newest model and no saved
              edit yet (then the sessions edited only under an older model). The GUI's 'next in
              queue' button walks through them; Save marks a session done.
    status    list the sessions by state, change nothing.

The sessions are the `inputs` / `inputs_file` of the config, or --sessions (a session, its suite2p
output folder, or a folder above many sessions).
"""
import argparse

from . import curated
from . import io_utils as io
from .pipeline import expand_sessions

STATES = [("no_prediction", "no prediction of the current Suite2p run yet (run: predict)"),
          ("pending", "predicted by the newest model, not reviewed (Save in the GUI)"),
          ("stale", "reviewed only under an older model: review again"),
          ("reviewed", "edited file exists for the newest model"),
          ("no_suite2p", "no Suite2p output")]


def scan(sessions, cfg):
    """{state: [session, ...]} in the order of `sessions`."""
    out = {k: [] for k, _ in STATES}
    for s in sessions:
        st = curated.session_state(s)
        out.setdefault(st["state"], []).append(s)
    return out


def print_status(groups, cfg, show=("no_prediction", "pending", "stale", "no_suite2p")):
    print(f"{sum(len(v) for v in groups.values())} session(s):")
    for key, text in STATES:
        print(f"  {len(groups.get(key, [])):4d}  {key:14s} {text}")
    for key in show:
        for s in groups.get(key, []):
            print(f"      [{key}] {io.session_id(s, cfg['session_id_depth'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["predict", "review", "status"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--sessions", nargs="*",
                    help="session folder(s) or a folder above them; default: inputs of the config")
    ap.add_argument("--force", action="store_true",
                    help="predict: predict again even if this model already did")
    ap.add_argument("--model-id", default=None,
                    help="predict / review: use this saved model (e.g. 20261021-1432); default: the latest model")
    a = ap.parse_args(argv)

    cfg = io.load_config(a.config)
    if a.sessions:
        cfg["inputs"], cfg["inputs_file"] = a.sessions, None
    sessions = expand_sessions(cfg)
    if not sessions:
        raise SystemExit("no sessions: set inputs / inputs_file in the config or use --sessions")

    if a.command == "predict":
        from . import models
        bundle = models.load_model(cfg, a.model_id)
        if bundle is None:
            raise SystemExit(f"no model in {cfg['model_dir']}: label + train first")
        res = models.predict(sessions, cfg, bundle, force=a.force)
        for s, why in res["failed"]:
            print(f"  failed: {s}: {why}")
        return

    groups = scan(sessions, cfg)
    if a.command == "status":
        print_status(groups, cfg)
        return

    # review: pending first, then the ones edited only under an older model
    queue = groups["pending"] + groups["stale"]
    print_status(groups, cfg, show=("no_prediction",))
    if not queue:
        print("nothing to review" + (" (run: predict first)" if groups["no_prediction"] else ""))
        return
    print(f"opening the GUI with {len(queue)} session(s): {len(groups['pending'])} pending, "
          f"{len(groups['stale'])} stale")
    from .gui import label_gui
    label_gui(queue, cfg, model_id=a.model_id)


if __name__ == "__main__":
    main()
