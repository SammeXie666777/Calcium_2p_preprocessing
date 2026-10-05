"""Source extraction: runs ..\\run_suite2p_on_MC.py on each session."""
import json
import sys
from datetime import datetime
from pathlib import Path

from . import io_utils as io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # python\ folder


def run_extraction(sessions, cfg):
    import run_suite2p_on_MC as s2p      # imported here: needs suite2p installed

    p = cfg["suite2p"]
    # make_settings / run_session read these module-level constants
    s2p.FS, s2p.TAU, s2p.DIAMETER = float(p["fs"]), float(p["tau"]), int(p["diameter"])
    s2p.MARGIN, s2p.USE_MC_BAD_FRAMES = int(p["margin"]), bool(p["use_mc_bad_frames"])
    s2p.CHAN2_THRESHOLD, s2p.CELLPOSE_CHAN2 = float(p["chan2_threshold"]), bool(p["cellpose_chan2"])
    settings = s2p.make_settings()

    done, failed = [], []
    for sess in sessions:
        if (io.plane_dir(sess) / "stat.npy").is_file() and not p["overwrite"]:
            print(f"  suite2p: already done, skipping {sess}")
            done.append(sess)
            continue
        try:
            s2p.run_session(str(sess), settings)
            manifest = {k: p[k] for k in ("fs", "tau", "diameter", "margin",
                                          "use_mc_bad_frames", "chan2_threshold",
                                          "cellpose_chan2")}
            manifest["timestamp"] = datetime.now().isoformat(timespec="seconds")
            with open(io.out_dir(sess) / "suite2p_run.json", "w") as f:
                json.dump(manifest, f, indent=2)
            done.append(sess)
        except Exception as e:
            print(f"  ! suite2p FAILED for {sess}: {e}")
            failed.append(sess)
    print(f"suite2p: {len(done)}/{len(sessions)} session(s) ready")
    return done
