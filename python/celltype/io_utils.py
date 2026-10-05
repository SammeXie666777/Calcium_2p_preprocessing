"""Config, session discovery, Suite2p output loading and the label store."""
import copy
import hashlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

MC_TIF = "ConcatenatedMovie_MC.tif"
RED_TIF = "ref_field_MC_registeration.tif"
SHIFTS_MAT = "ConcatenatedMovie_MC_shifts.mat"
OUT_FOLDER = "celltype"

DEFAULTS = {
    "inputs": [], "inputs_file": None, "session_id_depth": 3,
    "suite2p": {"fs": 30.0, "tau": 1.0, "diameter": 12, "margin": 0,
                "use_mc_bad_frames": True, "chan2_threshold": 0.65,
                "cellpose_chan2": False, "overwrite": False},
    "um_per_px": 1.0, "cre_line": "D1",
    "feature_version": "v1", "neuropil_coef": 0.7,
    "baseline_window_s": 60, "baseline_percentile": 8,
    "event_z": 3.0, "event_min_dur_s": 0.1, "red_bleedthrough": False,
    "label_store": None, "model_dir": None, "labeler": "",
    "hard_rules": {"cin_min_diam_um": 20, "min_solidity": 0.80, "max_aspect": 2.5},
    "model1": "selftrain_logreg", "model2": "labelspreading",
    "min_labels_per_class": 20, "anomaly_quantile": [0.01, 0.05],
    "thresholds": {"p_spn_hi": 0.9, "p_spn_lo": 0.2, "p_red_hi": 0.9, "p_red_lo": 0.1},
    "use_iscell": True,
}

LABEL_COLUMNS = ["session_id", "session_dir", "roi_idx", "stage1", "stage2",
                 "labeler", "timestamp", "s2p_hash"]
STAGE1_LABELS = ["SPN", "interneuron", "junk", "unsure"]
STAGE2_LABELS = ["red+", "red-", "ambiguous"]


# ---------------------------------------------------------------- config
def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    with open(path, "r") as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    for k in ("label_store", "model_dir"):
        if not cfg.get(k):
            raise ValueError(f"set {k} in {path}")
    cfg["cre_line"] = str(cfg["cre_line"]).upper()
    if cfg["cre_line"] not in ("D1", "D2"):
        raise ValueError(f"cre_line must be D1 or D2, got {cfg['cre_line']}")
    return cfg


# ---------------------------------------------------------------- sessions
def resolve_sessions(cfg):
    """Session folders from cfg inputs; a movie path maps to its folder."""
    items = list(cfg.get("inputs") or [])
    if cfg.get("inputs_file"):
        with open(cfg["inputs_file"]) as f:
            items += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    sessions = []
    for it in items:
        p = Path(it)
        if p.suffix.lower() in (".tif", ".tiff"):
            p = p.parent
        if p not in sessions:
            sessions.append(p)
    return sessions


def session_id(sess_dir, depth=3):
    parts = Path(sess_dir).resolve().parts
    return "/".join(parts[-depth:])


def animal_of(sid):
    return sid.split("/")[0]


def plane_dir(sess_dir):
    return Path(sess_dir) / "suite2p" / "plane0"


def out_dir(sess_dir):
    d = Path(sess_dir) / OUT_FOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- suite2p output
def _load_dict(path):
    try:
        d = np.load(path, allow_pickle=True)
        d = d.item() if d.dtype == object and d.shape == () else None
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def s2p_hash(sess_dir):
    """Short hash of stat.npy: labels are tied to one Suite2p run."""
    h = hashlib.md5()
    with open(plane_dir(sess_dir) / "stat.npy", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def load_plane(sess_dir):
    pd_ = plane_dir(sess_dir)
    if not (pd_ / "stat.npy").is_file():
        raise FileNotFoundError(f"no Suite2p output in {pd_}")
    out = {
        "stat": np.load(pd_ / "stat.npy", allow_pickle=True),
        "F": np.load(pd_ / "F.npy").astype(np.float32),
        "Fneu": np.load(pd_ / "Fneu.npy").astype(np.float32),
        "spks": np.load(pd_ / "spks.npy").astype(np.float32),
        "iscell": np.load(pd_ / "iscell.npy"),
        "redcell": np.load(pd_ / "redcell.npy") if (pd_ / "redcell.npy").is_file() else None,
    }
    ops = {}
    # older Suite2p: ops.npy; newer: settings / reg_outputs / detect_outputs
    for name in ("ops.npy", "reg_outputs.npy", "detect_outputs.npy"):
        if (pd_ / name).is_file():
            ops.update(_load_dict(pd_ / name))
    out["ops"] = ops
    out["hash"] = s2p_hash(sess_dir)
    return out


def load_bad_frames(sess_dir, n_frames):
    path = Path(sess_dir) / SHIFTS_MAT
    if not path.is_file():
        return np.zeros(n_frames, bool)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "bad" not in f:
                return np.zeros(n_frames, bool)
            bad = np.asarray(f["bad"]).astype(bool).ravel()
    except Exception:
        return np.zeros(n_frames, bool)
    return bad if bad.size == n_frames else np.zeros(n_frames, bool)


def _mean_from_tif(path, max_frames=2000):
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
        acc = None
        for i in idx:
            fr = tif.pages[i].asarray().astype(np.float64)
            acc = fr if acc is None else acc + fr
    return (acc / len(idx)).astype(np.float32)


def load_images(sess_dir, plane):
    """Green mean image and red reference image, both (Ly, Lx) float32."""
    ops = plane["ops"]
    green = ops.get("meanImg")
    if green is None:
        cache = out_dir(sess_dir) / "meanImg_green.npy"
        if cache.is_file():
            green = np.load(cache)
        else:
            green = _mean_from_tif(Path(sess_dir) / MC_TIF)
            np.save(cache, green)
    green = np.asarray(green, np.float32)

    red = None
    red_path = Path(sess_dir) / RED_TIF
    if red_path.is_file():
        red = tifffile.imread(red_path).astype(np.float32)
        if red.ndim == 3:
            red = red.mean(axis=0)
    elif ops.get("meanImg_chan2") is not None:
        red = np.asarray(ops["meanImg_chan2"], np.float32)
    if red is not None and red.shape != green.shape:
        raise ValueError(f"{sess_dir}: red image {red.shape} != green {green.shape}")
    return green, red


def roi_masks(stat, shape):
    """Per-ROI pixel indices + an index image (-1 = no ROI; later ROIs on top)."""
    Ly, Lx = shape
    index_img = -np.ones((Ly, Lx), np.int32)
    for i, s in enumerate(stat):
        index_img[s["ypix"], s["xpix"]] = i
    return index_img


# ---------------------------------------------------------------- label store
def load_labels(path):
    if not path or not os.path.isfile(path):
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(path, dtype={"stage1": str, "stage2": str})
    # the store is append-only: the last entry per ROI wins
    df = df.drop_duplicates(["session_id", "roi_idx"], keep="last")
    for c in ("stage1", "stage2"):
        df[c] = df[c].where(df[c].notna(), "")
    return df.reset_index(drop=True)


def append_label(path, row):
    row = {c: row.get(c, "") for c in LABEL_COLUMNS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    new = not os.path.isfile(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LABEL_COLUMNS).to_csv(
        path, mode="a", header=new, index=False)
