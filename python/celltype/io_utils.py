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
MAX_TIF = "ConcatenatedMovie_MC_max_registered.tif"      # max projection of the registered movie, next to it
SHIFTS_MAT = "ConcatenatedMovie_MC_shifts.mat"
OUT_FOLDER = "celltype"

DEFAULTS = {
    "inputs": [], "inputs_file": None, "session_id_depth": 3,
    "um_per_px": 1.0, "cre_line": "D1",
    "neuropil_coef": 0.7, "edge_margin_px": 0,
    "baseline_window_s": 60, "baseline_percentile": 8,
    "event_z": 3.0, "event_min_dur_s": 0.1, "red_bleedthrough": False,
    "label_store": None, "model_dir": None, "labeler": "",
    "hard_rules": {"cin_min_diam_um": 50, "min_solidity": 0.80, "max_aspect": 2.5, "max_oob_frac": 0.0},
    "model1": "ensemble", "model2": "logreg",
    "min_labels_per_class": 20, "anomaly_quantile": [0.01, 0.05],
    "thresholds": {"p_spn_hi": 0.9, "p_spn_lo": 0.2, "p_red_hi": 0.9, "p_red_lo": 0.1},
    "use_iscell": True, "iscell_prob_min": None, "snapshot_features": True,
    "model_id": None,              # null: always the latest model (newest id); an id pins every command to that model
    "regularization": {"stage1_C": 1.0, "stage2_C": 1.0, "tune": False,
                       "C_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    "train_on_bulk": False,        # also train on labels added by 'accept all confident'
}

LABEL_COLUMNS = ["session_id", "session_dir", "roi_idx", "stage1", "stage2",
                 "labeler", "timestamp", "s2p_hash", "source"]
# where a label came from:
#   hand      typed by you
#   accepted  you reviewed a predicted label for this ROI and accepted it (key y)
#   bulk      "accept all confident" (not reviewed one by one)
# hand + accepted are the training set; bulk is only used if train_on_bulk is true in the
# config, because training on the model's own unreviewed calls just teaches it its own beliefs.
TRAIN_SOURCES = ("hand", "accepted")
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
    m = cfg.get("edge_margin_px")
    if m is not None and float(m) != int(float(m)):
        print(f"  ! edge_margin_px is a number of PIXELS (a whole number, e.g. 2); {m} is used as {int(float(m))}. "
              f"The share of an ROI allowed outside the image is hard_rules.max_oob_frac.")
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


# A "session reference" names an image folder AND the Suite2p output folder inside
# it: plain "<dir>" means <dir>\\suite2p\\plane0 (the default), and
# "<dir>::<name>" means <dir>\\<name>\\plane0 (e.g. suite2p_sourcerydetection, or a
# deeper relative path). Everything that takes a session folder accepts either.
DEFAULT_OUT = "suite2p"
REF_SEP = "::"


def split_ref(ref):
    """(image folder, output folder name) of a session reference."""
    s = str(ref)
    if REF_SEP in s:
        d, o = s.rsplit(REF_SEP, 1)
        return Path(d), o
    return Path(s), DEFAULT_OUT


def make_ref(image_dir, out_name=DEFAULT_OUT):
    return str(image_dir) if out_name == DEFAULT_OUT else f"{image_dir}{REF_SEP}{out_name}"


def image_dir(ref):
    """Folder holding the images (ConcatenatedMovie_MC.tif, red reference, ...)."""
    return split_ref(ref)[0]


def find_sessions(folder, max_depth=3):
    """Session references for whatever folder was picked, going up or down:
        <session>                         its output folders (suite2p, suite2p_*)
        <session>\\<output>               the output folder itself
        <session>\\<output>\\plane0       the plane folder
        a folder above sessions            every output folder found below it
    An output folder is any folder that holds plane0\\stat.npy; its images are in
    the nearest folder above it that holds ConcatenatedMovie_MC.tif."""
    p = Path(folder)
    has = lambda d: (d / "plane0" / "stat.npy").is_file()
    if p.name.lower().startswith("celltype"):
        # our own cache folder (celltype, celltype_<output>): use the output it belongs to
        sib = p.parent / (p.name[len("celltype"):].lstrip("_") or DEFAULT_OUT)
        if has(sib):
            p = sib
    outs = []
    if p.name.lower() == "plane0" and has(p.parent):
        outs.append(p.parent)
    elif has(p):
        outs.append(p)
    else:
        base_depth = len(p.parts)
        for root, dirs, _ in os.walk(p):
            r, keep = Path(root), []
            for d in sorted(dirs):
                c = r / d
                if has(c):
                    outs.append(c)
                elif (d.lower() != "plane0" and not d.lower().startswith("celltype")
                      and len(c.parts) - base_depth <= max_depth):
                    keep.append(d)
            dirs[:] = keep
    refs = []
    for o in outs:
        img = o.parent
        for anc in list(o.parents)[:3]:
            if (anc / MC_TIF).is_file():
                img = anc
                break
        ref = make_ref(img, o.relative_to(img).as_posix())
        if ref not in refs:
            refs.append(ref)
    return refs


def session_id(sess_dir, depth=3):
    d, out = split_ref(sess_dir)
    sid = "/".join(d.resolve().parts[-depth:])
    return sid if out == DEFAULT_OUT else f"{sid}@{out}"

def animal_of(sid):
    return sid.split("/")[0]


def plane_dir(sess_dir):
    d, out = split_ref(sess_dir)
    return d / out / "plane0"


def out_dir(sess_dir, create=True):
    """Per-session folder for our own files (feature cache, manual ROIs, edits).
    A separate folder per Suite2p output, so runs do not overwrite each other.
    create=False only gives the path (used when scanning many sessions)."""
    img, out = split_ref(sess_dir)
    name = OUT_FOLDER if out == DEFAULT_OUT else f"{OUT_FOLDER}_{out.replace('/', '_').replace(chr(92), '_')}"
    d = img / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------- suite2p output
def _load_dict(path): # _ before function means private function, not to be used outside this module
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
    # load suite2p outputs: stat.npy + F.npy + Fneu.npy + spks.npy + iscell.npy + redcell.npy (optional) + ops dict
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
    if "Ly" not in ops and ops.get("meanImg") is not None:
        ops["Ly"], ops["Lx"] = np.asarray(ops["meanImg"]).shape
    out["ops"] = ops
    out["dir"] = pd_                      # the plane folder (data.bin, outputs)
    # settings the run used (extraction, detection, classification ...); None if absent
    out["settings"] = (_load_dict(pd_ / "settings.npy") or None
                       if (pd_ / "settings.npy").is_file() else None)
    out["hash"] = s2p_hash(sess_dir)
    return out


def load_bad_frames(sess_dir, n_frames):
    path = image_dir(sess_dir) / SHIFTS_MAT
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


def load_mat(path, variables=None):
    """Load a MATLAB .mat file as a plain dict {variable name: value}."""
    from scipy.io import loadmat

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no .mat file at {path}")
    if isinstance(variables, str):
        variables = [variables]
    try:
        d = loadmat(path, squeeze_me=True, simplify_cells=True,
                    variable_names=variables)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except NotImplementedError:
        # v7.3 files are HDF5, which scipy does not read
        import mat73
        return mat73.loadmat(path, only_include=variables)


def _mean_from_tif(path, max_frames=2000):
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
        acc = None
        for i in idx:
            fr = tif.pages[i].asarray().astype(np.float64)
            acc = fr if acc is None else acc + fr
    return (acc / len(idx)).astype(np.float32)


def _padding_mask(im, min_area=20):
    """True where an image holds no data: non-finite pixels, the constant rows / columns at its
    borders, and zero regions of at least min_area pixels that touch the border (what cropping or
    shifting a registered image leaves behind)."""
    from scipy import ndimage
    a = np.asarray(im, np.float32)
    pad = ~np.isfinite(a)
    a = np.where(pad, 0.0, a)
    r = np.flatnonzero(np.ptp(a, axis=1) != 0)
    c = np.flatnonzero(np.ptp(a, axis=0) != 0)
    if len(r) == 0 or len(c) == 0:
        return np.ones(a.shape, bool)
    pad[:r[0], :] = True; pad[r[-1] + 1:, :] = True
    pad[:, :c[0]] = True; pad[:, c[-1] + 1:] = True
    lab, n = ndimage.label(a == 0)
    if n:
        sizes = ndimage.sum(np.ones(lab.shape), lab, index=np.arange(1, n + 1))
        touching = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        for l in touching:
            if sizes[int(l) - 1] >= min_area:
                pad |= lab == l
    return pad


def oob_limit(cfg):
    """The share of an ROI's pixels allowed outside the valid area of the red or the green image before the ROI is
    discarded as an edge ROI: hard_rules.max_oob_frac (default 0 = any pixel outside discards it). A fraction: 0.1 =
    more than 10 %."""
    lim = (cfg.get("hard_rules") or {}).get("max_oob_frac")
    return 0.0 if lim is None else float(lim)


def load_max_image(sess_dir, shape=None):
    """The max projection of the registered movie that sits next to it (MAX_TIF), as (Ly, Lx) float32; None
    if the file is not there or its size is not `shape` (the green mean image)."""
    path = image_dir(sess_dir) / MAX_TIF
    if not path.is_file():
        return None
    im = tifffile.imread(path).astype(np.float32)
    if im.ndim == 3:
        im = im.max(axis=0)
    if im.ndim != 2:
        return None
    if shape is not None and im.shape != tuple(shape):
        print(f"  ! {path.name} is {im.shape}, the green mean image {tuple(shape)}: max image not used")
        return None
    return im


def load_images(sess_dir, plane):
    """Green mean image and red reference image, both (Ly, Lx) float32."""
    return load_images_valid(sess_dir, plane)[:2]


def load_images_valid(sess_dir, plane):
    """(green, red, valid_green, valid_red): the two mean images as load_images returns them, and
    for each a boolean mask of where it holds data (False: padding left by cropping or shifting; for
    green also outside Suite2p's valid x / y range). valid_red is None when there is no red image.
    An ROI with pixels in the padding of either image is an edge ROI (see features.roi_oob_fraction)."""
    ops = plane["ops"]
    green = ops.get("meanImg")
    if green is None:
        cache = out_dir(sess_dir) / "meanImg_green.npy"
        if cache.is_file():
            green = np.load(cache)
        else:
            green = _mean_from_tif(image_dir(sess_dir) / MC_TIF)
            np.save(cache, green)
    green = np.asarray(green, np.float32)
    valid_g = ~_padding_mask(green)
    yr, xr = ops.get("yrange"), ops.get("xrange")
    if yr is not None and xr is not None:
        try:
            box = np.zeros(green.shape, bool)
            box[int(yr[0]):int(yr[1]), int(xr[0]):int(xr[1])] = True
            if box.mean() > 0.2:                            # ignore a nonsense range
                valid_g &= box
        except (TypeError, ValueError, IndexError):
            pass

    red = None
    red_path = image_dir(sess_dir) / RED_TIF
    if red_path.is_file():
        red = tifffile.imread(red_path).astype(np.float32)
        if red.ndim == 3:
            red = red.mean(axis=0)
    elif ops.get("meanImg_chan2") is not None:
        red = np.asarray(ops["meanImg_chan2"], np.float32)
    valid_r = None if red is None else ~_padding_mask(red)
    if red is not None and red.shape != green.shape:
        print(f"  ! {image_dir(sess_dir).name}: red ref {red.shape} != green {green.shape}; "
              f"mapping red onto green")
        red, valid_r = align_red_to_green(red, green, valid=valid_r)
    return green, red, valid_g, valid_r


def align_red_to_green(red, green, max_shift_px=20, valid=None):
    """Map the red reference onto the green (functional) frame, since ROIs were
    drawn on green. Step 1: resample to green's shape (assumes the same FOV).
    Step 2: refine with a translation from phase correlation of the two images
    (high-passed so somata, not background, drive the fit); shifts larger than
    max_shift_px are treated as spurious and ignored.
    valid: optional boolean mask of the red image that gets the same resampling and shift (and one
    pixel of erosion, since interpolation blends the edge); then (image, mask) is returned."""
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize
    from scipy import ndimage

    r = resize(red, green.shape, order=1, anti_aliasing=True, preserve_range=True)
    r = r.astype(np.float32)
    v = None
    if valid is not None:
        v = resize(valid.astype(np.float32), green.shape, order=0, anti_aliasing=False,
                   preserve_range=True) > 0.5
    hp = lambda im: im - ndimage.gaussian_filter(im, 8)
    try:
        shift, _, _ = phase_cross_correlation(hp(green), hp(r), upsample_factor=10)
        if np.abs(shift).max() <= max_shift_px:
            r = ndimage.shift(r, shift, order=1, mode="nearest")
            if v is not None:
                v = ndimage.shift(v.astype(np.float32), shift, order=0, mode="constant", cval=0.0) > 0.5
            print(f"    red resized + shifted by (dy,dx)=({shift[0]:.1f},{shift[1]:.1f}) px")
        else:
            print(f"    red resized; shift {shift} too large, not applied")
    except Exception as e:
        print(f"    red resized; shift refinement failed: {e}")
    if v is not None:
        return r, ndimage.binary_erosion(v, iterations=1, border_value=1)
    return r


def load_red_frames(sess_dir, green, max_frames=400):
    """Frames of the red reference tif (evenly spaced, at most max_frames), mapped onto the green
    frame like load_images does for the mean: (n, Ly, Lx) float32. None when the red reference is
    a single image (or missing): then there is nothing to project and the mean is all there is."""
    path = image_dir(sess_dir) / RED_TIF
    if not path.is_file():
        return None
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
        if len(shape) < 3 or shape[0] < 2:
            return None
        n = shape[0]
        idx = np.unique(np.linspace(0, n - 1, min(n, int(max_frames))).astype(int))
        if len(tif.pages) == n:                        # one page per frame: read only the ones needed
            stack = np.stack([tif.pages[i].asarray().astype(np.float32) for i in idx])
        else:                                          # (small) volume stored as one page
            stack = tif.asarray()[idx].astype(np.float32)
    if stack.shape[1:] != green.shape:                 # same mapping as the mean image gets
        from skimage.transform import resize
        from skimage.registration import phase_cross_correlation
        from scipy import ndimage
        stack = np.stack([resize(f, green.shape, order=1, anti_aliasing=True, preserve_range=True)
                          for f in stack]).astype(np.float32)
        try:
            hp = lambda im: im - ndimage.gaussian_filter(im, 8)
            shift, _, _ = phase_cross_correlation(hp(green), hp(stack.mean(0)), upsample_factor=10)
            if np.abs(shift).max() <= 20:
                stack = np.stack([ndimage.shift(f, shift, order=1, mode="nearest") for f in stack])
        except Exception:
            pass
    return stack


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
    df = pd.read_csv(path, dtype={"stage1": str, "stage2": str, "source": str})
    # the store is append-only: the last entry per ROI wins
    df = df.drop_duplicates(["session_id", "roi_idx"], keep="last")
    for c in ("stage1", "stage2"):
        df[c] = df[c].where(df[c].notna(), "")
    if "source" not in df.columns:            # stores written before the source column existed
        df["source"] = "hand"
    df["source"] = df["source"].where(df["source"].notna() & (df["source"] != ""), "hand")
    return df.reset_index(drop=True)


def training_labels(cfg):
    """The labels that go into the training set: hand + accepted (+ bulk if train_on_bulk)."""
    lab = load_labels(cfg["label_store"])
    ok = set(TRAIN_SOURCES) | ({"bulk"} if cfg.get("train_on_bulk") else set())
    return lab[lab["source"].isin(ok)].reset_index(drop=True)


def _upgrade_store(path):
    """An older store has no 'source' column; appending rows with one would misalign the csv.
    Rewrite it once with source = hand (a backup copy is kept next to it)."""
    header = pd.read_csv(path, nrows=0).columns
    if "source" in header:
        return
    import shutil
    shutil.copyfile(path, str(path) + ".bak_before_source")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["source"] = "hand"
    df.to_csv(path, index=False)


def append_label(path, row):
    row = {c: row.get(c, "") for c in LABEL_COLUMNS}
    row["source"] = row["source"] or "hand"
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    new = not os.path.isfile(path)
    if not new:
        _upgrade_store(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LABEL_COLUMNS).to_csv(
        path, mode="a", header=new, index=False)
