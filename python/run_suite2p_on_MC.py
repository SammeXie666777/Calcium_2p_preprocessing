"""
run_suite2p_on_MC.py: Suite2p cell detection and signal extraction on movies that
are already motion-corrected (batch_2p_Calcium.m / sbx_motion_correct), without
Suite2p's own registration.

INPUTS per session folder (<date>\\<exptype>_rec<rid>\\)
    ConcatenatedMovie_MC.tif          green (functional) movie
    ref_field_MC_registeration.tif    red reference image registered to the session
                                      (optional: gives each ROI a red / not-red label)
    ConcatenatedMovie_MC_shifts.mat   optional: its 'bad' frames are left out of
                                      detection (not of extraction)

OUTPUTS in <session>\\suite2p\\plane0\\
    stat F Fneu spks iscell redcell detect_outputs ops (.npy), Fall.mat (MATLAB:
    ypix / xpix are 0-based), roi_contours.png, suite2p_log.txt. Columns of F are the
    frames of ConcatenatedMovie_MC.tif. The combined mode adds stat_A / stat_B,
    combined_index.npz and combined_traces.npz / .mat.

PIPELINE
    1  tif -> data.bin (reused if present, REUSE_BINARY)
    2  detection: sparsery / sourcery / cellpose, or "combined" (Cellpose masks +
       an activity detector in one run, one binning, one extraction)
    3  extraction (F, neuropil), deconvolution, cell / not-cell classification
    4  red label per ROI: RED_METHOD "intensity" (red inside the ROI relative to its
       surround) or "blobs" (bright blobs of the red image, IoU with each ROI)

USAGE
    python run_suite2p_on_MC.py <folder> [<folder> ...]     or edit SESSIONS below
A folder is a session (it holds ConcatenatedMovie_MC.tif) or a folder above
sessions (date / animal), searched SEARCH_DEPTH levels down. Quote paths with spaces.
test_suite2p.py runs the same code on arbitrary tifs.

Written for the Suite2p main-branch API (nested settings["run"], ["detection"], ...).
"""

# =============================================================================
# 1. PACKAGES
# =============================================================================
import copy
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version as _pkg_version

import h5py
import numpy as np
import tifffile
import torch
from scipy import ndimage

import suite2p
from suite2p import default_db, default_settings, run_s2p
from suite2p.detection import chan2detect


# =============================================================================
# 2. USER INPUT
# =============================================================================
# --- 2.1 which sessions ------------------------------------------------------
# Session folders, or folders above them. One per line: r"..." and a comma.
SESSIONS = [r"O:\2p_animals\mouse1"]
ROOT_DIR = r"O:\2p_animals"    # used when SESSIONS is empty and no folder is given
EXCLUDE = []               # relative to ROOT_DIR: "mouse3", "mouse1\260930", ...
SEARCH_DEPTH = 3           # folder levels below a given folder to search
SKIP_EXISTING = False      # True: skip sessions that already have suite2p\plane0\stat.npy

# --- 2.2 imaging -------------------------------------------------------------
FS = 31.25                 # frame rate (Hz) of the concatenated movie
TAU = 0.7                  # indicator decay (s): ~0.7 GCaMP6f, ~1.0-1.25 GCaMP6s
DIAMETER = 12              # expected soma diameter (px)
MARGIN = 0                 # px ignored at every edge for detection
USE_MC_BAD_FRAMES = True   # leave frames flagged 'bad' by motion correction out of detection

# --- 2.3 green-channel detection ---------------------------------------------
DETECTION_ALGORITHM = "combined"   # "sparsery" | "sourcery" | "cellpose" | "combined"
# sparsery / sourcery grow ROIs from activity correlation (can pull dendrites in).
# threshold = 5 x scale x THRESHOLD_SCALING, as printed in the log.
SPATIAL_SCALE = 2          # 0 = Suite2p estimates it; 1..4 force ~6 / 12 / 24 / 48 px
THRESHOLD_SCALING = 1.0    # higher = fewer, stronger ROIs
# cellpose outlines cell bodies in one summary image.
CELLPOSE_IMG = "max_proj"          # "meanImg" | "max_proj"
CELLPOSE_FLOW_THRESHOLD = 0.4      # higher = keep more (looser) masks; 0 = no check
CELLPOSE_CELLPROB_THRESHOLD = -1   # lower = more / larger masks
CELLPOSE_HIGHPASS_SPATIAL = 0      # spatial high-pass of that image, x DIAMETER (0 = none)

# --- 2.4 combined mode (DETECTION_ALGORITHM = "combined") --------------------
# A = anatomical masks, B = activity masks. A B mask overlapping an A mask is not
# extracted (the A shape is, tagged "AB"); unmatched A are "A_only", unmatched B
# "B_only". Neither set is thinned here (each detector's own filters still apply).
COMBINED_ANATOMICAL = "cellpose"
COMBINED_FUNCTIONAL = "sparsery"
COMBINED_MIN_OVERLAP = 0.3         # B matches A at shared / smaller-ROI size >= this
COMBINED_REUSE_DETECTION = False   # True: reuse stat_A / stat_B of an earlier run in
                                   # plane0 (skips binning + detection; retry overlaps fast)

# --- 2.5 ROI filters and classifier ------------------------------------------
# Suite2p's own filters; the values are its defaults.
MAX_OVERLAP = 0.75         # drop ROIs overlapping others by more than this fraction
NPIX_NORM_MIN = 0.0        # drop ROIs smaller than this (size relative to the median ROI)
NPIX_NORM_MAX = 100.0      # drop ROIs larger than this
SOMA_CROP = True           # crop dendrites when computing size / compactness statistics
SNR_THRESHOLD = None       # None = Suite2p default; a number sets extraction's snr_threshold
CLASSIFIER_PATH = None     # your classifier .npy; None = GUI default, else built-in

# --- 2.6 red (structural) channel --------------------------------------------
RED_METHOD = "intensity"   # "intensity" | "blobs" | "off"
# intensity: red = signal inside the ROI / (inside + neuropil ring around it) > threshold
#            (0.5 = as bright as the surround). Suite2p's own rule.
RED_INTENSITY_THRESHOLD = 0.65
# blobs: bright blobs of the red image (band-pass, robust threshold, size limits) are
#        masks; an ROI is red when its best IoU with a blob exceeds the threshold.
RED_BLOB_IOU_THRESHOLD = 0.25
RED_BLOB_DIAMETER = None   # px, expected red soma size; None = DIAMETER
RED_BLOB_Z = 4.0           # detection threshold in robust SDs of the band-passed image
RED_BLOB_AREA_RANGE = (0.25, 4.0)  # allowed blob area, x the area of a DIAMETER disk
                                   # (the thresholded blob is a bit larger than the soma)

# --- 2.7 binary, speed, output -----------------------------------------------
FAST_DISK = []             # [] / None: binary beside the data; or a folder on another drive
DELETE_BIN = False         # delete data.bin after the run (keep it for the GUI / manual ROIs)
REUSE_BINARY = True        # skip the tif -> binary conversion if data.bin has the right size
TORCH_DEVICE = "cuda"      # "cuda" falls back to cpu when there is no GPU
BATCH_FRAMES = 2000        # frames per batch (conversion, extraction); lower if RAM runs out
PLOT_CONTOURS = True       # roi_contours.png per session
PLOT_NONCELLS = False      # also draw ROIs classified as not-cell
PLOT_ROI_NUMBERS = False   # 0-based ROI index next to each cell
SHOW_PLOTS = False         # open each figure (blocks until closed)
SUITE2P_LOG_LEVEL = logging.INFO

# --- 2.8 default file names inside a session folder --------------------------
MC_TIF = "ConcatenatedMovie_MC.tif"
RED_TIF = "ref_field_MC_registeration.tif"
SHIFTS_MAT = "ConcatenatedMovie_MC_shifts.mat"


# =============================================================================
# 3. FUNCTIONS
# =============================================================================
# --- 3.1 small helpers -------------------------------------------------------
class Tee:
    """Write to the terminal and a log file at once, and attach a handler so
    Suite2p's own logger messages go to both."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "a", encoding="utf-8")
        self.handler = logging.StreamHandler(self)
        self.handler.setFormatter(logging.Formatter("    [suite2p %(asctime)s] %(message)s",
                                                    "%H:%M:%S"))
        self.logger = logging.getLogger("suite2p")

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        self.handler.setLevel(SUITE2P_LOG_LEVEL)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(min(self.logger.level or logging.WARNING, SUITE2P_LOG_LEVEL))
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        sys.stdout = self.terminal
        self.file.close()


def fmt_size(path):
    """File size as text, or 'missing'."""
    if not os.path.isfile(path):
        return "missing"
    n = float(os.path.getsize(path))
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def fmt_time(sec):
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def describe_session(sess_dir):
    """(animal, date, recording) from <animal>\\<date>\\<exptype>_rec<rid>."""
    parts = os.path.normpath(sess_dir).split(os.sep)
    parts += [""] * (3 - len(parts))
    return parts[-3], parts[-2], parts[-1]


def torch_device():
    return torch.device("cuda" if TORCH_DEVICE == "cuda" and torch.cuda.is_available()
                        else "cpu")


def load_dict(path):
    """A dict saved with np.save (Suite2p's .npy outputs)."""
    return np.load(path, allow_pickle=True).item()


def save_npy(folder, **arrays):
    """np.save(folder\\<name>.npy, array) for each keyword."""
    for name, arr in arrays.items():
        np.save(os.path.join(folder, f"{name}.npy"), arr)


def remove_files(paths):
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


def movie_shape(path):
    """(frames, Ly, Lx) of a tif without loading it."""
    with tifffile.TiffFile(path) as tif:
        return len(tif.pages), *tif.pages[0].shape[:2]


def load_bad_frames(path, n_frames):
    """'bad' flags of a -v7.3 _shifts.mat as a bool vector (all False if unusable)."""
    with h5py.File(path, "r") as f:
        bad = np.asarray(f["bad"]).astype(bool).ravel() if "bad" in f else None
    if bad is None:
        return np.zeros(n_frames, bool)
    if bad.size != n_frames:
        print(f"  ! {os.path.basename(path)}: {bad.size} bad-frame flags for "
              f"{n_frames} frames; not used")
        return np.zeros(n_frames, bool)
    return bad


def pick_classifier(settings):
    """Classifier file Suite2p's pipeline would use (settings, GUI default, built-in)."""
    from suite2p import classification
    path = settings["classification"].get("classifier_path")
    if path is not None and os.path.exists(path):
        return path
    if (settings["classification"]["use_builtin_classifier"]
            or not classification.user_classfile.is_file()):
        return classification.builtin_classfile
    return classification.user_classfile


# --- 3.2 settings ------------------------------------------------------------
def _set(settings, path, value):
    """settings[a][b]... = value only if that key exists in this Suite2p release
    (Suite2p would otherwise silently ignore a renamed setting)."""
    node = settings
    for k in path[:-1]:
        node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            break
    if isinstance(node, dict) and path[-1] in node:
        node[path[-1]] = value
    else:
        print(f"  ! this Suite2p has no setting {'.'.join(path)}; not applied")


def make_settings():
    """Suite2p settings from the constants above. Nested keys are changed in place
    (run_s2p merges only the top level, so partial nested dicts would drop keys)."""
    s = default_settings()
    s.update(fs=FS, tau=TAU, diameter=[DIAMETER, DIAMETER], torch_device=TORCH_DEVICE)
    s["run"]["do_registration"] = 0
    s["io"].update(save_mat=True, delete_bin=DELETE_BIN)
    s["extraction"]["batch_size"] = BATCH_FRAMES
    s["detection"]["sparsery_settings"]["spatial_scale"] = SPATIAL_SCALE
    s["detection"]["threshold_scaling"] = THRESHOLD_SCALING
    # the saved settings carry the anatomical algorithm (later steps such as manual
    # ROIs use its weight rule); "combined" itself is run by run_combined
    algo = COMBINED_ANATOMICAL if DETECTION_ALGORITHM == "combined" else DETECTION_ALGORITHM
    for key, val in (("algorithm", algo), ("max_overlap", MAX_OVERLAP),
                     ("npix_norm_min", NPIX_NORM_MIN), ("npix_norm_max", NPIX_NORM_MAX),
                     ("soma_crop", SOMA_CROP)):
        _set(s, ("detection", key), val)
    for key, val in (("img", CELLPOSE_IMG), ("flow_threshold", CELLPOSE_FLOW_THRESHOLD),
                     ("cellprob_threshold", CELLPOSE_CELLPROB_THRESHOLD),
                     ("highpass_spatial", CELLPOSE_HIGHPASS_SPATIAL)):
        _set(s, ("detection", "cellpose_settings", key), val)
    if SNR_THRESHOLD is not None:
        _set(s, ("extraction", "snr_threshold"), SNR_THRESHOLD)
    if CLASSIFIER_PATH:
        if not os.path.isfile(CLASSIFIER_PATH):    # Suite2p would fall back silently
            raise FileNotFoundError(f"CLASSIFIER_PATH not found: {CLASSIFIER_PATH}")
        s["classification"]["classifier_path"] = CLASSIFIER_PATH
    # red labelling: our own detect_red replaces chan2detect.detect (3.3); these
    # entries record what was used, and tell manual ROIs which rule to follow
    s["red_method"] = RED_METHOD
    s["detection"]["cellpose_chan2"] = RED_METHOD == "blobs"
    s["detection"]["chan2_threshold"] = (RED_BLOB_IOU_THRESHOLD if RED_METHOD == "blobs"
                                         else RED_INTENSITY_THRESHOLD)
    return s


def describe_red():
    if RED_METHOD == "intensity":
        return f"intensity > {RED_INTENSITY_THRESHOLD}"
    if RED_METHOD == "blobs":
        return (f"blobs (z > {RED_BLOB_Z}, diameter {RED_BLOB_DIAMETER or DIAMETER}), "
                f"IoU > {RED_BLOB_IOU_THRESHOLD}")
    return "off"


def print_settings(indent=""):
    try:
        version = _pkg_version("suite2p")
    except Exception:
        version = getattr(suite2p, "__version__", "?")
    scale = {0: "auto"}.get(SPATIAL_SCALE, f"{SPATIAL_SCALE} (~{3 * 2 ** SPATIAL_SCALE} px)")
    algo = DETECTION_ALGORITHM
    if algo == "combined":
        algo += f" (A={COMBINED_ANATOMICAL} + B={COMBINED_FUNCTIONAL}, match >= {COMBINED_MIN_OVERLAP})"
    elif algo == "cellpose":
        algo += f" on {CELLPOSE_IMG}"
    for line in (
            f"suite2p {version}  |  python {sys.version.split()[0]}  |  {sys.executable}",
            f"imaging: fs={FS} Hz, tau={TAU} s, diameter={DIAMETER} px, margin={MARGIN} px, "
            f"bad frames={'on' if USE_MC_BAD_FRAMES else 'off'}",
            f"detection: {algo}; spatial scale={scale}, threshold scaling={THRESHOLD_SCALING}",
            f"filters: max overlap={MAX_OVERLAP}, npix norm=[{NPIX_NORM_MIN}, {NPIX_NORM_MAX}], "
            f"soma crop={SOMA_CROP}, classifier={CLASSIFIER_PATH or 'GUI default / built-in'}",
            f"red channel: {describe_red()}",
            f"speed: device={TORCH_DEVICE}, batch={BATCH_FRAMES} frames, binary in="
            f"{FAST_DISK or 'session folder'}, delete binary={DELETE_BIN}, "
            f"reuse binary={REUSE_BINARY}"):
        print(indent + line)


# --- 3.3 red channel ---------------------------------------------------------
# Suite2p labels ROIs red inside detection_wrapper by calling chan2detect.detect.
# detect_red replaces it (install_red_detector), so single-method and combined
# runs share one implementation and Cellpose is never used on the red image.
def detect_blobs(img, diameter, z_thr=4.0, area_range=(0.25, 4.0)):
    """Bright-blob masks of a structural image as a label image (0 = background).

    Band-pass (difference of Gaussians tuned to `diameter`), robust z-score of the
    band-passed image, threshold, split touching blobs by watershed (when skimage
    is available), keep blobs whose area is within area_range x the area of a disk
    of `diameter`."""
    img = np.asarray(img, np.float32)
    valid = img > 0                                  # registration padding is 0
    sigma = diameter / (2 * np.sqrt(2))
    dog = ndimage.gaussian_filter(img, sigma) - ndimage.gaussian_filter(img, 1.6 * sigma)
    med = np.median(dog[valid])
    mad = 1.4826 * np.median(np.abs(dog[valid] - med)) + 1e-9
    fg = ((dog - med) / mad > z_thr) & valid
    lab, _ = ndimage.label(fg)
    try:                                             # split touching blobs
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        pk = peak_local_max(dog, min_distance=max(2, int(diameter / 2)), labels=lab,
                            exclude_border=False)      # blobs at the image edge count
        markers = np.zeros(lab.shape, np.int32)
        markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lab = watershed(-dog, markers, mask=fg)
    except ImportError:
        pass
    nominal = np.pi / 4 * diameter ** 2
    areas = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    keep = np.flatnonzero((areas >= area_range[0] * nominal)
                          & (areas <= area_range[1] * nominal))
    keep = keep[keep > 0]
    remap = np.zeros(len(areas), np.int32)             # old label -> 1..n, others 0
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab]


def intensity_labels(img, stats, threshold, inner_radius=2, min_pixels=350):
    """Per-ROI red label: (is_red, redprob), one row per ROI.

    redprob = inside / (inside + surround): `inside` the lam-weighted red signal in
    the ROI, `surround` the mean of its neuropil ring (other cells excluded, same
    masks as the extraction). 0.5 = as bright as the surround. This is Suite2p's
    chan2detect.intensity_ratio, written with per-ROI sparse masks (Suite2p builds a
    dense ROIs x pixels matrix, ~10 GB for 3,600 ROIs)."""
    from suite2p.extraction import masks as s2m
    Ly, Lx = img.shape
    flat = np.asarray(img, np.float64).ravel()
    cell_pix = s2m.create_cell_pix(stats, Ly=Ly, Lx=Lx)
    ring = s2m.create_neuropil_masks(
        ypixs=[s["ypix"] for s in stats], xpixs=[s["xpix"] for s in stats],
        cell_pix=cell_pix, inner_neuropil_radius=inner_radius,
        min_neuropil_pixels=min_pixels)
    out = np.zeros((len(stats), 2), np.float32)
    for i, (st, nu) in enumerate(zip(stats, ring)):
        idx, w = s2m.create_cell_mask(st, Ly=Ly, Lx=Lx, allow_overlap=True)
        inside = max(1e-3, float(w @ flat[idx]))
        surround = float(flat[nu].mean()) if len(nu) else 0.0
        prob = inside / (inside + surround)
        out[i] = (prob > threshold, prob)
    return out


def iou_labels(stats, masks, threshold):
    """Per-ROI label from the best IoU with any mask of a label image."""
    n = int(masks.max())
    sizes = np.bincount(masks.ravel(), minlength=n + 1)[1:]
    out = np.zeros((len(stats), 2), np.float32)
    for i, st in enumerate(stats):
        if n:
            shared = np.bincount(masks[st["ypix"], st["xpix"]], minlength=n + 1)[1:]
            iou = float((shared / (st["ypix"].size + sizes - shared)).max())
        else:
            iou = 0.0
        out[i] = (iou > threshold, iou)
    return out


def detect_red(meanImg, meanImg_chan2, stats, *args, inner_neuropil_radius=2,
               min_neuropil_pixels=350, **kwargs):
    """Drop-in for chan2detect.detect: returns (masks, redcell) with redcell of shape
    (n_rois, 2) = [is_red, score]. masks is the blob label image, or None for the
    intensity method. Other arguments of Suite2p's call are accepted and ignored."""
    red = np.asarray(meanImg_chan2, np.float32)
    if RED_METHOD == "blobs":
        masks = detect_blobs(red, RED_BLOB_DIAMETER or DIAMETER, RED_BLOB_Z,
                             RED_BLOB_AREA_RANGE)
        return masks, iou_labels(stats, masks, RED_BLOB_IOU_THRESHOLD)
    return None, intensity_labels(red, stats, RED_INTENSITY_THRESHOLD,
                                  inner_neuropil_radius, min_neuropil_pixels)


def install_red_detector():
    chan2detect.detect = detect_red


def detect_red_separately(red, plane_dir):
    """Reference and movie on different pixel grids cannot be compared per ROI.
    Blobs of the reference are still detected on its own grid (red_masks.npy)."""
    if RED_METHOD != "blobs":
        print("  red labels need the reference on the movie's grid: none computed")
        return
    masks = detect_blobs(red, RED_BLOB_DIAMETER or DIAMETER, RED_BLOB_Z, RED_BLOB_AREA_RANGE)
    save_npy(plane_dir, red_masks=masks)
    print(f"  red reference: {masks.max()} blobs on its own grid {masks.shape} -> red_masks.npy")


# --- 3.4 binary --------------------------------------------------------------
def reuse_existing_binary(db, settings, plane_dir, n_frames, Ly, Lx):
    """True when plane0\\data.bin can be used as it is (REUSE_BINARY).

    The binary is the converted movie and does not depend on detection settings.
    Suite2p itself skips the conversion only when plane0 also holds db.npy and
    settings.npy (written at the end of a conversion). So:
      right size, sidecars missing -> recreate them, skip the conversion
      wrong size (interrupted)     -> remove the sidecars so Suite2p converts again
      REUSE_BINARY False           -> same removal: always convert again"""
    binf = os.path.join(plane_dir, "data.bin")
    sidecars = [os.path.join(plane_dir, n) for n in ("db.npy", "settings.npy")]
    if not os.path.isfile(binf):
        return False
    size, expected = os.path.getsize(binf), n_frames * Ly * Lx * 2
    if not REUSE_BINARY or size != expected:
        remove_files(sidecars)
        print("  REUSE_BINARY is False: data.bin is converted again" if not REUSE_BINARY else
              f"  ! data.bin is {size / 1e9:.2f} GB, expected {expected / 1e9:.2f} GB for "
              f"{n_frames} frames: converting again")
        return False
    if not all(os.path.isfile(f) for f in sidecars):
        from suite2p import io as s2p_io
        d = s2p_io.init_dbs({**default_db(), **db})[0]
        d.update(nframes=n_frames, Ly=Ly, Lx=Lx)
        np.save(d["db_path"], d)
        np.save(d["settings_path"], settings)
        print("  data.bin is complete, db.npy / settings.npy were missing: recreated")
    print(f"  data.bin already exists ({size / 1e9:.1f} GB, {n_frames} frames): "
          f"conversion skipped")
    return True


def fast_disk_for(sess_dir, mc_path, animal, date, rec):
    """Per-session folder on FAST_DISK if it has room for the binary, else None
    (the binary then goes beside the data, after a free-space check)."""
    need = os.path.getsize(mc_path) * 1.1
    target = FAST_DISK or sess_dir
    os.makedirs(target, exist_ok=True)
    free = shutil.disk_usage(target).free
    if not FAST_DISK:
        print(f"  free space on {os.path.splitdrive(sess_dir)[0] or sess_dir}: "
              f"{free / 1e9:.1f} GB ({(free - need) / 1e9:.1f} GB left while it runs)")
        if free < need:
            raise OSError(f"not enough space for the binary: need ~{need / 1e9:.1f} GB, "
                          f"{free / 1e9:.1f} GB free")
        return None
    if free < need:
        print(f"  ! FAST_DISK has {free / 1e9:.1f} GB free, binary needs "
              f"~{need / 1e9:.1f} GB: using the session folder instead")
        return None
    return os.path.join(FAST_DISK, f"{animal}_{date}_{rec}")


def cleanup_fast_disk(fast):
    """Remove the session's FAST_DISK folder (the binary is kept unless DELETE_BIN)."""
    if not fast or not os.path.isdir(fast):
        return
    files = [os.path.join(r, f) for r, _, fs in os.walk(fast) for f in fs]
    if DELETE_BIN or not files:
        shutil.rmtree(fast, ignore_errors=True)
    else:
        print(f"  binary kept on fast disk: {', '.join(files)}")


# --- 3.5 combined detection --------------------------------------------------
def merge_detections(stat_A, stat_B, Ly, Lx, min_overlap):
    """Merge two segmentations of one field: A whole, plus the B masks that do not
    overlap an A mask (shared / smaller size >= min_overlap counts as overlap).

    Each merged stat carries combined_source ("AB" | "A_only" | "B_only"), a_index,
    b_index (first matched B, -1 if none), n_b_matched. lam is scaled to max 1 per ROI
    so the two methods' weights are comparable in Suite2p's cell_pix map.
    Returns (merged list, info dict)."""
    nA, nB = len(stat_A), len(stat_B)
    owner = -np.ones((Ly, Lx), np.int32)                 # top A mask of each pixel
    for i, st in enumerate(stat_A):
        owner[st["ypix"], st["xpix"]] = i
    size_A = np.array([st["ypix"].size for st in stat_A])
    best_a, coef = -np.ones(nB, int), np.zeros(nB)
    for j, st in enumerate(stat_B):
        o = owner[st["ypix"], st["xpix"]]
        o = o[o >= 0]
        if o.size == 0:
            continue
        cnt = np.bincount(o, minlength=nA)
        i = int(cnt.argmax())
        coef[j] = cnt[i] / min(st["ypix"].size, size_A[i])
        if coef[j] >= min_overlap:
            best_a[j] = i
    b_of_a = {}
    for j in np.flatnonzero(best_a >= 0):
        b_of_a.setdefault(int(best_a[j]), []).append(int(j))

    def tag(st, source, a, b, n):
        d = dict(st)
        d["lam"] = (np.asarray(d["lam"], np.float32) / np.max(d["lam"])).astype(np.float32)
        d.update(combined_source=source, a_index=a, b_index=b, n_b_matched=n)
        return d

    merged = [tag(st, "AB" if i in b_of_a else "A_only", i,
                  b_of_a[i][0] if i in b_of_a else -1, len(b_of_a.get(i, [])))
              for i, st in enumerate(stat_A)]
    merged += [tag(stat_B[j], "B_only", -1, int(j), 0) for j in np.flatnonzero(best_a < 0)]
    info = {"n_A": nA, "n_B": nB, "n_AB": len(b_of_a), "n_A_only": nA - len(b_of_a),
            "n_B_only": int((best_a < 0).sum()), "b_matched_to_a": best_a,
            "b_overlap_coef": coef}
    return merged, info


def assign_priority_overlaps(stat, Ly, Lx):
    """Overlap flags with A taking priority, so an A shape is kept whole: a pixel
    of an A ROI is flagged only when another A ROI covers it; a B_only ROI is
    flagged against everything. (Suite2p drops shared pixels from both masks.)"""
    is_a = np.array([st["combined_source"] != "B_only" for st in stat])
    n_a, n_all = np.zeros((Ly, Lx), np.int32), np.zeros((Ly, Lx), np.int32)
    for st, a in zip(stat, is_a):
        n_all[st["ypix"], st["xpix"]] += 1
        if a:
            n_a[st["ypix"], st["xpix"]] += 1
    for st, a in zip(stat, is_a):
        st["overlap"] = ((n_a if a else n_all)[st["ypix"], st["xpix"]] > 1).astype(bool)
    return stat


def detect_a_and_b(f_reg, settings, reg_outputs, classfile, device, plane_dir):
    """Bin the movie once and run both detectors on it. Returns
    (detect_outputs of A, stat_A, stat_B); everything is saved for reuse."""
    from suite2p import detection
    det = settings["detection"]
    diameter = np.array(settings["diameter"], float)
    n_frames = f_reg.shape[0]
    bin_size = det.get("bin_size") or int(max(1, n_frames // det["nbins"],
                                              np.round(settings["tau"] * settings["fs"])))
    t0 = time.time()
    print(f"  combined: binning once (bins of {bin_size} frames) ...")
    mov = detection.bin_movie(f_reg, bin_size, yrange=reg_outputs["yrange"],
                              xrange=reg_outputs["xrange"], badframes=reg_outputs["badframes"],
                              nbins=det["nbins"])
    print(f"    binned {mov.shape} in {fmt_time(time.time() - t0)}")

    def run(algo):
        s = copy.deepcopy(settings)
        s["detection"]["algorithm"] = algo
        t = time.time()
        out = detection.detection_wrapper(
            f_reg, diameter=diameter, tau=settings["tau"], fs=settings["fs"],
            meanImg_chan2=None,                     # red labels: once, on the merged set
            yrange=reg_outputs["yrange"], xrange=reg_outputs["xrange"],
            badframes=reg_outputs["badframes"], mov=mov,
            preclassify=settings["classification"]["preclassify"],
            classifier_path=classfile, settings=s["detection"], device=device)
        print(f"    {algo}: {len(out[1])} ROIs in {fmt_time(time.time() - t)}")
        return out

    print(f"  combined: A = {COMBINED_ANATOMICAL}, B = {COMBINED_FUNCTIONAL}")
    detect_outputs, stat_A, _ = run(COMBINED_ANATOMICAL)
    stat_B = run(COMBINED_FUNCTIONAL)[1]
    stat_A, stat_B = list(stat_A), list(stat_B)
    for name, arr in (("stat_A", np.array(stat_A, dtype=object)),
                      ("stat_B", np.array(stat_B, dtype=object)),
                      ("detect_outputs_combined", detect_outputs)):
        np.save(os.path.join(plane_dir, f"{name}.npy"), arr)
    return detect_outputs, stat_A, stat_B


def extract_and_classify(stat, f_reg, settings, classfile, device, Ly, Lx):
    """Suite2p's pipeline after detection: extraction (second pass if an SNR
    threshold removes ROIs), activity statistics, deconvolution, classification.
    Returns (stat, F, Fneu, spks, iscell)."""
    from scipy.stats import skew
    from suite2p import classification, extraction
    ex = settings["extraction"]
    t0 = time.time()
    snr_threshold = ex["snr_threshold"]
    for step in range(1 + (snr_threshold > 0)):
        F, Fneu, _, _ = extraction.extraction_wrapper(stat, f_reg, settings=ex, device=device)
        dF = F - ex["neuropil_coefficient"] * Fneu
        snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)
        keep = snr > snr_threshold
        if step == 0 and snr_threshold > 0 and (~keep).sum() > 0:
            print(f"    removing {int((~keep).sum())} ROIs with snr < {snr_threshold}")
            stat = assign_priority_overlaps(stat[keep], Ly, Lx)
        else:
            break
    for st, sk, sd, sn in zip(stat, skew(dF, axis=1), np.std(dF, axis=1), snr):
        st["snr"], st["skew"], st["std"] = sn, sk, sd
    print(f"    extraction of {len(stat)} ROIs: {fmt_time(time.time() - t0)}")
    if settings["run"]["do_deconvolution"]:
        dFp = extraction.preprocess(F=dF.copy(), fs=settings["fs"], batch_size=ex["batch_size"],
                                    device=device, **settings["dcnv_preprocess"])
        spks = extraction.oasis(F=dFp, batch_size=ex["batch_size"], tau=settings["tau"],
                                fs=settings["fs"])
    else:
        spks = np.zeros_like(F)
    iscell = classification.classify(stat=stat, classfile=classfile)
    return stat, F, Fneu, spks, iscell


def save_combined_outputs(plane_dir, stat, F, Fneu, spks, iscell, info, stat_A, stat_B):
    """combined_index.npz (where each ROI came from) and combined_traces.npz / .mat
    (traces of AB, A_only and B_only; the .mat also holds both segmentations)."""
    from scipy.io import savemat
    source = np.array([st["combined_source"] for st in stat])
    a_idx = np.array([st["a_index"] for st in stat])
    b_idx = np.array([st["b_index"] for st in stat])
    np.savez(os.path.join(plane_dir, "combined_index.npz"), source=source, a_index=a_idx,
             b_index=b_idx, n_b_matched=np.array([st["n_b_matched"] for st in stat]),
             b_matched_to_a=info["b_matched_to_a"], b_overlap_coef=info["b_overlap_coef"],
             min_overlap=COMBINED_MIN_OVERLAP, anatomical=COMBINED_ANATOMICAL,
             functional=COMBINED_FUNCTIONAL)
    groups = {}
    for name, tag in (("AB", "AB"), ("Aonly", "A_only"), ("Bonly", "B_only")):
        ix = np.flatnonzero(source == tag)
        groups.update({f"idx_{name}": ix, f"F_{name}": F[ix], f"Fneu_{name}": Fneu[ix],
                       f"spks_{name}": spks[ix], f"iscell_{name}": iscell[ix],
                       f"a_index_{name}": a_idx[ix], f"b_index_{name}": b_idx[ix]})
    np.savez(os.path.join(plane_dir, "combined_traces.npz"), **groups)
    try:
        savemat(os.path.join(plane_dir, "combined_traces.mat"),
                {**groups, "source": source, "stat_A": np.array(stat_A, dtype=object),
                 "stat_B": np.array(stat_B, dtype=object)}, do_compression=True)
    except Exception as e:
        print(f"  ! combined_traces.mat not written: {type(e).__name__}: {e}")


def run_combined(db, settings, plane_dir):
    """Cellpose (A) + activity (B) detection in one run, one extraction.

    1  run_s2p with detection off: converts / locates the binary
    2  bin once, run both detectors on it (or reuse the saved A / B)
    3  merge; overlap flags with A priority (nothing is dropped)
    4  extraction, deconvolution, classification, red labels (3.3)
    5  the usual Suite2p files plus the combined outputs
    Suite2p's run_plane(stat=...) cannot be used: with a given stat it fails at
    the end (detect_outputs undefined)."""
    from suite2p import io as s2p_io
    device = torch_device()
    prep = copy.deepcopy(settings)
    prep["run"]["do_detection"] = False
    prep["io"].update(save_mat=False, save_ops_orig=False, delete_bin=False, move_bin=False)
    print("  combined: preparing the binary ...")
    run_s2p(db=db, settings=prep)

    db_p = load_dict(os.path.join(plane_dir, "db.npy"))
    reg_outputs = load_dict(os.path.join(plane_dir, "reg_outputs.npy"))
    n_frames, Ly, Lx = db_p["nframes"], db_p["Ly"], db_p["Lx"]
    classfile = pick_classifier(settings)
    saved = [os.path.join(plane_dir, f"{n}.npy")
             for n in ("stat_A", "stat_B", "detect_outputs_combined")]

    with s2p_io.BinaryFile(Ly=Ly, Lx=Lx, filename=db_p["reg_file"], n_frames=n_frames,
                           write=False) as f_reg:
        if COMBINED_REUSE_DETECTION and all(os.path.isfile(p) for p in saved):
            print("  combined: reusing the saved A / B detections")
            stat_A = list(np.load(saved[0], allow_pickle=True))
            stat_B = list(np.load(saved[1], allow_pickle=True))
            detect_outputs = load_dict(saved[2])
        else:
            detect_outputs, stat_A, stat_B = detect_a_and_b(
                f_reg, settings, reg_outputs, classfile, device, plane_dir)
        merged, info = merge_detections(stat_A, stat_B, Ly, Lx, COMBINED_MIN_OVERLAP)
        # each ROI keeps the statistics of its own detector: npix_norm is "size
        # relative to this method's typical ROI" and the classifier uses it
        stat = assign_priority_overlaps(np.array(merged, dtype=object), Ly, Lx)
        print(f"  combined: {info['n_A']} A + {info['n_B']} B -> {len(stat)} ROIs "
              f"({info['n_AB']} AB, {info['n_A_only']} A_only, {info['n_B_only']} B_only)")
        stat, F, Fneu, spks, iscell = extract_and_classify(
            stat, f_reg, settings, classfile, device, Ly, Lx)

    redcell = None
    if reg_outputs.get("meanImg_chan2") is not None:
        masks, redcell = detect_red(detect_outputs["meanImg_crop"],
                                    reg_outputs["meanImg_chan2"], stat)
        detect_outputs["chan2_masks"] = masks

    save_npy(plane_dir, stat=stat, F=F, Fneu=Fneu, spks=spks, iscell=iscell,
             detect_outputs=detect_outputs, zcorr=np.zeros((0,)), settings=settings)
    if redcell is not None:
        save_npy(plane_dir, redcell=redcell)
    ops = {**db_p, **copy.deepcopy(settings), **reg_outputs, **detect_outputs}  # save_mat edits nested dicts
    if settings["io"]["save_mat"]:
        s2p_io.save_mat(ops, stat, F, Fneu, spks, iscell, redcell, None, None)
    if settings["io"]["save_ops_orig"]:
        save_npy(plane_dir, ops=ops)
    save_combined_outputs(plane_dir, stat, F, Fneu, spks, iscell, info, stat_A, stat_B)
    if settings["io"]["delete_bin"]:
        remove_files([db_p[k] for k in ("reg_file", "reg_file_chan2", "raw_file",
                                        "raw_file_chan2") if k in db_p])
    return stat


# --- 3.6 plots and summary ---------------------------------------------------
def mean_projection(out_dir, mc_path, n_frames, Ly, Lx):
    """Mean image of the movie: Suite2p's meanImg, else from data.bin, else the tif."""
    for name in ("ops.npy", "reg_outputs.npy", "detect_outputs.npy"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            try:
                d = load_dict(path)
                if isinstance(d, dict) and d.get("meanImg") is not None:
                    return np.asarray(d["meanImg"], np.float32), f"Suite2p meanImg ({name})"
            except Exception:
                pass
    acc = np.zeros((Ly, Lx), np.float64)
    bin_path = os.path.join(out_dir, "data.bin")
    if os.path.isfile(bin_path) and os.path.getsize(bin_path) == n_frames * Ly * Lx * 2:
        mov = np.memmap(bin_path, dtype=np.int16, mode="r", shape=(n_frames, Ly, Lx))
        for i in range(0, n_frames, 1000):
            acc += mov[i:i + 1000].sum(axis=0, dtype=np.float64)
        return (acc / n_frames).astype(np.float32), "mean of all frames (data.bin)"
    with tifffile.TiffFile(mc_path) as tif:
        for page in tif.pages:
            acc += page.asarray()
    return (acc / n_frames).astype(np.float32), f"mean of all frames ({os.path.basename(mc_path)})"


def _draw_outline(ax, ypix, xpix, color, lw, ls="solid"):
    """Contour of one mask given by its pixel coordinates."""
    y0, x0 = ypix.min() - 1, xpix.min() - 1
    m = np.zeros((ypix.max() - y0 + 2, xpix.max() - x0 + 2))
    m[ypix - y0, xpix - x0] = 1
    ax.contour(np.arange(m.shape[1]) + x0, np.arange(m.shape[0]) + y0, m,
               levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def _load_red_masks(out_dir):
    """Blob masks of the red image: on the movie's grid (detect_outputs) or on its
    own grid (red_masks.npy). None for the intensity method."""
    path = os.path.join(out_dir, "red_masks.npy")
    if os.path.isfile(path):
        return np.load(path).astype(np.int32)
    path = os.path.join(out_dir, "detect_outputs.npy")
    if os.path.isfile(path):
        try:
            m = load_dict(path).get("chan2_masks")
            if m is not None and np.ndim(m) == 2:
                return np.asarray(m).astype(np.int32)
        except Exception as e:
            print(f"  ! could not read the red masks: {type(e).__name__}: {e}")
    return None


def plot_contours(out_dir, mc_path, red, n_frames, Ly, Lx, title):
    """roi_contours.png: ROI outlines on the mean projection and on the red image.
    Colours: cells by origin (combined) or red / not red; red blobs in red."""
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stat = np.load(os.path.join(out_dir, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(out_dir, "iscell.npy"))[:, 0].astype(bool)
    red_path = os.path.join(out_dir, "redcell.npy")
    isred = (np.nan_to_num(np.load(red_path)[:, 0]) > 0.5 if os.path.isfile(red_path)
             else np.zeros(len(stat), bool))
    red_masks = _load_red_masks(out_dir)
    mean_img, source = mean_projection(out_dir, mc_path, n_frames, Ly, Lx)
    comb = os.path.join(out_dir, "combined_index.npz")
    src = np.load(comb)["source"] if os.path.isfile(comb) else None
    src_color = {"AB": "yellow", "A_only": "lime", "B_only": "cyan"}

    images = [(mean_img, f"mean projection, functional\n{source}")]
    if red is not None:
        images.append((red, "red reference"))
    fig, axes = plt.subplots(1, len(images), figsize=(8 * len(images), 8.6), squeeze=False)
    for ax, (img, sub) in zip(axes[0], images):
        lo, hi = np.percentile(img, [1, 99.7])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        same_grid = img.shape == mean_img.shape
        if not same_grid:           # reference on its own grid: ROI pixels do not apply
            sub += "\n(own pixel grid: movie ROIs not drawn)"
        for i, s in enumerate(stat if same_grid else []):
            if not iscell[i] and not PLOT_NONCELLS:
                continue
            color = ("0.6" if not iscell[i] else src_color[str(src[i])] if src is not None
                     else "magenta" if isred[i] else "lime")
            _draw_outline(ax, s["ypix"], s["xpix"], color,
                          0.6 if iscell[i] else 0.4, "solid" if iscell[i] else "dotted")
            if PLOT_ROI_NUMBERS and iscell[i]:
                ax.text(s["xpix"].mean(), s["ypix"].mean(), str(i), color="yellow",
                        fontsize=4, ha="center", va="center")
        if red_masks is not None and red_masks.shape == img.shape:
            for k, sl in enumerate(ndimage.find_objects(red_masks), 1):
                if sl is not None:
                    yy, xx = np.nonzero(red_masks[sl] == k)
                    _draw_outline(ax, yy + sl[0].start, xx + sl[1].start, "red", 1.0)
        ax.set_title(sub, fontsize=9)
        ax.axis("off")

    n_cell, n_red = int(iscell.sum()), int((iscell & isred).sum())
    if src is not None:
        handles = [Line2D([], [], color=c, label=f"{k} cells ({int(((src == k) & iscell).sum())})")
                   for k, c in src_color.items()]
    else:
        handles = [Line2D([], [], color="magenta", label=f"cell, red ({n_red})"),
                   Line2D([], [], color="lime", label=f"cell, not red ({n_cell - n_red})")]
    if red_masks is not None:
        handles.append(Line2D([], [], color="red", label=f"red blobs ({int(red_masks.max())})"))
    if PLOT_NONCELLS:
        handles.append(Line2D([], [], color="0.6", ls=":", label=f"not cell ({len(stat) - n_cell})"))
    axes[0][0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, "roi_contours.png")
    fig.savefig(png, dpi=200)
    print(f"  contour plot: {png}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def summarize_outputs(plane_dir):
    """ROI counts and the output files after a run."""
    try:
        iscell = np.load(os.path.join(plane_dir, "iscell.npy"))
        msg = f"  ROIs: {len(iscell)} detected, {int(iscell[:, 0].sum())} classified as cells"
        red_path = os.path.join(plane_dir, "redcell.npy")
        if os.path.isfile(red_path):
            red = np.load(red_path)
            if np.isnan(red[:, 0]).all():
                msg += ", red labels: none"
            else:
                isr = np.nan_to_num(red[:, 0]) > 0.5
                msg += (f", {int(isr.sum())} red (all ROIs) / "
                        f"{int((isr & (iscell[:, 0] > 0)).sum())} red among cells")
        print(msg)
    except Exception as e:
        print(f"  ! could not read iscell/redcell: {e}")
    print(f"  output folder: {plane_dir}")
    for name in sorted(os.listdir(plane_dir)):
        path = os.path.join(plane_dir, name)
        print(f"    {name:<28s} {'<folder>' if os.path.isdir(path) else fmt_size(path)}")


# --- 3.7 one session ---------------------------------------------------------
def run_session(sess_dir, settings, label="", mc_path=None, red_path=None,
                shifts_path=None, title=None):
    """
    Run one session; everything printed also goes to plane0\\suite2p_log.txt.

    sess_dir     folder that receives suite2p\\plane0 (and where the default input
                 names MC_TIF / RED_TIF / SHIFTS_MAT are looked for)
    mc_path, red_path, shifts_path
                 explicit inputs anywhere on disk. None = the default name in
                 sess_dir; False (red / shifts) = deliberately none
    title        heading of the log and of the contour plot
    """
    install_red_detector()
    # Suite2p's save_mat replaces None by empty arrays IN PLACE in the nested dicts
    # of the settings it gets; one settings object reused across sessions would
    # break the next session. Every session gets its own copy.
    settings = copy.deepcopy(settings)
    sess_dir = os.path.abspath(sess_dir)
    if mc_path is None and not os.path.isdir(sess_dir):
        raise FileNotFoundError(f"session folder not found: {sess_dir}")
    if mc_path is not None and not os.path.isfile(mc_path):
        raise FileNotFoundError(f"motion-corrected movie not found: {mc_path}")
    plane_dir = os.path.join(sess_dir, "suite2p", "plane0")
    os.makedirs(plane_dir, exist_ok=True)
    log_path = os.path.join(plane_dir, "suite2p_log.txt")
    with Tee(log_path):
        try:
            _run_session(sess_dir, plane_dir, settings, label, mc_path, red_path,
                         shifts_path, title)
        except Exception as e:      # the failure goes to the session log too
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
        finally:
            print(f"  log: {log_path}")


def _load_red_reference(red_path, Ly, Lx):
    """(image, same_grid): the reference as a 2-D float array (a stack is averaged)."""
    red = tifffile.imread(red_path).astype(np.float32)
    if red.ndim == 3:
        red = red.mean(axis=0)
    same = red.shape == (Ly, Lx)
    if not same:
        print(f"  ! {os.path.basename(red_path)} is {red.shape} but the movie frames are "
              f"{(Ly, Lx)}: no per-ROI red labels (different pixel grids)")
    return red, same


def _run_session(sess_dir, plane_dir, settings, label, mc_path, red_path, shifts_path, title):
    t0 = time.time()
    # None = default name in sess_dir; False = deliberately none
    use_red = red_path is not False and RED_METHOD != "off"
    use_shifts = shifts_path is not False
    mc_path = mc_path or os.path.join(sess_dir, MC_TIF)
    red_path = red_path or os.path.join(sess_dir, RED_TIF)
    shifts_path = shifts_path or os.path.join(sess_dir, SHIFTS_MAT)
    animal, date, rec = describe_session(sess_dir)

    print(f"\n=== {label}{title or f'animal {animal} | date {date} | recording {rec}'}")
    print(f"  folder:  {sess_dir}")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print_settings(indent="  ")
    print("  inputs:")
    for p in (mc_path, red_path if use_red else None, shifts_path if use_shifts else None):
        if p:
            where = os.path.dirname(os.path.abspath(p))
            print(f"    {os.path.basename(p):<34s} {fmt_size(p)}"
                  f"{'' if where == sess_dir else '  <- ' + where}")
    if not os.path.isfile(mc_path):
        raise FileNotFoundError(mc_path)
    n_frames, Ly, Lx = movie_shape(mc_path)
    print(f"  {os.path.basename(mc_path)}: {n_frames} frames, {Ly} x {Lx} px "
          f"({n_frames / FS / 60:.1f} min at {FS} Hz)")

    # what Suite2p reads instead of registering (do_registration = 0)
    reg_outputs = {"yrange": [MARGIN, Ly - MARGIN], "xrange": [MARGIN, Lx - MARGIN],
                   "badframes": np.zeros(n_frames, bool)}
    if USE_MC_BAD_FRAMES and use_shifts and os.path.isfile(shifts_path):
        reg_outputs["badframes"] = load_bad_frames(shifts_path, n_frames)
        nb = int(reg_outputs["badframes"].sum())
        print(f"  bad frames excluded from detection: {nb} ({100 * nb / n_frames:.1f}%)")

    red, red_separate = None, False
    if use_red and os.path.isfile(red_path):
        red, same_grid = _load_red_reference(red_path, Ly, Lx)
        red_separate = not same_grid
        if same_grid:        # the presence of this key switches red labels on
            reg_outputs["meanImg_chan2"] = red
        print(f"  red reference {os.path.basename(red_path)}: min/median/max = "
              f"{red.min():.0f} / {np.median(red):.0f} / {red.max():.0f}")
    elif use_red:
        print(f"  ! {os.path.basename(red_path)} not found: no red labels")
    save_npy(plane_dir, reg_outputs=reg_outputs)

    db = default_db()
    db.update(data_path=[os.path.dirname(os.path.abspath(mc_path))],
              file_list=[os.path.basename(mc_path)], save_path0=sess_dir,
              save_folder="suite2p", nchannels=1, keep_movie_raw=False,
              input_format="tif", batch_size=BATCH_FRAMES)
    reused = reuse_existing_binary(db, settings, plane_dir, n_frames, Ly, Lx)
    fast = None if reused else fast_disk_for(sess_dir, mc_path, animal, date, rec)
    if fast:
        db["fast_disk"] = fast
    print(f"  binary (data.bin, ~{os.path.getsize(mc_path) / 1e9:.1f} GB): "
          f"{os.path.join(fast or sess_dir, 'suite2p', 'plane0')}"
          f"{' (existing)' if reused else ''}{' (deleted after the run)' if DELETE_BIN else ''}")

    print("  running suite2p (detection, extraction, deconvolution, classification) ...")
    try:
        if DETECTION_ALGORITHM == "combined":
            run_combined(db, settings, plane_dir)
        else:
            run_s2p(db=db, settings=settings)
    finally:                        # also on failure: do not leave a 20 GB binary behind
        cleanup_fast_disk(fast)
    print(f"  suite2p done in {fmt_time(time.time() - t0)}")

    if red_separate:
        try:
            detect_red_separately(red, plane_dir)
        except Exception as e:      # the green results are already saved
            print(f"  ! red detection failed: {type(e).__name__}: {e}")
    if PLOT_CONTOURS:
        try:
            plot_contours(plane_dir, mc_path, red, n_frames, Ly, Lx,
                          title or f"{animal} | {date} | {rec}")
        except Exception as e:      # a failed plot does not fail the session
            print(f"  ! contour plot failed: {type(e).__name__}: {e}")
    summarize_outputs(plane_dir)
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S} (total {fmt_time(time.time() - t0)})")


# --- 3.8 finding sessions ----------------------------------------------------
def is_excluded(path):
    """True if path is, or is below, a folder listed in EXCLUDE (relative to ROOT_DIR)."""
    try:
        rel = os.path.relpath(path, ROOT_DIR).lower()
    except ValueError:                      # different drive
        return False
    for e in EXCLUDE:
        e = os.path.normpath(e).lower().strip(os.sep)
        if rel == e or rel.startswith(e + os.sep):
            return True
    return False


def find_sessions(folders):
    """Session folders (those holding MC_TIF) in or below the given folders."""
    sessions = []
    for folder in folders:
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            print(f"  ! not a folder, skipped: {folder}")
            continue
        found = []
        if os.path.isfile(os.path.join(folder, MC_TIF)):
            found = [folder]
        else:
            base_depth = folder.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(folder):
                dirs.sort()
                if MC_TIF in files:
                    found.append(root)
                    dirs[:] = []                    # a session holds no sessions
                elif root.count(os.sep) - base_depth >= SEARCH_DEPTH:
                    dirs[:] = []
                else:                               # skip output folders and exclusions
                    dirs[:] = [d for d in dirs if d.lower() not in ("suite2p", "caiman", "celltype")
                               and not is_excluded(os.path.join(root, d))]
            print(f"  {folder}: {len(found)} session(s) found" if found
                  else f"  ! no {MC_TIF} found in or below {folder}")
        for s in found:
            if is_excluded(s):
                print(f"  excluded, skipped: {s}")
            elif s not in sessions:
                sessions.append(s)
    if SKIP_EXISTING:
        done = [s for s in sessions
                if os.path.isfile(os.path.join(s, "suite2p", "plane0", "stat.npy"))]
        for s in done:
            print(f"  already has Suite2p output, skipped: {s}")
        sessions = [s for s in sessions if s not in done]
    return sessions


# --- 3.9 main ----------------------------------------------------------------
def main(argv=None):
    """Process the folders given on the command line (or SESSIONS, or ROOT_DIR)."""
    args = sys.argv[1:] if argv is None else argv
    folders = args or SESSIONS or [ROOT_DIR]
    if not (args or SESSIONS):
        print(f"No session list given: searching all of {ROOT_DIR}")
    sessions = find_sessions(folders)
    if not sessions:
        sys.exit("No sessions to process.")
    print(f"{len(sessions)} session(s) to process:")
    for k, s in enumerate(sessions, 1):
        print(f"  [{k}] {s}")
    print_settings()
    settings = make_settings()
    failed, t_all = [], time.time()
    for k, s in enumerate(sessions, 1):
        try:
            run_session(s, settings, label=f"[{k}/{len(sessions)}] ")
        except Exception as e:     # report and continue with the next session
            print(f"  ! FAILED [{k}/{len(sessions)}] {s}: {type(e).__name__}: {e}")
            failed.append(s)
    print(f"\n{len(sessions) - len(failed)}/{len(sessions)} session(s) done "
          f"in {fmt_time(time.time() - t_all)}")
    for s in failed:
        print(f"  failed: {s}")


# =============================================================================
# 4. RUN
# =============================================================================
if __name__ == "__main__":
    main()
