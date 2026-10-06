"""
run_suite2p_on_MC.py

Run Suite2p cell detection, signal extraction, deconvolution and
classification on sessions already motion-corrected by batch_2p_Calcium.m
(sbx_motion_correct), skipping Suite2p's own registration.

Per session folder (<date>\\<exptype>_rec<rid>\\) it uses:
    ConcatenatedMovie_MC.tif           functional (green) movie, already registered
    ref_field_MC_registeration.tif     red reference mean image, already registered
                                       to this session (batch step 2e); passed
                                       to Suite2p as the channel-2 mean image so
                                       ROIs get a red / not-red label (redcell)
    ConcatenatedMovie_MC_shifts.mat    optional: frames flagged 'bad' during
                                       motion correction are excluded from
                                       detection (not from extraction)

Outputs go to <session>\\suite2p\\plane0\\ (stat.npy, F.npy, Fneu.npy, spks.npy,
iscell.npy, redcell.npy, ops.npy, and Fall.mat for MATLAB). Columns of F
match the frames of ConcatenatedMovie_MC.tif, i.e. the rows of
output_behaviorConcat.mat. In Fall.mat, stat{i}.ypix / xpix are 0-based:
add 1 for MATLAB indexing. Also written there:
    suite2p_log.txt     everything printed for this session (also shown in
                        the terminal), including Suite2p's own messages
    roi_contours.png    all ROI contours on the mean projection of the
                        functional movie (and on the red reference)

Speed: Suite2p copies the movie into its own binary (data.bin, about the size
of the tif) and reads that binary several times. By default the binary is
written to the session's own folder (same drive as the data; free space is
checked first) and deleted after the run (DELETE_BIN). Nothing is written
to C:. FAST_DISK can point it at another drive instead.

How it works: with settings["run"]["do_registration"] = 0, Suite2p converts the
tif to its binary and, instead of registering, loads reg_outputs.npy from the
plane folder if it exists. This script writes that file first, with the
valid region, the bad frames and the red mean image ("meanImg_chan2"), which
detection then uses to compute redcell. No second-channel movie is needed,
so F_chan2 is not computed.

Written for the suite2p main branch API as of June 2026 (nested settings:
settings["run"], ["detection"], ["io"], ...). Older releases (ops dict) need
different keys.

Usage:
    python run_suite2p_on_MC.py <folder> [<folder> ...]
or edit SESSIONS below and run without arguments.

Each <folder> is either a session folder (it holds ConcatenatedMovie_MC.tif)
or a folder above the sessions, e.g. a date folder Y:\\mouse1\\260929 or an
animal folder Y:\\mouse1: every session found below it (up to SEARCH_DEPTH
levels down) is processed. Put paths with spaces in quotes.
"""
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

import suite2p
from suite2p import default_db, default_settings, run_s2p

#%%%%%%%%%%%%%%%%
#%% USER INPUT %%
#%%%%%%%%%%%%%%%%
# Session folders, or folders above them (date / animal folders are searched
# for sessions). One per line, each with r"..." and a comma, e.g.
#     SESSIONS = [
#         r"Y:\mouse1\260929\932_3dir_rec000",    # one session
#         r"Y:\mouse1\260930",                    # every session of that date
#         r"Y:\mouse2",                           # every session of that animal
#     ]
SESSIONS = [
    r"Y:\mouse1\260929\932_3dir_rec000",
]
# If SESSIONS is empty (and no folders are given on the command line), every
# session found under ROOT_DIR is processed.
ROOT_DIR = "Y:\\"
# Folders to skip, relative to ROOT_DIR: an animal ("mouse3"), a date
# ("mouse1\260930") or a session ("mouse1\260930\932_3dir_rec000"). Anything
# below a listed folder is skipped too.
EXCLUDE = []
SEARCH_DEPTH = 3           # how many folder levels below a given folder to search
SKIP_EXISTING = True      # True: skip sessions that already have Suite2p output
                           # (suite2p\plane0\stat.npy), e.g. to resume a batch

FS = 31.25            # imaging frame rate (Hz) of the concatenated movie -- set this
TAU = 0.7            # indicator decay (s): ~0.7 GCaMP6f, ~1.0-1.25 GCaMP6s
DIAMETER = 12        # expected soma diameter in px (SPN somata ~10-15 um)
MARGIN = 0           # px ignored at every edge for detection (edge-replicated
                     # pixels of extreme frames, if any)
USE_MC_BAD_FRAMES = True   # exclude frames flagged 'bad' by sbx_motion_correct
                           # from detection

# ROI detection (sparsery). Check roi_contours.png after changing these.
SPATIAL_SCALE = 1          # cell size used for detection: 0 = Suite2p estimates it
                           # (DIAMETER is not used for this); 1, 2, 3, 4 force
                           # ~6, 12, 24, 48 px. Use 2 for ~12 px somata if the
                           # log says "estimated spatial scale ~6 pixels" and
                           # the ROIs look like soma fragments
THRESHOLD_SCALING = 1.0    # detection threshold: higher = fewer, stronger ROIs;
                           # lower = more, weaker ROIs (Suite2p default 1.0)

# Cell / not-cell classifier (fills iscell.npy)
CLASSIFIER_PATH = None     # your own classifier .npy (Suite2p GUI: Classifier >
                           # Build, from curated iscell.npy files). None = the
                           # default saved in the GUI (classifier_user.npy in
                           # ~/.suite2p/classifiers) if there is one, else
                           # Suite2p's built-in classifier

# Red labelling. Suite2p's intensity method: redprob = red signal inside the
# ROI / (inside + surrounding neuropil), both weighted means, so ~0.5 means
# "as bright as the surround". Suite2p's settings default (0.25) is meant
# for the Cellpose IoU method and labels almost every ROI red with the
# intensity method; 0.65 is the intensity function's own default. Check the
# redprob histogram per FOV and adjust.
CHAN2_THRESHOLD = 0.65
CELLPOSE_CHAN2 = False     # True: Cellpose finds cells in the red image and
                           # redcell = IoU with them > CHAN2_THRESHOLD (then
                           # use ~0.25)

# Speed / memory
FAST_DISK = r"E:\suite2p_tmp"  # None: Suite2p's binary goes in <session>\suite2p\plane0\,
                           # on the drive the data is on (needs free space ~ movie
                           # size). Or a folder on another drive, e.g. r"E:\suite2p_tmp"
DELETE_BIN = True          # delete the binary after the run (it is a full copy
                           # of the movie; keep it only to browse it in the GUI)
TORCH_DEVICE = "cpu"       # "cuda" only with an NVIDIA GPU (none on this PC)
BATCH_FRAMES = 2000        # frames per batch for tif->binary conversion and
                           # extraction (default 500); ~1 MB RAM/frame at 512x512
                           # float32, so 2000 ~ 2 GB. Lower if memory runs out.

PLOT_CONTOURS = True       # save roi_contours.png per session
PLOT_NONCELLS = True       # also draw ROIs Suite2p classified as not-cell (gray)
PLOT_ROI_NUMBERS = False   # write the 0-based ROI index next to each cell
SHOW_PLOTS = False         # also open each figure (blocks until closed)
SUITE2P_LOG_LEVEL = logging.INFO   # Suite2p's own messages; logging.DEBUG for more

MC_TIF = "ConcatenatedMovie_MC.tif"
RED_TIF = "ref_field_MC_registeration.tif"
SHIFTS_MAT = "ConcatenatedMovie_MC_shifts.mat"
#%%%%%%%%%%%%%%%%%%%%%
#%% END: USER INPUT %%
#%%%%%%%%%%%%%%%%%%%%%


class Tee:
    """Write to the terminal and a log file at the same time. Suite2p reports
    progress through the 'suite2p' logger, which has no handler when run from
    a script, so a handler writing here is attached for the duration."""

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
    """File size as a readable string, or 'missing'."""
    if not os.path.isfile(path):
        return "missing"
    n = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def fmt_time(sec):
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def describe_session(sess_dir):
    """animal / date / recording from <animal>\\<date>\\<exptype>_rec<rid>."""
    parts = os.path.normpath(sess_dir).split(os.sep)
    parts += [""] * (3 - len(parts))
    return parts[-3], parts[-2], parts[-1]


def print_settings(indent=""):
    try:
        s2p_version = _pkg_version("suite2p")
    except Exception:
        s2p_version = getattr(suite2p, "__version__", "?")
    print(f"{indent}suite2p {s2p_version}  |  "
          f"python {sys.version.split()[0]}  |  {sys.executable}")
    print(f"{indent}settings: fs={FS} Hz, tau={TAU} s, diameter={DIAMETER} px, margin={MARGIN} px, "
          f"bad frames={'on' if USE_MC_BAD_FRAMES else 'off'}, "
          f"red: {'cellpose IoU' if CELLPOSE_CHAN2 else 'intensity'} > {CHAN2_THRESHOLD}")
    scale = {0: "auto"}.get(SPATIAL_SCALE, f"{SPATIAL_SCALE} (~{3 * 2 ** SPATIAL_SCALE} px)")
    print(f"{indent}detection: spatial scale={scale}, threshold scaling={THRESHOLD_SCALING}, "
          f"classifier={CLASSIFIER_PATH or 'GUI default / built-in'}")
    print(f"{indent}speed: device={TORCH_DEVICE}, batch={BATCH_FRAMES} frames, "
          f"binary in={FAST_DISK or 'session folder'}, delete binary={DELETE_BIN}")


def mean_projection(out_dir, mc_path, n_frames, Ly, Lx):
    """Mean image of the functional movie: Suite2p's meanImg if it saved one,
    else computed from its binary (data.bin), else from the tif."""
    for name in ("ops.npy", "reg_outputs.npy", "detect_outputs.npy"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            try:
                d = np.load(path, allow_pickle=True).item()
                if isinstance(d, dict) and d.get("meanImg") is not None:
                    return np.asarray(d["meanImg"], np.float32), f"Suite2p meanImg ({name})"
            except Exception:
                pass
    bin_path = os.path.join(out_dir, "data.bin")
    if os.path.isfile(bin_path) and os.path.getsize(bin_path) == n_frames * Ly * Lx * 2:
        mov = np.memmap(bin_path, dtype=np.int16, mode="r", shape=(n_frames, Ly, Lx))
        acc = np.zeros((Ly, Lx), np.float64)
        for i in range(0, n_frames, 1000):
            acc += mov[i:i + 1000].sum(axis=0, dtype=np.float64)
        return (acc / n_frames).astype(np.float32), "mean of all frames (data.bin)"
    with tifffile.TiffFile(mc_path) as tif:
        acc = np.zeros((Ly, Lx), np.float64)
        for page in tif.pages:
            acc += page.asarray()
    return (acc / n_frames).astype(np.float32), f"mean of all frames ({MC_TIF})"


def plot_contours(out_dir, mc_path, red, n_frames, Ly, Lx, title):
    """All ROI contours on the mean projection (and on the red reference)."""
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stat = np.load(os.path.join(out_dir, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(out_dir, "iscell.npy"))[:, 0].astype(bool)
    red_path = os.path.join(out_dir, "redcell.npy")
    isred = (np.load(red_path)[:, 0].astype(bool) if os.path.isfile(red_path)
             else np.zeros(len(stat), bool))
    mean_img, source = mean_projection(out_dir, mc_path, n_frames, Ly, Lx)

    images = [(mean_img, f"mean projection, functional\n{source}")]
    if red is not None:
        images.append((red, f"red reference\n{RED_TIF}"))
    fig, axes = plt.subplots(1, len(images), figsize=(8 * len(images), 8.6), squeeze=False)
    for ax, (img, sub) in zip(axes[0], images):
        lo, hi = np.percentile(img, [1, 99.7])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        for i, s in enumerate(stat):
            if not iscell[i] and not PLOT_NONCELLS:
                continue
            y0, x0 = s["ypix"].min() - 1, s["xpix"].min() - 1
            m = np.zeros((s["ypix"].max() - y0 + 2, s["xpix"].max() - x0 + 2))
            m[s["ypix"] - y0, s["xpix"] - x0] = 1
            color = ("magenta" if isred[i] else "lime") if iscell[i] else "0.6"
            ax.contour(np.arange(m.shape[1]) + x0, np.arange(m.shape[0]) + y0, m,
                       levels=[0.5], colors=[color], linewidths=0.6 if iscell[i] else 0.4,
                       linestyles="solid" if iscell[i] else "dotted")
            if PLOT_ROI_NUMBERS and iscell[i]:
                ax.text(s["xpix"].mean(), s["ypix"].mean(), str(i), color="yellow",
                        fontsize=4, ha="center", va="center")
        ax.set_title(sub, fontsize=9)
        ax.axis("off")
    n_cell, n_red = int(iscell.sum()), int((iscell & isred).sum())
    handles = [Line2D([], [], color="magenta", label=f"cell, red ({n_red})"),
               Line2D([], [], color="lime", label=f"cell, not red ({n_cell - n_red})")]
    if PLOT_NONCELLS:
        handles.append(Line2D([], [], color="0.6", ls=":",
                              label=f"not cell ({len(stat) - n_cell})"))
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
    """ROI counts and output files after a run."""
    try:
        iscell = np.load(os.path.join(plane_dir, "iscell.npy"))
        msg = f"  ROIs: {len(iscell)} detected, {int(iscell[:, 0].sum())} classified as cells"
        red_path = os.path.join(plane_dir, "redcell.npy")
        if os.path.isfile(red_path):
            red = np.load(red_path)
            msg += (f", {int(red[:, 0].sum())} red (all ROIs) / "
                    f"{int((red[:, 0] * iscell[:, 0]).sum())} red among cells")
        print(msg)
    except Exception as e:
        print(f"  ! could not read iscell/redcell: {e}")
    print(f"  output folder: {plane_dir}")
    for name in sorted(os.listdir(plane_dir)):
        path = os.path.join(plane_dir, name)
        print(f"    {name:<20s} {'<folder>' if os.path.isdir(path) else fmt_size(path)}")


def movie_shape(path):
    """Number of frames and frame size of a tif without loading it."""
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        Ly, Lx = tif.pages[0].shape[:2]
    return n, Ly, Lx


def load_bad_frames(path, n_frames):
    """'bad' from the -v7.3 _shifts.mat (HDF5), as a bool vector."""
    with h5py.File(path, "r") as f:
        if "bad" not in f:
            return np.zeros(n_frames, bool)
        bad = np.asarray(f["bad"]).astype(bool).ravel()
    if bad.size != n_frames:
        print(f"  ! {os.path.basename(path)}: {bad.size} bad-frame flags but "
              f"{n_frames} frames; not using them")
        return np.zeros(n_frames, bool)
    return bad


def make_settings():
    # start from the defaults and change nested keys in place: run_s2p merges
    # settings only at the top level, so a partial nested dict would drop keys
    settings = default_settings()
    settings["fs"] = FS
    settings["tau"] = TAU
    settings["diameter"] = [DIAMETER, DIAMETER]
    settings["run"]["do_registration"] = 0
    settings["detection"]["chan2_threshold"] = CHAN2_THRESHOLD
    settings["detection"]["cellpose_chan2"] = CELLPOSE_CHAN2
    settings["detection"]["sparsery_settings"]["spatial_scale"] = SPATIAL_SCALE
    settings["detection"]["threshold_scaling"] = THRESHOLD_SCALING
    if CLASSIFIER_PATH:
        # Suite2p silently falls back to another classifier if the file is missing
        if not os.path.isfile(CLASSIFIER_PATH):
            raise FileNotFoundError(f"CLASSIFIER_PATH not found: {CLASSIFIER_PATH}")
        settings["classification"]["classifier_path"] = CLASSIFIER_PATH
    settings["io"]["save_mat"] = True
    settings["io"]["delete_bin"] = DELETE_BIN
    settings["torch_device"] = TORCH_DEVICE
    settings["extraction"]["batch_size"] = BATCH_FRAMES
    return settings


def run_session(sess_dir, settings, label=""):
    """Run one session; everything printed also goes to plane0's suite2p_log.txt."""
    sess_dir = os.path.abspath(sess_dir)
    if not os.path.isdir(sess_dir):
        raise FileNotFoundError(f"session folder not found: {sess_dir}")
    plane_dir = os.path.join(sess_dir, "suite2p", "plane0")
    os.makedirs(plane_dir, exist_ok=True)
    log_path = os.path.join(plane_dir, "suite2p_log.txt")
    with Tee(log_path):
        try:
            _run_session(sess_dir, plane_dir, settings, label)
        except Exception as e:      # log the failure in the session log too
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
        finally:
            print(f"  log: {log_path}")


def _fast_disk_for(sess_dir, mc_path, animal, date, rec):
    """Per-session folder on FAST_DISK if it has room for the binary, else None."""
    need = os.path.getsize(mc_path) * 1.1
    if not FAST_DISK:               # binary in the session folder: check that drive
        drive = os.path.splitdrive(sess_dir)[0] or sess_dir
        free = shutil.disk_usage(sess_dir).free
        print(f"  free space on {drive}: {free / 1e9:.1f} GB "
              f"({free / 1e9 - need / 1e9:.1f} GB left while it runs)")
        if free < need:
            raise OSError(f"not enough space on {drive} for the binary: need "
                          f"~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
        return None
    os.makedirs(FAST_DISK, exist_ok=True)
    free = shutil.disk_usage(FAST_DISK).free
    if free < need:
        print(f"  ! FAST_DISK {FAST_DISK} has {free / 1e9:.1f} GB free, binary needs "
              f"~{need / 1e9:.1f} GB: writing it to the session folder instead")
        return None
    return os.path.join(FAST_DISK, f"{animal}_{date}_{rec}")


def _cleanup_fast_disk(fast):
    """Remove the session's FAST_DISK folder (binary included if DELETE_BIN)."""
    if not fast or not os.path.isdir(fast):
        return
    files = [os.path.join(r, f) for r, _, fs in os.walk(fast) for f in fs]
    if DELETE_BIN or not files:
        shutil.rmtree(fast, ignore_errors=True)
    else:
        print(f"  binary kept on fast disk: {', '.join(files)}")


def _run_session(sess_dir, plane_dir, settings, label):
    t0 = time.time()
    animal, date, rec = describe_session(sess_dir)
    print(f"\n=== {label}animal {animal} | date {date} | recording {rec}")
    print(f"  folder:  {sess_dir}")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print_settings(indent="  ")
    print("  inputs:")
    for name in (MC_TIF, RED_TIF, SHIFTS_MAT):
        print(f"    {name:<34s} {fmt_size(os.path.join(sess_dir, name))}")

    mc_path = os.path.join(sess_dir, MC_TIF)
    if not os.path.isfile(mc_path):
        raise FileNotFoundError(mc_path)
    n_frames, Ly, Lx = movie_shape(mc_path)
    print(f"  {MC_TIF}: {n_frames} frames, {Ly} x {Lx} px "
          f"({n_frames / FS / 60:.1f} min at {FS} Hz)")

    reg_outputs = {
        "yrange": [MARGIN, Ly - MARGIN],     # Python ranges: [start, stop)
        "xrange": [MARGIN, Lx - MARGIN],
        "badframes": np.zeros(n_frames, bool),
    }

    shifts_path = os.path.join(sess_dir, SHIFTS_MAT)
    if USE_MC_BAD_FRAMES and os.path.isfile(shifts_path):
        reg_outputs["badframes"] = load_bad_frames(shifts_path, n_frames)
        nb = int(reg_outputs["badframes"].sum())
        print(f"  bad frames excluded from detection: {nb} ({100 * nb / n_frames:.1f}%)")

    red = None
    red_path = os.path.join(sess_dir, RED_TIF)
    if os.path.isfile(red_path):
        red = tifffile.imread(red_path).astype(np.float32)
        if red.shape != (Ly, Lx):
            raise ValueError(f"{RED_TIF} is {red.shape}, movie frames are {(Ly, Lx)}")
        reg_outputs["meanImg_chan2"] = red
        print(f"  red reference: {RED_TIF} (redcell will be computed), intensity "
              f"min/median/max = {red.min():.0f} / {np.median(red):.0f} / {red.max():.0f}")
    else:
        print(f"  ! no {RED_TIF}: no red labels for this session")

    # Suite2p loads this from its plane folder instead of registering
    # (do_registration = 0), so it has to go into plane0 before the run
    np.save(os.path.join(plane_dir, "reg_outputs.npy"), reg_outputs)

    db = default_db()
    db["data_path"] = [sess_dir]
    db["file_list"] = [MC_TIF]     # only this tif; the folder holds other tifs
    db["save_path0"] = sess_dir
    db["save_folder"] = "suite2p"
    db["nchannels"] = 1
    db["keep_movie_raw"] = False
    db["input_format"] = "tif"
    db["batch_size"] = BATCH_FRAMES
    fast = _fast_disk_for(sess_dir, mc_path, animal, date, rec)
    if fast:
        db["fast_disk"] = fast
    print(f"  binary (data.bin, ~{os.path.getsize(mc_path) / 1e9:.1f} GB): "
          f"{os.path.join(fast or sess_dir, 'suite2p', 'plane0')}"
          f"{' (deleted after the run)' if DELETE_BIN else ''}")

    print("  running suite2p (detection, extraction, deconvolution, classification) ...")
    try:
        run_s2p(db=db, settings=settings)
    finally:                        # also on failure: do not leave 20 GB behind
        _cleanup_fast_disk(fast)
    print(f"  suite2p done in {fmt_time(time.time() - t0)}")

    if PLOT_CONTOURS:
        try:
            plot_contours(plane_dir, mc_path, red, n_frames, Ly, Lx,
                          f"{animal} | {date} | {rec}")
        except Exception as e:      # a failed plot should not fail the session
            print(f"  ! contour plot failed: {type(e).__name__}: {e}")
    summarize_outputs(plane_dir)
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S} (total {fmt_time(time.time() - t0)})")


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
        if os.path.isfile(os.path.join(folder, MC_TIF)):
            found = [folder]
        else:
            found = []
            base_depth = folder.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(folder):
                dirs.sort()
                if MC_TIF in files:
                    found.append(root)
                    dirs[:] = []                    # a session has no sessions inside
                elif root.count(os.sep) - base_depth >= SEARCH_DEPTH:
                    dirs[:] = []
                else:                               # never look inside output folders
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


if __name__ == "__main__":
    folders = sys.argv[1:] or SESSIONS or [ROOT_DIR]
    if not (sys.argv[1:] or SESSIONS):
        print(f"No session list given: searching all of {ROOT_DIR}")
    sessions = find_sessions(folders)
    if not sessions:
        sys.exit("No sessions to process.")
    print(f"{len(sessions)} session(s) to process:")
    for k, s in enumerate(sessions, 1):
        print(f"  [{k}] {s}")
    print_settings()
    settings = make_settings()
    failed = []
    t_all = time.time()
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
