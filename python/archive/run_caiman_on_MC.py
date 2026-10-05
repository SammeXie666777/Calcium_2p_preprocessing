"""
run_caiman_on_MC.py

Run CaImAn (CNMF) source extraction on sessions already motion-corrected by
batch_2p_Calcium.m (sbx_motion_correct). CaImAn's own motion correction is
skipped. Only the functional (green) movie is used: no red channel.

Per session folder (<animal>\\<date>\\<exptype>_rec<rid>\\) it uses:
    ConcatenatedMovie_MC.tif           functional (green) movie, already registered
    ConcatenatedMovie_MC_shifts.mat    optional: the number of frames flagged
                                       'bad' is reported (CNMF has no bad-frame
                                       option, so they are NOT excluded)

Outputs go to <session>\\caiman\\:
    caiman_results.hdf5   the full CaImAn object (load with
                          caiman.source_extraction.cnmf.cnmf.load_CNMF)
    caiman_results.mat    for MATLAB: A (pixels x components, sparse; pixel
                          index is column-major like MATLAB, so
                          reshape(A(:,i), dims) is the footprint), C, S, F_dff,
                          YrA, accepted (logical), snr, r_values, centers
                          (row, col; 1-based), dims, fs
    A.npz, C.npy, S.npy, F_dff.npy, YrA.npy, accepted.npy, stats.csv
                          the same for Python (A: scipy.sparse.load_npz; pixel
                          index is Fortran order: A[:, i].reshape(dims, order="F"))
    mean_image.npy        mean projection of the movie
    roi_contours.png      contours of all components on the mean projection
    caiman_log.txt        everything printed for this session (also shown in
                          the terminal), including CaImAn's own messages
Columns of C / S / F_dff match the frames of ConcatenatedMovie_MC.tif, i.e.
the rows of output_behaviorConcat.mat.

How it works: CaImAn needs the movie as a float32 memory-mapped file
(pixels x frames). CaImAn's own converter loads a whole tif into RAM, which
does not fit for a 20-30 GB movie, so this script writes that file itself,
streaming the tif in chunks, in CaImAn's format and naming. The file is
TWICE the size of a 16-bit tif (float32). It is written to the session's own
folder (<session>\\caiman\\memmap\\, same drive as the data; free space is
checked first) and deleted after the run. Nothing is written to C:.

Install (CaImAn is NOT on pip: "pip install caiman" is an unrelated package):
    install Miniforge, then in the Miniforge Prompt
        mamba create -n caiman -c conda-forge caiman
        conda activate caiman
    Works on Windows, Linux and macOS. Keep it in its own environment.

Usage (in the caiman environment):
    python run_caiman_on_MC.py <session folder> [<session folder> ...]
or edit SESSIONS below and run without arguments.

Written for CaImAn 1.11-1.12 (CNMFParams with a flat params_dict).
"""
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime

import h5py
import numpy as np
import scipy.sparse
import tifffile
from scipy.io import savemat

#%%%%%%%%%%%%%%%%
#%% USER INPUT %%
#%%%%%%%%%%%%%%%%
SESSIONS = [
    # r"Y:\mouse1\260928\932_3dir_rec000",
]

FS = 31.25           # imaging frame rate (Hz) of the concatenated movie
DECAY_TIME = 0.4     # length of a typical transient (s): ~0.4 GCaMP6f, ~1.0+ GCaMP6s
UM_PER_PX = 1.0      # pixel size (um); only stored in the results
MARGIN = 0           # px ignored at every edge (edge-replicated pixels, if any)

# Cell size and density
GSIG = 6             # expected HALF-size of a soma in px (soma diameter / 2)
RF = 25              # half-size of the patches the FOV is split into (px);
                     # at least 3-4 x GSIG. None = no patches (whole FOV at once)
STRIDE = 12          # overlap between patches (px); about one soma diameter
K = 6                # components to look for PER PATCH: set from cell density,
                     # a bit above the real number in a (2*RF) x (2*RF) px patch

# CNMF
P = 1                # order of the calcium dynamics model (1: instant rise; 0: no deconvolution)
GNB = 2              # number of background components
MERGE_THR = 0.85     # merge components whose traces correlate above this
SSUB, TSUB = 2, 2    # spatial / temporal downsampling for the initialization only
REFIT = True         # second CNMF pass seeded with the first (better, ~2x time)

# Component quality (accepted = passes these; rejected ones are kept and flagged)
MIN_SNR = 2.0        # transient signal-to-noise
RVAL_THR = 0.85      # spatial correlation between footprint and raw data
USE_CNN = False      # CaImAn's CNN shape classifier (needs tensorflow + its
                     # model files: run "caimanmanager install" once)

# Speed / memory
N_PROCESSES = 4      # worker processes (this PC: 6 cores, 32 GB). None = cores - 1
MEMMAP_DIR = None    # None: the float32 memmap goes in <session>\caiman\memmap\, on
                     # the drive the data is on (needs free space = 2 x a 16-bit
                     # tif). Or a folder on another drive, e.g. r"E:\caiman_tmp"
DELETE_MEMMAP = True # delete the memmap after the run
CHUNK_FRAMES = 1000  # frames read from the tif at a time while writing the memmap

DFF_QUANTILE = 8     # dF/F baseline: this percentile over a running window of
DFF_WINDOW_S = 60    # this many seconds

PLOT_CONTOURS = True       # save caiman\roi_contours.png per session
PLOT_REJECTED = True       # also draw rejected components (gray)
PLOT_ROI_NUMBERS = False   # write the 0-based component index next to each cell
SHOW_PLOTS = False         # also open each figure (blocks until closed)
CONTOUR_ENERGY = 0.9       # contour encloses this fraction of a footprint's energy
CAIMAN_LOG_LEVEL = logging.INFO   # CaImAn's own messages; logging.WARNING for fewer

MC_TIF = "ConcatenatedMovie_MC.tif"
SHIFTS_MAT = "ConcatenatedMovie_MC_shifts.mat"
OUT_FOLDER = "caiman"
#%%%%%%%%%%%%%%%%%%%%%
#%% END: USER INPUT %%
#%%%%%%%%%%%%%%%%%%%%%


class Tee:
    """Write to the terminal and a log file at the same time. CaImAn reports
    progress through the logging module, so a handler writing here is attached
    to the root logger for the duration."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "a", encoding="utf-8")
        self.handler = logging.StreamHandler(self)
        self.handler.setFormatter(logging.Formatter("    [caiman %(asctime)s] %(message)s",
                                                    "%H:%M:%S"))
        self.logger = logging.getLogger()

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        self.handler.setLevel(CAIMAN_LOG_LEVEL)
        self.old_level = self.logger.level
        self.logger.addHandler(self.handler)
        self.logger.setLevel(CAIMAN_LOG_LEVEL)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.old_level)
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
        import caiman
        cm_version = getattr(caiman, "__version__", "?")
    except Exception as e:
        cm_version = f"NOT IMPORTABLE ({type(e).__name__})"
    print(f"{indent}caiman {cm_version}  |  python {sys.version.split()[0]}  |  {sys.executable}")
    print(f"{indent}settings: fs={FS} Hz, decay={DECAY_TIME} s, gSig={GSIG} px, "
          f"patches rf={RF} stride={STRIDE} K={K}, p={P}, gnb={GNB}, merge={MERGE_THR}, "
          f"refit={REFIT}")
    print(f"{indent}quality: min SNR={MIN_SNR}, r >= {RVAL_THR}, CNN={'on' if USE_CNN else 'off'}")
    print(f"{indent}speed: processes={N_PROCESSES or 'cores-1'}, memmap dir="
          f"{MEMMAP_DIR or 'session folder'}, delete memmap={DELETE_MEMMAP}")


def movie_shape(path):
    """Number of frames and frame size of a tif without loading it."""
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        Ly, Lx = tif.pages[0].shape[:2]
    return n, Ly, Lx


def count_bad_frames(path, n_frames):
    """Number of frames flagged 'bad' in the -v7.3 _shifts.mat, or None."""
    try:
        with h5py.File(path, "r") as f:
            if "bad" not in f:
                return None
            bad = np.asarray(f["bad"]).astype(bool).ravel()
        return int(bad.sum()) if bad.size == n_frames else None
    except Exception:
        return None


def write_memmap(mc_path, mmap_dir, n_frames, Ly, Lx):
    """Stream the tif into a CaImAn memmap; return its path and the mean image.

    CaImAn's format: float32, shape (pixels, frames) stored in C order, with
    each frame flattened in Fortran (column-major) order, and the dimensions
    encoded in the file name.
    """
    os.makedirs(mmap_dir, exist_ok=True)
    fname = os.path.join(
        mmap_dir, f"memmap__d1_{Ly}_d2_{Lx}_d3_1_order_C_frames_{n_frames}.mmap")
    mm = np.memmap(fname, mode="w+", dtype=np.float32, shape=(Ly * Lx, n_frames), order="C")
    acc = np.zeros((Ly, Lx), np.float64)
    t0 = time.time()
    with tifffile.TiffFile(mc_path) as tif:
        for a in range(0, n_frames, CHUNK_FRAMES):
            b = min(a + CHUNK_FRAMES, n_frames)
            chunk = tif.asarray(key=range(a, b)).astype(np.float32)
            chunk = chunk.reshape(b - a, Ly, Lx)        # a single frame comes back 2-D
            acc += chunk.sum(axis=0, dtype=np.float64)
            mm[:, a:b] = chunk.transpose(1, 2, 0).reshape(Ly * Lx, b - a, order="F")
            if (a // CHUNK_FRAMES) % 5 == 4 or b == n_frames:
                print(f"    memmap: {b}/{n_frames} frames, {fmt_time(time.time() - t0)}")
    mm.flush()
    del mm
    return fname, (acc / n_frames).astype(np.float32)


def check_memmap(images, mc_path, n_frames, Ly, Lx):
    """Make sure CaImAn reads the memmap back as the same movie."""
    if images.shape != (n_frames, Ly, Lx):
        raise RuntimeError(f"memmap loads as {images.shape}, expected {(n_frames, Ly, Lx)}")
    with tifffile.TiffFile(mc_path) as tif:
        for k in (0, n_frames // 2, n_frames - 1):
            if not np.array_equal(np.asarray(images[k]), tif.pages[k].asarray().astype(np.float32)):
                raise RuntimeError(f"memmap frame {k} differs from the tif: memmap layout "
                                   f"does not match this CaImAn version")


def footprint_mask(A, i, dims):
    """Pixels (rows, cols) holding CONTOUR_ENERGY of component i's footprint."""
    lo, hi = A.indptr[i], A.indptr[i + 1]
    idx, val = A.indices[lo:hi], A.data[lo:hi]
    if len(idx) == 0:
        return np.array([], int), np.array([], int)
    order = np.argsort(-val)
    energy = np.cumsum(val[order] ** 2)
    keep = idx[order][:np.searchsorted(energy, CONTOUR_ENERGY * energy[-1]) + 1]
    return np.unravel_index(keep, dims, order="F")


def component_centers(A, dims):
    """Centre of mass (row, col), 0-based, of every footprint."""
    A = scipy.sparse.csc_matrix(A)
    rows, cols = np.unravel_index(A.indices, dims, order="F")
    comp = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
    w = np.bincount(comp, A.data, A.shape[1]) + 1e-12
    return np.c_[np.bincount(comp, A.data * rows, A.shape[1]) / w,
                 np.bincount(comp, A.data * cols, A.shape[1]) / w]


def plot_contours(out_dir, A, dims, accepted, mean_img, title):
    """All component contours on the mean projection of the functional movie."""
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    A = scipy.sparse.csc_matrix(A)
    fig, ax = plt.subplots(figsize=(8 * dims[1] / max(dims), 8.6 * dims[0] / max(dims) + 0.6))
    lo, hi = np.percentile(mean_img, [1, 99.7])
    ax.imshow(mean_img, cmap="gray", vmin=lo, vmax=hi)
    for i in range(A.shape[1]):
        if not accepted[i] and not PLOT_REJECTED:
            continue
        y, x = footprint_mask(A, i, dims)
        if len(y) == 0:
            continue
        y0, x0 = y.min() - 1, x.min() - 1
        m = np.zeros((y.max() - y0 + 2, x.max() - x0 + 2))
        m[y - y0, x - x0] = 1
        ax.contour(np.arange(m.shape[1]) + x0, np.arange(m.shape[0]) + y0, m, levels=[0.5],
                   colors=["lime" if accepted[i] else "0.6"],
                   linewidths=0.6 if accepted[i] else 0.4,
                   linestyles="solid" if accepted[i] else "dotted")
        if PLOT_ROI_NUMBERS and accepted[i]:
            ax.text(x.mean(), y.mean(), str(i), color="yellow", fontsize=4,
                    ha="center", va="center")
    n_acc = int(np.sum(accepted))
    handles = [Line2D([], [], color="lime", label=f"accepted ({n_acc})")]
    if PLOT_REJECTED:
        handles.append(Line2D([], [], color="0.6", ls=":",
                              label=f"rejected ({A.shape[1] - n_acc})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title(f"{title}\nCaImAn components on the mean projection, functional", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    png = os.path.join(out_dir, "roi_contours.png")
    fig.savefig(png, dpi=200)
    print(f"  contour plot: {png}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def export_results(out_dir, est, dims, mean_img):
    """Write the estimates as .npy/.npz/.csv and one .mat; return A, accepted."""
    A = scipy.sparse.csc_matrix(est.A)
    n = A.shape[1]
    accepted = np.zeros(n, bool)
    if getattr(est, "idx_components", None) is not None:
        accepted[np.asarray(est.idx_components, int)] = True
    else:
        accepted[:] = True
    snr = np.asarray(getattr(est, "SNR_comp", None) if getattr(est, "SNR_comp", None) is not None
                     else np.full(n, np.nan), float)
    rval = np.asarray(getattr(est, "r_values", None) if getattr(est, "r_values", None) is not None
                      else np.full(n, np.nan), float)
    centers = component_centers(A, dims)
    arrays = {"C": est.C, "S": est.S, "YrA": est.YrA, "F_dff": getattr(est, "F_dff", None)}
    arrays = {k: np.asarray(v, np.float32) for k, v in arrays.items() if v is not None}

    scipy.sparse.save_npz(os.path.join(out_dir, "A.npz"), A)
    for k, v in arrays.items():
        np.save(os.path.join(out_dir, f"{k}.npy"), v)
    np.save(os.path.join(out_dir, "accepted.npy"), accepted)
    np.save(os.path.join(out_dir, "mean_image.npy"), mean_img)
    with open(os.path.join(out_dir, "stats.csv"), "w") as f:
        f.write("component,accepted,snr,r_value,center_row,center_col,npix\n")
        for i in range(n):
            f.write(f"{i},{int(accepted[i])},{snr[i]:.4f},{rval[i]:.4f},"
                    f"{centers[i, 0]:.2f},{centers[i, 1]:.2f},{A.indptr[i + 1] - A.indptr[i]}\n")
    try:
        savemat(os.path.join(out_dir, "caiman_results.mat"), {
            "A": A.astype(np.float64), **arrays, "accepted": accepted, "snr": snr,
            "r_values": rval, "centers": centers + 1, "dims": np.array(dims, float),
            "mean_image": mean_img, "fs": FS, "um_per_px": UM_PER_PX})
    except Exception as e:          # e.g. a variable over the 2 GB limit of -v5 files
        print(f"  ! caiman_results.mat not written ({type(e).__name__}: {e}); "
              f"use the .npy / .hdf5 files")
    return A, accepted


def summarize_outputs(out_dir, accepted):
    print(f"  components: {len(accepted)} found, {int(accepted.sum())} accepted "
          f"(SNR >= {MIN_SNR}, r >= {RVAL_THR}{', CNN' if USE_CNN else ''})")
    print(f"  output folder: {out_dir}")
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        print(f"    {name:<22s} {'<folder>' if os.path.isdir(path) else fmt_size(path)}")


def make_params(mmap_path):
    from caiman.source_extraction.cnmf import params
    return params.CNMFParams(params_dict={
        "fnames": [mmap_path], "fr": FS, "decay_time": DECAY_TIME,
        "dxy": (UM_PER_PX, UM_PER_PX),
        "p": P, "nb": GNB, "merge_thr": MERGE_THR,
        "rf": RF, "stride": STRIDE, "K": K, "gSig": [GSIG, GSIG],
        "method_init": "greedy_roi", "rolling_sum": True, "only_init": True,
        "ssub": SSUB, "tsub": TSUB, "border_pix": MARGIN,
        "min_SNR": MIN_SNR, "rval_thr": RVAL_THR, "use_cnn": USE_CNN,
    })


def run_session(sess_dir, label=""):
    """Run one session; everything printed also goes to caiman\\caiman_log.txt."""
    sess_dir = os.path.abspath(sess_dir)
    if not os.path.isdir(sess_dir):
        raise FileNotFoundError(f"session folder not found: {sess_dir}")
    out_dir = os.path.join(sess_dir, OUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "caiman_log.txt")
    with Tee(log_path):
        try:
            _run_session(sess_dir, out_dir, label)
        except Exception as e:      # log the failure in the session log too
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
        finally:
            print(f"  log: {log_path}")


def _run_session(sess_dir, out_dir, label):
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf

    t0 = time.time()
    animal, date, rec = describe_session(sess_dir)
    print(f"\n=== {label}animal {animal} | date {date} | recording {rec}")
    print(f"  folder:  {sess_dir}")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print_settings(indent="  ")
    print("  inputs:")
    for name in (MC_TIF, SHIFTS_MAT):
        print(f"    {name:<34s} {fmt_size(os.path.join(sess_dir, name))}")

    mc_path = os.path.join(sess_dir, MC_TIF)
    if not os.path.isfile(mc_path):
        raise FileNotFoundError(mc_path)
    n_frames, Ly, Lx = movie_shape(mc_path)
    dims = (Ly, Lx)
    print(f"  {MC_TIF}: {n_frames} frames, {Ly} x {Lx} px "
          f"({n_frames / FS / 60:.1f} min at {FS} Hz)")
    nbad = count_bad_frames(os.path.join(sess_dir, SHIFTS_MAT), n_frames)
    if nbad:
        print(f"  note: {nbad} frames ({100 * nbad / n_frames:.1f}%) were flagged 'bad' by "
              f"motion correction; CNMF uses them anyway")

    # float32 memmap on a local disk
    need = n_frames * Ly * Lx * 4
    mmap_dir = (os.path.join(MEMMAP_DIR, f"{animal}_{date}_{rec}") if MEMMAP_DIR
                else os.path.join(out_dir, "memmap"))
    os.makedirs(mmap_dir, exist_ok=True)
    drive = os.path.splitdrive(mmap_dir)[0] or mmap_dir
    free = shutil.disk_usage(mmap_dir).free
    print(f"  memmap (float32, {need / 1e9:.1f} GB): {mmap_dir}"
          f"{' (deleted after the run)' if DELETE_MEMMAP else ''}")
    print(f"  free space on {drive}: {free / 1e9:.1f} GB "
          f"({free / 1e9 - need / 1e9:.1f} GB left while it runs)")
    if free < need * 1.02:
        shutil.rmtree(mmap_dir, ignore_errors=True)
        raise OSError(f"not enough space on {drive} for the memmap: need "
                      f"{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free. "
                      f"Free up space or point MEMMAP_DIR at another drive.")

    dview = images = Yr = None
    try:
        mmap_path, mean_img = write_memmap(mc_path, mmap_dir, n_frames, Ly, Lx)
        print(f"  memmap written in {fmt_time(time.time() - t0)}")
        Yr, mm_dims, T = cm.load_memmap(mmap_path)
        images = np.reshape(Yr.T, [T] + list(mm_dims), order="F")
        check_memmap(images, mc_path, n_frames, Ly, Lx)

        _, dview, n_proc = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=N_PROCESSES, single_thread=False)
        print(f"  running CNMF on {n_proc} processes ...")
        t1 = time.time()
        cnm = cnmf.CNMF(n_proc, params=make_params(mmap_path), dview=dview)
        cnm = cnm.fit(images)
        print(f"  first pass: {cnm.estimates.A.shape[1]} components, "
              f"{fmt_time(time.time() - t1)}")
        if REFIT:
            t1 = time.time()
            cnm = cnm.refit(images, dview=dview)
            print(f"  refit: {cnm.estimates.A.shape[1]} components, "
                  f"{fmt_time(time.time() - t1)}")
        t1 = time.time()
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
        cnm.estimates.detrend_df_f(quantileMin=DFF_QUANTILE,
                                   frames_window=int(round(DFF_WINDOW_S * FS)))
        print(f"  component evaluation + dF/F: {fmt_time(time.time() - t1)}")
        cnm.save(os.path.join(out_dir, "caiman_results.hdf5"))
        est = cnm.estimates
    finally:                              # also on failure: stop workers, free disk
        if dview is not None:
            cm.stop_server(dview=dview)
        images = Yr = None                # release the memmap before deleting it
        if DELETE_MEMMAP:
            shutil.rmtree(mmap_dir, ignore_errors=True)
            if os.path.isdir(mmap_dir):   # Windows: a file still open cannot be deleted
                print(f"  ! could not delete the memmap ({need / 1e9:.1f} GB), "
                      f"delete it by hand: {mmap_dir}")
    print(f"  caiman done in {fmt_time(time.time() - t0)}")

    A, accepted = export_results(out_dir, est, dims, mean_img)
    if PLOT_CONTOURS:
        try:
            plot_contours(out_dir, A, dims, accepted, mean_img, f"{animal} | {date} | {rec}")
        except Exception as e:      # a failed plot should not fail the session
            print(f"  ! contour plot failed: {type(e).__name__}: {e}")
    summarize_outputs(out_dir, accepted)
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S} (total {fmt_time(time.time() - t0)})")


if __name__ == "__main__":      # required on Windows: CaImAn starts worker processes
    sessions = sys.argv[1:] or SESSIONS
    if not sessions:
        sys.exit("Give session folders as arguments or fill in SESSIONS.")
    try:
        import caiman  # noqa: F401
        from caiman.source_extraction.cnmf import cnmf  # noqa: F401
    except ImportError as e:
        sys.exit(f"CaImAn is not available in this Python ({sys.executable}): {e}\n"
                 f"Run this in the caiman conda environment (see the top of this file).")
    print_settings()
    failed = []
    t_all = time.time()
    for k, s in enumerate(sessions, 1):
        try:
            run_session(s, label=f"[{k}/{len(sessions)}] ")
        except Exception as e:     # report and continue with the next session
            print(f"  ! FAILED [{k}/{len(sessions)}] {s}: {type(e).__name__}: {e}")
            failed.append(s)
    print(f"\n{len(sessions) - len(failed)}/{len(sessions)} session(s) done "
          f"in {fmt_time(time.time() - t_all)}")
    for s in failed:
        print(f"  failed: {s}")
