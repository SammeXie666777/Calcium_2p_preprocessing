# manual_roi.py: treat a hand-drawn polygon like a Suite2p ROI.
#
# The polygon is turned into a Suite2p "stat" entry and then run through Suite2p's
# own functions, with the settings of the run that produced the session:
#   mask -> stat       suite2p.detection.anatomical.masks_to_stats
#   stat statistics    suite2p.detection.roi_stats, assign_overlaps
#   cell / neuropil    suite2p.extraction.masks.create_cell_pix / create_cell_mask /
#                      create_neuropil_masks
#   F, Fneu            suite2p.extraction.extract.extract_traces
#   spks               suite2p.extraction.preprocess / oasis
#   iscell             suite2p.classification.classify
#   redcell            the run's red rule (settings["red_method"]): "intensity" =
#                      inside / (inside + neuropil ring) on the red image, "blobs" =
#                      IoU with the saved blob masks (detect_outputs "chan2_masks");
#                      NaN when the run has no red image / masks
# Only the glue is ours. The GUI draws the polygon and calls manual_roi_traces.

import pickle
from pathlib import Path

import numpy as np
from skimage.draw import polygon

from . import features as ft
from . import io_utils as io

VERSION = 2     # manual ROI result format; older pickles (other scale) are ignored


def polygon_to_pixels(polygon_coords, image_shape):
    """Pixel indices (rr, cc) inside a polygon of (row, col) vertices.
    shape= clips vertices dragged past the image edge."""
    pts = np.asarray(polygon_coords, float)
    return polygon(pts[:, 0], pts[:, 1], shape=image_shape)


# ---------------------------------------------------------------- the movie
class TifMovie:
    """Lazy (frames, Ly, Lx) view of a tif read page by page, for tifs that cannot
    be memory-mapped. Supports .shape/.ndim and movie[start:stop]."""

    def __init__(self, path):
        import tifffile
        self._tif = tifffile.TiffFile(path)
        first = self._tif.pages[0].asarray()
        self.shape = (len(self._tif.pages),) + first.shape
        self.ndim, self.dtype = 3, first.dtype

    def __getitem__(self, sl):
        if not isinstance(sl, slice):
            raise TypeError("TifMovie only supports movie[start:stop]")
        return np.stack([self._tif.pages[i].asarray()
                         for i in range(*sl.indices(self.shape[0]))])


def open_movie(sess_dir, plane):
    """Movie as a (frames, Ly, Lx) read-only array, not loaded into RAM.
    Uses Suite2p's data.bin if it exists; otherwise ConcatenatedMovie_MC.tif (the
    same movie, since registration was off). Raises if the frame count does not
    match F.npy."""
    ops = plane["ops"]
    n_expected = plane["F"].shape[1]
    binf = Path(plane.get("dir", io.plane_dir(sess_dir))) / "data.bin"
    if binf.is_file():
        Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
        movie = np.memmap(binf, dtype="int16", mode="r",
                          shape=(binf.stat().st_size // (2 * Ly * Lx), Ly, Lx))
        src = "data.bin"
    else:
        import tifffile
        tif = io.image_dir(sess_dir) / io.MC_TIF
        if not tif.is_file():
            raise FileNotFoundError(f"no data.bin and no {io.MC_TIF} in {sess_dir}")
        try:
            movie = tifffile.memmap(tif)
        except ValueError:                        # "image data are not memory-mappable"
            movie = TifMovie(tif)
        src = io.MC_TIF
        print(f"  (no data.bin; manual ROI reads {src})")
    if movie.ndim != 3 or movie.shape[0] != n_expected:
        raise ValueError(f"{src} has shape {movie.shape} but Suite2p traces have "
                         f"{n_expected} frames: movie and F.npy are not the same frames")
    return movie


class WindowCache:
    """Keeps the pixels of one image window (all frames) in RAM, as int16 exactly as
    Suite2p's binary holds them (uint16 tif frames are floor-divided by 2, as in
    suite2p/io/tiff.py). The first ROI in a region reads the movie once; later ROIs
    inside the window read RAM only. ~ frames x h x w x 2 bytes."""
    MIN_SIZE = 128      # window is at least this big, so neighbouring ROIs fit
    PAD = 16            # extra border around the pixels first needed

    def __init__(self):
        self.win, self.data, self.key = None, None, None

    def covers(self, key, bbox):
        if self.win is None or self.key != key:
            return False
        y0, y1, x0, x1 = self.win
        return bbox[0] >= y0 and bbox[1] <= y1 and bbox[2] >= x0 and bbox[3] <= x1

    def load(self, movie, key, bbox, chunk=1000):
        Ly, Lx = movie.shape[1:]
        wins = []
        for lo, hi, size in ((bbox[0], bbox[1], Ly), (bbox[2], bbox[3], Lx)):
            lo, hi = max(0, lo - self.PAD), min(size, hi + self.PAD)
            grow = max(0, self.MIN_SIZE - (hi - lo))
            lo, hi = max(0, lo - grow // 2), min(size, hi + grow - grow // 2)
            wins += [lo, hi]
        y0, y1, x0, x1 = wins
        self.data = None                                      # free the old window first
        out = np.empty((movie.shape[0], y1 - y0, x1 - x0), np.int16)
        for t0 in range(0, movie.shape[0], chunk):
            blk = np.asarray(movie[t0:t0 + chunk])[:, y0:y1, x0:x1]
            if blk.dtype == np.uint16:
                blk = blk // 2                                # Suite2p's tif -> int16 rule
            out[t0:t0 + chunk] = blk.astype(np.int16)
        self.win, self.data, self.key = (y0, y1, x0, x1), out, key


# ---------------------------------------------------------------- settings
def run_settings(plane):
    """Settings of the Suite2p run (settings.npy in plane0); defaults if missing."""
    s = plane.get("settings")
    if not s:
        from suite2p import default_settings
        s = default_settings()
        print("  ! no settings.npy for this session: using Suite2p defaults")
    return s


def classifier_file(settings):
    """Classifier file the run used (same rule as suite2p/pipeline_s2p.py)."""
    import os
    from suite2p import classification
    cf = settings["classification"].get("classifier_path", None)
    if cf is not None and os.path.exists(cf):
        return cf
    if (settings["classification"]["use_builtin_classifier"]
            or not classification.user_classfile.is_file()):
        return classification.builtin_classfile
    return classification.user_classfile


# ---------------------------------------------------------------- polygon -> stat
def build_stat(poly, plane, stat_all, settings):
    """Stat entry for a polygon, the way Suite2p builds one from an anatomical mask:
    masks_to_stats (pixels, lam, med) -> roi_stats (compact, radius, npix_soma, ...)
    -> npix_norm relative to the session's ROIs -> overlap flags.

    lam uses Suite2p's anatomical weight rule (the image selected in the run's
    cellpose settings; the mean-image rule if the run used Sparsery / Sourcery).
    Only Cellpose runs make this identical to Suite2p; for the other methods lam is
    learned from activity during detection and cannot be recomputed for a mask."""
    from suite2p.detection import anatomical, assign_overlaps, roi_stats
    ops, det = plane["ops"], settings["detection"]
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    mean_img = np.asarray(ops.get("meanImg_crop", ops.get("meanImg")))
    max_proj = np.asarray(ops.get("max_proj", mean_img))
    yr = ops.get("yrange", [0, Ly])
    xr = ops.get("xrange", [0, Lx])
    algo = det.get("algorithm", "sparsery")
    img = det["cellpose_settings"]["img"] if algo == "cellpose" else "meanImg"
    if img == "meanImg":       # same formulas as anatomical.select_rois
        lo, hi = np.percentile(mean_img, 1), np.percentile(mean_img, 99)
        weights = 0.1 + np.clip((mean_img - lo) / (hi - lo), 0, 1)
    else:
        weights = max_proj

    rr, cc = polygon_to_pixels(np.asarray(poly) - [yr[0], xr[0]], mean_img.shape)
    if len(rr) < 3:
        raise ValueError("polygon is too small / outside the image")
    label = np.zeros(mean_img.shape, np.int32)
    label[rr, cc] = 1
    st = anatomical.masks_to_stats(label, weights)[0]
    st["ypix"], st["xpix"] = st["ypix"] + int(yr[0]), st["xpix"] + int(xr[0])
    st["med"] = [st["med"][0] + int(yr[0]), st["med"][1] + int(xr[0])]

    use_median = algo == "cellpose"
    st = roi_stats(np.array([st], dtype=object), Ly, Lx, diameter=settings["diameter"],
                   max_overlap=None, do_soma_crop=det.get("soma_crop", 1),
                   npix_norm_min=-1, npix_norm_max=np.inf, median=use_median)[0]
    # npix_norm is relative to the session's ROIs (roi_stats saw only this one)
    soma = np.array([s.get("npix_soma", s["npix"]) for s in stat_all], float)
    npix = np.array([s["npix"] for s in stat_all], float)
    pick = np.median if use_median else (lambda a: np.median(a[:100]))
    st["npix_norm"] = st["npix_soma"] / (pick(soma) + 1e-10)
    st["npix_norm_no_crop"] = st["npix"] / (pick(npix) + 1e-10)
    # overlap flags against the session's ROIs (on copies: stat_all is not changed)
    copies = [dict(s) for s in stat_all] + [st]
    st["overlap"] = assign_overlaps(copies, Ly, Lx)[-1]["overlap"]
    return st


# ---------------------------------------------------------------- masks + traces
def roi_masks(stats_incl, st, ex, Ly, Lx):
    """Cell mask (indices, weights) and neuropil indices of st, from Suite2p's own
    mask functions. stats_incl is every ROI of the session including st (as in
    create_masks, where the neuropil excludes all cells). Falls back to
    allow_overlap=True when every pixel of st overlaps another ROI (an empty cell
    mask would give F = 0)."""
    from suite2p.extraction import masks as s2m
    cell_pix = s2m.create_cell_pix(stats_incl, Ly, Lx, lam_percentile=ex["lam_percentile"])
    allow = bool(ex["allow_overlap"])
    cell_idx, cell_w = s2m.create_cell_mask(st, Ly, Lx, allow_overlap=allow)
    if len(cell_idx) == 0 and not allow:
        cell_idx, cell_w = s2m.create_cell_mask(st, Ly, Lx, allow_overlap=True)
        allow = True
    neu = None
    if ex["neuropil_extract"]:
        neu = s2m.create_neuropil_masks(
            ypixs=[st["ypix"]], xpixs=[st["xpix"]], cell_pix=cell_pix,
            inner_neuropil_radius=ex["inner_neuropil_radius"],
            min_neuropil_pixels=ex["min_neuropil_pixels"],
            circular=ex["circular_neuropil"])[0]
    return cell_idx, cell_w, neu, allow, cell_pix


def extract_roi(movie, cache, cell_idx, cell_w, neu_idx, ex, device):
    """F and Fneu of one ROI with Suite2p's extract_traces, on the cached window
    (indices are remapped from the full frame to the window)."""
    from suite2p.extraction.extract import extract_traces
    shape = movie.shape[1:]
    parts = [np.asarray(cell_idx, np.int64)]
    if neu_idx is not None:
        parts.append(np.asarray(neu_idx, np.int64))
    ys, xs = np.unravel_index(np.concatenate(parts), shape)
    bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    key = id(movie)
    hit = cache.covers(key, bbox)
    if not hit:
        cache.load(movie, key, bbox)
    y0, y1, x0, x1 = cache.win
    w = x1 - x0

    def to_win(idx):
        idx = np.asarray(idx, np.int64)
        return ((idx // shape[1]) - y0) * w + ((idx % shape[1]) - x0)
    F, Fneu = extract_traces(
        cache.data, [(to_win(cell_idx), np.asarray(cell_w))],
        None if neu_idx is None else [to_win(neu_idx)],
        batch_size=ex["batch_size"], device=device)
    return F[0], Fneu[0], hit


def manual_roi_traces(movie, poly, plane, cfg, cache=None, device=None):
    """Everything Suite2p would have produced for a hand-drawn polygon.

    movie: (frames, Ly, Lx); poly: (N, 2) (row, col) vertices; plane: the
    io.load_plane dict, with plane["stat"] = Suite2p stat plus earlier manual ROIs.
    Returns a dict: stat, F, Fneu, spks, iscell, redcell, and the celltype dF/F."""
    import torch
    from scipy.stats import skew
    from suite2p import classification
    from suite2p.extraction import oasis, preprocess

    device = device or torch.device("cpu")
    cache = cache if cache is not None else WindowCache()
    settings = run_settings(plane)
    ex = settings["extraction"]
    Ly, Lx = movie.shape[1:]
    stat_all = plane["stat"]

    st = build_stat(poly, plane, stat_all, settings)
    stats_incl = list(stat_all) + [st]
    cell_idx, cell_w, neu_idx, allow, cell_pix = roi_masks(stats_incl, st, ex, Ly, Lx)
    F, Fneu, hit = extract_roi(movie, cache, cell_idx, cell_w, neu_idx, ex, device)

    # activity statistics used by the classifier (as in pipeline_s2p.py)
    dF = (F - ex["neuropil_coefficient"] * Fneu)[None]
    st["snr"] = float(1 - 0.5 * np.diff(dF, axis=1).var(axis=1)[0] / dF.var(axis=1)[0])
    st["skew"], st["std"] = float(skew(dF, axis=1)[0]), float(np.std(dF, axis=1)[0])

    # spikes: the same two calls as the pipeline
    bs = ex["batch_size"]
    dFp = preprocess(F=dF.copy(), fs=settings["fs"], batch_size=bs, device=device,
                     **settings["dcnv_preprocess"])
    spks = oasis(F=dFp, batch_size=bs, tau=settings["tau"], fs=settings["fs"])[0]

    iscell = classification.classify(stat=np.array([st], dtype=object),
                                     classfile=classifier_file(settings))[0]

    # red label by the rule the run used (same formulas as run_suite2p_on_MC.py)
    redcell = np.array([np.nan, np.nan])
    method = settings.get("red_method", "intensity")
    thr = settings["detection"].get("chan2_threshold", 0.65)
    ops = plane["ops"]
    if method == "blobs":
        masks = ops.get("chan2_masks")
        if masks is not None and np.shape(masks) == (Ly, Lx):
            masks = np.asarray(masks, np.int64)
            n = int(masks.max())
            iou = 0.0
            if n:
                shared = np.bincount(masks[st["ypix"], st["xpix"]], minlength=n + 1)[1:]
                sizes = np.bincount(masks.ravel(), minlength=n + 1)[1:]
                iou = float((shared / (len(st["ypix"]) + sizes - shared)).max())
            redcell = np.array([float(iou > thr), iou])
    elif method == "intensity":
        red = ops.get("meanImg_chan2")
        if red is not None and np.shape(red) == (Ly, Lx):
            from suite2p.extraction import masks as s2m
            cp = cell_pix if ex["lam_percentile"] == 50.0 else s2m.create_cell_pix(
                stats_incl, Ly, Lx)
            ci, cw = s2m.create_cell_mask(st, Ly, Lx, allow_overlap=True)
            ring = s2m.create_neuropil_masks(
                [st["ypix"]], [st["xpix"]], cp, ex["inner_neuropil_radius"],
                ex["min_neuropil_pixels"])[0]
            flat = np.asarray(red, np.float64).ravel()
            inside = max(1e-3, float(np.asarray(cw) @ flat[ci]))
            prob = inside / (inside + float(flat[ring].mean()))
            redcell = np.array([float(prob > thr), prob])

    dff, _ = ft.compute_dff({"F": F[None], "Fneu": Fneu[None]}, cfg)
    return {"version": VERSION, "poly": np.asarray(poly), "stat": st,
            "ypix": st["ypix"], "xpix": st["xpix"],
            "n_pix": len(cell_idx), "n_neuropil": 0 if neu_idx is None else len(neu_idx),
            "allow_overlap_used": allow, "cache_hit": hit,
            "algorithm": settings["detection"].get("algorithm", "sparsery"),
            "F": F, "Fneu": Fneu, "spks": spks, "dff": dff[0],
            "iscell": np.asarray(iscell, float), "redcell": redcell}


def verify_against_suite2p(movie, plane, i, cache=None, device=None):
    """Test: push detected ROI i's own stat entry through the same mask + extraction
    code and compare with the saved F.npy / Fneu.npy. Returns max abs difference
    relative to the trace range; ~0 means the movie and the code reproduce Suite2p."""
    import torch
    device = device or torch.device("cpu")
    settings = run_settings(plane)
    ex = settings["extraction"]
    Ly, Lx = movie.shape[1:]
    stat = plane["stat"]
    cell_idx, cell_w, neu_idx, _, _ = roi_masks(list(stat), stat[i], ex, Ly, Lx)
    F, Fneu, _ = extract_roi(movie, cache if cache is not None else WindowCache(),
                             cell_idx, cell_w, neu_idx, ex, device)
    return {n: float(np.abs(m - r).max() / (np.ptp(r) + 1e-9))
            for n, m, r in (("F", F, plane["F"][i]), ("Fneu", Fneu, plane["Fneu"][i]))}


# ---------------------------------------------------------------- persistence
def load_manual_rois(path, s2p_hash):
    """Manual ROI results saved for this Suite2p run ([] if none / other run).
    Results from before the Suite2p-function rewrite (no version) are skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        old = pickle.load(f)
    if old.get("s2p_hash") != s2p_hash:
        return []
    rois = [r for r in old["rois"] if r.get("version") == VERSION]
    if len(rois) < len(old["rois"]):
        print(f"  ! {len(old['rois']) - len(rois)} manual ROI(s) from an older version "
              f"skipped in {path.name}; redraw them")
    return rois


def save_manual_roi(path, res, s2p_hash):
    """Append one result to the pickle at path (earlier ROIs of this Suite2p run
    are kept). Returns the number of ROIs stored."""
    path = Path(path)
    rois = []
    if path.is_file():
        with open(path, "rb") as f:
            old = pickle.load(f)
        if old.get("s2p_hash") == s2p_hash:
            rois = old["rois"]
    rois.append(res)
    with open(path, "wb") as f:
        pickle.dump({"s2p_hash": s2p_hash, "rois": rois}, f)
    return len(rois)
