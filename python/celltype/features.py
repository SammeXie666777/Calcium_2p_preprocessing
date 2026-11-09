"""Per-ROI features from Suite2p output + the green / red mean images.

Three groups, used by different stages:
    MORPH, ACTIVITY  -> stage 1 (SPN vs non-SPN)
    RED              -> stage 2 (red+ vs red-); never used by stage 1, and stage 2
                        never sees activity, so later dSPN/iSPN activity
                        comparisons are not biased by how cells were labeled.
Features are made FOV-relative where brightness matters (*_rel), so models can
be trained across sessions without per-session z-scoring.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import skew
from skimage.measure import regionprops

from . import io_utils as io

# image features of the green MEAN and MAX projections (max: also cells that are dim and only rarely come on).
# Computed and kept in the table (INFO), but NOT model inputs: the event snapshot below does their job better.
GREEN_MAX = ["green_z", "green_max_z", "green_max_rel", "green_max_blob", "max_mean_logratio"]
# event snapshot: the movie around the ROI's own strongest events minus quiet frames (see SnapContext), in noise units
SNAP = ["snap_peak_z", "snap_mean_z", "snap_inside_frac", "snap_corr_mask", "snap_offset", "snap_spread"]
MORPH = ["equiv_diam_um", "area_um2", "eccentricity", "solidity", "aspect",
         "circularity", "nucleus_shadow", "green_rel", "s2p_compact", "green_blob"] + SNAP
ACTIVITY = ["skew", "baseline_rel", "baseline_over_neuropil", "noise_sd",
            "event_rate", "event_amp_med", "event_amp_cv", "frac_active",
            "acf_1s", "acf_5s", "snr", "neuropil_corr", "common_mode_corr", "rise_decay_asym"]
RED = ["red_ring_ratio", "red_z", "red_fill", "red_corr", "red_mean_rel",
       "z_core", "z_coreB", "red_blob"]
# kept in the table for display, the CV report and the edge rule, but NOT model inputs:
# s2p_redprob = Suite2p's own red-cell probability; *_oob_frac = share of the ROI's pixels outside
# the valid area (padding) of the red / green image
INFO = ["s2p_redprob", "red_oob_frac", "green_oob_frac"] + GREEN_MAX
META = ["session_id", "session_dir", "animal", "roi_idx", "s2p_hash", "iscell",
        "s2p_iscell_prob", "cy", "cx"]


# ---------------------------------------------------------------- traces
def rolling_baseline(X, fs, window_s, pct):
    """Rolling percentile per row, computed on 1-s bins then interpolated."""
    n, T = X.shape
    b = max(1, int(round(fs)))
    nb = T // b
    if nb < 3:
        return np.repeat(np.percentile(X, pct, axis=1)[:, None], T, axis=1)
    binned = X[:, :nb * b].reshape(n, nb, b).mean(axis=2)
    size = max(1, min(nb, int(round(window_s))))
    base = ndimage.percentile_filter(binned, pct, size=(1, size), mode="nearest")
    centers = np.arange(nb) * b + (b - 1) / 2
    t = np.arange(T)
    return np.vstack([np.interp(t, centers, row) for row in base]).astype(np.float32)


def compute_dff(plane, cfg):
    """dF/F of neuropil-corrected F. Denominator floored at 10% of the raw-F
    baseline so cells whose corrected baseline is ~0 do not blow up."""
    fs = cfg["suite2p"]["fs"]
    F, Fneu = plane["F"], plane["Fneu"]
    Fc = F - cfg["neuropil_coef"] * Fneu
    F0c = rolling_baseline(Fc, fs, cfg["baseline_window_s"], cfg["baseline_percentile"])
    F0 = rolling_baseline(F, fs, cfg["baseline_window_s"], cfg["baseline_percentile"])
    denom = np.maximum(F0c, 0.1 * np.abs(F0) + 1e-6)
    return (Fc - F0c) / denom, F0


def noise_sd(dff):
    """Robust noise SD from frame-to-frame differences (insensitive to transients)."""
    d = np.diff(dff, axis=1)
    return np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1) / 0.6745 / np.sqrt(2)


def _acf(x, lag):
    if lag >= x.shape[1]:
        return np.full(x.shape[0], np.nan)
    x = x - x.mean(axis=1, keepdims=True)
    den = (x * x).sum(axis=1) + 1e-12
    return (x[:, :-lag] * x[:, lag:]).sum(axis=1) / den


def _events(row, z_row, thr, min_len):
    above = np.concatenate([[False], z_row > thr, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts, stops = edges[::2], edges[1::2]
    keep = (stops - starts) >= min_len
    return [row[a:b].max() for a, b in zip(starts[keep], stops[keep])]


def trace_corr_features(dff, F, Fneu, fs, iscell, chunk=256, n_ref=300, seed=0):
    """Three trace features (n ROIs x T frames in), computed in blocks of ROIs to keep memory low:
      common_mode_corr  correlation (after removing drift slower than 2 s) with the median trace of the
                        cells: shared brain-wide activity or motion
      neuropil_corr     the same with the ROI's own neuropil trace: how much of the ROI is its
                        surround (F against Fneu)
      rise_decay_asym   skewness of the frame-to-frame changes of the lightly smoothed (0.2 s) dF/F:
                        calcium transients rise fast and decay slowly (positive), noise is symmetric (~0)"""
    n = dff.shape[0]
    sm, k = max(3, int(round(2 * fs))), max(3, int(round(0.2 * fs)))
    hp = lambda x: x - ndimage.uniform_filter1d(x, size=sm, axis=1, mode="nearest")
    zs = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    out = {c: np.full(n, np.nan) for c in ("common_mode_corr", "neuropil_corr", "rise_decay_asym")}
    cells = np.flatnonzero(np.asarray(iscell) > 0)
    ref = None
    if len(cells):                                          # the reference from (at most) n_ref cells
        pick = cells if len(cells) <= n_ref else np.random.default_rng(seed).choice(cells, n_ref, replace=False)
        ref = zs(np.median(hp(dff[pick].astype(np.float32)), axis=0)[None])[0]
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        d = dff[sl].astype(np.float32)
        if ref is not None:
            out["common_mode_corr"][sl] = (zs(hp(d)) * ref).mean(1)
        out["neuropil_corr"][sl] = (zs(hp(F[sl].astype(np.float32))) * zs(hp(Fneu[sl].astype(np.float32)))).mean(1)
        out["rise_decay_asym"][sl] = skew(np.diff(ndimage.uniform_filter1d(d, size=k, axis=1), axis=1), axis=1)
    return out


def activity_features(plane, cfg, bad, dff_F0=None):
    """The activity features of every ROI. dff_F0: the (dff, F0) of compute_dff if the caller has them already."""
    fs = cfg["suite2p"]["fs"]
    dff, F0 = compute_dff(plane, cfg) if dff_F0 is None else dff_F0
    good = ~bad
    dff = dff[:, good]
    F, Fneu = plane["F"][:, good], plane["Fneu"][:, good]
    T = dff.shape[1]
    sd = noise_sd(dff) + 1e-9
    z = dff / sd[:, None]
    min_len = max(1, int(round(cfg["event_min_dur_s"] * fs)))
    rate, amp_med, amp_cv = [], [], []
    for i in range(dff.shape[0]):
        amps = _events(dff[i], z[i], cfg["event_z"], min_len)
        rate.append(len(amps) / (T / fs / 60.0))
        amp_med.append(np.median(amps) if amps else 0.0)
        amp_cv.append(np.std(amps) / np.mean(amps) if len(amps) > 1 and np.mean(amps) > 0 else np.nan)
    f0_med = np.median(F0[:, good], axis=1)
    tc = trace_corr_features(dff, F, Fneu, fs, plane["iscell"][:, 0])
    return pd.DataFrame({
        "skew": skew(dff, axis=1),
        "baseline_rel": f0_med / (np.median(f0_med) + 1e-9),
        "baseline_over_neuropil": np.median(F, axis=1) / (np.median(Fneu, axis=1) + 1e-9),
        "noise_sd": sd,
        "event_rate": rate,
        "event_amp_med": amp_med,
        "event_amp_cv": amp_cv,
        "frac_active": (z > 2).mean(axis=1),
        "acf_1s": _acf(dff, int(round(fs))),
        "acf_5s": _acf(dff, int(round(5 * fs))),
        "snr": np.percentile(dff, 95, axis=1) / sd,
        **tc,
    })


# ---------------------------------------------------------------- images
def _crop(s, shape, pad):
    y0 = max(0, s["ypix"].min() - pad); y1 = min(shape[0], s["ypix"].max() + pad + 1)
    x0 = max(0, s["xpix"].min() - pad); x1 = min(shape[1], s["xpix"].max() + pad + 1)
    return y0, y1, x0, x1


def morph_features(stat, green, um_per_px):
    g = green - np.percentile(green, 1)
    rows = []
    for s in stat:
        y0, y1, x0, x1 = _crop(s, g.shape, 2)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[s["ypix"] - y0, s["xpix"] - x0] = True
        rp = regionprops(m.astype(np.uint8))[0]
        area = float(m.sum())
        maj, mnr = rp.axis_major_length, max(rp.axis_minor_length, 1e-6)
        perim = max(rp.perimeter, 1e-6)
        # dark-nucleus score: GCaMP is cytosolic, so SPN somata have a dim centre
        cy, cx = rp.centroid
        yy, xx = np.indices(m.shape)
        r = np.hypot(yy - cy, xx - cx) / np.sqrt(area / np.pi)
        gc = g[y0:y1, x0:x1]
        inner, outer = m & (r < 0.4), m & (r > 0.6)
        shadow = (gc[outer].mean() / (gc[inner].mean() + 1e-6)
                  if inner.sum() >= 2 and outer.sum() >= 2 else np.nan)
        rows.append({
            "area_um2": area * um_per_px ** 2,
            "equiv_diam_um": 2 * np.sqrt(area / np.pi) * um_per_px,
            "eccentricity": rp.eccentricity,
            "solidity": rp.solidity,
            "aspect": maj / mnr,
            "circularity": min(4 * np.pi * area / perim ** 2, 1.0),
            "nucleus_shadow": shadow,
            "green_mean": gc[m].mean(),
            "s2p_compact": s.get("compact", np.nan),
            "cy": cy + y0, "cx": cx + x0,
        })
    df = pd.DataFrame(rows)
    df["green_rel"] = df["green_mean"] / (df["green_mean"].median() + 1e-9)
    return df.drop(columns="green_mean")


def red_image(green, red, bleedthrough):
    """Red image with an optional green bleed-through correction, floored at 0."""
    r = red.astype(np.float32).copy()
    if bleedthrough:
        sel = r < np.percentile(r, 75)          # fit on dim pixels (not red somata)
        A = np.vstack([green[sel], np.ones(sel.sum())]).T
        alpha = max(0.0, np.linalg.lstsq(A, r[sel], rcond=None)[0][0])
        r = r - alpha * green
    return r - np.percentile(r, 1)


def blob_map(red, diameter):
    """Soma-shaped bright spots of the red image, for looking at (not a model input): a
    scale-normalised Laplacian of Gaussian matched to a soma of `diameter` pixels, in noise units
    (median and MAD over the tissue only, so an empty dark corner does not set the scale).
    A red soma is a peak (about > 3), flat or noisy areas are about 0, dark gaps are negative."""
    from skimage.filters import threshold_otsu
    img = np.asarray(red, np.float32)
    sigma = max(1.0, diameter / (2 * np.sqrt(2)))
    L = -sigma ** 2 * ndimage.gaussian_laplace(img, sigma)
    sm = ndimage.gaussian_filter(img, diameter)
    try:
        tissue = sm > threshold_otsu(sm)
        if tissue.sum() < 100:
            tissue = np.ones(img.shape, bool)
    except ValueError:                                  # a flat image
        tissue = np.ones(img.shape, bool)
    med = np.median(L[tissue])
    return ((L - med) / (1.4826 * np.median(np.abs(L[tissue] - med)) + 1e-6)).astype(np.float32)


def _ring_z(stat, index_img, img, diameter):
    """Per ROI: how far its mean in `img` is above the ring around it, in noise units (the red_z recipe)."""
    outer_r = max(6, int(round(0.6 * diameter)))
    union = ndimage.binary_dilation(index_img >= 0, iterations=1)
    out = np.full(len(stat), np.nan)
    for i, s in enumerate(stat):
        y0, y1, x0, x1 = _crop(s, img.shape, outer_r + 2)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[s["ypix"] - y0, s["xpix"] - x0] = True
        ring = ndimage.binary_dilation(m, iterations=outer_r) & ~ndimage.binary_dilation(m, iterations=2)
        clean = ring & ~union[y0:y1, x0:x1]
        if clean.sum() >= 10:
            ring = clean
        if ring.sum() < 3:
            continue
        rc = img[y0:y1, x0:x1]
        bg = rc[ring]
        med = np.median(bg)
        out[i] = (rc[m].mean() - med) / (1.4826 * np.median(np.abs(bg - med)) + 1e-6)
    return out


def green_max_features(stat, index_img, green, max_img, diameter):
    """Image features of the green MEAN and MAX projections, for SPN vs not. The mean shows what is bright
    on average; the max also shows cells that are dim and only rarely come on.
      green_z            ROI mean above its surround ring in the mean image (noise units)
      green_max_z        the same in the max image
      green_max_rel      ROI mean of the max image / the median of that over the ROIs (brightness in the max image
                         relative to the field)
      green_max_blob     strongest soma-shaped spot of the max image near the ROI centre (blob_peaks)
      max_mean_logratio  log of (ROI mean of the max image / ROI mean of the mean image), each minus its 1st
                         percentile: high for a cell that is dim on average but lights up now and then
    Without a max image the max features are NaN (the models fill them with the median)."""
    n = len(stat)
    out = {"green_z": _ring_z(stat, index_img, green, diameter)}
    nan = np.full(n, np.nan)
    if max_img is None:
        out.update(green_max_z=nan, green_max_rel=nan, green_max_blob=nan, max_mean_logratio=nan)
        return out
    mx, mn = max_img - np.percentile(max_img, 1), green - np.percentile(green, 1)
    roi_max = np.array([mx[s["ypix"], s["xpix"]].mean() for s in stat])
    roi_mean = np.array([mn[s["ypix"], s["xpix"]].mean() for s in stat])
    floor = 0.05 * (np.median(roi_mean) + 1e-9)
    out["green_max_z"] = _ring_z(stat, index_img, max_img, diameter)
    out["green_max_rel"] = roi_max / (np.median(roi_max) + 1e-9)
    out["green_max_blob"] = blob_peaks(max_img, stat, diameter)
    out["max_mean_logratio"] = np.log(np.maximum(roi_max, floor) / np.maximum(roi_mean, floor))
    return out


def blob_peaks(img, stat, diameter):
    """Per ROI: the strongest soma-shaped bright spot of `img` within half a diameter of the ROI centre
    (blob_map, in noise units); NaN everywhere if there is no image."""
    if img is None:
        return np.full(len(stat), np.nan)
    bm = blob_map(img, diameter)
    rad = max(2, int(round(0.5 * diameter)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    mx = ndimage.maximum_filter(bm, footprint=(yy ** 2 + xx ** 2 <= rad ** 2), mode="nearest")
    cy = np.array([s["ypix"].mean() for s in stat]).round().astype(int).clip(0, bm.shape[0] - 1)
    cx = np.array([s["xpix"].mean() for s in stat]).round().astype(int).clip(0, bm.shape[1] - 1)
    return mx[cy, cx]


def roi_oob_fraction(stat, valid, margin=0):
    """Per ROI: the share of its pixels that lie outside the image or in the padding of `valid` (the
    mask from io.load_images_valid), counting `margin` pixels around the ROI as well. NaN everywhere
    when valid is None (no such image)."""
    if valid is None:
        return np.full(len(stat), np.nan)
    v = ndimage.binary_erosion(valid, iterations=int(margin), border_value=1) if margin > 0 else valid
    Ly, Lx = v.shape
    out = np.empty(len(stat))
    for i, s in enumerate(stat):
        y, x = np.asarray(s["ypix"]), np.asarray(s["xpix"])
        inb = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
        ok = np.zeros(len(y), bool)
        ok[inb] = v[y[inb], x[inb]]
        out[i] = 1.0 - ok.mean() if len(y) else np.nan
    return out


def red_features(stat, index_img, red, diameter):
    nan = {k: np.nan for k in RED}
    if red is None:
        return pd.DataFrame([nan] * len(stat))
    outer_r = max(6, int(round(0.6 * diameter)))
    inner_r = 2
    union = ndimage.binary_dilation(index_img >= 0, iterations=1)
    rows = []
    for s in stat:
        y0, y1, x0, x1 = _crop(s, red.shape, outer_r + 2)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[s["ypix"] - y0, s["xpix"] - x0] = True
        lam = np.zeros(m.shape, np.float32)
        lam[s["ypix"] - y0, s["xpix"] - x0] = s.get("lam", np.ones(len(s["ypix"])))
        rc = red[y0:y1, x0:x1]
        # centre of the filled mask: a disc of half the radius, so a bright neighbour that only
        # overlaps the edge of the mask does not count (the dark nucleus is filled in)
        mf = ndimage.binary_fill_holes(m)
        cy, cx = ndimage.center_of_mass(mf)
        yy, xx = np.indices(m.shape)
        core = (yy - cy) ** 2 + (xx - cx) ** 2 <= max(1.5, 0.5 * np.sqrt(mf.sum() / np.pi)) ** 2
        big = ndimage.binary_dilation(m, iterations=outer_r)
        ring = big & ~ndimage.binary_dilation(m, iterations=inner_r)
        ring_clean = ring & ~union[y0:y1, x0:x1]
        if ring_clean.sum() >= 10:              # exclude other ROIs if possible
            ring = ring_clean
        if ring.sum() < 3:
            rows.append(nan); continue
        bg = rc[ring]
        med = np.median(bg)
        sd = 1.4826 * np.median(np.abs(bg - med)) + 1e-6
        vals = rc[m]
        # second ring, further out (5 px gap) to stay clear of the cell's own halo
        ringB = ndimage.binary_dilation(mf, iterations=outer_r + 4) & ~ndimage.binary_dilation(mf, iterations=5)
        cleanB = ringB & ~union[y0:y1, x0:x1]
        bgB = rc[cleanB if cleanB.sum() >= 10 else ringB]
        medB = np.median(bgB)
        sdB = 1.4826 * np.median(np.abs(bgB - medB)) + 1e-6
        lv, rv = lam[big], rc[big]
        corr = np.corrcoef(lv, rv)[0, 1] if lv.std() > 0 and rv.std() > 0 else np.nan
        rows.append({
            "red_ring_ratio": vals.mean() / (bg.mean() + 1e-6),
            "red_z": (vals.mean() - med) / sd,
            "red_fill": float((vals > med + 2 * sd).mean()),
            "red_corr": corr,
            "red_mean": vals.mean(),
            "z_core": (rc[core].mean() - med) / sd,
            "z_coreB": (rc[core].mean() - medB) / sdB,
        })
    df = pd.DataFrame(rows)
    df["red_mean_rel"] = df["red_mean"] / (df["red_mean"].median() + 1e-9)
    return df.drop(columns="red_mean")


# ---------------------------------------------------------------- per session
def apply_iscell_threshold(df, cfg):
    """cfg['iscell_prob_min'] = null: keep Suite2p's own binary iscell. A number: a ROI is a cell
    when its Suite2p iscell probability is >= that number. Done after loading, so changing the
    number needs no recompute of the cached feature table. Every later step (which ROIs the
    models train on and label, 'not_cell') reads this column."""
    thr = cfg.get("iscell_prob_min")
    if thr is not None:
        df = df.copy()
        df["iscell"] = (df["s2p_iscell_prob"].to_numpy(float) >= float(thr)).astype(int)
    return df


def session_features(sess_dir, cfg, force=False, model_id=None):
    """The feature table of a session (cached in <session>\\celltype\\features.csv).
    model_id given (training / prediction): the cache must have been made FOR THAT MODEL (same id, same feature
    settings), else the features are computed again and stamped with it. Not given (GUI, notebooks, --mode features):
    any cache made with the current feature settings is used."""
    df = _session_features(sess_dir, cfg, force, model_id)
    return apply_iscell_threshold(df, cfg)


def stamp_features(sess_dir, cfg, model_id):
    """Mark the cached features of a session as made for model `model_id` (right after that model was trained
    on exactly these features)."""
    vfile = io.out_dir(sess_dir) / "features.version"
    if (io.out_dir(sess_dir) / "features.csv").is_file():
        vfile.write_text(_cache_tag(cfg, model_id))


# ---------------------------------------------------------------- event snapshots (stage 1)
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


class SnapContext:
    """One session's data for the event snapshots (dF/F, bad frames, noise, the registered movie), opened once.
    Reuses dff / bad / movie when given; the movie is data.bin (fast) or the registered tif."""

    def __init__(self, plane, cfg, sdir, dff=None, bad=None, movie=None, size=32, k_events=6, min_events=3):
        from . import manual_roi as mr            # imported here: manual_roi imports this module
        self.plane, self.stat, self.cfg = plane, plane["stat"], cfg
        self.size, self.k, self.min_events = size, k_events, min_events
        self.fs = cfg["suite2p"]["fs"]
        self.dff = compute_dff(plane, cfg)[0] if dff is None else dff
        self.T = self.dff.shape[1]
        self.bad = io.load_bad_frames(sdir, self.T) if bad is None else bad
        self.sd = noise_sd(self.dff) + 1e-9
        self.movie = mr.open_movie(sdir, plane) if movie is None else movie
        self.Ly, self.Lx = self.movie.shape[1:]
        self.half = size // 2
        self.yy, self.xx = np.mgrid[:size, :size]

    def crop(self, t, cy, cx):
        half = self.half
        ya, yb, xa, xb = max(cy - half, 0), min(cy + half, self.Ly), max(cx - half, 0), min(cx + half, self.Lx)
        c = np.asarray(self.movie[t:t + 1][0][ya:yb, xa:xb], np.float32)
        return np.pad(c, ((ya - (cy - half), (cy + half) - yb), (xa - (cx - half), (cx + half) - xb)), mode="edge")

    def _zimage(self, frames, base, cy, cx):
        """(mean of the 3 frames around each event frame) - base, in noise units (divided by its robust spread over the
        crop), smoothed by 1 px."""
        ev = np.mean([np.mean([self.crop(int(np.clip(t + d, 0, self.T - 1)), cy, cx) for d in (-1, 0, 1)], axis=0) for t in frames], axis=0)
        d = ev - base
        return ndimage.gaussian_filter(d / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-6), 1.0)

    def one(self, i):
        """The snapshot of ROI i: dict(n_events, peaks, top, z, z_peak, feats, center). z is the average of its top events,
        z_peak the strongest single event, both minus the quiet frames, in noise units. They are None, and feats NaN, when the ROI
        has fewer than min_events events (or too few quiet frames)."""
        from scipy.signal import find_peaks
        s = self.stat[i]; y, x = np.asarray(s["ypix"]), np.asarray(s["xpix"])
        cy, cx = int(round(y.mean())), int(round(x.mean()))
        feats = {c: np.nan for c in SNAP}
        trace = self.dff[i].copy(); trace[self.bad] = -np.inf
        pk, pr = find_peaks(trace, height=self.cfg["event_z"] * self.sd[i], distance=max(1, int(self.fs)))
        res = {"n_events": len(pk), "peaks": pk, "top": None, "z": None, "z_peak": None, "feats": feats, "center": (cy, cx)}
        if len(pk) < self.min_events:
            return res
        top = pk[np.argsort(pr["peak_heights"])[::-1][:self.k]]
        quiet = np.flatnonzero((self.dff[i] < np.median(self.dff[i])) & ~self.bad)
        if len(quiet) < 5:
            return res
        base = np.mean([self.crop(int(t), cy, cx) for t in quiet[np.linspace(0, len(quiet) - 1, 2 * len(top)).astype(int)]], axis=0)
        z = self._zimage(top, base, cy, cx)
        res.update(top=top, z=z, z_peak=self._zimage(top[:1], base, cy, cx))
        size, half, yy, xx = self.size, self.half, self.yy, self.xx
        m = np.zeros((size, size), bool)
        my, mx_ = y - cy + half, x - cx + half
        ok = (my >= 0) & (my < size) & (mx_ >= 0) & (mx_ < size)
        m[my[ok], mx_[ok]] = True
        if m.sum() < 3:
            return res
        ccy, ccx, req = y.mean() - cy + half, x.mean() - cx + half, np.sqrt(len(y) / np.pi)
        near = np.hypot(yy - ccy, xx - ccx) / req < 2.0
        feats["snap_peak_z"], feats["snap_mean_z"] = z[m].max(), z[m].mean()
        feats["snap_corr_mask"] = _corr(z[near], m[near])
        zp = np.clip(z, 0, None) * near
        tot = zp.sum()
        if tot > 1e-6:
            my_c, mx_c = (zp * yy).sum() / tot, (zp * xx).sum() / tot
            feats["snap_inside_frac"] = zp[m].sum() / tot
            feats["snap_offset"] = np.hypot(my_c - ccy, mx_c - ccx) / req
            feats["snap_spread"] = np.sqrt((zp * ((yy - my_c) ** 2 + (xx - mx_c) ** 2)).sum() / tot) / req
        return res


def snapshot_table(ctx, rois, progress=None):
    """The six snapshot features of the ROIs `rois` (DataFrame with roi_idx, SNAP and snap_n_events) and their snapshots:
    (table, z_avg, z_peak, centers) with the images (N, size, size) and the window centres (N, 2) in image pixels."""
    rois = [int(i) for i in rois]
    n, size = len(rois), ctx.size
    cols = {c: np.full(n, np.nan) for c in SNAP}
    n_ev = np.zeros(n)
    za = np.full((n, size, size), np.nan, np.float32); zp = za.copy(); centers = np.zeros((n, 2), np.int32)
    for r, i in enumerate(rois):
        o = ctx.one(i)
        n_ev[r], centers[r] = o["n_events"], o["center"]
        if o["z"] is not None:
            za[r], zp[r] = o["z"], o["z_peak"]
        for c in SNAP:
            cols[c][r] = o["feats"][c]
        if progress and (r + 1) % 200 == 0:
            progress(r + 1, n)
    return pd.DataFrame({"roi_idx": rois, **cols, "snap_n_events": n_ev}), za, zp, centers


def save_snapshots(sess_dir, plane_hash, rois, za, zp, centers):
    """celltype/snapshots.npz: the snapshot images of the ROIs (float16), so the GUI shows them without reading the movie."""
    np.savez_compressed(io.out_dir(sess_dir) / "snapshots.npz", roi_idx=np.asarray(rois, np.int32), z_avg=za.astype(np.float16),
                        z_peak=zp.astype(np.float16), center=np.asarray(centers, np.int32), s2p_hash=str(plane_hash))


def load_snapshots(sess_dir, plane_hash):
    """The saved snapshots of a session as {'index': {roi_idx: row}, 'z_avg', 'z_peak', 'center'}, or None if there are none for
    this Suite2p run."""
    path = io.out_dir(sess_dir, create=False) / "snapshots.npz"
    if not path.is_file():
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["s2p_hash"]) != str(plane_hash):
            return None
        return {"index": {int(r): k for k, r in enumerate(d["roi_idx"])}, "z_avg": d["z_avg"].astype(np.float32),
                "z_peak": d["z_peak"].astype(np.float32), "center": d["center"]}
    except Exception:
        return None


def feature_settings(cfg):
    """The settings the feature table depends on besides the code (saved with every model, so a prediction computes the
    features the way the model was trained): the snapshots on / off, the edge margin and the share of an ROI allowed
    outside the image (it decides which ROIs get a snapshot)."""
    return {"snapshot_features": bool(cfg.get("snapshot_features", True)),
            "edge_margin_px": int(cfg.get("edge_margin_px", 0) or 0),
            "max_oob_frac": float(io.oob_limit(cfg))}


def _settings_tag(cfg):
    s = feature_settings(cfg)
    return "|".join([f"snap={int(s['snapshot_features'])}", f"iscell={cfg.get('iscell_prob_min')}",
                     f"margin={s['edge_margin_px']}", f"oob={s['max_oob_frac']:g}"])


def _cache_tag(cfg, model_id=None):
    """What the cached feature table was made for: the model id (or '-' when no model asked for it) and the feature settings."""
    return f"{model_id or '-'}|{_settings_tag(cfg)}"


def _session_features(sess_dir, cfg, force=False, model_id=None):
    """Feature table for one session, cached in <session>\\celltype\\."""
    sess_dir = Path(sess_dir)
    # one cache file per session; what it was made for sits next to it (features.version) and it is overwritten
    # when that no longer fits: a model asking for its own features gets them recomputed once per model.
    cache = io.out_dir(sess_dir) / "features.csv"
    vfile = io.out_dir(sess_dir) / "features.version"
    cur_hash = io.s2p_hash(sess_dir)
    if cache.is_file() and not force and vfile.is_file():
        have = vfile.read_text().strip()
        ok = have == _cache_tag(cfg, model_id) if model_id else have.split("|", 1)[-1] == _settings_tag(cfg)
        if ok:
            df = pd.read_csv(cache)
            if len(df) and str(df["s2p_hash"].iloc[0]) == cur_hash:
                return df

    plane = io.load_plane(sess_dir)
    stat = plane["stat"]
    green, red, valid_g, valid_r = io.load_images_valid(sess_dir, plane)
    index_img = io.roi_masks(stat, green.shape)
    bad = io.load_bad_frames(sess_dir, plane["F"].shape[1])
    if cfg["suite2p"].get("use_mc_bad_frames", True) is False:
        bad[:] = False

    morph = morph_features(stat, green, cfg["um_per_px"])
    dff_full, F0_full = compute_dff(plane, cfg)
    act = activity_features(plane, cfg, bad, (dff_full, F0_full))
    redimg = red_image(green, red, cfg["red_bleedthrough"]) if red is not None else None
    redf = red_features(stat, index_img, redimg, cfg["suite2p"]["diameter"])
    redf["s2p_redprob"] = (plane["redcell"][:, 1] if plane["redcell"] is not None
                           else np.nan)
    D, margin = cfg["suite2p"]["diameter"], int(cfg.get("edge_margin_px", 0))
    morph["green_blob"] = blob_peaks(green, stat, D)
    max_img = io.load_max_image(sess_dir, green.shape)
    if max_img is None:
        print(f"  {sess_dir.name}: no {io.MAX_TIF} of the size of the mean image: the max-image features are NaN (median-filled)")
    for k, v in green_max_features(stat, index_img, green, max_img, D).items():
        morph[k] = v
    redf["red_blob"] = blob_peaks(red, stat, D)
    redf["red_oob_frac"] = roi_oob_fraction(stat, valid_r, margin)
    redf["green_oob_frac"] = roi_oob_fraction(stat, valid_g, margin)
    lim = io.oob_limit(cfg)
    for name, col in (("red", "red_oob_frac"), ("green", "green_oob_frac")):
        n_any, n_drop = int((redf[col] > 0).sum()), int((redf[col] > lim).sum())
        if n_any:
            print(f"  {sess_dir.name}: {n_any} ROI(s) have pixels outside the valid area of the {name} image; "
                  f"{n_drop} have more than {lim:g} of their pixels outside and are discarded as edge ROIs "
                  f"(non-SPN by the hard rule)")

    # event snapshots, for the ROIs that are cells and not edge ROIs only (the others are not predicted); NaN elsewhere
    n_roi = len(stat)
    snap = pd.DataFrame({c: np.full(n_roi, np.nan) for c in SNAP})
    if cfg.get("snapshot_features", True):
        thr = cfg.get("iscell_prob_min")
        is_cell = (plane["iscell"][:, 1] >= float(thr)) if thr is not None else (plane["iscell"][:, 0] > 0)
        edge = ((np.nan_to_num(redf["red_oob_frac"].to_numpy(float)) > lim)
                | (np.nan_to_num(redf["green_oob_frac"].to_numpy(float)) > lim))
        todo = np.flatnonzero(is_cell & ~edge)
        try:
            t0 = time.time()
            ctx = SnapContext(plane, cfg, sess_dir, dff=dff_full, bad=bad)
            tab, za, zp, centers = snapshot_table(ctx, todo, progress=lambda k, n: print(f"    snapshots {k}/{n}"))
            snap.loc[todo, SNAP] = tab[SNAP].to_numpy()
            save_snapshots(sess_dir, plane["hash"], todo, za, zp, centers)
            print(f"  {sess_dir.name}: event snapshots of {len(todo)} cell ROIs ({int((tab.snap_n_events >= 3).sum())} with >= 3 events) "
                  f"in {time.time() - t0:.0f} s")
        except Exception as e:
            print(f"  ! {sess_dir.name}: event snapshots not computed ({e}); the snap_* features are NaN (median-filled)")

    sid = io.session_id(sess_dir, cfg["session_id_depth"])
    meta = pd.DataFrame({
        "session_id": sid, "session_dir": str(sess_dir), "animal": io.animal_of(sid),
        "roi_idx": np.arange(len(stat)), "s2p_hash": plane["hash"],
        "iscell": plane["iscell"][:, 0].astype(int),
        "s2p_iscell_prob": plane["iscell"][:, 1],
    })
    df = pd.concat([meta, morph, act, redf, snap], axis=1)
    df = df[META + MORPH + ACTIVITY + RED + INFO]
    df.to_csv(cache, index=False)
    vfile.write_text(_cache_tag(cfg, model_id))
    return df


def extract_features(sessions, cfg, force=False):
    out = {}
    for s in sessions:
        try:
            out[str(s)] = session_features(s, cfg, force)
            print(f"  features: {s} ({len(out[str(s)])} ROIs)")
        except Exception as e:
            print(f"  ! features FAILED for {s}: {e}")
    return out
