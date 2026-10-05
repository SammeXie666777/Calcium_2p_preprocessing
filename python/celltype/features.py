"""Per-ROI features from Suite2p output + the green / red mean images.

Three groups, used by different stages:
    MORPH, ACTIVITY  -> stage 1 (SPN vs non-SPN)
    RED              -> stage 2 (red+ vs red-); never used by stage 1, and stage 2
                        never sees activity, so later dSPN/iSPN activity
                        comparisons are not biased by how cells were labeled.
Features are made FOV-relative where brightness matters (*_rel), so models can
be trained across sessions without per-session z-scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import skew
from skimage.measure import regionprops

from . import io_utils as io

MORPH = ["equiv_diam_um", "area_um2", "eccentricity", "solidity", "aspect",
         "circularity", "nucleus_shadow", "green_rel", "s2p_compact"]
ACTIVITY = ["skew", "baseline_rel", "baseline_over_neuropil", "noise_sd",
            "event_rate", "event_amp_med", "event_amp_cv", "frac_active",
            "acf_1s", "acf_5s", "snr"]
RED = ["red_ring_ratio", "red_z", "red_fill", "red_corr", "red_mean_rel",
       "s2p_redprob"]
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


def activity_features(plane, cfg, bad):
    fs = cfg["suite2p"]["fs"]
    dff, F0 = compute_dff(plane, cfg)
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


def red_features(stat, index_img, red, diameter):
    nan = {k: np.nan for k in RED[:-1]}
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
        lv, rv = lam[big], rc[big]
        corr = np.corrcoef(lv, rv)[0, 1] if lv.std() > 0 and rv.std() > 0 else np.nan
        rows.append({
            "red_ring_ratio": vals.mean() / (bg.mean() + 1e-6),
            "red_z": (vals.mean() - med) / sd,
            "red_fill": float((vals > med + 2 * sd).mean()),
            "red_corr": corr,
            "red_mean": vals.mean(),
        })
    df = pd.DataFrame(rows)
    df["red_mean_rel"] = df["red_mean"] / (df["red_mean"].median() + 1e-9)
    return df.drop(columns="red_mean")


# ---------------------------------------------------------------- per session
def session_features(sess_dir, cfg, force=False):
    """Feature table for one session, cached in <session>\\celltype\\."""
    sess_dir = Path(sess_dir)
    cache = io.out_dir(sess_dir) / f"features_{cfg['feature_version']}.csv"
    cur_hash = io.s2p_hash(sess_dir)
    if cache.is_file() and not force:
        df = pd.read_csv(cache)
        if len(df) and str(df["s2p_hash"].iloc[0]) == cur_hash:
            return df

    plane = io.load_plane(sess_dir)
    stat = plane["stat"]
    green, red = io.load_images(sess_dir, plane)
    index_img = io.roi_masks(stat, green.shape)
    bad = io.load_bad_frames(sess_dir, plane["F"].shape[1])
    if cfg["suite2p"].get("use_mc_bad_frames", True) is False:
        bad[:] = False

    morph = morph_features(stat, green, cfg["um_per_px"])
    act = activity_features(plane, cfg, bad)
    redimg = red_image(green, red, cfg["red_bleedthrough"]) if red is not None else None
    redf = red_features(stat, index_img, redimg, cfg["suite2p"]["diameter"])
    redf["s2p_redprob"] = (plane["redcell"][:, 1] if plane["redcell"] is not None
                           else np.nan)

    sid = io.session_id(sess_dir, cfg["session_id_depth"])
    meta = pd.DataFrame({
        "session_id": sid, "session_dir": str(sess_dir), "animal": io.animal_of(sid),
        "roi_idx": np.arange(len(stat)), "s2p_hash": plane["hash"],
        "iscell": plane["iscell"][:, 0].astype(int),
        "s2p_iscell_prob": plane["iscell"][:, 1],
    })
    df = pd.concat([meta, morph, act, redf], axis=1)
    df = df[META + MORPH + ACTIVITY + RED]
    df.to_csv(cache, index=False)
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
