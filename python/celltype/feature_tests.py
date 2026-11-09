"""Experimental stage-1 image features, to be tested in a notebook before they go into features.py:

    lam_features        features of the Suite2p weights (lam) of each ROI: no movie needed
    snapshot_features   event-triggered snapshot of each ROI: the movie around the ROI's own strongest events minus
                        quiet frames, in noise units, reduced to a few numbers (where its activity is, and whether it
                        sits on the ROI); needs the registered movie
    rg_features         stage-2 features that combine the red and the green image (ratio, pixel correlation, overlap)
    snapshot_examples / plot_snapshot_examples   look at good and bad snapshots next to the mean image and the activity
    feature_table / plot_feature_report / plot_compare   the AUC and held-out importance of every feature, and held-out
                        comparisons of feature sets
Nothing here is used by the pipeline.
"""
import numpy as np
import pandas as pd

from . import features as ft
from . import io_utils as io

LAM_COLS = ["lam_peak", "lam_cv", "lam_offset", "lam_corr_mean", "lam_corr_max"]
SNAP_COLS = ft.SNAP            # the event-snapshot features are now computed by features.py (SnapContext, snapshot_table)
RG_COLS = ["rg_ratio", "rg_corr", "rg_overlap"]

# the labels the notebook cells use: stage 1 = SPN (1) vs interneuron / junk (0), everything else -1
LABELS = {1: lambda d: np.where(d.stage1 == "SPN", 1, np.where(d.stage1.isin(["interneuron", "junk"]), 0, -1)),
          2: lambda d: np.where((d.stage1 == "SPN") & (d.stage2 == "red+"), 1,
                                np.where((d.stage1 == "SPN") & (d.stage2 == "red-"), 0, -1))}


def ensure_columns(rows, cfg, cols):
    """`rows` (ROIs of the feature table: session_dir, roi_idx, ...) with all feature columns `cols`. A table made by an
    older version of features.py lacks the newer ones: those that can be computed are computed here, with the same
    functions as the pipeline, for the sessions (and ROIs) in rows, and merged on:
        the event snapshot features (ft.SNAP; they read the movie), the whole-frame max features (ft.GREEN_MAX), green_blob,
        and the activity features (the trace ones such as neuropil_corr, common_mode_corr, rise_decay_asym)
    Any other missing column raises an error telling to recompute the table. Returns rows unchanged when nothing is missing."""
    missing = [c for c in cols if c not in rows.columns]
    if not missing:
        return rows
    need_act = [c for c in missing if c in ft.ACTIVITY]
    need_max = [c for c in missing if c in ft.GREEN_MAX]
    need_snap = [c for c in missing if c in ft.SNAP]
    need_blob = "green_blob" in missing
    unknown = [c for c in missing if c not in need_act + need_max + need_snap + ["green_blob"]]
    if unknown:
        raise RuntimeError(f"the table lacks {unknown}, which this helper cannot compute: rerun cell 1 of the training "
                           f"notebook so the whole table is recomputed with the current features.py")
    D, parts = cfg["suite2p"]["diameter"], []
    for sdir in rows["session_dir"].unique():
        plane = io.load_plane(sdir); stat = plane["stat"]
        g, _ = io.load_images(sdir, plane)
        d = {"session_dir": sdir, "roi_idx": np.arange(len(stat))}
        if need_max:
            mx = ft.green_max_features(stat, io.roi_masks(stat, g.shape), g, io.load_max_image(sdir, g.shape), D)
            d.update({c: mx[c] for c in need_max})
        if need_blob:
            d["green_blob"] = ft.blob_peaks(g, stat, D)
        part = pd.DataFrame(d)
        if need_act:
            bad = io.load_bad_frames(sdir, plane["F"].shape[1])
            if cfg["suite2p"].get("use_mc_bad_frames", True) is False:
                bad[:] = False
            act = ft.activity_features(plane, cfg, bad)
            for c in need_act:
                part[c] = act[c].to_numpy()
        if need_snap:                                           # only the ROIs of rows: the snapshots read the movie
            ids = rows.loc[rows.session_dir == sdir, "roi_idx"].astype(int).tolist()
            tab, _, _, _ = ft.snapshot_table(ft.SnapContext(plane, cfg, sdir), ids)
            part = part.merge(tab[["roi_idx"] + need_snap], on="roi_idx", how="left")
        parts.append(part)
    print(f"  computed {missing} for {len(parts)} session(s): the table was made by an older features.py")
    return rows.merge(pd.concat(parts), on=["session_dir", "roi_idx"], how="left")


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def lam_features(sdir, cfg, rois=None):
    """Features of the Suite2p weights (`lam`) of each ROI, which say where its activity is; no movie needed.
      lam_peak       max weight / mean weight: how concentrated the footprint is
      lam_cv         SD / mean of the weights
      lam_offset     distance between the weighted and the plain centre of the ROI, in ROI radii
      lam_corr_mean  correlation of the weights with the brightness of the mean image inside the ROI
      lam_corr_max   the same with the max image (NaN without one)
    DataFrame with roi_idx and these columns."""
    plane = io.load_plane(sdir); stat = plane["stat"]
    g, _ = io.load_images(sdir, plane); mx = io.load_max_image(sdir, g.shape)
    rois = list(range(len(stat))) if rois is None else list(rois)
    out = {c: np.full(len(rois), np.nan) for c in LAM_COLS}
    for r, i in enumerate(rois):
        s = stat[i]; y, x = np.asarray(s["ypix"]), np.asarray(s["xpix"])
        lam = np.asarray(s.get("lam", np.ones(len(y))), float)
        if len(y) < 3 or lam.sum() <= 0:
            continue
        out["lam_peak"][r] = lam.max() / lam.mean()
        out["lam_cv"][r] = lam.std() / lam.mean()
        w = lam / lam.sum()
        out["lam_offset"][r] = np.hypot((w * y).sum() - y.mean(), (w * x).sum() - x.mean()) / np.sqrt(len(y) / np.pi)
        out["lam_corr_mean"][r] = _corr(lam, g[y, x])
        if mx is not None:
            out["lam_corr_max"][r] = _corr(lam, mx[y, x])
    return pd.DataFrame({"roi_idx": rois, **out})


def snapshot_features(sdir, cfg, rois, size=32, k_events=6, min_events=3, seed=0, return_images=False):
    """The six event-snapshot features (features.SNAP, computed by features.SnapContext) of the ROIs `rois` of one session, as the
    pipeline computes them, plus snap_n_events; DataFrame with roi_idx. With return_images also the (N, size, size) snapshots
    (average of the top events, noise units). Needs the movie (data.bin: fast; the registered tif: slow)."""
    ctx = ft.SnapContext(io.load_plane(sdir), cfg, sdir, size=size, k_events=k_events, min_events=min_events)
    tab, za, _, _ = ft.snapshot_table(ctx, list(rois))
    return (tab, za) if return_images else tab


def snapshot_examples(pairs, cfg, size=32, k_events=6):
    """Everything needed to look at the snapshots of the (session_dir, roi_idx) pairs, in the order given: the snapshot z image
    (None without enough events), the event frames used (top) and all detected peaks, the dF/F trace, the ROI mask and the crops
    of the mean image and (if the session has one) the whole-frame max image around the ROI, all `size` x `size`.
    The movie of a session is opened once and dropped when its last pair is done."""
    pairs = [(s, int(i)) for s, i in pairs]
    out = [None] * len(pairs)
    for sdir in dict.fromkeys(p[0] for p in pairs):
        ctx = ft.SnapContext(io.load_plane(sdir), cfg, sdir, size=size, k_events=k_events)
        g, _ = io.load_images(sdir, ctx.plane); mx = io.load_max_image(sdir, g.shape)
        half = size // 2

        def cut(im, cy, cx):
            ya, yb, xa, xb = max(cy - half, 0), min(cy + half, im.shape[0]), max(cx - half, 0), min(cx + half, im.shape[1])
            return np.pad(im[ya:yb, xa:xb], ((ya - (cy - half), (cy + half) - yb), (xa - (cx - half), (cx + half) - xb)), mode="edge")
        for k, (s2, i) in enumerate(pairs):
            if s2 != sdir:
                continue
            o = ctx.one(i); cy, cx = o["center"]; st = ctx.stat[i]
            mask = np.zeros((size, size), bool)
            my, mx_ = np.asarray(st["ypix"]) - cy + half, np.asarray(st["xpix"]) - cx + half
            ok = (my >= 0) & (my < size) & (mx_ >= 0) & (mx_ < size)
            mask[my[ok], mx_[ok]] = True
            out[k] = {**o, "session": sdir, "roi": i, "mean_crop": cut(g, cy, cx), "max_crop": None if mx is None else cut(mx, cy, cx),
                      "mask": mask, "dff": ctx.dff[i].copy(), "fs": ctx.fs}
        del ctx
    return out


def plot_snapshot_examples(examples, labels, title):
    """One row per example: the mean image with the ROI outline, the whole-frame max image, the snapshot (noise units), the whole
    dF/F trace with the events used (red) and the other detected peaks (grey), and a close-up of the strongest event (the 3 frames
    averaged are shaded). The title of each row gives the label and the snapshot features."""
    import matplotlib.pyplot as plt
    n = len(examples)
    fig = plt.figure(figsize=(19, 2.5 * n)); gs = fig.add_gridspec(n, 5, width_ratios=[1, 1, 1, 3.2, 1.6], wspace=.25, hspace=.55)
    fig.suptitle(title, fontsize=12)
    for r, (e, lab) in enumerate(zip(examples, labels)):
        for c, (key, name) in enumerate((("mean_crop", "mean image"), ("max_crop", "whole-frame max"))):
            a = fig.add_subplot(gs[r, c]); a.axis("off")
            if e[key] is not None:
                a.imshow(e[key], cmap="gray"); a.contour(e["mask"], levels=[.5], colors="c", linewidths=.8)
            a.set_title(name if r == 0 else "", fontsize=9)
        a = fig.add_subplot(gs[r, 2]); a.axis("off")
        if e["z"] is not None:
            v = max(5.0, float(np.nanpercentile(np.abs(e["z"]), 99.5)))
            im = a.imshow(e["z"], cmap="coolwarm", vmin=-v, vmax=v); a.contour(e["mask"], levels=[.5], colors="k", linewidths=.8)
            if r == 0:
                a.set_title("snapshot (noise units)", fontsize=9)
        else:
            a.text(.5, .5, f"no snapshot\n({e['n_events']} events)", ha="center", va="center", transform=a.transAxes)
        f = e["feats"]
        a = fig.add_subplot(gs[r, 3]); t = np.arange(len(e["dff"])) / e["fs"] / 60
        a.plot(t, e["dff"], "k", lw=.4)
        if len(e["peaks"]):
            a.plot(t[e["peaks"]], e["dff"][e["peaks"]], ".", color="0.6", ms=4)
        if e["top"] is not None:
            a.plot(t[e["top"]], e["dff"][e["top"]], "o", color="tab:red", ms=4)
        a.set_xlabel("min", fontsize=7); a.tick_params(labelsize=7)
        a.set_title(f"{lab}  ROI {e['roi']}  |  mean z {f['snap_mean_z']:.1f}  peak z {f['snap_peak_z']:.1f}  inside {f['snap_inside_frac']:.2f}  "
                    f"corr {f['snap_corr_mask']:.2f}  offset {f['snap_offset']:.2f}  spread {f['snap_spread']:.2f}  ({e['n_events']} events)", fontsize=8, loc="left")
        a = fig.add_subplot(gs[r, 4])
        if e["top"] is not None:
            t0 = int(e["top"][int(np.argmax(e["dff"][e["top"]]))]); w = int(6 * e["fs"])
            lo, hi = max(t0 - w, 0), min(t0 + w, len(e["dff"]))
            a.plot((np.arange(lo, hi) - t0) / e["fs"], e["dff"][lo:hi], "k", lw=.6); a.axvspan(-1 / e["fs"], 1 / e["fs"], color="tab:red", alpha=.25)
            a.set_xlabel("s from the strongest event", fontsize=7); a.tick_params(labelsize=7)
        else:
            a.axis("off")
    plt.show()


def _ring_stats(stat, index_img, img, diameter, rois):
    """Per ROI in rois: (mean inside, median of the surround ring, robust SD of the ring); NaN where the ring is too small."""
    from scipy import ndimage
    outer_r = max(6, int(round(0.6 * diameter)))
    union = ndimage.binary_dilation(index_img >= 0, iterations=1)
    out = np.full((len(rois), 3), np.nan)
    for r, i in enumerate(rois):
        s = stat[i]
        y0, y1, x0, x1 = ft._crop(s, img.shape, outer_r + 2)
        m = np.zeros((y1 - y0, x1 - x0), bool); m[s["ypix"] - y0, s["xpix"] - x0] = True
        ring = ndimage.binary_dilation(m, iterations=outer_r) & ~ndimage.binary_dilation(m, iterations=2)
        clean = ring & ~union[y0:y1, x0:x1]
        ring = clean if clean.sum() >= 10 else ring
        if ring.sum() < 3:
            continue
        rc = img[y0:y1, x0:x1]; bg = rc[ring]; med = np.median(bg)
        out[r] = (rc[m].mean(), med, 1.4826 * np.median(np.abs(bg - med)) + 1e-6)
    return out


def rg_features(sdir, cfg, rois=None):
    """Stage-2 features that combine the red and the green mean image (red aligned to green), for the ROIs in rois:
      rg_ratio    log of (the ROI's red above its surround) / (the ROI's green above its surround), each with a floor of 5 %
                  of the session median: how red the cell is relative to its own green, which discounts brightness and focus
      rg_corr     Pearson correlation of the red and the green image over the ROI and its surround (the dilated mask)
      rg_overlap  Jaccard overlap of the red-bright and the green-bright pixels there (bright = more than 2 robust SD above the
                  surround ring of that image)
    NaN without a red image. DataFrame with roi_idx and these columns."""
    from scipy import ndimage
    plane = io.load_plane(sdir); stat = plane["stat"]
    g, r = io.load_images(sdir, plane)
    rois = list(range(len(stat))) if rois is None else list(rois)
    out = {c: np.full(len(rois), np.nan) for c in RG_COLS}
    if r is None:
        return pd.DataFrame({"roi_idx": rois, **out})
    D = cfg["suite2p"]["diameter"]; outer_r = max(6, int(round(0.6 * D)))
    idx = io.roi_masks(stat, g.shape)
    sg, sr = _ring_stats(stat, idx, g, D, rois), _ring_stats(stat, idx, r, D, rois)
    sig_g, sig_r = sg[:, 0] - sg[:, 1], sr[:, 0] - sr[:, 1]
    fl_g, fl_r = 0.05 * np.nanmedian(np.abs(sig_g)), 0.05 * np.nanmedian(np.abs(sig_r))
    out["rg_ratio"] = np.log(np.maximum(sig_r, fl_r) / np.maximum(sig_g, fl_g))
    for k, i in enumerate(rois):
        s = stat[i]
        y0, y1, x0, x1 = ft._crop(s, g.shape, outer_r + 2)
        m = np.zeros((y1 - y0, x1 - x0), bool); m[s["ypix"] - y0, s["xpix"] - x0] = True
        region = ndimage.binary_dilation(m, iterations=outer_r)
        rc, gc = r[y0:y1, x0:x1], g[y0:y1, x0:x1]
        out["rg_corr"][k] = _corr(rc[region], gc[region])
        if np.isfinite(sr[k]).all() and np.isfinite(sg[k]).all():
            R = region & (rc > sr[k, 1] + 2 * sr[k, 2]); G = region & (gc > sg[k, 1] + 2 * sg[k, 2])
            u = (R | G).sum()
            if u:
                out["rg_overlap"][k] = (R & G).sum() / u
    return pd.DataFrame({"roi_idx": rois, **out})


# ---------------------------------------------------------------- AUC and held-out importance of every feature, and comparisons
def feature_table(res):
    """One row per feature of a run_cv result (run with importance=True): its AUC over all labeled ROIs, its AUC on each
    held-out session (mean and SD over the sessions that have at least 3 ROIs of each class), and the permutation importance
    measured on the held-out data of each fold (fall in the held-out AUC when the feature is shuffled; mean and SD over folds).
    NaN in a feature is filled with its median, as the models do."""
    from sklearn.metrics import roc_auc_score
    d, y, cols = res["d"], res["y"], res["cols"]
    sess = d["session_id"].to_numpy()
    rows = []
    for c in cols:
        x = d[c].to_numpy(float)
        x = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x)
        per = [roc_auc_score(y[sess == s], x[sess == s]) for s in pd.unique(sess)
               if y[sess == s].sum() >= 3 and (1 - y[sess == s]).sum() >= 3]
        rows.append((c, roc_auc_score(y, x), np.mean(per) if per else np.nan, np.std(per) if per else np.nan))
    t = pd.DataFrame(rows, columns=["feature", "AUC", "AUC held-out sessions", "AUC SD"]).set_index("feature")
    imp = res["imp_s"]
    t["importance"], t["importance SD"] = imp.mean(), imp.std()
    return t


def plot_feature_report(t, title):
    """Three panels in the same feature order (sorted by held-out importance): the AUC of the single feature over all labeled ROIs,
    its AUC on each held-out session (mean, SD over sessions), and its permutation importance on the held-out data (mean, SD over
    folds). AUC below 0.5 (blue) means the feature is higher in the negative class."""
    import matplotlib.pyplot as plt
    t = t.sort_values("importance")
    y = np.arange(len(t))
    fig, ax = plt.subplots(1, 3, figsize=(17, 0.28 * len(t) + 2), sharey=True)
    col = lambda v: ["tab:red" if x >= 0.5 else "tab:blue" for x in v]
    ax[0].barh(y, t["AUC"] - 0.5, left=0.5, color=col(t["AUC"])); ax[0].set_title("AUC of the single feature (all labeled ROIs)")
    ax[1].barh(y, t["AUC held-out sessions"] - 0.5, left=0.5, xerr=t["AUC SD"], color=col(t["AUC held-out sessions"]), ecolor="0.4")
    ax[1].set_title("AUC on held-out sessions (mean, SD over sessions)")
    for a in ax[:2]:
        a.axvline(0.5, color="k", lw=.8); a.set_xlim(0.2, 1.0)
    ax[2].barh(y, t["importance"], xerr=t["importance SD"], color=["tab:green" if v > 0 else "0.6" for v in t["importance"]], ecolor="0.4")
    ax[2].axvline(0, color="k", lw=.8); ax[2].set_title("permutation importance on held-out data\n(fall in AUC when shuffled; mean, SD over folds)")
    ax[0].set_yticks(y, t.index, fontsize=7)
    fig.suptitle(title, fontsize=11); plt.tight_layout(); plt.show()


def plot_compare(results, title):
    """results: {label: run_cv result}. Held-out ROC curves of the label sets, and the held-out AUC of each session per set."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    first = next(iter(results.values()))
    for lab, r in results.items():
        ok = ~np.isnan(r["oof"]); fpr, tpr, _ = roc_curve(r["y"][ok], r["oof"][ok])
        ax[0].plot(fpr, tpr, label=f"{lab} ({roc_auc_score(r['y'][ok], r['oof'][ok]):.3f})")
    ax[0].plot([0, 1], [0, 1], ":", color="gray"); ax[0].legend(fontsize=7); ax[0].set_title("held-out ROC"); ax[0].set_xlim(0, .6); ax[0].set_ylim(.4, 1)
    sessions = first["folds"].dropna(subset=["AUC"])["held out"].tolist()
    w = 0.8 / len(results)
    for k, (lab, r) in enumerate(results.items()):
        f = r["folds"].set_index("held out")["AUC"]
        ax[1].bar(np.arange(len(sessions)) + k * w, [f.get(s, np.nan) for s in sessions], w, label=lab)
    ax[1].set_xticks(np.arange(len(sessions)) + 0.4 - w / 2, [str(s)[-22:] for s in sessions], rotation=60, fontsize=6)
    ax[1].set_ylim(.5, 1.0); ax[1].set_title("AUC of each held-out session")
    fig.suptitle(title); plt.tight_layout(); plt.show()
