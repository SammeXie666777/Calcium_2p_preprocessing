"""Two-stage models, training with leave-one-animal-out evaluation, prediction.

Stage 1  SPN vs non-SPN   MORPH + ACTIVITY features, semi-supervised (self-training)
                          or, with too few non-SPN labels, an anomaly detector
                          fit on SPNs only.
Stage 2  red+ vs red-     RED features only, label spreading among SPNs.
Hard rules (size / shape) override stage 1.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from . import features as ft
from . import io_utils as io

F1 = ft.MORPH + ft.ACTIVITY
F2 = ft.RED
NONSPN = ("interneuron", "junk")
MODEL_FILE = "celltype_model.joblib"


# ---------------------------------------------------------------- data
def training_table(cfg):
    """Features of every labeled session joined with the labels (unlabeled
    ROIs kept, with empty stage1/stage2, for the semi-supervised fits)."""
    labels = io.load_labels(cfg["label_store"])
    if labels.empty:
        raise RuntimeError(f"no labels in {cfg['label_store']}")
    tables = []
    for sdir, lab in labels.groupby("session_dir"):
        try:
            df = ft.session_features(sdir, cfg)
        except Exception as e:
            print(f"  ! skipping {sdir}: {e}")
            continue
        stale = lab["s2p_hash"].astype(str) != str(df["s2p_hash"].iloc[0])
        if stale.any():
            print(f"  ! {sdir}: {stale.sum()} labels are from an older Suite2p run, ignored")
            lab = lab[~stale]
        df = df.merge(lab[["roi_idx", "stage1", "stage2"]], on="roi_idx", how="left")
        tables.append(df)
    if not tables:
        raise RuntimeError("no usable labeled sessions")
    df = pd.concat(tables, ignore_index=True)
    df[["stage1", "stage2"]] = df[["stage1", "stage2"]].fillna("")
    return df


def hard_rule_flag(df, cfg):
    r = cfg["hard_rules"]
    return ((df["equiv_diam_um"] > r["cin_min_diam_um"])
            | (df["solidity"] < r["min_solidity"])
            | (df["aspect"] > r["max_aspect"])).to_numpy()


# ---------------------------------------------------------------- fitting
def _prep():
    return [SimpleImputer(strategy="median"), StandardScaler()]


def _p1(clf, X):
    return clf.predict_proba(X)[:, list(clf.classes_).index(1)]


def fit_models(df, cfg, verbose=True):
    """Fit stage 2 then stage 1 on a training table. Returns the model bundle."""
    log = print if verbose else (lambda *a, **k: None)
    n_min = cfg["min_labels_per_class"]
    pool = (df["iscell"] == 1) | (df["stage1"] != "")

    # ---- stage 2: red+ / red- among SPNs ----
    y2 = np.full(len(df), -1)
    is_spn_lab = df["stage1"] == "SPN"
    y2[(is_spn_lab & (df["stage2"] == "red+")).to_numpy()] = 1
    y2[(is_spn_lab & (df["stage2"] == "red-")).to_numpy()] = 0
    n_pos2, n_neg2 = (y2 == 1).sum(), (y2 == 0).sum()
    if min(n_pos2, n_neg2) < 5:
        raise RuntimeError(f"stage 2 needs >= 5 red+ and red- SPN labels "
                           f"(have {n_pos2} / {n_neg2})")
    sel2 = pool.to_numpy() & ~((df["stage1"].isin(NONSPN)).to_numpy())
    X2 = df.loc[sel2, F2].to_numpy(float)
    if cfg["model2"] == "logreg":
        m2 = make_pipeline(*_prep(), LogisticRegression(class_weight="balanced", max_iter=2000))
        lab = y2[sel2] >= 0
        m2.fit(X2[lab], y2[sel2][lab])
    else:
        m2 = make_pipeline(*_prep(), LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=200))
        m2.fit(X2, y2[sel2])
    log(f"  stage 2 ({cfg['model2']}): {n_pos2} red+, {n_neg2} red- labels")

    # ---- stage 1: SPN / non-SPN ----
    y1 = np.full(len(df), -1)
    y1[(df["stage1"] == "SPN").to_numpy()] = 1
    y1[df["stage1"].isin(NONSPN).to_numpy()] = 0
    # free positives: unlabeled cells that are clearly red are SPNs
    # (D1-Cre: all red cells are dSPNs; D2-Cre: CINs are red too -> size gate)
    unl = (y1 == -1) & pool.to_numpy()
    p_red_unl = _p1(m2, df.loc[unl, F2].to_numpy(float))
    pseudo = np.zeros(len(df), bool)
    pseudo[np.flatnonzero(unl)[p_red_unl > 0.97]] = True
    pseudo &= ~hard_rule_flag(df, cfg)
    y1[pseudo] = 1
    n_pos1, n_neg1 = (y1 == 1).sum(), (y1 == 0).sum()

    sel1 = pool.to_numpy() | (y1 >= 0)
    X1 = df.loc[sel1, F1].to_numpy(float)
    bundle = {"model2": m2, "pseudo_positives": int(pseudo.sum())}
    if n_neg1 >= n_min and (y1 == 1).sum() >= n_min:
        base = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                               max_iter=200, class_weight="balanced")
                if cfg["model1"] == "selftrain_hgb"
                else LogisticRegression(class_weight="balanced", max_iter=2000))
        m1 = SelfTrainingClassifier(make_pipeline(*_prep(), base), threshold=0.95, max_iter=10)
        m1.fit(X1, y1[sel1])
        bundle.update(mode1="classifier", model1=m1)
        log(f"  stage 1 ({cfg['model1']}): {n_pos1} SPN ({pseudo.sum()} from red), "
            f"{n_neg1} non-SPN labels")
    else:
        Xp = df.loc[y1 == 1, F1].to_numpy(float)
        m1 = make_pipeline(*_prep(), IsolationForest(n_estimators=300, random_state=0))
        m1.fit(Xp)
        bundle.update(mode1="anomaly", model1=m1,
                      ref_scores=np.sort(m1.decision_function(Xp)))
        log(f"  stage 1 (anomaly, only {n_neg1} non-SPN labels < {n_min}): "
            f"fit on {n_pos1} SPNs ({pseudo.sum()} from red)")
    bundle.update(features1=F1, features2=F2, cre_line=cfg["cre_line"],
                  feature_version=cfg["feature_version"],
                  thresholds=dict(cfg["thresholds"]), hard_rules=dict(cfg["hard_rules"]),
                  anomaly_quantile=list(cfg["anomaly_quantile"]))
    return bundle


def predict_probs(bundle, df):
    X1 = df[bundle["features1"]].to_numpy(float)
    if bundle["mode1"] == "classifier":
        p_spn = _p1(bundle["model1"], X1)
    else:   # percentile of the score among training SPNs
        s = bundle["model1"].decision_function(X1)
        p_spn = np.searchsorted(bundle["ref_scores"], s) / len(bundle["ref_scores"])
    p_red = _p1(bundle["model2"], df[bundle["features2"]].to_numpy(float))
    return p_spn, p_red


def assign_labels(df, p_spn, p_red, bundle, use_iscell=True):
    th = bundle["thresholds"]
    if bundle["mode1"] == "anomaly":
        lo, hi = bundle["anomaly_quantile"]
    else:
        lo, hi = th["p_spn_lo"], th["p_spn_hi"]
    red_name, nored_name = ("dSPN", "iSPN") if bundle["cre_line"] == "D1" else ("iSPN", "dSPN")
    rule = hard_rule_flag(df, {"hard_rules": bundle["hard_rules"]})
    lab = np.where(p_red > th["p_red_hi"], red_name,
                   np.where(p_red < th["p_red_lo"], nored_name, "SPN_ambiguous")).astype(object)
    lab[p_spn < hi] = "uncertain"
    lab[(p_spn < lo) | rule] = "nonSPN"
    if use_iscell:
        lab[df["iscell"].to_numpy() == 0] = "not_cell"
    return lab, rule


# ---------------------------------------------------------------- evaluation
def _splits(df, y_mask):
    animals, sessions = df["animal"].nunique(), df["session_id"].nunique()
    if animals >= 2:
        groups, name = df["animal"].to_numpy(), "leave-one-animal-out"
    elif sessions >= 2:
        groups, name = df["session_id"].to_numpy(), "leave-one-session-out"
    else:
        return _roi_folds(df, y_mask)
    n = len(np.unique(groups))
    return list(GroupKFold(n_splits=n).split(df, groups=groups)), name


def _roi_folds(df, y_mask):
    # one session only: stratified folds over labeled ROIs, unlabeled always in train
    lab = np.flatnonzero(y_mask)
    y = df["stage1"].to_numpy()[lab]
    skf = StratifiedKFold(n_splits=min(5, max(2, pd.Series(y).value_counts().min())),
                          shuffle=True, random_state=0)
    folds = []
    for tr, te in skf.split(lab, y):
        test = lab[te]
        train = np.setdiff1d(np.arange(len(df)), test)
        folds.append((train, test))
    return folds, "5-fold over ROIs (single session: optimistic)"


def evaluate(df, cfg):
    labeled1 = (df["stage1"].isin(["SPN", *NONSPN])).to_numpy()
    folds, scheme = _splits(df, labeled1)
    p_spn = np.full(len(df), np.nan)
    p_red = np.full(len(df), np.nan)
    mode = None
    for tr, te in folds:
        try:
            b = fit_models(df.iloc[tr].reset_index(drop=True), cfg, verbose=False)
        except RuntimeError as e:
            print(f"  ! fold skipped: {e}")
            continue
        mode = b["mode1"]
        ps, pr = predict_probs(b, df.iloc[te])
        p_spn[te], p_red[te] = ps, pr

    th = cfg["thresholds"]
    lo, hi = (cfg["anomaly_quantile"] if mode == "anomaly" else (th["p_spn_lo"], th["p_spn_hi"]))
    rep = {"scheme": scheme, "stage1_mode": mode}

    # stage 1
    m = labeled1 & ~np.isnan(p_spn)
    y = (df["stage1"].to_numpy()[m] == "SPN").astype(int)
    p = p_spn[m]
    rule = hard_rule_flag(df, cfg)[m]
    pred = np.where((p < lo) | rule, 0, np.where(p >= hi, 1, -1))
    cov = pred >= 0
    rep["stage1"] = {
        "n": int(m.sum()), "n_SPN": int(y.sum()), "n_nonSPN": int((1 - y).sum()),
        "coverage": float(cov.mean()) if m.any() else None,
        "balanced_acc_confident": (float(balanced_accuracy_score(y[cov], pred[cov]))
                                   if cov.sum() and len(np.unique(y[cov])) > 1 else None),
        "acc_confident": float((y[cov] == pred[cov]).mean()) if cov.sum() else None,
        "confusion_confident[true nonSPN,SPN][pred]": (
            confusion_matrix(y[cov], pred[cov], labels=[0, 1]).tolist() if cov.sum() else None),
    }

    # stage 2
    m2 = ((df["stage1"] == "SPN") & df["stage2"].isin(["red+", "red-"])).to_numpy() & ~np.isnan(p_red)
    y2 = (df["stage2"].to_numpy()[m2] == "red+").astype(int)
    pr = p_red[m2]
    pred2 = np.where(pr > th["p_red_hi"], 1, np.where(pr < th["p_red_lo"], 0, -1))
    cov2 = pred2 >= 0
    s2 = {"n": int(m2.sum()), "n_red+": int(y2.sum()), "n_red-": int((1 - y2).sum()),
          "coverage": float(cov2.mean()) if m2.any() else None,
          "balanced_acc_confident": (float(balanced_accuracy_score(y2[cov2], pred2[cov2]))
                                     if cov2.sum() and len(np.unique(y2[cov2])) > 1 else None),
          "balanced_acc_all_at_0.5": (float(balanced_accuracy_score(y2, (pr > 0.5).astype(int)))
                                      if len(np.unique(y2)) > 1 else None)}
    s2p = df["s2p_redprob"].to_numpy()[m2]
    if np.isfinite(s2p).all() and len(np.unique(y2)) > 1:
        s2["suite2p_redcell_balanced_acc"] = float(balanced_accuracy_score(
            y2, (s2p > cfg["suite2p"]["chan2_threshold"]).astype(int)))
    rep["stage2"] = s2
    cv = df[["session_id", "roi_idx", "stage1", "stage2"]].copy()
    cv["p_spn_cv"], cv["p_red_cv"] = p_spn, p_red
    return rep, cv


# ---------------------------------------------------------------- train / predict
def train(cfg):
    df = training_table(cfg)
    print(f"training table: {len(df)} ROIs from {df['session_id'].nunique()} session(s), "
          f"{df['animal'].nunique()} animal(s)")
    print("cross-validating ...")
    rep, cv = evaluate(df, cfg)
    print("fitting final model on all labels ...")
    bundle = fit_models(df, cfg)
    if bundle["mode1"] != rep["stage1_mode"]:
        rep["note"] = (f"CV folds used stage-1 mode '{rep['stage1_mode']}' (fewer labels per "
                       f"fold); the final model uses '{bundle['mode1']}'")
    rep["final_stage1_mode"] = bundle["mode1"]
    bundle.update(trained_on=sorted(df["session_id"].unique().tolist()),
                  created=datetime.now().isoformat(timespec="seconds"), cv_report=rep)
    mdir = Path(cfg["model_dir"]); mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, mdir / MODEL_FILE)
    with open(mdir / "cv_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    cv.to_csv(mdir / "cv_predictions.csv", index=False)
    print(json.dumps(rep, indent=2))
    print(f"model saved -> {mdir / MODEL_FILE}")
    return bundle


def load_model(cfg):
    path = Path(cfg["model_dir"]) / MODEL_FILE
    if not path.is_file():
        return None
    b = joblib.load(path)
    if b["feature_version"] != cfg["feature_version"]:
        raise RuntimeError(f"model was trained on features {b['feature_version']}, "
                           f"config says {cfg['feature_version']}: retrain")
    return b


def predict(sessions, cfg, bundle=None):
    bundle = bundle or load_model(cfg)
    if bundle is None:
        raise RuntimeError(f"no model in {cfg['model_dir']}: label + train first")
    for sess in sessions:
        try:
            df = ft.session_features(sess, cfg)
        except Exception as e:
            print(f"  ! {sess}: {e}")
            continue
        p_spn, p_red = predict_probs(bundle, df)
        lab, rule = assign_labels(df, p_spn, p_red, bundle, cfg["use_iscell"])
        out = pd.DataFrame({
            "roi_idx": df["roi_idx"], "roi_idx_matlab": df["roi_idx"] + 1,
            "iscell": df["iscell"], "p_spn": p_spn, "p_red": p_red,
            "hard_rule": rule, "final_label": lab,
        })
        od = io.out_dir(sess)
        out.to_csv(od / "labels.csv", index=False)
        savemat(od / "labels.mat", {"celltype": {
            "roi_idx": out["roi_idx_matlab"].to_numpy(float),
            "p_spn": p_spn, "p_red": p_red,
            "final_label": np.array(lab.tolist(), dtype=object),
            "cre_line": bundle["cre_line"], "model_created": bundle["created"]}})
        counts = out.loc[out["final_label"] != "not_cell", "final_label"].value_counts()
        n = counts.sum()
        print(f"  {io.session_id(sess, cfg['session_id_depth'])}: "
              + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in counts.items()))
        try:
            qc_figure(sess, df, out, od / "labels_qc.png")
        except Exception as e:
            print(f"    (QC figure failed: {e})")


COLORS = {"dSPN": "#e0457b", "iSPN": "#2ca25f", "SPN_ambiguous": "#f0c330",
          "nonSPN": "#3b8bd9", "uncertain": "#bbbbbb", "not_cell": "#555555"}


def qc_figure(sess, df, out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    plane = io.load_plane(sess)
    green, red = io.load_images(sess, plane)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax[0].imshow(green, cmap="gray", vmin=np.percentile(green, 1), vmax=np.percentile(green, 99.5))
    for s, lab in zip(plane["stat"], out["final_label"]):
        y0, x0 = s["ypix"].min() - 1, s["xpix"].min() - 1
        m = np.zeros((s["ypix"].max() - y0 + 2, s["xpix"].max() - x0 + 2))
        m[s["ypix"] - y0, s["xpix"] - x0] = 1
        for c in find_contours(m, 0.5):
            ax[0].plot(c[:, 1] + x0, c[:, 0] + y0, color=COLORS[lab], lw=0.8)
    ax[0].set_title(io.session_id(sess)); ax[0].axis("off")
    cells = out["final_label"] != "not_cell"
    ax[1].hist(out.loc[cells, "p_spn"], bins=30, color="#3b8bd9"); ax[1].set_title("p(SPN)")
    ax[2].hist(out.loc[cells & (out["p_spn"] > 0.5), "p_red"], bins=30, color="#e0457b")
    ax[2].set_title("p(red+) | SPN")
    handles = [plt.Line2D([], [], color=c, lw=3, label=f"{k} ({(out['final_label'] == k).sum()})")
               for k, c in COLORS.items()]
    ax[0].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
