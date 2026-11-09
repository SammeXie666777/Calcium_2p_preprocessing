"""Predicted labels vs hand labels, and the adjusted cell table.

No GUI and no model code in here (so it can be tested alone). The labeling GUI uses it to

  * put your hand labels and the model's predictions in ONE vocabulary
        hand  (SPN, red+)  <->  predicted dSPN  (D1-Cre; iSPN for D2-Cre)
        hand  (SPN, red-)  <->  predicted iSPN  (dSPN for D2-Cre)
        hand  (SPN, ambiguous) <-> SPN_ambiguous,   interneuron / junk <-> nonSPN
  * say whether the two agree, disagree, or cannot be compared
  * build the per-ROI "adjusted cell table": your label where you gave one, otherwise the
    prediction, plus where each label came from and whether it is in the training set.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# classes written by models.predict (final_label) and their colours (same as models.COLORS)
PRED_CLASSES = ["dSPN", "iSPN", "SPN_ambiguous", "nonSPN", "uncertain", "not_cell"]
PRED_HEX = {"dSPN": "#e0457b", "iSPN": "#2ca25f", "SPN_ambiguous": "#eec12b",
            "nonSPN": "#3b8bd9", "uncertain": "#bbbbbb", "not_cell": "#555555"}
# a prediction that says something about the cell (uncertain / not_cell are "no call")
DEFINITE = ("dSPN", "iSPN", "SPN_ambiguous", "nonSPN")
SPN_CLASSES = ("dSPN", "iSPN", "SPN_ambiguous")


def red_names(cre_line):
    """(class for red+, class for red-): red+ is dSPN in D1-Cre mice, iSPN in D2-Cre."""
    return ("dSPN", "iSPN") if str(cre_line).upper() == "D1" else ("iSPN", "dSPN")


def hand_to_class(st1, st2, cre_line):
    """Hand label (stage 1, stage 2) in the vocabulary of the predictions.
    '' = no usable label (nothing set, or 'unsure'); 'SPN' = SPN without a red call yet."""
    red, nored = red_names(cre_line)
    if st1 == "SPN":
        return {"red+": red, "red-": nored, "ambiguous": "SPN_ambiguous"}.get(st2, "SPN")
    if st1 in ("interneuron", "junk"):
        return "nonSPN"
    return ""


def class_to_hand(pred, cre_line):
    """(stage1, stage2) that reproduces a predicted class, or None when a single key cannot
    say it: nonSPN could be interneuron or junk (you decide), uncertain / not_cell are no call."""
    red, nored = red_names(cre_line)
    if pred == red:
        return "SPN", "red+"
    if pred == nored:
        return "SPN", "red-"
    if pred == "SPN_ambiguous":
        return "SPN", "ambiguous"
    return None


def compare(hand_class, pred):
    """'agree' / 'disagree', or '' when one side has no definite call."""
    if not hand_class or pred not in DEFINITE:
        return ""
    if hand_class == "SPN":                       # SPN without a red call: only stage 1 can be compared
        return "agree" if pred in SPN_CLASSES else "disagree"
    return "agree" if hand_class == pred else "disagree"


def load_predictions(path, plane_hash=None, n_roi=None):
    """labels.csv written by models.predict -> (DataFrame indexed by roi_idx, message).
    (None, reason) if there is no usable file: missing, made for another Suite2p run, or
    a different number of ROIs."""
    path = Path(path)
    if not path.is_file():
        return None, f"no {path.name} in {path.parent}"
    df = pd.read_csv(path, dtype={"s2p_hash": str})
    if not {"roi_idx", "final_label"} <= set(df.columns):
        return None, f"{path.name} has no roi_idx / final_label columns"
    if "s2p_hash" in df.columns and plane_hash is not None and \
            str(df["s2p_hash"].iloc[0]) != str(plane_hash):
        return None, f"{path.name} was made for another Suite2p run (rerun predict)"
    if n_roi is not None and len(df) != n_roi:
        return None, f"{path.name} has {len(df)} ROIs, the session has {n_roi} (rerun predict)"
    df["final_label"] = df["final_label"].fillna("")
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    note = "" if "s2p_hash" in df.columns else " (no Suite2p hash in it: cannot check it is current)"
    return df.set_index("roi_idx"), f"{path.name} from {when}{note}"


def adjusted_table(n_roi, n_orig, deleted, hand, hand_src, pred, cre_line, train_sources):
    """One row per ROI (Suite2p ROIs, then manual ones), in Suite2p index order.

    hand      {roi_idx: [stage1, stage2]}      your labels for this Suite2p run
    hand_src  {roi_idx: 'hand' | 'accepted' | 'bulk'}
    pred      DataFrame indexed by roi_idx with final_label, p_spn, p_red (or None)
    deleted   set of deleted roi_idx
    train_sources  label sources that go into the training set

    final_label: your label where you gave one ('SPN' if you gave no red call; 'uncertain' for
    'unsure'), else the prediction. final_source says which ('hand' / 'model' / '').
    used_in_training: your label is in the training set (source in train_sources, stage 1 set)."""
    rows = []
    for i in range(n_roi):
        st1, st2 = hand.get(i, ["", ""])
        src = hand_src.get(i, "") if (st1 or st2) else ""
        hcls = hand_to_class(st1, st2, cre_line)
        has_pred = pred is not None and i in pred.index
        plab = str(pred.at[i, "final_label"]) if has_pred else ""
        ps = float(pred.at[i, "p_spn"]) if has_pred and "p_spn" in pred else np.nan
        pr = float(pred.at[i, "p_red"]) if has_pred and "p_red" in pred else np.nan
        if i in deleted:
            final, fsrc = "deleted", ""
        elif hcls:
            final, fsrc = hcls, "hand"
        elif st1 == "unsure":
            final, fsrc = "uncertain", "hand"
        elif plab:
            final, fsrc = plab, "model"
        else:
            final, fsrc = "", ""
        rows.append({
            "roi_idx": i, "roi_idx_matlab": i + 1, "manual": int(i >= n_orig), "deleted": int(i in deleted),
            "hand_stage1": st1, "hand_stage2": st2, "hand_source": src,
            "pred_label": plab, "p_spn": ps, "p_red": pr,
            "agreement": compare(hcls, plab) if has_pred else "",
            "final_label": final, "final_source": fsrc,
            "used_in_training": int(bool(st1) and src in train_sources and i not in deleted),
        })
    return pd.DataFrame(rows)


def save_adjusted(table, folder, cre_line="", model_created=""):
    """labels_adjusted.csv and labels_adjusted.mat (struct 'celltype', 1-based roi_idx like
    labels.mat) in the session's celltype folder. Overwrites: it is regenerated from the
    label store and the predictions every time. Returns the two paths."""
    from scipy.io import savemat
    folder = Path(folder)
    csv, mat = folder / "labels_adjusted.csv", folder / "labels_adjusted.mat"
    table.to_csv(csv, index=False)
    cell = lambda col: np.array(table[col].astype(str).tolist(), dtype=object)
    savemat(mat, {"celltype": {
        "roi_idx": table["roi_idx_matlab"].to_numpy(float),
        "final_label": cell("final_label"), "final_source": cell("final_source"),
        "pred_label": cell("pred_label"), "hand_stage1": cell("hand_stage1"),
        "hand_stage2": cell("hand_stage2"), "hand_source": cell("hand_source"),
        "p_spn": table["p_spn"].to_numpy(float), "p_red": table["p_red"].to_numpy(float),
        "deleted": table["deleted"].to_numpy(float), "manual": table["manual"].to_numpy(float),
        "used_in_training": table["used_in_training"].to_numpy(float),
        "cre_line": str(cre_line), "model_created": str(model_created)}}, do_compression=True)
    return csv, mat
