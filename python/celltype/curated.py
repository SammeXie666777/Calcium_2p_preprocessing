"""File names, model ids and the review state of a session's celltype folder, and the edited
(curated) file written by the GUI's Save button. No file here holds traces: they stay in the Suite2p
Fall.mat, and every row carries the index that finds its trace there.

Per session, in <session>/celltype/ (nothing goes into the suite2p folder):

    celltype_model_<id>.csv / .mat    what the model <id> predicted (no traces): one row per Suite2p
                                      ROI, in Suite2p order (row k = stat{k} of Fall.mat), with
                                      s2p_hash, model_id, p_spn, p_red, final_label ...  Written by
                                      the batch prediction; never changed afterwards.
    celltype_curated_<id>.mat / .csv  the edited version, written by the GUI's Save, built on the
                                      prediction of model <id>: the struct 'celltype' (every column of the
                                      table, one row per kept ROI) + index arrays tying each row to Fall.mat
                                      (fall_row, orig_index, deleted_roi_idx, the pixels of manual ROIs in
                                      manual_stat). Overwritten by later Saves of the same <id>.
    celltype_status.json              summary of the two above (state, latest model id, edited ids)

<id> is the training time of the model (20261021-1432). The prediction files and the edited
files of an older model stay; the latest model's id is the newest one. Every file carries the
Suite2p hash (s2p_hash, the same one in the edited .mat), so it can be matched to the Fall.mat of
the run it belongs to; its `fall_file` field gives that path.

The edited file is rebuilt from sources that are never touched: the Suite2p outputs, the label
store (your labels), roi_edits.json (deleted ROIs), manual_rois.pkl (manual ROIs) and the
predictions. So your edits carry over to the prediction of a newer model.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils as io
from . import review

MODEL_CSV = "celltype_model_{id}.csv"
MODEL_MAT = "celltype_model_{id}.mat"
MODEL_QC = "celltype_model_{id}_qc.png"
CURATED_MAT = "celltype_curated_{id}.mat"
CURATED_CSV = "celltype_curated_{id}.csv"
STATUS_JSON = "celltype_status.json"
EDITS_JSON = "roi_edits.json"          # deleted ROI indices, per Suite2p run (written by the GUI)
MANUAL_PKL = "manual_rois.pkl"         # manual ROIs, per Suite2p run (written by the GUI)


def kept_table(adjusted, keep, n_orig):
    """Rows of the adjusted table (review.adjusted_table) for the kept ROIs, with `row` (the
    position in every array of the .mat, 1-based) and `orig_index` (-1 = manual) added."""
    tab = adjusted.set_index("roi_idx").loc[list(keep)].reset_index()
    tab.insert(0, "row", np.arange(1, len(tab) + 1))
    tab.insert(3, "orig_index", [i if i < n_orig else -1 for i in tab["roi_idx"]])
    # the row of Fall.mat (1-based: F(fall_row, :), stat{fall_row}, iscell(fall_row, :)) of a Suite2p ROI;
    # NaN for a manual ROI, which is not in Fall.mat (manual_id counts them, their pixels are in the .mat)
    tab.insert(4, "fall_row", [i + 1 if i < n_orig else np.nan for i in tab["roi_idx"]])
    tab.insert(5, "manual_id", np.cumsum(tab["orig_index"].to_numpy() < 0) * (tab["orig_index"].to_numpy() < 0))
    return tab


def light_arrays(stat, n_orig, keep, deleted, s2p_hash, fall_file):
    """What the edited .mat holds besides the cell-type struct. No traces and no Suite2p stat (they are
    in Fall.mat): a Suite2p ROI is found by its `fall_row` / `orig_index`, an added (manual) ROI, which is
    not in Fall.mat, by its pixels in `manual_stat` (ypix, xpix 0-based, lam; one entry per manual ROI in
    the order of `manual_id`), from which its trace can be extracted again.
      orig_index       per kept ROI: 0-based Suite2p ROI index, -1 for a manual ROI
      is_manual        per kept ROI
      deleted_roi_idx  Suite2p ROI indices that were deleted (left out of the table)
      s2p_hash, fall_file   the Suite2p run and the Fall.mat these indices belong to"""
    manual = [i for i in keep if i >= n_orig]
    ms = np.empty(len(manual), dtype=object)
    for k, i in enumerate(manual):
        s = stat[i]
        ms[k] = {"manual_id": float(k + 1), "roi_idx": float(i),
                 "ypix": np.asarray(s["ypix"], float), "xpix": np.asarray(s["xpix"], float),
                 "lam": np.asarray(s.get("lam", np.ones(len(s["ypix"]))), float)}
    return {"orig_index": np.array([i if i < n_orig else -1 for i in keep], float),
            "is_manual": np.array([i >= n_orig for i in keep], np.uint8),
            "deleted_roi_idx": np.array(sorted(deleted), float), "manual_stat": ms,
            "s2p_hash": str(s2p_hash), "fall_file": str(fall_file)}


def celltype_struct(tab, keep, cre_line, deleted, model_created="", model_id="", s2p_hash="",
                    fall_file=""):
    """{'celltype': struct, 'deleted_roi_idx': ...} for savemat: every column of the table as a
    field (text columns become cell arrays), row-aligned with the table."""
    assert list(tab["roi_idx"]) == list(keep), "table and curated ROI set are not the same ROIs"
    d = {}
    for c in tab.columns:
        if pd.api.types.is_numeric_dtype(tab[c]):
            d[c] = tab[c].to_numpy(float)
        else:
            d[c] = np.array(tab[c].astype(str).tolist(), dtype=object)
    d["cre_line"] = str(cre_line)
    d["model_created"] = str(model_created or "")
    d["model_id"] = str(model_id or "")
    d["s2p_hash"] = str(s2p_hash or "")
    d["fall_file"] = str(fall_file or "")
    d["saved"] = pd.Timestamp.now().isoformat(timespec="seconds")
    return {"celltype": d, "deleted_roi_idx": np.array(sorted(deleted), float)}


# ---------------------------------------------------------------- model ids and file names
def model_id(created):
    """Training time of a model (bundle['created'], '2026-10-21T14:32:05') -> '20261021-1432'."""
    try:
        return pd.Timestamp(str(created)).strftime("%Y%m%d-%H%M")
    except Exception:
        return "noid"


def model_files(folder, mid):
    """(csv, mat, qc figure) of the prediction of model mid."""
    folder = Path(folder)
    return (folder / MODEL_CSV.format(id=mid), folder / MODEL_MAT.format(id=mid),
            folder / MODEL_QC.format(id=mid))


def curated_files(folder, mid):
    """(mat, csv) of the edited file built on the prediction of model mid."""
    folder = Path(folder)
    return folder / CURATED_MAT.format(id=mid), folder / CURATED_CSV.format(id=mid)


_ID = re.compile(r"\d{8}-\d{4}|noid")


def _ids(folder, pattern, prefix):
    """Model ids found in file names like celltype_model_<id>.csv (oldest first)."""
    out = []
    for p in Path(folder).glob(pattern):
        mid = p.stem[len(prefix):]
        if _ID.fullmatch(mid):
            out.append(mid)
    return sorted(set(out), key=lambda m: "" if m == "noid" else m)


def model_file_ok(sess_dir, mid, plane_hash):
    """Is there a complete prediction of model mid for this Suite2p run (same s2p_hash)?"""
    csv, mat, _ = model_files(io.out_dir(sess_dir, create=False), mid)
    if not (csv.is_file() and mat.is_file()):
        return False
    try:
        h = pd.read_csv(csv, nrows=1, dtype={"s2p_hash": str})["s2p_hash"].iloc[0]
    except Exception:
        return False
    return str(h) == str(plane_hash)


def latest_model(sess_dir, plane_hash):
    """(id, csv path) of the newest model prediction that belongs to this Suite2p run,
    or (None, None)."""
    folder = io.out_dir(sess_dir, create=False)
    for mid in reversed(_ids(folder, "celltype_model_*.csv", "celltype_model_")):
        if model_file_ok(sess_dir, mid, plane_hash):
            return mid, model_files(folder, mid)[0]
    return None, None


def session_state(sess_dir):
    """Where a session stands (read from the files, nothing is created):

      no_prediction  no prediction of any model for this Suite2p run (or a Suite2p rerun made the
                     old ones stale): run the batch prediction
      pending        the newest model has predicted it, nothing saved yet
      stale          like pending, but an edited file exists for an OLDER model: review again
                     (the edits carry over: hand labels, manual and deleted ROIs are reused)
      reviewed       an edited file exists for the newest model's prediction
      no_suite2p     no Suite2p output found

    Returns a dict: state, latest_id, edited_ids, hash."""
    try:
        h = io.s2p_hash(sess_dir)
    except Exception as e:
        return {"state": "no_suite2p", "latest_id": None, "edited_ids": [], "hash": None,
                "note": str(e)}
    folder = io.out_dir(sess_dir, create=False)
    mid, _ = latest_model(sess_dir, h)
    edited = [m for m in _ids(folder, "celltype_curated_*.csv", "celltype_curated_")
              if all(p.is_file() for p in curated_files(folder, m))]
    if mid is None:
        state = "no_prediction"
    elif mid in edited:
        state = "reviewed"
    elif edited:
        state = "stale"
    else:
        state = "pending"
    return {"state": state, "latest_id": mid, "edited_ids": edited, "hash": h}


def write_status(sess_dir):
    """celltype_status.json: a summary of session_state for people and scripts to read. The files
    themselves are the source of truth; this is rewritten after every prediction and Save."""
    st = session_state(sess_dir)
    folder = io.out_dir(sess_dir)
    mid = st["latest_id"]
    rec = {"state": st["state"], "s2p_hash": st["hash"], "latest_model_id": mid,
           "model_file": model_files(folder, mid)[0].name if mid else None,
           "edited_model_ids": st["edited_ids"],
           "edited_file_for_latest": curated_files(folder, mid)[1].name if mid in st["edited_ids"] else None,
           "fall_file": str(io.plane_dir(sess_dir) / "Fall.mat"),
           "updated": pd.Timestamp.now().isoformat(timespec="seconds")}
    (folder / STATUS_JSON).write_text(json.dumps(rec, indent=1))
    return rec


def write_model_file(sess_dir, out, bundle):
    """celltype_model_<id>.csv / .mat for a session: the predictions of `bundle` (a DataFrame with
    roi_idx, roi_idx_matlab, iscell, p_spn, p_red, hard_rule, final_label, s2p_hash), no traces.
    Rows are in Suite2p order, row k = stat{k} of Fall.mat. Written under temporary names and
    swapped in together. Returns (csv, mat, model id)."""
    from scipy.io import savemat
    folder = io.out_dir(sess_dir)
    mid = bundle.get("model_id") or model_id(bundle["created"])
    csv, mat, _ = model_files(folder, mid)
    out = out.copy()
    out["model_id"] = mid
    plane_dir = io.plane_dir(sess_dir)
    struct = {"roi_idx": out["roi_idx_matlab"].to_numpy(float),       # 1-based, = row of Fall.mat
              "roi_idx_0based": out["roi_idx"].to_numpy(float),
              "iscell": out["iscell"].to_numpy(float),
              "p_spn": out["p_spn"].to_numpy(float), "p_red": out["p_red"].to_numpy(float),
              "hard_rule": out["hard_rule"].to_numpy(float),
              "edge_roi": (out["edge_roi"].to_numpy(float) if "edge_roi" in out else np.zeros(len(out))),
              "final_label": np.array(out["final_label"].astype(str).tolist(), dtype=object),
              "s2p_hash": str(out["s2p_hash"].iloc[0]), "model_id": mid,
              "model_created": str(bundle["created"]), "cre_line": str(bundle["cre_line"]),
              "n_rois": float(len(out)),
              "fall_file": str(plane_dir / "Fall.mat"), "suite2p_plane_dir": str(plane_dir)}
    tmp_csv = folder / f"celltype_model_{mid}_tmp.csv"
    tmp_mat = folder / f"celltype_model_{mid}_tmp.mat"
    try:
        out.to_csv(tmp_csv, index=False)
        savemat(tmp_mat, {"celltype": struct}, do_compression=True)
        os.replace(tmp_csv, csv)
        os.replace(tmp_mat, mat)
    except Exception:
        for t in (tmp_csv, tmp_mat):
            try:
                t.unlink()
            except OSError:
                pass
        raise
    return csv, mat, mid
