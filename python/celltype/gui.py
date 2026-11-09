"""napari labeling tool for building / extending the training set.

Keys (Qt shortcuts: they work wherever you last clicked, except while typing in a number or text
box; while a manual ROI is being drawn, d / z / s ... go to napari's shape layer instead):
    stage 1   s = SPN   i = interneuron   j = junk (not a soma)   u = unsure
    stage 2   r = red+  g = red-  a = ambiguous      (sets stage 1 = SPN if empty)
    x = clear this ROI      n / b = next / previous ROI in the queue
    Right / Left arrow = next / previous ROI in the order of the 'order' menu, without labeling
        anything (wraps around at the ends); e.g. order "p(red): most uncertain first" and look
        through the predictions with the arrows
    m = compute the last manual ROI drawn (button 'draw manual ROI' to start)
    q = stop drawing, go back to clicking existing ROIs
    p = show the side panel again if it was closed (also Window menu)
    d = delete this ROI (hidden; kept in memory + roi_edits.json)   z = undo last delete
    y = accept the predicted label for this ROI (dSPN / iSPN / SPN_ambiguous; for nonSPN
        press i or j to say which kind)
    'Save' (top of the panel) writes celltype_curated_<model id>.csv / .mat into the session's
        celltype folder (overwriting the previous Save of the same model id, never the model's
        own celltype_model_<id> prediction; see save_celltype); nothing goes into suite2p
    'next in queue' loads the next session that has a prediction of the newest model and no Save
        yet (sessions with only an older model's edits come after those)
Click an ROI in the image to jump to it. Every label is appended to the label
store immediately, so closing the window never loses work.

Predicted vs your labels ('show' menu): "my labels" = outline colour as before, "predicted" =
the model's class, "both" = fill is the predicted class and the outline is your label
(a yellow outline on a pink fill is an SPN you called red+ that the model calls dSPN, etc.).
'disagreements only' hides ROIs where the two agree or one has no call; the order menu can
list disagreements first. The predictions are read from the newest
<session>/celltype/celltype_model_<id>.csv (made by the batch prediction); without one the old
labels.csv is used, and without that, if a model exists, they are computed live.
Left dock 'view display': the green layer can show the mean image or a max / std projection of the
movie (a subsample of evenly spaced frames; napari's own 'projection mode' control only acts on
3-D data, so it does nothing on these 2-D images), and 'auto-contrast view' stretches the green /
red layers to the percentiles of what is in view (zoom in first), with an optional auto gamma and
a 'follow zoom / pan' switch that redoes it whenever you move.
'crops under the view' (dock menu): the zoomed crops of the selected ROI show the event snapshot of the ROI (the movie around its own
strongest events minus quiet frames, auto contrast / gamma; from celltype/snapshots.npz, instant) instead of, or next to, the green mean image.
Editing a predicted session: y / bulk-accept / retype labels, then
  'Save'                  writes <session>/celltype/celltype_curated_<model id>.csv and .mat
                          (overwritten by the next Save of the same model id; <model id> is the id of
                          the prediction being edited): the struct 'celltype' with every column of the
                          cell-type table (your label where you gave one, else the prediction, with the
                          source of each and whether it is in the training set), plus the index arrays that
                          match every row to the Suite2p Fall.mat (fall_row, orig_index, deleted ROIs, the
                          pixels of manual ROIs). No traces: they stay in Fall.mat. The history (label store, roi_edits.json,
                          manual_rois.pkl, celltype_model_<id>.* of every model) is left as it is.
  'Retrain'               trains on the whole label store (hand + accepted labels) and writes this
                          session's celltype_model_<new id>.* with the new model (older ones stay).
Label sources in the store: hand (typed) and accepted (y) are training labels; bulk ('accept
all confident') is not, unless train_on_bulk: true in the config.

display: dF/F trace (full session) and most active 60 s, plus zoomed-in green / red /overlap images with ROI contour. The ROI queue can be ordered by:
    unlabeled first (random), most uncertain first (needs model), disagreements first, p(red) or p(SPN) most uncertain / most certain first (p(red) also high to low; for p(red) the ROIs predicted to be SPNs come first), largest first, highest baseline first, most red first, or ROI index.
    Also display velocity and position traces for the session if available.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from . import features as ft
from . import manual_roi as mr
from . import io_utils as io
from . import models
from . import review
from . import curated

BEHAV_MAT = "output_behaviorConcat.mat"
EDITS_JSON = "roi_edits.json"          # deleted ROI indices, per Suite2p run
MANUAL_EDGE = (0.7, 0.2, 1.0, 1.0)     # purple: manually drawn ROIs
BALL = ("ballPitch", "ballYaw", "ballRoll")
BALL_SPEED = "ball speed (|pitch|+|yaw|+|roll|)"
NONE = "(none)"
VEL_GUESS = ("rotaryEncoderVelocity", BALL_SPEED, "velocity", "speed")
LICK_GUESS = ("lick", "licks", "lickSensor")
EDGE = {"": (0.85, 0.85, 0.85, 0.45), "SPN": (1.0, 0.85, 0.1, 1.0),
        "red+": (1.0, 0.25, 0.75, 1.0), "red-": (0.2, 1.0, 0.35, 1.0),
        "ambiguous": (1.0, 0.6, 0.1, 1.0), "interneuron": (0.2, 0.75, 1.0, 1.0),
        "junk": (0.6, 0.1, 0.1, 1.0), "unsure": (1.0, 0.55, 0.0, 1.0),
        "not_cell": (0.4, 0.4, 0.4, 0.35)}
ORDERS = ["unlabeled first (random)", "most uncertain first (needs model)",
          "disagreements first (mine vs predicted)",
          "p(red): most uncertain first", "p(red): most certain first", "p(red): high to low",
          "p(SPN): most uncertain first", "p(SPN): most certain first",
          "largest first", "highest baseline first", "most red first", "ROI index"]
# what the green layer shows (left dock, 'green layer shows')
PROJECTIONS = ["mean image", "max projection", "std projection"]
CROP_MODES = ["snapshot / red / overlap", "green / red / overlap", "green / snapshot / red / overlap"]
SNAP_MODES = ["strongest event", "average of the top events"]
# what the outlines / fills show: my labels, the predicted classes, or both overlaid
VIEWS = ["my labels", "predicted", "both: fill = predicted, outline = mine", "none"]
VIEW_MINE, VIEW_PRED, VIEW_BOTH, VIEW_NONE = range(4)
PRED_FILL_ALPHA = 0.42


find_sessions = io.find_sessions   # kept under its old name


def _needs_session(fn):
    """Handlers that use the current session do nothing (with a hint) until one
    has been opened. Qt passes extra arguments to slots (a button's clicked sends
    'checked'): they are dropped when the handler does not accept them, as a plain
    bound method would be."""
    import inspect
    ps = list(inspect.signature(fn).parameters.values())
    var_pos = any(p.kind is p.VAR_POSITIONAL for p in ps)
    n_pos = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps) - 1  # minus self

    def wrapper(self, *a, **k):
        if getattr(self, "S", None) is None:
            self.viewer.status = "open a session first ('open session...')"
            return None
        if not var_pos:
            a = a[:n_pos]
        return fn(self, *a, **k)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def _contour(s):
    from skimage.measure import find_contours
    y0, x0 = s["ypix"].min() - 1, s["xpix"].min() - 1
    m = np.zeros((s["ypix"].max() - y0 + 2, s["xpix"].max() - x0 + 2))
    m[s["ypix"] - y0, s["xpix"] - x0] = 1
    cs = find_contours(m, 0.5)
    c = max(cs, key=len) if cs else np.array([[0, 0], [0, 1], [1, 1]], float)
    return c + [y0, x0]




class Session:
    """Everything the GUI needs for one session, loaded once."""
   # __init__ (double underscore on both sides) called automatically when an object of the class is created.
    def __init__(self, sess_dir, cfg, bundle):
        self.dir = sess_dir
        self.sid = io.session_id(sess_dir, cfg["session_id_depth"])
        self.plane = io.load_plane(sess_dir)
        self.stat = self.plane["stat"]
        self.green, self.red = io.load_images(sess_dir, self.plane)
        self.index_img = io.roi_masks(self.stat, self.green.shape)
        self.feat = ft.session_features(sess_dir, cfg)
        self.dff, _ = ft.compute_dff(self.plane, cfg)
        self.contours = [_contour(s) for s in self.stat]
        self._cfg = cfg
        self.snaps = ft.load_snapshots(sess_dir, self.plane["hash"])       # celltype/snapshots.npz, made with the features
        self._snap_ctx, self._snap_memo = None, {}
        self.p_spn = self.p_red = None
        if bundle is not None:
            self.p_spn, self.p_red = models.predict_probs(bundle, self.feat)
        self.behav, self.behav_file = {}, None
        self.win_cache = mr.WindowCache()        # manual-ROI pixels kept in RAM
        # --- edits: manual ROIs are appended after the Suite2p ROIs (index n_orig + k)
        self.n_orig = len(self.stat)
        self.extra = []                          # manual ROI results, in index order
        for res in mr.load_manual_rois(self._manual_path(), self.plane["hash"]):
            self.add_manual(res)
        self.deleted, self.deleted_order = set(), []
        ep = self._edits_path()
        if ep.is_file():
            try:
                d = json.loads(ep.read_text())
                if d.get("s2p_hash") == self.plane["hash"]:
                    self.deleted_order = [int(i) for i in d.get("deleted", [])
                                          if int(i) < len(self.stat)]
                    self.deleted = set(self.deleted_order)
            except Exception as e:
                print(f"  (could not read {ep.name}: {e})")
        if self.deleted:
            self.rebuild_index()
        default = io.image_dir(sess_dir) / BEHAV_MAT
        if default.is_file():
            try:
                self.load_behavior(default)
            except Exception as e:
                print(f"  (behavior not loaded for {sess_dir}: {e})")

    # ---- edits (manual ROIs, deletions)
    def _manual_path(self):
        return io.out_dir(self.dir) / "manual_rois.pkl"

    def _edits_path(self):
        return io.out_dir(self.dir) / EDITS_JSON

    def add_manual(self, res):
        """Append a manual ROI (result of mr.manual_roi_traces) to every per-ROI
        structure so it behaves like a Suite2p ROI. Returns its ROI index."""
        i = len(self.stat)
        st = res["stat"]                       # full Suite2p stat entry (see manual_roi)
        st["manual"] = True
        ypix, xpix = st["ypix"], st["xpix"]
        self.stat = np.concatenate([self.stat, np.array([st], dtype=object)])
        self.index_img[ypix, xpix] = i
        self.contours.append(_contour(st))
        row = {c: np.nan for c in self.feat.columns}
        row.update(session_id=self.sid, session_dir=str(self.dir), animal=io.animal_of(self.sid),
                   roi_idx=i, s2p_hash=self.plane["hash"], iscell=int(res["iscell"][0]),
                   s2p_iscell_prob=float(res["iscell"][1]), s2p_redprob=float(res["redcell"][1]),
                   cy=float(st["med"][0]), cx=float(st["med"][1]))
        self.feat = pd.concat([self.feat, pd.DataFrame([row])], ignore_index=True)
        if self.p_spn is not None:
            self.p_spn, self.p_red = np.append(self.p_spn, np.nan), np.append(self.p_red, np.nan)
        self.extra.append(res)
        return i

    def rebuild_index(self):
        """index_img maps each pixel to one ROI (the top one where ROIs overlap).
        Rebuild it from the ROIs that are not deleted, later ROIs on top, so a
        pixel shared with a deleted ROI points to the ROI underneath again."""
        self.index_img[:] = -1
        for i, s in enumerate(self.stat):
            if i not in self.deleted:
                self.index_img[s["ypix"], s["xpix"]] = i

    def dff_of(self, i):
        return self.dff[i] if i < self.n_orig else self.extra[i - self.n_orig]["dff"]

    def _save_edits(self):
        self._edits_path().write_text(json.dumps(
            {"s2p_hash": self.plane["hash"], "deleted": [int(i) for i in self.deleted_order]}))

    def delete(self, i):
        if i not in self.deleted:
            self.deleted.add(int(i))
            self.deleted_order.append(int(i))
            self._save_edits()
            self.rebuild_index()

    def restore_last(self):
        """Undo the most recent delete; returns the ROI index or None."""
        if not self.deleted_order:
            return None
        i = self.deleted_order.pop()
        self.deleted.discard(i)
        self._save_edits()
        self.rebuild_index()
        return i

    def red_projection(self, kind, n_frames=400):
        """Same for the red layer, from the frames of the red reference tif. None when that tif is a
        single image (nothing to project): the red layer then keeps showing its mean."""
        if kind.startswith("mean"):
            return self.red
        key = ("red" + kind[:3], int(n_frames))
        cache = self.__dict__.setdefault("_proj", {})
        if key not in cache:
            stack = io.load_red_frames(self.dir, self.green, n_frames)
            cache[key] = None if stack is None else _project(stack, kind)
        return cache[key]

    def projection(self, kind, n_frames=400):
        """Image shown in the green layer: 'mean image' (the one loaded with the session), the 'max'
        projection (read from ConcatenatedMovie_MC_max_registered.tif in the session folder; without that
        file, the max of `n_frames` evenly spaced frames, a lower bound of the true max) or the 'std'
        projection (computed from those frames; it highlights the active pixels). self.proj_source says which."""
        if kind.startswith("mean"):
            return self.green
        key = (kind[:3], int(n_frames))
        cache = self.__dict__.setdefault("_proj", {})
        src = self.__dict__.setdefault("_proj_src", {})
        if key in cache:
            self.proj_source = src[key]
            return cache[key]
        if kind.startswith("max"):                          # the max projection that is in the session folder
            img = io.load_max_image(self.dir, self.green.shape)
            if img is not None:
                cache[key], src[key] = img, f"file {io.MAX_TIF}"
                self.proj_source = src[key]
                return img
        out = _project(self.get_movie(), kind, n_frames)
        if out.shape != self.green.shape:
            raise ValueError(f"movie frames are {out.shape} but the green image is {self.green.shape}")
        cache[key], src[key] = out.astype(np.float32), f"{int(n_frames)} frames of the movie"
        self.proj_source = src[key]
        return cache[key]

    def snapshot(self, i):
        """(z_avg, z_peak, (cy, cx)) of ROI i, the event snapshots in noise units (average of its top events / its strongest
        event) and the centre of their 32 x 32 window: read from celltype/snapshots.npz (instant), else computed for this ROI from
        the movie (once, then remembered). None for a manual ROI, a ROI with fewer than 3 events, or without a movie."""
        i = int(i)
        if i >= self.n_orig:
            return None
        z = self.snaps
        if z is not None and i in z["index"]:
            k = z["index"][i]
            return z["z_avg"][k], z["z_peak"][k], tuple(int(v) for v in z["center"][k])
        if i in self._snap_memo:
            return self._snap_memo[i]
        res = None
        try:
            if self._snap_ctx is None:
                self._snap_ctx = ft.SnapContext(self.plane, self._cfg, self.dir, dff=self.dff, movie=self.get_movie())
            o = self._snap_ctx.one(i)
            res = None if o["z"] is None else (o["z"], o["z_peak"], o["center"])
        except Exception as e:
            print(f"  (snapshot of ROI {i} not computed: {e})")
        self._snap_memo[i] = res
        return res

    def set_probs(self, bundle):
        """(Re)compute p(SPN) / p(red) of the Suite2p ROIs with `bundle` (manual ROIs get NaN)."""
        p1, p2 = models.predict_probs(bundle, self.feat.iloc[:self.n_orig])
        nm = len(self.stat) - self.n_orig
        self.p_spn = np.concatenate([p1, np.full(nm, np.nan)])
        self.p_red = np.concatenate([p2, np.full(nm, np.nan)])

    def kept(self):
        """ROI indices of a curated set: every ROI except the deleted ones, in index order
        (the Suite2p ROIs, then the manual ones)."""
        return [i for i in range(len(self.stat)) if i not in self.deleted]

    def curated_mat(self, path, celltype=None):
        """Write the edited file: the cell-type struct plus the index arrays that tie every row to the Suite2p
        Fall.mat (see curated.light_arrays). No traces and no Suite2p stat: those stay in Fall.mat, and a row
        finds its trace through `fall_row`. Manual ROIs, which are not in Fall.mat, carry their pixels.
        celltype: optional function keep -> dict of extra variables (the cell types).
        Returns (n_kept, n_manual, n_deleted)."""
        from scipy.io import savemat
        keep = self.kept()
        out = curated.light_arrays(self.stat, self.n_orig, keep, self.deleted, self.plane["hash"],
                                   io.plane_dir(self.dir) / "Fall.mat")
        if celltype is not None:
            out.update(celltype(keep))
        savemat(path, out, do_compression=True)
        return len(keep), sum(1 for i in keep if i >= self.n_orig), len(self.deleted)

    def get_movie(self):
        """Registered movie, opened on first use (a memmap, so cheap)."""
        if getattr(self, "movie", None) is None:
            self.movie = mr.open_movie(self.dir, self.plane)
        return self.movie

    def load_behavior(self, path):
        """Per-frame behavior fields from output_behaviorConcat.mat."""
        d = io.load_mat(path)
        d = d.get("concatenatedBehavior", d)
        T = self.dff.shape[1]
        vecs = {}
        for k, v in d.items():
            try:
                a = np.asarray(v)
                if a.ndim == 2 and 1 in a.shape:
                    a = a.ravel()
                if a.ndim == 1 and a.dtype.kind in "fiub" and a.size > 1:
                    vecs[k] = a.astype(np.float64)
            except Exception:
                pass
        if not vecs:
            raise ValueError("no per-frame numeric fields found")
        # per-frame fields share one length: the most common vector length
        lens = [a.size for a in vecs.values()]
        n = max(set(lens), key=lens.count)
        vecs = {k: a for k, a in vecs.items() if a.size == n}
        if n != T:
            vecs = {k: a[np.minimum(np.arange(T), n - 1)] for k, a in vecs.items()}
        if all(f in vecs for f in BALL):
            vecs[BALL_SPEED] = sum(np.abs(vecs[f]) for f in BALL)
        self.behav, self.behav_file = vecs, str(path)


class LabelTool:
    def __init__(self, sessions, cfg, model_id=None):
        import napari
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
                                    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox,
                                    QTabWidget, QVBoxLayout, QWidget)
        self._Figure, self._Canvas = Figure, FigureCanvasQTAgg

        class MsgLabel(QLabel):
            """Message line that takes no room while it is empty."""
            def __init__(self):
                super().__init__()
                self.setWordWrap(True)
                self.hide()

            def setText(self, text):
                super().setText(text)
                self.setVisible(bool(text))

        self.cfg, self.store = cfg, cfg["label_store"]
        if not self.store:
            raise ValueError("set label_store in the config")
        try:
            self.bundle = models.load_model(cfg, model_id)
            if self.bundle is not None:
                print(f"  model for live predictions: {self.bundle.get('model_id') or curated.model_id(self.bundle['created'])}")
        except Exception as e:
            print(f"  (model not used in GUI: {e})")
            self.bundle = None
        self.sessions = [str(s) for s in sessions]
        self.cache = {}
        self.S = None                 # current Session; None until one is opened
        self.viewer = napari.Viewer(title="celltype labeling")
        # These are 2-D images and 2-D ROI outlines. In napari's 3-D display mode (the 2D/3D
        # button under the layer controls) the outline layers cannot be updated any more
        # ("input array dimensions ... must match"), so the view is kept in 2-D.
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.events.ndisplay.connect(self._force_2d)
        try:
            self.viewer.window._qt_viewer.viewerButtons.ndisplayButton.setEnabled(False)
        except Exception:
            pass                                    # the button moved in this napari version

        w = QWidget(); lay = QVBoxLayout(w)
        self.sess_box = QComboBox(); self.sess_box.addItems(self.sessions)
        self.order_box = QComboBox(); self.order_box.addItems(ORDERS)
        # Suite2p-style classifier threshold: show ROIs with iscell probability >= value
        self.thr_spin = QDoubleSpinBox(); self.thr_spin.setRange(0, 1)
        self.thr_spin.setSingleStep(0.05); self.thr_spin.setDecimals(2)
        self.thr_spin.setKeyboardTracking(False)   # update on Enter / focus-out, not per keystroke
        thr0 = float(cfg.get("gui_iscell_threshold", cfg.get("iscell_prob_min") or 0.5))
        self.thr_spin.setValue(thr0)
        self.thr_count = QLabel()
        # ROI diameter range: only ROIs with min <= equivalent diameter <= max are shown
        # (equiv_diam_um of the feature table: um, i.e. px when um_per_px = 1)
        self.dmin_spin, self.dmax_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.dmin_spin, 0.0), (self.dmax_spin, 1000.0)):
            sp.setRange(0, 1000); sp.setDecimals(1); sp.setSingleStep(1.0)
            sp.setKeyboardTracking(False)           # update on Enter / focus-out
            sp.setValue(v)
            sp.setToolTip("ROIs whose equivalent diameter (um; px if um_per_px = 1) is "
                          "outside min..max are hidden. 0 and 1000 = no limit. "
                          "Manual ROIs are always shown.")
        self.open_btn = QPushButton("open session...")
        self.next_btn = QPushButton("next in queue")
        self.beh_btn = QPushButton("load behavior file...")
        self.beh_lbl = QLabel(); self.beh_lbl.setWordWrap(True)
        self.vel_box = QComboBox(); self.lick_box = QComboBox()
        self.draw_btn = QPushButton("draw manual ROI")
        self.calc_btn = QPushButton("compute manual ROI (m)")
        self.stop_btn = QPushButton("back to selecting ROIs (q)")
        self.man_lbl = MsgLabel()
        self.del_btn = QPushButton("delete this ROI (d)")
        self.undo_btn = QPushButton("undo last delete (z)")
        self.save_btn = QPushButton("Save")
        self.edit_lbl = MsgLabel()
        # predicted labels vs mine: what to show, how to accept / retrain / save
        self.view_box = QComboBox(); self.view_box.addItems(VIEWS)
        self.view_box.setCurrentIndex(VIEW_BOTH)
        self.diff_chk = QCheckBox("disagreements only")
        self.pred_btn = QPushButton("load predicted")
        self.accept_btn = QPushButton("accept prediction (y)")
        self.accept_all_btn = QPushButton("accept all confident...")
        self.retrain_btn = QPushButton("retrain + re-predict")
        self.pred_lbl = MsgLabel()
        self.legend = QLabel(); self.legend.setWordWrap(True)
        self.legend.setText(
            "fill = predicted: " + " ".join(
                f"<span style='color:{review.PRED_HEX[c]}'>&#9632;</span>{c}"
                for c in review.PRED_CLASSES)
            + "<br>outline = mine: "
            + " ".join(f"<span style='color:rgb({int(255 * EDGE[k][0])},{int(255 * EDGE[k][1])},"
                       f"{int(255 * EDGE[k][2])})'>&#9632;</span>{k}"
                       for k in ("SPN", "red+", "red-", "ambiguous", "interneuron", "junk", "unsure")))
        self.auto_adv = QCheckBox("auto-advance"); self.auto_adv.setChecked(True)
        # when you step through the ROIs while zoomed in, the view follows the selected ROI
        self.follow_roi_chk = QCheckBox("Follow ROI"); self.follow_roi_chk.setChecked(True)
        self.follow_roi_chk.setToolTip(
            "keep the selected ROI in view when you step through the ROIs (arrows, n / b, after a "
            "label). Only acts while you are zoomed in; zoomed out, nothing moves. Clicking an ROI "
            "in the image never moves the view.")
        self.min_fov = QSpinBox(); self.min_fov.setRange(40, 1000); self.min_fov.setValue(120)
        self.min_fov.setSuffix(" px"); self.min_fov.setKeyboardTracking(False)
        self.min_fov.setToolTip("Follow ROI never shows less than this many pixels on the shorter "
                                "side of the view: if you are zoomed in further, it zooms out to this.")
        self.info = QLabel(); self.info.setWordWrap(True)
        self.counts = QLabel(); self.counts.setWordWrap(True)
        self.fig = Figure(figsize=(5, 6)); self.canvas = FigureCanvasQTAgg(self.fig)
        # compact buttons: short text here, the full name and the key in the tooltip
        for btn, txt, tip in (
                (self.open_btn, "Open...", "open a session folder (or a folder above several sessions)"),
                (self.next_btn, "Next >", "load the next session in the list above that has a prediction "
                                          "of the newest model and no Save yet (sessions edited only "
                                          "under an older model come last)"),
                (self.beh_btn, "Behavior...", "load a behavior .mat file (output_behaviorConcat.mat)"),
                (self.draw_btn, "Draw ROI", "draw a manual ROI polygon (double-click or Esc to finish)"),
                (self.calc_btn, "Compute (m)", "compute the drawn manual ROI(s), key m"),
                (self.stop_btn, "Select (q)", "back to selecting ROIs by clicking, key q"),
                (self.del_btn, "Delete (d)", "delete this ROI, key d"),
                (self.undo_btn, "Undo (z)", "undo the last delete, key z"),
                (self.save_btn, "Save", "write celltype_curated_<model id>.csv / .mat in the session's "
                                        "celltype folder (the next Save of the same model id overwrites "
                                        "it; the model's own prediction file is never touched): the "
                                        "cell-type table with the row of Fall.mat of every ROI (no "
                                        "traces). Nothing is written to the suite2p folder."),
                (self.pred_btn, "Load predicted", "read the predicted labels of this session (the newest "
                                                  "celltype_model_<id>.csv, or live from the model)"),
                (self.accept_btn, "Accept (y)", "accept the predicted label of this ROI as mine, key y"),
                (self.accept_all_btn, "Accept all...", "label every shown, unlabeled dSPN / iSPN ROI from the "
                                                       "prediction (source 'bulk')"),
                (self.retrain_btn, "Retrain", "retrain on the label store and re-predict this session")):
            btn.setText(txt)
            btn.setToolTip(tip)
        self.view_box.setToolTip("what the outlines / fills show: my labels, the predicted classes, or both")
        self.diff_chk.setText("only disagreements")
        self.diff_chk.setToolTip("hide ROIs where my label and the prediction agree or one has no call")
        self.auto_adv.setToolTip("go to the next ROI after a stage-2 label / non-SPN label")
        keys = QLabel("keys (hover)")
        keys.setToolTip(__doc__.split("Click")[0].split("Keys")[1])

        def page(*rows):
            """One tab: rows of (widget, col, colspan); tight spacing."""
            pg = QWidget()
            g = QGridLayout(pg)
            g.setContentsMargins(4, 4, 4, 4); g.setHorizontalSpacing(4); g.setVerticalSpacing(3)
            for r, row in enumerate(rows):
                for wdg, col, span in row:
                    g.addWidget(wdg, r, col, 1, span)
            for c in range(6):
                g.setColumnStretch(c, 1)
            return pg

        L = QLabel
        tabs = QTabWidget()
        tabs.addTab(page(
            [(L("order"), 0, 1), (self.order_box, 1, 5)],
            [(L("iscell >="), 0, 1), (self.thr_spin, 1, 1), (L("diam"), 2, 1), (self.dmin_spin, 3, 1),
             (L("to"), 4, 1), (self.dmax_spin, 5, 1)],
            [(self.thr_count, 0, 3), (self.auto_adv, 3, 2), (keys, 5, 1)],
            [(self.beh_btn, 0, 2), (L("vel"), 2, 1), (self.vel_box, 3, 1), (L("lick"), 4, 1),
             (self.lick_box, 5, 1)],
            [(self.beh_lbl, 0, 6)]), "Filter")
        tabs.addTab(page(
            [(self.draw_btn, 0, 2), (self.calc_btn, 2, 2), (self.stop_btn, 4, 2)],
            [(self.del_btn, 0, 2), (self.undo_btn, 2, 2)]), "ROIs")
        tabs.addTab(page(
            [(L("show"), 0, 1), (self.view_box, 1, 3), (self.diff_chk, 4, 2)],
            [(self.pred_btn, 0, 2), (self.accept_btn, 2, 2), (self.accept_all_btn, 4, 2)],
            [(self.retrain_btn, 0, 3)],
            [(self.legend, 0, 6)]), "Predicted")
        self.tabs = tabs
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(4)
        head.addWidget(self.sess_box, 1)
        head.addWidget(self.open_btn)
        head.addWidget(self.next_btn)
        head.addWidget(self.follow_roi_chk)
        head.addWidget(self.save_btn)
        lay.addLayout(head)
        lay.addWidget(tabs)
        for lbl in (self.pred_lbl, self.man_lbl, self.edit_lbl):    # messages: shown only when there is one
            lay.addWidget(lbl)
        w.setStyleSheet(
            "QPushButton{padding:1px 6px;min-height:17px;font-size:8pt;}"
            "QComboBox,QDoubleSpinBox,QSpinBox{min-height:17px;font-size:8pt;}"
            "QLabel,QCheckBox{font-size:8pt;}"
            "QTabBar::tab{padding:2px 10px;font-size:8pt;}")
        for lbl in (self.info, self.counts):
            lbl.setStyleSheet("font-size: 9pt;")
        # the plot takes all remaining height; it is redrawn after a resize (its layout depends on
        # the width, see _plot)
        self.canvas.setMinimumHeight(420)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, 1)                 # stretch factor 1
        lay.addWidget(self.info)
        lay.addWidget(self.counts)
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._replot_after_resize)
        self.canvas.mpl_connect("resize_event", lambda e: self._resize_timer.start(150))
        w.setMinimumWidth(480)
        self.panel = w            # keep it: closing the dock detaches the widget, and
                                  # without a reference Python would delete it
        self.viewer.window.add_dock_widget(self.panel, area="right", name="ROI")
        try:                      # a menu entry too, in case the panel is closed
            self.viewer.window.window_menu.addAction("Show ROI panel (p)", self._show_panel)
        except Exception:
            pass
        self._build_display_dock()                 # left dock: projection, auto-contrast, gamma

        self.sess_box.currentIndexChanged.connect(lambda _: self.load_session())
        self.order_box.currentIndexChanged.connect(lambda _: self.build_queue())
        self.thr_spin.valueChanged.connect(self._set_thr)
        self.dmin_spin.valueChanged.connect(self._set_thr)
        self.dmax_spin.valueChanged.connect(self._set_thr)
        self.beh_btn.clicked.connect(self._pick_behavior)
        self.open_btn.clicked.connect(self._open_session)
        self.next_btn.clicked.connect(self.next_in_queue)
        self.draw_btn.clicked.connect(self._start_manual_roi)
        self.calc_btn.clicked.connect(self._draw_roi_manual)
        self.stop_btn.clicked.connect(self._stop_manual_roi)
        self.del_btn.clicked.connect(self.delete_current)
        self.undo_btn.clicked.connect(self.undo_delete)
        self.save_btn.clicked.connect(self.save_celltype)
        self.view_box.currentIndexChanged.connect(lambda _: self._refresh_edges())
        self.diff_chk.stateChanged.connect(self._set_thr)        # also rebuilds the ROI queue
        self.pred_btn.clicked.connect(self._load_pred_clicked)
        self.accept_btn.clicked.connect(self.accept_current)
        self.accept_all_btn.clicked.connect(self.accept_all_confident)
        self.retrain_btn.clicked.connect(self.retrain_and_predict)
        self.vel_box.currentIndexChanged.connect(lambda _: self._plot())
        self.lick_box.currentIndexChanged.connect(lambda _: self._plot())
        # The keys (s, r, g, y, ...) only reach the viewer when the image canvas has the keyboard
        # focus. A click on a panel button / checkbox / menu would take the focus away and the
        # keys would stop working, so those widgets never take focus, and the spin boxes give it
        # back to the canvas when you finish typing in them.
        from qtpy.QtCore import Qt
        for kind in (QPushButton, QCheckBox, QComboBox):
            for child in self.panel.findChildren(kind):
                child.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)           # nor the trace plot
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)             # clicking a tab must not
        self.tabs.tabBar().setFocusPolicy(Qt.FocusPolicy.NoFocus)    # take the keys either
        for sp in (self.thr_spin, self.dmin_spin, self.dmax_spin):
            sp.editingFinished.connect(self._focus_canvas)
        self._bind_keys()
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.load_session()

    # ------------------------------------------------------------ session
    def load_session(self):
        if not self.sessions or self.sess_box.currentIndex() < 0:
            return
        path = self.sessions[self.sess_box.currentIndex()]
        if path not in self.cache:
            self.viewer.status = f"loading {path} ..."
            self.cache[path] = Session(path, self.cfg, self.bundle)
        self.S = S = self.cache[path]
        lab = io.load_labels(self.store)
        lab = lab[lab["session_id"] == S.sid]
        lab = lab[lab["s2p_hash"].astype(str) == S.plane["hash"]]
        self.labels = {int(r.roi_idx): [r.stage1, r.stage2] for r in lab.itertuples()
                       if r.stage1 or r.stage2}
        # where each label came from: hand / accepted / bulk (see io_utils.TRAIN_SOURCES)
        self.label_src = {int(r.roi_idx): (r.source or "hand") for r in lab.itertuples()
                          if r.stage1 or r.stage2}
        self.pred, self.pred_msg = None, ""
        self._load_pred()              # newest celltype_model_<id>.csv, else labels.csv, else live

        v = self.viewer
        for name in ("green", "red", "red blob", "green blob", "predicted", "ROIs", "selected",
                     "manual ROI"):
            if name in v.layers:
                v.layers.remove(name)
        pct = lambda im, q: float(np.percentile(im, q))
        self.l_green = v.add_image(S.green, name="green", colormap="green",
                                   blending="additive",
                                   contrast_limits=(pct(S.green, 1), pct(S.green, 99.7)))
        self.l_red = None
        if S.red is not None:
            self.l_red = v.add_image(S.red, name="red", colormap="red", blending="additive",
                                     contrast_limits=(pct(S.red, 1), pct(S.red, 99.7)))
        self.l_blob_red = self.l_blob_green = None
        self._update_blob(create=True)             # 'red blob' / 'green blob' layers (hidden until ticked)
        self._sync_display()                       # projection menu back to 'mean', gamma of both layers
        # predicted classes as a fill under the outlines (outline = my label, see 'show')
        self.l_pred = v.add_shapes(S.contours, shape_type="polygon", name="predicted",
                                   edge_width=0, edge_color=[0, 0, 0, 0],
                                   face_color=self._pred_face_colors())
        self.l_rois = v.add_shapes(S.contours, shape_type="polygon", name="ROIs",
                                   edge_width=0.6, face_color=[0, 0, 0, 0],
                                   edge_color=self._edge_colors())
        self.l_sel = v.add_shapes([S.contours[0]], shape_type="polygon", name="selected",
                                  edge_width=1.5, edge_color="yellow", face_color=[0, 0, 0, 0])
        for l in (self.l_pred, self.l_rois, self.l_sel):
            l.editable = False
        self.pred_lbl.setText(f"predicted labels: {self.pred_msg}")
        self._ensure_man_layer()
        v.layers.selection.active = self.l_green
        self._fill_behavior_boxes()
        self.build_queue()

    def _update_blob(self, create=False):
        """The 'red blob' and 'green blob' image layers: ft.blob_map of whatever the red / green
        layer shows (the mean, or the projection when one is chosen). Meant for checking by eye
        that somata are the bright spots here; not used by the models. Hidden by default."""
        d = float(self.cfg["suite2p"]["diameter"])
        for chan, src, cmap in (("red", "l_red", "magma"), ("green", "l_green", "viridis")):
            layer = getattr(self, src, None)
            if layer is None:
                continue
            attr = f"l_blob_{chan}"
            try:
                blob = ft.blob_map(np.asarray(layer.data), d)
                lim = (0.0, max(float(np.percentile(blob, 99.9)), 3.0))
                cur = None if create else getattr(self, attr, None)
                if cur is None:
                    setattr(self, attr, self.viewer.add_image(
                        blob, name=f"{chan} blob", colormap=cmap, blending="additive",
                        contrast_limits=lim, visible=False))
                else:
                    cur.data = blob
                    cur.contrast_limits = lim
            except Exception as e:                   # a display aid only: never block the session
                self.viewer.status = f"({chan} blob layer not computed: {e})"

    # ------------------------------------------------------------ display: projection, contrast, gamma
    def _build_display_dock(self):
        """Dock on the LEFT (under napari's layer controls): which image the green layer shows,
        and auto-contrast + gamma of the green / red layers for the part of the image in view."""
        from qtpy.QtCore import QTimer, Qt
        from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel,
                                    QPushButton, QSpinBox, QVBoxLayout, QWidget)
        w = QWidget()
        lay = QVBoxLayout(w)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        lay.addLayout(grid)
        lay.addStretch(1)

        self.proj_box = QComboBox(); self.proj_box.addItems(PROJECTIONS)
        self.proj_n = QSpinBox(); self.proj_n.setRange(20, 20000); self.proj_n.setValue(400)
        self.proj_n.setSingleStep(100); self.proj_n.setKeyboardTracking(False)
        self.proj_lbl = QLabel(); self.proj_lbl.setWordWrap(True)
        self.pct_lo, self.pct_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, rng, v in ((self.pct_lo, (0, 49), 1.0), (self.pct_hi, (51, 100), 99.7)):
            sp.setRange(*rng); sp.setDecimals(1); sp.setSingleStep(0.5); sp.setValue(v)
            sp.setKeyboardTracking(False)
        self.auto_gamma = QCheckBox("auto gamma (median to mid-grey)"); self.auto_gamma.setChecked(True)
        self.follow_chk = QCheckBox("follow zoom / pan (live)")
        self.gamma_g, self.gamma_r = QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.gamma_g, self.gamma_r):
            sp.setRange(0.2, 5.0); sp.setDecimals(2); sp.setSingleStep(0.05); sp.setValue(1.0)
            sp.setKeyboardTracking(False)
        b_both = QPushButton("auto-contrast view (green + red)")
        b_g, b_r = QPushButton("green only"), QPushButton("red only")
        self.disp_lbl = QLabel(); self.disp_lbl.setWordWrap(True)
        self.crop_box = QComboBox(); self.crop_box.addItems(CROP_MODES)
        self.crop_box.setToolTip("the zoomed crops under the napari view: the green mean image can be replaced by (or shown next to) "
                                 "the event snapshot of the ROI")
        self.snap_box = QComboBox(); self.snap_box.addItems(SNAP_MODES)
        self.snap_box.setToolTip("strongest event: the 3 frames around the ROI's single strongest event minus quiet frames; "
                                 "average: the mean over its top events (what the stage-1 features use). Contrast and gamma are set "
                                 "automatically for each ROI")
        for lbl in (self.proj_lbl, self.disp_lbl):
            lbl.setStyleSheet("font-size: 9pt;")

        grid.addWidget(QLabel("<b>green + red layers show</b>"), 0, 0, 1, 3)
        grid.addWidget(self.proj_box, 1, 0, 1, 2); grid.addWidget(self.proj_n, 1, 2)
        grid.addWidget(QLabel("max / std use this many evenly spaced frames"), 2, 0, 1, 3)
        grid.addWidget(self.proj_lbl, 3, 0, 1, 3)
        grid.addWidget(QLabel("<b>contrast of the area in view</b>"), 4, 0, 1, 3)
        grid.addWidget(QLabel("percentiles"), 5, 0); grid.addWidget(self.pct_lo, 5, 1)
        grid.addWidget(self.pct_hi, 5, 2)
        grid.addWidget(b_both, 6, 0, 1, 3)
        grid.addWidget(b_g, 7, 0, 1, 1); grid.addWidget(b_r, 7, 1, 1, 2)
        grid.addWidget(self.auto_gamma, 8, 0, 1, 3)
        grid.addWidget(self.follow_chk, 9, 0, 1, 3)
        grid.addWidget(QLabel("gamma green"), 10, 0); grid.addWidget(self.gamma_g, 10, 1)
        grid.addWidget(QLabel("gamma red"), 11, 0); grid.addWidget(self.gamma_r, 11, 1)
        grid.addWidget(self.disp_lbl, 12, 0, 1, 3)
        grid.addWidget(QLabel("<b>Follow ROI</b>: smallest view"), 13, 0, 1, 2)
        grid.addWidget(self.min_fov, 13, 2)
        grid.addWidget(QLabel("<b>crops under the view</b>"), 14, 0, 1, 3)
        grid.addWidget(self.crop_box, 15, 0, 1, 3)
        grid.addWidget(QLabel("snapshot shows"), 16, 0); grid.addWidget(self.snap_box, 16, 1, 1, 2)
        for kind in (QPushButton, QCheckBox, QComboBox):          # keep the keys working (see __init__)
            for child in w.findChildren(kind):
                child.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.disp_panel = w
        self.viewer.window.add_dock_widget(w, area="left", name="view display")

        self._view_timer = QTimer(); self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self.auto_contrast)
        self.proj_box.currentIndexChanged.connect(self._set_projection)
        self.proj_n.valueChanged.connect(lambda _: self._set_projection() if self.proj_box.currentIndex() else None)
        b_both.clicked.connect(lambda *_: self.auto_contrast())
        b_g.clicked.connect(lambda *_: self.auto_contrast(which=("green",)))
        b_r.clicked.connect(lambda *_: self.auto_contrast(which=("red",)))
        self.gamma_g.valueChanged.connect(lambda v: self._on_gamma("green", v))
        self.gamma_r.valueChanged.connect(lambda v: self._on_gamma("red", v))
        self.follow_chk.stateChanged.connect(self._follow_toggled)
        self.crop_box.currentIndexChanged.connect(lambda *_: self._plot())
        self.snap_box.currentIndexChanged.connect(lambda *_: self._plot())
        for sp in (self.proj_n, self.pct_lo, self.pct_hi, self.gamma_g, self.gamma_r, self.min_fov):
            sp.editingFinished.connect(self._focus_canvas)

    def _layer(self, name):
        return getattr(self, "l_green", None) if name == "green" else getattr(self, "l_red", None)

    def _sync_display(self):
        """After a session (re)loads its layers: projection menu back to 'mean image', and the
        gamma boxes applied to the new layers."""
        self.proj_box.blockSignals(True)
        self.proj_box.setCurrentIndex(0)
        self.proj_box.blockSignals(False)
        self.proj_lbl.setText("layers: mean image")
        for name, sp in (("green", self.gamma_g), ("red", self.gamma_r)):
            layer = self._layer(name)
            if layer is not None:
                layer.gamma = float(sp.value())

    def _camera(self):
        v = self.viewer
        return v.scene.camera if hasattr(v, "scene") else v.camera      # napari >= 0.9 / older

    def _canvas_hw(self):
        """Canvas size in pixels as (height, width), across napari versions."""
        v = self.viewer
        for get in (lambda: v.canvas.size, lambda: v._canvas_size):
            try:
                h, w = get()
                return float(h), float(w)
            except Exception:
                pass
        nat = v.window._qt_viewer.canvas.native
        return float(nat.height()), float(nat.width())

    def _view_crop(self, shape, min_px=8):
        """Rows and columns of the image that are in view right now (the zoomed-in area),
        as (r0, r1, c0, c1); the whole image when it is not zoomed."""
        cam = self._camera()
        cy, cx = (float(c) for c in cam.center[-2:])
        zoom = max(float(cam.zoom), 1e-6)
        h, w = self._canvas_hw()
        hh, hw = h / (2 * zoom), w / (2 * zoom)

        def span(c, half, n):
            a = int(np.clip(np.floor(c - half), 0, n - 1))
            b = int(np.clip(np.ceil(c + half) + 1, a + 1, n))
            if b - a < min_px:                                  # never a degenerate patch:
                a = max(0, a - (min_px - (b - a)) // 2)         # grow it around the same centre
                b = min(n, a + min_px)
                a = max(0, b - min_px)
            return a, b
        r0, r1 = span(cy, hh, shape[0])
        c0, c1 = span(cx, hw, shape[1])
        return r0, r1, c0, c1

    def auto_contrast(self, *_, which=("green", "red")):
        """Contrast limits (and, if ticked, gamma) of the green / red layer from the pixels in the
        current field of view: low..high percentile of what you see, so after zooming in the
        display is stretched to that region. Gamma is set so the median of the view is mid-grey."""
        lo_p, hi_p = float(self.pct_lo.value()), float(self.pct_hi.value())
        done = []
        for name in which:
            layer = self._layer(name)
            if layer is None:
                continue
            img = np.asarray(layer.data)
            r0, r1, c0, c1 = self._view_crop(img.shape)
            tile = img[r0:r1, c0:c1].astype(np.float32)
            if not np.isfinite(tile).any():
                continue
            lo, hi = np.nanpercentile(tile, [lo_p, hi_p])
            if not hi > lo:
                hi = lo + 1.0
            layer.contrast_limits = (float(lo), float(hi))
            msg = f"{name}: {lo:.0f}-{hi:.0f}"
            if self.auto_gamma.isChecked():
                m = float(np.clip((np.nanmedian(tile) - lo) / (hi - lo), 0.02, 0.98))
                g = float(np.clip(np.log(0.5) / np.log(m), 0.3, 3.0))
                layer.gamma = g
                sp = self.gamma_g if name == "green" else self.gamma_r
                sp.blockSignals(True); sp.setValue(g); sp.blockSignals(False)
                msg += f", gamma {g:.2f}"
            done.append(msg)
        if done:
            self.disp_lbl.setText(f"view rows {r0}-{r1}, cols {c0}-{c1}: " + "; ".join(done))

    def _on_gamma(self, name, value):
        layer = self._layer(name)
        if layer is not None:
            layer.gamma = float(value)

    def _set_projection(self, *_):
        """Show the mean image, or a max / std projection of the movie, in the green layer.
        (napari's own 'projection mode' control only acts on 3-D data; these are 2-D images.)"""
        from qtpy.QtWidgets import QApplication
        S, kind = self.S, self.proj_box.currentText()
        if not kind.startswith("mean"):
            self.proj_lbl.setText(f"loading / computing the {kind} (once per setting) ...")
            QApplication.processEvents()
        try:
            img = S.projection(kind, self.proj_n.value())
        except Exception as e:
            self.proj_lbl.setText(f"<span style='color:red'>could not compute: {e}</span>")
            return
        self.l_green.data = img
        note = ""
        if self.l_red is not None:                       # the red layer follows the same menu
            try:
                rimg = S.red_projection(kind, self.proj_n.value())
            except Exception as e:
                rimg, note = None, f"; red: could not compute ({e})"
            if rimg is not None:
                self.l_red.data = rimg
            else:
                self.l_red.data = S.red
                if not kind.startswith("mean") and not note:
                    note = "; red: single image, nothing to project (shows its mean)"
        self._update_blob()                              # the blob layers follow the red / green layers
        self.proj_lbl.setText((f"layers: {kind}" if kind.startswith("mean") else
                               f"layers: {kind} ({getattr(S, 'proj_source', '')})") + note)
        self.auto_contrast(which=("green", "red"))

    def _follow_toggled(self, state):
        cam = self._camera()
        for ev in (cam.events.zoom, cam.events.center):
            try:
                if state:
                    ev.connect(self._view_changed)
                else:
                    ev.disconnect(self._view_changed)
            except Exception:
                pass
        if state:
            self.auto_contrast()

    def _view_changed(self, *_):
        """Zoom / pan: re-stretch the display a moment after the movement stops."""
        self._view_timer.start(200)

    def _replot_after_resize(self):
        """The plot layout depends on the canvas width (square crops as wide as possible)."""
        if getattr(self, "S", None) is not None and hasattr(self, "cur"):
            self._plot()

    def _focus_canvas(self, *_):
        """Give the keyboard focus back to the napari canvas so the label keys work."""
        try:
            self.viewer.window._qt_viewer.canvas.native.setFocus()
        except Exception:
            pass

    def _show_panel(self, *_):
        """Bring the side panel back after it was closed (its x button detaches it)."""
        win = self.viewer.window
        try:
            inner = win.dock_widgets.get("ROI") if hasattr(win.dock_widgets, "get") else None
        except Exception:
            inner = None
        if inner is not None:                      # still docked: just make it visible
            dock = inner.parent()
            dock.show()
            dock.raise_()
            return
        win.add_dock_widget(self.panel, area="right", name="ROI")

    def _open_session(self, *_):
        """Pick a session folder, any Suite2p output folder inside it (suite2p,
        suite2p_sourcerydetection, ...), its plane0, or a folder above several
        sessions. Every output found becomes one entry in the session list; the
        images come from the folder holding ConcatenatedMovie_MC.tif (above the
        output folder), the results from <output>\\plane0."""
        from qtpy.QtWidgets import QApplication, QFileDialog
        start = str(io.image_dir(self.S.dir).parent) if self.S is not None else ""
        folder = QFileDialog.getExistingDirectory(
            self.viewer.window._qt_window,
            "session folder (or its suite2p / plane0 folder)", start)
        if not folder:
            return
        found = io.find_sessions(folder)
        if not found:
            self.viewer.status = f"no <output>\\plane0\\stat.npy in or below {folder}"
            self.man_lbl.setText(f"<span style='color:red'>no &lt;output&gt;\\plane0\\stat.npy "
                                 f"in or below {folder}</span>")
            return
        self.viewer.status = f"loading {found[0]} (computing features the first time) ..."
        QApplication.processEvents()
        first_idx = None
        for sp in found:
            if sp not in self.sessions:
                self.sessions.append(sp)
                self.sess_box.addItem(sp)         # the first item of an empty box loads it
            if first_idx is None:
                first_idx = self.sessions.index(sp)
        self.sess_box.setCurrentIndex(first_idx)
        self.man_lbl.setText(f"{len(found)} session(s) opened")

    def _pick_behavior(self):
        # open a file dialog to pick a behavior .mat file, load it, and update the GUI
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self.viewer.window._qt_window, "behavior .mat", str(io.image_dir(self.S.dir)),
            "MAT files (*.mat)")
        if not path:
            return
        try:
            self.S.load_behavior(path)
        except Exception as e:
            self.beh_lbl.setText(f"<span style='color:red'>could not load: {e}</span>")
            return
        self._fill_behavior_boxes()
        self._plot()

    def _fill_behavior_boxes(self):
        S = self.S
        names = sorted(S.behav)
        for box, guess in ((self.vel_box, VEL_GUESS), (self.lick_box, LICK_GUESS)):
            box.blockSignals(True); box.clear()
            box.addItems([NONE] + names)
            pick = next((g for g in guess if g in S.behav), None)
            if pick is None:       # fall back to a field whose name contains the stem
                stem = "lick" if box is self.lick_box else "vel"
                pick = next((n for n in names if stem in n.lower()), None)
            box.setCurrentText(pick or NONE)
            box.blockSignals(False)
        self.beh_lbl.setText(f"behavior: {Path(S.behav_file).name}" if S.behav_file
                             else "no behavior loaded")

    def _label_of(self, i):
        st1, st2 = self.labels.get(i, ["", ""])
        if st1 == "SPN" and st2:
            return st2
        if st1:
            return st1
        return "not_cell" if self.S.feat["iscell"].iloc[i] == 0 else ""

    # ---- predicted labels next to mine
    def _pred_label_of(self, i):
        """Predicted class of ROI i ('' = none: no predictions loaded, or a manual ROI)."""
        p = getattr(self, "pred", None)
        if p is None or i not in p.index:
            return ""
        return str(p.at[i, "final_label"])

    def _hand_class_of(self, i):
        """My label of ROI i in the vocabulary of the predictions (dSPN, iSPN, ..., '' = none)."""
        st1, st2 = self.labels.get(i, ["", ""])
        return review.hand_to_class(st1, st2, self.cfg["cre_line"])

    def _compare(self, i):
        """'agree' / 'disagree' / '' (no definite call on one side)."""
        return review.compare(self._hand_class_of(i), self._pred_label_of(i))

    def _pred_rgba(self, label, alpha):
        return to_rgba(review.PRED_HEX[label], alpha) if label in review.PRED_HEX else EDGE[""]

    def _load_pred(self):
        """Predictions for the current session, from the newest celltype_model_<id>.csv of this
        Suite2p run (self.pred_id = its model id); else the labels.csv of the older scheme; else
        computed live from the model (pred_id = that model's id)."""
        S = self.S
        mid, csv = curated.latest_model(S.dir, S.plane["hash"])
        df, msg, self.pred_id = None, "", None
        if csv is not None:
            df, msg = review.load_predictions(csv, S.plane["hash"], S.n_orig)
            if df is not None:
                self.pred_id = mid
        if df is None:
            df2, msg2 = review.load_predictions(io.out_dir(S.dir) / "labels.csv", S.plane["hash"], S.n_orig)
            if df2 is not None:
                df, msg = df2, msg2 + " (older scheme)"
            elif csv is None:
                msg = f"no celltype_model_<id>.csv in {io.out_dir(S.dir)} (run the batch prediction)"
        if df is None and self.bundle is not None and S.p_spn is not None:
            self.pred_id = curated.model_id(self.bundle["created"])
            n = S.n_orig
            lab, _ = models.assign_labels(S.feat.iloc[:n], S.p_spn[:n], S.p_red[:n],
                                          self.bundle, self.cfg["use_iscell"])
            df = pd.DataFrame({"final_label": lab, "p_spn": S.p_spn[:n], "p_red": S.p_red[:n]},
                              index=pd.Index(S.feat["roi_idx"].to_numpy()[:n], name="roi_idx"))
            msg = f"computed live by the current model ({msg})"
        self.pred, self.pred_msg = df, msg

    def _load_pred_clicked(self, *_):
        self._load_pred()
        self.pred_lbl.setText(f"predicted labels: {self.pred_msg}" if self.pred is not None
                              else f"<span style='color:red'>no predictions: {self.pred_msg}; "
                                   f"run --mode predict or train a model first</span>")
        self._refresh_edges()
        self.build_queue()

    def _visible(self):
        vis = self.S.feat["s2p_iscell_prob"].to_numpy() >= self.thr_spin.value()
        lo, hi = sorted((self.dmin_spin.value(), self.dmax_spin.value()))
        diam = self.S.feat["equiv_diam_um"].to_numpy(dtype=float)
        vis &= ~np.isfinite(diam) | ((diam >= lo) & (diam <= hi))    # NaN: not filtered
        if self.diff_chk.isChecked():         # only ROIs where my label and the prediction disagree
            vis &= np.array([self._compare(i) == "disagree" for i in range(len(vis))])
        vis[self.S.n_orig:] = True            # manual ROIs: shown unless deleted
        if self.S.deleted:
            vis[list(self.S.deleted)] = False       # deleted ROIs are hidden
        return vis

    def _edge_colors(self):
        """Outline colours: my label ('my labels', 'both'), the predicted class ('predicted'),
        or a faint grey ('none')."""
        S = self.S
        vis = self._visible()
        mode = self.view_box.currentIndex()
        if mode in (VIEW_MINE, VIEW_BOTH):
            cols = np.array([EDGE[self._label_of(i)] for i in range(len(S.stat))])
        elif mode == VIEW_PRED:
            cols = np.array([self._pred_rgba(self._pred_label_of(i), 1.0) for i in range(len(S.stat))])
        else:
            cols = np.tile(np.array(EDGE[""]), (len(S.stat), 1))
        cols[S.n_orig:] = MANUAL_EDGE       # manually drawn ROIs are purple
        cols[~vis, 3] = 0                   # hidden: below the threshold or deleted
        return cols

    def _pred_face_colors(self):
        """Fill colours of the 'predicted' layer: the predicted class (semi-transparent) in the
        'predicted' and 'both' views; nothing otherwise, for 'not_cell', and for hidden ROIs."""
        S = self.S
        cols = np.zeros((len(S.stat), 4))
        if self.view_box.currentIndex() in (VIEW_PRED, VIEW_BOTH) and self.pred is not None:
            vis = self._visible()
            for i in range(S.n_orig):
                lab = self._pred_label_of(i)
                if lab and lab != "not_cell":
                    cols[i] = self._pred_rgba(lab, PRED_FILL_ALPHA)
            cols[~vis, 3] = 0
        return cols

    def _update_overlay(self):
        """Redraw outlines (mine / predicted) and the predicted fill."""
        self.l_rois.edge_color = self._edge_colors()
        if hasattr(self, "l_pred"):
            self.l_pred.face_color = self._pred_face_colors()

    def _set_thr(self, val):
        if hasattr(self, "l_rois"):
            self._update_overlay()
        self.build_queue()

    def _ensure_man_layer(self):
        """The manual-ROI layer must be in the viewer's layer list before it can be
        selected; (re)create it if it is missing."""
        l = getattr(self, "l_man", None)
        if l is None or l not in self.viewer.layers:
            self.l_man = self.viewer.add_shapes(
                name="manual ROI", edge_width=1.0, edge_color="purple",
                face_color=[0, 0, 0, 0])
            self.l_man.events.mode.connect(self._sync_shortcuts)    # drawing on / off
        return self.l_man

    def _start_manual_roi(self):
        """Switch to the manual-ROI layer in polygon mode; click to place vertices,
        double-click / Esc to finish."""
        self._ensure_man_layer()
        self.viewer.layers.selection.active = self.l_man
        self.l_man.mode = "add_polygon"

    def _stop_manual_roi(self, *_):
        """Leave drawing mode so clicks select existing ROIs again."""
        self._ensure_man_layer().mode = "pan_zoom"
        self.viewer.layers.selection.active = self.l_green

    def _draw_roi_manual(self, *_):
        """Compute every polygon currently on the 'manual ROI' layer (those drawn and
        not yet calculated), one after another in the order drawn. Each finished
        polygon joins the ROI list (purple); a polygon that fails stays on the layer."""
        from qtpy.QtWidgets import QApplication
        man = self._ensure_man_layer()
        polys = [np.asarray(d)[:, -2:] for d in man.data]       # (row, col) vertices
        if not polys:
            self.man_lbl.setText("draw a polygon first")
            return
        S = self.S
        done, failed, notes = [], [], []
        for k, poly in enumerate(polys, 1):
            cached = S.win_cache.win is not None
            self.man_lbl.setText(
                f"computing {k} / {len(polys)} ..." if cached else
                f"computing {k} / {len(polys)}: loading region from the movie "
                f"(once per region) ...")
            QApplication.processEvents()          # show the message before the long read
            try:
                # S.stat includes earlier manual ROIs: they are excluded from the neuropil
                res = mr.manual_roi_traces(S.get_movie(), poly, dict(S.plane, stat=S.stat),
                                           self.cfg, S.win_cache)
            except Exception as e:
                failed.append((poly, str(e)))
                continue
            n = mr.save_manual_roi(io.out_dir(S.dir) / "manual_rois.pkl", res, S.plane["hash"])
            i = S.add_manual(res)
            self.l_rois.add(S.contours[i], shape_type="polygon", edge_width=0.6,
                            face_color=[0, 0, 0, 0])
            self.l_pred.add(S.contours[i], shape_type="polygon", edge_width=0,
                            face_color=[0, 0, 0, 0], edge_color=[0, 0, 0, 0])   # keeps the layers aligned
            self.queue.insert(self.pos + 1 + len(done), i)
            done.append(i)
            if res["allow_overlap_used"]:
                notes.append(f"#{n}: all pixels overlap another ROI, overlap kept")
        # the drawing layer keeps only the polygons that failed
        man.data = []
        for poly, _ in failed:
            man.add(poly, shape_type="polygon", edge_width=1.0, edge_color="purple",
                    face_color=[0, 0, 0, 0])
        man.mode = "pan_zoom"
        self._update_overlay()
        self.thr_count.setText(f"{self._visible().sum()} / {len(S.stat)} ROIs shown")

        msg = f"{len(done)} manual ROI(s) computed and saved"
        if done:
            r = S.extra[-1]
            msg += (f"; last: {r['n_pix']} px, neuropil {r['n_neuropil']} px, "
                    f"iscell {r['iscell'][1]:.2f}")
            if r["algorithm"] != "cellpose":
                msg += (f". weights: image rule (run used {r['algorithm']}: "
                        f"lam not reproducible)")
        if notes:
            msg += ". " + "; ".join(notes)
        if failed:
            msg += (f". <span style='color:red'>{len(failed)} failed and kept on the layer: "
                    f"{failed[0][1]}</span>")
        self.man_lbl.setText(msg)
        if done:
            self.select(done[-1])

    # ------------------------------------------------------------ delete / save
    def _refresh_edges(self):
        self._update_overlay()
        self.thr_count.setText(f"{self._visible().sum()} / {len(self.S.stat)} ROIs shown")

    def delete_current(self, *_):
        """Hide the current ROI (Suite2p or manual) and remember it; the list is kept
        in memory, in roi_edits.json, and applied by 'Save'."""
        S, i = self.S, self.cur
        if i in S.deleted:
            return
        S.delete(i)
        self.queue = [q for q in self.queue if q != i]
        self._refresh_edges()
        n = len(S.deleted)
        self.edit_lbl.setText(f"{n} ROI(s) deleted (not yet in a .mat; press 'Save')")
        if self.queue:
            self.select(self.queue[min(self.pos, len(self.queue) - 1)])

    def undo_delete(self, *_):
        S = self.S
        i = S.restore_last()
        if i is None:
            self.edit_lbl.setText("nothing to undo")
            return
        self._refresh_edges()
        if self._visible()[i]:
            self.queue.insert(self.pos, i)
            self.select(i)
        self.edit_lbl.setText(f"restored ROI {i}; {len(S.deleted)} deleted")

    def _train_sources(self):
        """Label sources that go into the training set (see io_utils.TRAIN_SOURCES)."""
        return io.TRAIN_SOURCES + (("bulk",) if self.cfg.get("train_on_bulk") else ())

    def _adjusted(self):
        """Adjusted cell table of the current session (a DataFrame, one row per ROI)."""
        S = self.S
        return review.adjusted_table(len(S.stat), S.n_orig, S.deleted, self.labels, self.label_src,
                                     self.pred, self.cfg["cre_line"], self._train_sources())

    def _save_id(self):
        """Model id the edits are saved under: the id of the prediction being edited (a prediction
        file, or the live model); the loaded model's id if there is none; else 'noid'."""
        if getattr(self, "pred_id", None):
            return self.pred_id
        if self.bundle is not None:
            return curated.model_id(self.bundle["created"])
        return "noid"

    def _celltype_struct(self, tab, keep):
        """The 'celltype' struct of celltype_curated_<id>.mat (see curated.celltype_struct)."""
        mid = self._save_id()
        created = (self.bundle["created"] if self.bundle is not None
                   and curated.model_id(self.bundle["created"]) == mid else "")
        return curated.celltype_struct(tab, keep, self.cfg["cre_line"], self.S.deleted, created,
                                       model_id=mid, s2p_hash=self.S.plane["hash"],
                                       fall_file=io.plane_dir(self.S.dir) / "Fall.mat")

    def save_celltype(self, *_):
        """The one Save button. Writes two files in <session>\\celltype\\, named after the model id
        of the prediction being edited (see _save_id) and overwritten by the next Save of that id;
        the model's own celltype_model_<id> prediction is never touched:

          celltype_curated_<id>.mat  the struct 'celltype' with every column of the cell-type table (deleted
                                ROIs left out, manual ROIs last) and the index arrays that tie every row to the
                                Suite2p Fall.mat (fall_row, orig_index, deleted_roi_idx, the pixels of the
                                manual ROIs); no traces
          celltype_curated_<id>.csv  the same table, one row per kept ROI

        Nothing goes into the suite2p folder. Both files are written under temporary names and
        swapped in together, so a failed save leaves the previous pair untouched. The history
        (label store, roi_edits.json, manual_rois.pkl, the celltype_model_<id>.* files) is not
        touched."""
        import os
        from qtpy.QtWidgets import QApplication
        S = self.S
        folder = io.out_dir(S.dir)
        mid = self._save_id()
        mat, csv = curated.curated_files(folder, mid)
        tmp_mat = folder / f"celltype_curated_{mid}_tmp.mat"
        tmp_csv = folder / f"celltype_curated_{mid}_tmp.csv"
        keep = S.kept()
        tab = curated.kept_table(self._adjusted(), keep, S.n_orig)
        self.edit_lbl.setText("saving ...")
        QApplication.processEvents()
        try:
            n_keep, n_man, n_del = S.curated_mat(tmp_mat, celltype=lambda k: self._celltype_struct(tab, k))
            tab.to_csv(tmp_csv, index=False)
            os.replace(tmp_mat, mat)
            os.replace(tmp_csv, csv)
        except Exception as e:
            for t in (tmp_mat, tmp_csv):
                try:
                    t.unlink()
                except OSError:
                    pass
            self.edit_lbl.setText(f"<span style='color:red'>save failed: {e}</span>")
            return
        try:
            curated.write_status(S.dir)
        except Exception as e:
            print(f"    (celltype_status.json not updated: {e})")
        fs = tab["final_source"].value_counts().to_dict()
        live = (" (predictions were computed live from the model, no celltype_model file yet)"
                if self.pred is not None and "live" in self.pred_msg else "")
        self.edit_lbl.setText(
            f"saved {csv.name} and {mat.name} in {folder}: {n_keep} ROIs ({n_man} manual), "
            f"{n_del} deleted left out; {fs.get('hand', 0)} cell types from my labels, "
            f"{fs.get('model', 0)} from the model, {int(tab['used_in_training'].sum())} in the "
            f"training set{live}. {self._queue_note()}")

    def _queue_note(self):
        """'3 sessions left to review (1 only edited under an older model)' for the sessions in
        the list above (not counting the current one)."""
        n = len(self.sessions)
        cur = self.sess_box.currentIndex()
        pend = stale = 0
        for k in range(1, n):
            st = curated.session_state(self.sessions[(cur + k) % n])["state"]
            pend += st == "pending"
            stale += st == "stale"
        left = pend + stale
        if not left:
            return "Queue: nothing left to review."
        return (f"Queue: {left} session{'s' if left != 1 else ''} left to review"
                + (f" ({stale} only edited under an older model)" if stale else "") + ".")

    def next_in_queue(self, *_):
        """Load the next session of the list that has a prediction of the newest model and no
        saved edit yet; sessions edited only under an older model come after those. Walks the list
        from the current session onwards and wraps around. A session you only looked at stays in
        the queue until you Save it."""
        if not self.sessions:
            return
        n, cur = len(self.sessions), self.sess_box.currentIndex()
        first = {"pending": None, "stale": None}
        for k in range(1, n):
            idx = (cur + k) % n
            st = curated.session_state(self.sessions[idx])["state"]
            if st in first and first[st] is None:
                first[st] = idx
        target = first["pending"] if first["pending"] is not None else first["stale"]
        if target is None:
            self.edit_lbl.setText("Queue: nothing left to review (every session has a saved edit for "
                                  "the newest model, or no prediction yet: run the batch prediction).")
            return
        self.sess_box.setCurrentIndex(target)               # loads the session
        older = first["pending"] is None                     # only stale sessions are left
        self.edit_lbl.setText(("Edited under an older model only: " if older else "") + self._queue_note())

    def _probs(self):
        """p(SPN) and p(red) of every ROI as arrays (NaN: manual ROIs, or no model / predictions).
        The loaded predictions are used when there are any (labels.csv or live), else the
        probabilities of the model loaded at start."""
        S = self.S
        n = len(S.stat)
        p_spn, p_red = np.full(n, np.nan), np.full(n, np.nan)
        if self.pred is not None and {"p_spn", "p_red"} <= set(self.pred.columns):
            idx = self.pred.index.to_numpy()
            ok = idx < n
            p_spn[idx[ok]] = self.pred["p_spn"].to_numpy(float)[ok]
            p_red[idx[ok]] = self.pred["p_red"].to_numpy(float)[ok]
        elif S.p_spn is not None:
            m = min(n, len(S.p_spn))
            p_spn[:m], p_red[:m] = S.p_spn[:m], S.p_red[:m]
        return p_spn, p_red

    def _order_by_prob(self, idx, order):
        """Order the shown ROIs `idx` by p(red) or p(SPN):
             'most uncertain first'  closest to 0.5 first (the ones the model is unsure about)
             'most certain first'    farthest from 0.5 first
             'high to low'           p(red) from red+ to red-
        For p(red) the ROIs predicted to be SPNs come first (p(red) means nothing for the
        others); ROIs without a probability (manual ROIs) are always last."""
        p_spn, p_red = self._probs()
        red = order.startswith("p(red)")
        p = p_red if red else p_spn
        if np.isnan(p[idx]).all():
            self.viewer.status = "no probabilities yet: press 'Load predicted' or train a model"
            return idx
        if "uncertain" in order:
            key = np.abs(p - 0.5)
        elif "certain" in order:
            key = -np.abs(p - 0.5)
        else:                                         # high to low
            key = -p
        if red:
            is_spn = np.array([self._pred_label_of(i) in review.SPN_CLASSES for i in range(len(p))])
            if is_spn.any():
                key = key + (~is_spn) * 10.0          # non-SPN after the SPNs (NaN stays last)
        return idx[np.argsort(key[idx], kind="stable")]

    def build_queue(self):
        S, f = self.S, self.S.feat
        vis = self._visible()
        self.thr_count.setText(f"{vis.sum()} / {len(vis)} ROIs shown")
        idx = np.flatnonzero(vis)
        if len(idx) == 0:
            self.queue = []
            return
        order = self.order_box.currentText()
        unl = np.array([i not in self.labels for i in idx])
        if order.startswith("unlabeled"):
            rng = np.random.default_rng(0)
            idx = np.concatenate([rng.permutation(idx[unl]), idx[~unl]])
        elif order.startswith("most uncertain"):
            if S.p_spn is None:
                self.viewer.status = "no trained model yet: using random order"
                idx = np.random.default_rng(0).permutation(idx)
            else:
                u1 = np.abs(S.p_spn[idx] - 0.5)
                u2 = np.where(S.p_spn[idx] > 0.5, np.abs(S.p_red[idx] - 0.5), np.inf)
                score = np.minimum(u1, u2) + (~unl) * 10      # labeled ones last
                idx = idx[np.argsort(score)]
        elif order.startswith("disagreements"):
            # my label vs the prediction: disagreements, then ROIs I have not labeled, then the rest
            if self.pred is None:
                self.viewer.status = "no predictions loaded: press 'load predicted'"
            dis = np.array([self._compare(i) == "disagree" for i in idx])
            idx = np.concatenate([idx[dis], idx[~dis & unl], idx[~dis & ~unl]])
        elif order.startswith(("p(red)", "p(SPN)")):
            idx = self._order_by_prob(idx, order)
        elif order.startswith("largest"):
            idx = idx[np.argsort(-f["equiv_diam_um"].to_numpy()[idx])]
        elif order.startswith("highest baseline"):
            idx = idx[np.argsort(-f["baseline_rel"].to_numpy()[idx])]
        elif order.startswith("most red"):
            idx = idx[np.argsort(-np.nan_to_num(f["red_z"].to_numpy()[idx], nan=-1e9))]
        self.queue, self.pos = list(idx), 0
        if self.queue:
            self.select(self.queue[0])

    # ------------------------------------------------------------ display
    def select(self, i, follow=True):
        """Select ROI i. follow: also move the view to it (the 'Follow ROI' box); a click on an
        ROI in the image passes follow=False so the view does not jump under the mouse."""
        self.cur = int(i)
        if self.cur in self.queue:
            self.pos = self.queue.index(self.cur)
        S = self.S
        try:
            self.l_sel.data = [S.contours[self.cur]]
        except ValueError:                          # viewer was switched to 3-D: back to 2-D, retry
            self._force_2d()
            self.l_sel.data = [S.contours[self.cur]]
        self.l_sel.edge_color = "yellow"
        self.viewer.layers.selection.active = self.l_green
        if follow:
            self._follow_selected()
        self._plot()

    def _follow_selected(self):
        """Move the view to the selected ROI, if 'Follow ROI' is on and the view is zoomed in:

          * zoomed out (the whole image is in view): nothing moves, so zooming out never gets
            undone or shifted by stepping through the ROIs
          * zoomed in: the view is centred on the ROI at the current zoom, but never shows less
            than `min view` pixels on its shorter side (it zooms out to that), and never past the
            edge of the image (an ROI near the border sits off-centre instead of leaving blank space)
        A failure here only writes to the status bar; selecting the ROI always works."""
        try:
            if not self.follow_roi_chk.isChecked():
                return
            S = self.S
            s = S.stat[self.cur]
            cy, cx = float(np.mean(s["ypix"])), float(np.mean(s["xpix"]))
            if not np.isfinite([cy, cx]).all():
                return
            n_y, n_x = S.green.shape
            cam = self._camera()
            h, w = self._canvas_hw()
            zoom = float(cam.zoom)
            if not (zoom > 0 and h > 0 and w > 0 and np.isfinite([zoom, h, w]).all()):
                return
            vis_y, vis_x = h / zoom, w / zoom                      # data pixels in view
            if vis_y >= n_y and vis_x >= n_x:
                return
            short, wanted = min(vis_y, vis_x), float(self.min_fov.value())
            if short < wanted:
                zoom = zoom * short / wanted
                cam.zoom = zoom
                vis_y, vis_x = h / zoom, w / zoom
            pos = []
            for c, vis, n in ((cy, vis_y, n_y), (cx, vis_x, n_x)):
                pos.append((n - 1) / 2 if vis >= n else float(np.clip(c, vis / 2, n - 1 - vis / 2)))
            cam.center = tuple(cam.center[:-2]) + tuple(pos)
        except Exception as e:                                     # never break the selection
            self.viewer.status = f"(view did not follow: {e})"

    def _force_2d(self, *_):
        """Keep the viewer in 2-D display (see __init__)."""
        if self.viewer.dims.ndisplay != 2:
            self.viewer.dims.ndisplay = 2
            self.viewer.status = "this tool only works in the 2-D view"

    def _plot(self):
        S, i, fig = self.S, self.cur, self.fig
        fig.clear()
        s = S.stat[i]
        cy, cx = int(np.mean(s["ypix"])), int(np.mean(s["xpix"]))
        h = 20
        y0, y1 = max(0, cy - h), min(S.green.shape[0], cy + h)
        x0, x1 = max(0, cx - h), min(S.green.shape[1], cx + h)
        c = S.contours[i] - [y0, x0]

        def norm(im):
            # contrast set per channel from the zoomed crop (not the whole FOV)
            lo, hi = np.percentile(im, [1, 99.8])
            return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)
        g = norm(S.green[y0:y1, x0:x1])
        r = norm(S.red[y0:y1, x0:x1]) if S.red is not None else np.zeros_like(g)
        z = np.zeros_like(g)
        # pure channel colours (no grey); overlap = red + green -> yellow
        g_panel, r_panel, o_panel = ("green", np.dstack([z, g, z]), {}), ("red", np.dstack([r, z, z]), {}), ("overlap", np.dstack([r, g, z]), {})
        mode = self.crop_box.currentIndex()
        sn, sp_note = None, None
        if mode == 1:                                   # the old three: green / red / overlap
            panels = [g_panel, r_panel, o_panel]
        else:                                           # the event snapshot instead of, or next to, the green mean image
            sn = S.snapshot(i)
            if sn is None:
                sp = ("snapshot", np.zeros_like(g), {"cmap": "gray", "vmin": 0, "vmax": 1})
                sp_note = "n/a (manual ROI, or fewer than 3 events)"
            else:
                zs, (scy, scx) = (sn[1] if self.snap_box.currentIndex() == 0 else sn[0]), sn[2]
                n_s = zs.shape[0]; top_, left_ = scy - n_s // 2 - y0 - 0.5, scx - n_s // 2 - x0 - 0.5
                sp = ("snapshot: " + ("strongest event" if self.snap_box.currentIndex() == 0 else "top-event average"), snapshot_display(zs),
                      {"cmap": "magma", "vmin": 0, "vmax": 1, "extent": (left_, left_ + n_s, top_ + n_s, top_)})
                sp_note = None
            panels = [sp, r_panel, o_panel] if mode == 0 else [g_panel, sp, r_panel, o_panel]
        fs = self.cfg["suite2p"]["fs"]
        d = S.dff_of(i)
        vel = S.behav.get(self.vel_box.currentText())
        lick = S.behav.get(self.lick_box.currentText())

        def behav_ax(ax, t, v, l):
            """velocity line + lick ticks on a shared time axis t (v, l binned like t)."""
            if v is not None:
                ax.plot(t, v, color="tab:blue", lw=0.6)
                ax.set_ylabel("velocity", fontsize=7, color="tab:blue")
            if l is not None:
                top = ax.get_ylim()[1] if v is not None else 1.0
                on = np.flatnonzero(l > 0)
                if len(on):
                    ax.vlines(t[on], top * 0.9, top, color="tab:orange", lw=0.6)
                ax.set_ylim(top=top)
                if v is None:
                    ax.set_yticks([]); ax.set_ylabel("licks", fontsize=7, color="tab:orange")
            ax.tick_params(labelsize=6)
            if v is None and l is None:
                ax.text(0.5, 0.5, "no behavior loaded", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])

        # The three crops use the full width of the canvas: squares with a thin gap, the titles
        # written inside the images instead of above them. The traces share the rest of the height.
        fw, fh = fig.get_size_inches() * fig.dpi
        mg, gap = 0.006, 0.01                          # outer margin, gap (fractions of the width)
        pw = (1 - 2 * mg - (len(panels) - 1) * gap) / len(panels)       # width of one crop
        ph = min(pw * fw / fh, 0.42)                   # square in pixels (at most 42 % of the height)
        top = 1 - mg * fw / fh
        for k, (title, im, kw) in enumerate(panels):
            ax = fig.add_axes([mg + k * (pw + gap), top - ph, pw, ph])
            ax.imshow(im, **kw)
            if "extent" in kw:                          # the snapshot window is smaller than the crop: same frame as the others
                ax.set_xlim(-0.5, x1 - x0 - 0.5); ax.set_ylim(y1 - y0 - 0.5, -0.5)
            if title == "snapshot" and sp_note:
                ax.text(0.5, 0.5, sp_note, transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.7", wrap=True)
            ax.plot(c[:, 1], c[:, 0], color="white", lw=0.8)
            ax.text(0.04, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color="white", bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0))
            ax.axis("off")
        gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.8, 1.2, 0.8], left=0.085, right=0.985,
                              bottom=0.055, top=top - ph - 0.035, hspace=0.5)

        # overview of the whole session
        b = max(1, len(d) // 4000)
        n = len(d) // b
        env = d[:n * b].reshape(n, b)
        t = np.arange(n) * b / fs / 60
        ax = fig.add_subplot(gs[0])
        ax.fill_between(t, env.min(1), env.max(1), color="k", lw=0)
        ax.set_ylabel("dF/F", fontsize=7); ax.tick_params(labelsize=6)
        axb = fig.add_subplot(gs[1], sharex=ax)
        binned = lambda x, fn: None if x is None else fn(x[:n * b].reshape(n, b), axis=1)
        behav_ax(axb, t, binned(vel, np.mean), binned(lick, np.max))
        axb.set_xlabel("min", fontsize=7)

        # most active 60 s
        z = min(len(d), int(60 * fs))
        st = int(np.argmax(np.convolve(d, np.ones(z) / z, "valid"))) if len(d) > z else 0
        tt = np.arange(st, st + z) / fs
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(tt, d[st:st + z], color="k", lw=0.6)
        ax2.set_title("most active 60 s", fontsize=7); ax2.tick_params(labelsize=6)
        ax3 = fig.add_subplot(gs[3], sharex=ax2)
        sl = lambda x: None if x is None else x[st:st + z]
        behav_ax(ax3, tt, sl(vel), sl(lick))
        ax3.set_xlabel("s", fontsize=7)
        self.canvas.draw_idle()

        f = S.feat.iloc[i]
        st1, st2 = self.labels.get(i, ["", ""])
        src = self.label_src.get(i, "")
        mine = f"<b>{st1 or '-'} / {st2 or '-'}</b>" + (f" [{src}]" if src else "")
        pl = self._pred_label_of(i)
        if pl:
            ps = self.pred.at[i, "p_spn"] if "p_spn" in self.pred else np.nan
            pr = self.pred.at[i, "p_red"] if "p_red" in self.pred else np.nan
            cmp = self._compare(i)
            tag = {"agree": " <span style='color:#2a9d3a'>(agrees with mine)</span>",
                   "disagree": " <span style='color:#d62828'>(DISAGREES with mine)</span>"}.get(cmp, "")
            pred_txt = (f"<br>predicted: <span style='color:{review.PRED_HEX.get(pl, '#888')}'>"
                        f"<b>{pl}</b></span>  p(SPN)={ps:.2f}  p(red)={pr:.2f}{tag}")
        elif S.p_spn is not None:
            pred_txt = f"  p(SPN)={S.p_spn[i]:.2f}  p(red)={S.p_red[i]:.2f}"
        else:
            pred_txt = ""
        self.info.setText(
            f"<b>ROI {i}{' (manual)' if i >= S.n_orig else ''}</b> "
            f"({self.pos + 1}/{len(self.queue)})  label: {mine}{pred_txt}<br>"
            f"diam {f['equiv_diam_um']:.1f} um, solidity {f['solidity']:.2f}, aspect {f['aspect']:.2f}, "
            f"nucleus {f['nucleus_shadow']:.2f}<br>"
            f"skew {f['skew']:.2f}, baseline_rel {f['baseline_rel']:.2f}, "
            f"events {f['event_rate']:.2f}/min, active {f['frac_active']:.3f}<br>"
            f"red z {f['red_z']:.1f}, core z {f['z_core']:.1f}, corr {f['red_corr']:.2f}, "
            f"ring ratio {f['red_ring_ratio']:.2f}, s2p redprob {f['s2p_redprob']:.2f}")
        self._update_counts()

    def _update_counts(self):
        vals = list(self.labels.values())
        c1 = pd.Series([v[0] for v in vals if v[0]]).value_counts().to_dict()
        c2 = pd.Series([v[1] for v in vals if v[1]]).value_counts().to_dict()
        allc = io.load_labels(self.store)
        srcs = pd.Series(list(self.label_src.values())).value_counts().to_dict()
        agree = ""
        if self.pred is not None:
            cmp = [self._compare(i) for i in range(self.S.n_orig)]
            agree = (f"<br>mine vs predicted: {cmp.count('agree')} agree, "
                     f"{cmp.count('disagree')} disagree")
        self.counts.setText(
            f"this session: {c1} {c2}  sources {srcs}{agree}<br>"
            f"all sessions: {allc['stage1'].value_counts().drop('', errors='ignore').to_dict()} "
            f"{allc['stage2'].value_counts().drop('', errors='ignore').to_dict()}")

    # ------------------------------------------------------------ labeling
    def set_label(self, stage1=None, stage2=None, clear=False, source="hand"):
        """source: 'hand' (typed), 'accepted' (a predicted label accepted with y); both are
        training labels. ('bulk' labels are written by accept_all_confident.)"""
        i = self.cur
        st1, st2 = self.labels.get(i, ["", ""])
        if clear:
            st1 = st2 = ""
        if stage1 is not None:
            st1 = stage1
            if stage1 != "SPN":
                st2 = ""
        if stage2 is not None:
            st2 = stage2
            st1 = "SPN"
        self.labels[i] = [st1, st2]
        self.label_src[i] = source
        io.append_label(self.store, {
            "session_id": self.S.sid, "session_dir": self.S.dir, "roi_idx": i,
            "stage1": st1, "stage2": st2, "labeler": self.cfg.get("labeler", ""),
            "s2p_hash": self.S.plane["hash"], "source": source})
        if clear:
            del self.labels[i]
            self.label_src.pop(i, None)
        self._update_overlay()
        advance = stage2 is not None or (stage1 is not None and stage1 != "SPN")
        if advance and self.auto_adv.isChecked():
            self.step(1)
        else:
            self._plot()

    def step(self, d, wrap=False):
        """Move d ROIs along the queue (the order of the 'order' menu). Nothing is labeled.
        wrap: past the last ROI go back to the first (and the other way round); otherwise stop."""
        if self.queue:
            new = self.pos + d
            if wrap and not 0 <= new < len(self.queue):
                self.viewer.status = ("end of the list: back to the first ROI" if d > 0
                                      else "start of the list: on to the last ROI")
            self.pos = int(new % len(self.queue)) if wrap else int(np.clip(new, 0, len(self.queue) - 1))
            self.select(self.queue[self.pos])

    # ------------------------------------------------------------ predicted -> labels -> retrain
    def accept_current(self, *_):
        """Key y: make the predicted class of this ROI my label (source 'accepted', a training
        label). A prediction of nonSPN is not accepted automatically: it could be an
        interneuron or junk, so press i or j. uncertain / not_cell are no call."""
        i = self.cur
        pred = self._pred_label_of(i)
        hand = review.class_to_hand(pred, self.cfg["cre_line"])
        if hand is None:
            what = {"nonSPN": "predicted nonSPN: press i (interneuron) or j (junk) to say which",
                    "": "no prediction loaded for this ROI ('load predicted')"}.get(
                        pred, f"predicted {pred}: no definite call to accept")
            self.pred_lbl.setText(what)
            self.viewer.status = f"ROI {i}: {what}"       # also in the status bar under the image
            return
        msg = f"ROI {i}: accepted {pred} -> {hand[0]} / {hand[1]}"
        self.pred_lbl.setText(msg)
        self.viewer.status = msg
        self.set_label(stage2=hand[1], source="accepted")
        self._focus_canvas()

    def accept_all_confident(self, *_):
        """Label every shown, still unlabeled ROI that the model calls dSPN or iSPN, in one go.
        Written as source 'bulk': they appear as my labels but are NOT used for training
        (unless train_on_bulk is true in the config), because training on the model's own
        calls that nobody looked at only teaches it its own beliefs."""
        from qtpy.QtWidgets import QMessageBox
        S = self.S
        vis = self._visible()
        todo = [i for i in range(S.n_orig)
                if vis[i] and i not in self.labels and self._pred_label_of(i) in ("dSPN", "iSPN")]
        if not todo:
            self.pred_lbl.setText("nothing to accept: no shown, unlabeled ROI is predicted dSPN / iSPN")
            return
        yes = QMessageBox.StandardButton.Yes
        self._focus_canvas()
        ask = QMessageBox.question(
            self.viewer.window._qt_window, "accept all confident",
            f"Label {len(todo)} unlabeled, shown ROIs with their predicted dSPN / iSPN class?\n\n"
            f"They are stored as 'bulk' labels: visible as your labels, but not used for "
            f"training unless train_on_bulk: true in the config. Reviewing ROIs one by one "
            f"(key y) makes them training labels.")
        if ask != yes:
            return
        for i in todo:
            st1, st2 = review.class_to_hand(self._pred_label_of(i), self.cfg["cre_line"])
            self.labels[i] = [st1, st2]
            self.label_src[i] = "bulk"
            io.append_label(self.store, {
                "session_id": S.sid, "session_dir": S.dir, "roi_idx": i, "stage1": st1, "stage2": st2,
                "labeler": self.cfg.get("labeler", ""), "s2p_hash": S.plane["hash"], "source": "bulk"})
        self._update_overlay()
        self._plot()
        self.pred_lbl.setText(f"{len(todo)} ROIs labeled from the prediction (source bulk, not in the training set)")

    def retrain_and_predict(self, *_):
        """Train on the whole label store (hand + accepted labels of ALL sessions, so this
        session's corrections join the training set automatically), then rewrite this
        session's celltype_model_<new id> files with the new model and show it."""
        from qtpy.QtWidgets import QApplication
        n = len(io.training_labels(self.cfg))
        self.pred_lbl.setText(f"training on {n} labeled ROIs (hand + accepted, all sessions) and "
                              f"predicting this session ... the window can freeze for a while")
        QApplication.processEvents()
        try:
            bundle = models.train(self.cfg)
            self.bundle = bundle
            for s in self.cache.values():              # every opened session: new probabilities
                s.set_probs(bundle)
            models.predict([self.S.dir], self.cfg, bundle)
        except Exception as e:
            self.pred_lbl.setText(f"<span style='color:red'>retrain failed: {e}</span>")
            return
        self._load_pred()
        self._refresh_edges()
        self.build_queue()
        self.pred_lbl.setText(f"retrained on {n} labeled ROIs; this session has a new prediction file "
                              f"({self.pred_msg}); older ones stay. Other sessions get the new "
                              f"model's files when you run the batch prediction.")

    def _bind_keys(self):
        keys = {"s": lambda: self.set_label(stage1="SPN"),
                "y": lambda: self.accept_current(),
                "i": lambda: self.set_label(stage1="interneuron"),
                "j": lambda: self.set_label(stage1="junk"),
                "u": lambda: self.set_label(stage1="unsure"),
                "r": lambda: self.set_label(stage2="red+"),
                "g": lambda: self.set_label(stage2="red-"),
                "a": lambda: self.set_label(stage2="ambiguous"),
                "x": lambda: self.set_label(clear=True),
                "m": lambda: self._draw_roi_manual(),
                "q": lambda: self._stop_manual_roi(),
                "d": lambda: self.delete_current(),
                "z": lambda: self.undo_delete(),
                "p": lambda: self._show_panel(),
                "n": lambda: self.step(1), "b": lambda: self.step(-1),
                # arrow keys: browse the ROIs in the current order without labeling, round and round
                "Right": lambda: self.step(1, wrap=True), "Left": lambda: self.step(-1, wrap=True)}
        # Application-wide Qt shortcuts rather than napari key bindings: napari only sees a key while
        # its canvas has the keyboard focus, and a click anywhere else (the trace plot, the layer
        # list, a panel) moves the focus away. Qt shortcuts fire wherever the focus is; only text
        # boxes (number boxes, name fields) keep the keys they need for typing.
        try:
            self._shortcuts = install_shortcuts(self.viewer.window._qt_window, keys,
                                                repeat=("Right", "Left", "n", "b"))
            self._sync_shortcuts()
        except Exception as e:
            print(f"  (Qt shortcuts not available: {e}; using napari key bindings)")
            self._shortcuts = {}
            for k, fn in keys.items():
                self.viewer.bind_key(k, (lambda f: lambda viewer: f())(fn), overwrite=True)

    # While a manual ROI is being drawn, napari's own keys of the shape layer (d, z, a, ...) must
    # reach it, so only these of our shortcuts stay on.
    DRAW_KEYS = ("m", "q", "p", "n", "b", "Right", "Left")

    def _drawing(self):
        l = getattr(self, "l_man", None)
        return l is not None and l.mode != "pan_zoom"

    def _sync_shortcuts(self, *_):
        drawing = self._drawing()
        for k, sc in getattr(self, "_shortcuts", {}).items():
            sc.setEnabled(not drawing or k in self.DRAW_KEYS)

    def _on_click(self, viewer, event):
        if getattr(self, "l_man", None) is not None and self.l_man.mode != "pan_zoom":      # drawing a manual ROI: clicks are vertices
            return
        y, x = (int(round(v)) for v in event.position[-2:])
        if 0 <= y < self.S.index_img.shape[0] and 0 <= x < self.S.index_img.shape[1]:
            i = self.S.index_img[y, x]
            if i >= 0 and self._visible()[i]:
                self.select(i, follow=False)


for _name in ("build_queue", "select", "_plot", "set_label", "step", "delete_current",
              "undo_delete", "save_celltype", "_draw_roi_manual", "_start_manual_roi",
              "_stop_manual_roi", "_pick_behavior", "_on_click", "_set_thr",
              "_fill_behavior_boxes", "_refresh_edges", "accept_current",
              "accept_all_confident", "retrain_and_predict",
              "_load_pred_clicked", "auto_contrast", "_set_projection"):
    setattr(LabelTool, _name, _needs_session(getattr(LabelTool, _name)))


def _project(mov, kind, n_frames=None):
    """'max' or 'std' projection over (at most n_frames) evenly spaced frames of mov[t] (array,
    memmap or TifMovie), one frame at a time."""
    n_all = mov.shape[0]
    idx = np.arange(n_all) if n_frames is None else         np.unique(np.linspace(0, n_all - 1, min(int(n_frames), n_all)).astype(int))
    mx = np.full(mov.shape[1:], -np.inf, np.float32)
    s1 = np.zeros(mov.shape[1:], np.float64)
    s2 = np.zeros(mov.shape[1:], np.float64)
    for i in idx:
        fr = np.asarray(mov[i:i + 1])[0].astype(np.float32)
        mx = np.maximum(mx, fr)
        s1 += fr
        s2 += fr.astype(np.float64) ** 2
    n = len(idx)
    out = mx if kind.startswith("max") else np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))
    return out.astype(np.float32)


def snapshot_display(z, lo_pct=5, hi_pct=99.8):
    """Contrast and gamma of a snapshot (noise units) for display: stretched between its low and high percentiles, then a gamma that
    brings the mean of the stretched image to about 0.3, so a weak blob is visible and a strong one does not saturate. 0..1."""
    z = np.asarray(z, np.float32)
    lo, hi = np.nanpercentile(z, [lo_pct, hi_pct])
    if not hi > lo:
        return np.zeros_like(z)
    n = np.clip((z - lo) / (hi - lo), 0, 1)
    m = float(np.nanmean(n))
    gamma = float(np.clip(np.log(0.3) / np.log(m), 0.4, 1.6)) if 0 < m < 1 else 1.0
    return n ** gamma


def install_shortcuts(window, callbacks, repeat=()):
    """One application-wide Qt shortcut per key name ('s', 'Right', ...): callbacks[key]() runs
    whichever widget has the focus, except one that needs the key for typing (a text or number
    box). Keys in `repeat` repeat while held. Returns {key: QShortcut}."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeySequence
    try:
        from qtpy.QtGui import QShortcut
    except ImportError:                                  # PyQt5 keeps it in QtWidgets
        from qtpy.QtWidgets import QShortcut
    out = {}
    for k, fn in callbacks.items():
        sc = QShortcut(QKeySequence(k), window)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.setAutoRepeat(k in repeat)
        sc.activated.connect(fn)
        out[k] = sc
    return out


def label_gui(sessions, cfg, model_id=None):
    import napari
    tool = LabelTool(sessions, cfg, model_id=model_id)
    napari.run()
    return tool
