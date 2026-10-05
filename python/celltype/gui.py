"""napari labeling tool for building / extending the training set.

Keys (viewer must have focus):
    stage 1   s = SPN   i = interneuron   j = junk (not a soma)   u = unsure
    stage 2   r = red+  g = red-  a = ambiguous      (sets stage 1 = SPN if empty)
    x = clear this ROI      n / b = next / previous ROI in the queue
Click an ROI in the image to jump to it. Every label is appended to the label
store immediately, so closing the window never loses work.
"""
import numpy as np
import pandas as pd

from . import features as ft
from . import io_utils as io
from . import models

EDGE = {"": (0.85, 0.85, 0.85, 0.45), "SPN": (1.0, 0.85, 0.1, 1.0),
        "red+": (1.0, 0.25, 0.75, 1.0), "red-": (0.2, 1.0, 0.35, 1.0),
        "ambiguous": (1.0, 0.6, 0.1, 1.0), "interneuron": (0.2, 0.75, 1.0, 1.0),
        "junk": (0.6, 0.1, 0.1, 1.0), "unsure": (1.0, 0.55, 0.0, 1.0),
        "not_cell": (0.4, 0.4, 0.4, 0.35)}
ORDERS = ["unlabeled first (random)", "most uncertain first (needs model)",
          "largest first", "highest baseline first", "most red first", "ROI index"]


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
        self.p_spn = self.p_red = None
        if bundle is not None:
            self.p_spn, self.p_red = models.predict_probs(bundle, self.feat)


class LabelTool:
    def __init__(self, sessions, cfg):
        import napari
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtWidgets import (QCheckBox, QComboBox, QLabel, QVBoxLayout, QWidget)

        self.cfg, self.store = cfg, cfg["label_store"]
        if not self.store:
            raise ValueError("set label_store in the config")
        try:
            self.bundle = models.load_model(cfg)
        except Exception as e:
            print(f"  (model not used in GUI: {e})")
            self.bundle = None
        self.sessions = [str(s) for s in sessions]
        self.cache = {}
        self.viewer = napari.Viewer(title="celltype labeling")

        w = QWidget(); lay = QVBoxLayout(w)
        self.sess_box = QComboBox(); self.sess_box.addItems(self.sessions)
        self.order_box = QComboBox(); self.order_box.addItems(ORDERS)
        self.include_noncell = QCheckBox("include Suite2p iscell = 0")
        self.auto_adv = QCheckBox("auto-advance"); self.auto_adv.setChecked(True)
        self.info = QLabel(); self.info.setWordWrap(True)
        self.counts = QLabel(); self.counts.setWordWrap(True)
        self.fig = Figure(figsize=(5, 6)); self.canvas = FigureCanvasQTAgg(self.fig)
        for wd in (QLabel("session"), self.sess_box, QLabel("ROI order"), self.order_box,
                   self.include_noncell, self.auto_adv, self.canvas, self.info, self.counts,
                   QLabel(__doc__.split("Click")[0].split("Keys")[1])):
            lay.addWidget(wd)
        self.viewer.window.add_dock_widget(w, area="right", name="ROI")

        self.sess_box.currentIndexChanged.connect(lambda _: self.load_session())
        self.order_box.currentIndexChanged.connect(lambda _: self.build_queue())
        self.include_noncell.stateChanged.connect(lambda _: self.build_queue())
        self._bind_keys()
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.load_session()

    # ------------------------------------------------------------ session
    def load_session(self):
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

        v = self.viewer
        for name in ("green", "red", "ROIs", "selected"):
            if name in v.layers:
                v.layers.remove(name)
        pct = lambda im, q: float(np.percentile(im, q))
        self.l_green = v.add_image(S.green, name="green", colormap="gray",
                                   contrast_limits=(pct(S.green, 1), pct(S.green, 99.7)))
        if S.red is not None:
            v.add_image(S.red, name="red", colormap="magenta", blending="additive",
                        contrast_limits=(pct(S.red, 1), pct(S.red, 99.7)))
        self.l_rois = v.add_shapes(S.contours, shape_type="polygon", name="ROIs",
                                   edge_width=0.6, face_color=[0, 0, 0, 0],
                                   edge_color=self._edge_colors())
        self.l_sel = v.add_shapes([S.contours[0]], shape_type="polygon", name="selected",
                                  edge_width=1.5, edge_color="yellow", face_color=[0, 0, 0, 0])
        for l in (self.l_rois, self.l_sel):
            l.editable = False
        v.layers.selection.active = self.l_green
        self.build_queue()

    def _label_of(self, i):
        st1, st2 = self.labels.get(i, ["", ""])
        if st1 == "SPN" and st2:
            return st2
        if st1:
            return st1
        return "not_cell" if self.S.feat["iscell"].iloc[i] == 0 else ""

    def _edge_colors(self):
        return np.array([EDGE[self._label_of(i)] for i in range(len(self.S.stat))])

    def build_queue(self):
        S, f = self.S, self.S.feat
        idx = np.arange(len(f))
        if not self.include_noncell.isChecked():
            idx = idx[f["iscell"].to_numpy()[idx] == 1]
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
    def select(self, i):
        self.cur = int(i)
        if self.cur in self.queue:
            self.pos = self.queue.index(self.cur)
        S = self.S
        self.l_sel.data = [S.contours[self.cur]]
        self.l_sel.edge_color = "yellow"
        self.viewer.layers.selection.active = self.l_green
        self._plot()

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
            lo, hi = np.percentile(im, [1, 99.7])
            return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)
        g = norm(S.green)[y0:y1, x0:x1]
        r = norm(S.red)[y0:y1, x0:x1] if S.red is not None else np.zeros_like(g)
        panels = [("green", g, "gray"), ("red", np.dstack([r, 0 * r, r]), None),
                  ("merge", np.dstack([r, g, r]), None)]
        for k, (title, im, cmap) in enumerate(panels):
            ax = fig.add_subplot(3, 3, k + 1)
            ax.imshow(im, cmap=cmap) if cmap else ax.imshow(im)
            ax.plot(c[:, 1], c[:, 0], color="yellow", lw=0.8)
            ax.set_title(title, fontsize=8); ax.axis("off")

        ax = fig.add_subplot(3, 1, 2)
        d = S.dff[i]
        b = max(1, len(d) // 4000)
        n = len(d) // b
        env = d[:n * b].reshape(n, b)
        t = np.arange(n) * b / self.cfg["suite2p"]["fs"] / 60
        ax.fill_between(t, env.min(1), env.max(1), color="k", lw=0)
        ax.set_xlabel("min", fontsize=7); ax.set_ylabel("dF/F", fontsize=7)
        ax.tick_params(labelsize=6)
        ax2 = fig.add_subplot(3, 1, 3)
        z = min(len(d), int(60 * self.cfg["suite2p"]["fs"]))
        st = int(np.argmax(np.convolve(d, np.ones(z) / z, "valid"))) if len(d) > z else 0
        tt = np.arange(st, st + z) / self.cfg["suite2p"]["fs"]
        ax2.plot(tt, d[st:st + z], color="k", lw=0.6)
        ax2.set_title("most active 60 s", fontsize=7); ax2.tick_params(labelsize=6)
        fig.tight_layout()
        self.canvas.draw_idle()

        f = S.feat.iloc[i]
        st1, st2 = self.labels.get(i, ["", ""])
        probs = (f"  p(SPN)={S.p_spn[i]:.2f}  p(red)={S.p_red[i]:.2f}"
                 if S.p_spn is not None else "")
        self.info.setText(
            f"<b>ROI {i}</b> ({self.pos + 1}/{len(self.queue)})  label: "
            f"<b>{st1 or '-'} / {st2 or '-'}</b>{probs}<br>"
            f"diam {f['equiv_diam_um']:.1f} um, solidity {f['solidity']:.2f}, aspect {f['aspect']:.2f}, "
            f"nucleus {f['nucleus_shadow']:.2f}<br>"
            f"skew {f['skew']:.2f}, baseline_rel {f['baseline_rel']:.2f}, "
            f"events {f['event_rate']:.2f}/min, active {f['frac_active']:.3f}<br>"
            f"red z {f['red_z']:.1f}, ring ratio {f['red_ring_ratio']:.2f}, "
            f"s2p redprob {f['s2p_redprob']:.2f}")
        self._update_counts()

    def _update_counts(self):
        vals = list(self.labels.values())
        c1 = pd.Series([v[0] for v in vals if v[0]]).value_counts().to_dict()
        c2 = pd.Series([v[1] for v in vals if v[1]]).value_counts().to_dict()
        allc = io.load_labels(self.store)
        self.counts.setText(
            f"this session: {c1} {c2}<br>"
            f"all sessions: {allc['stage1'].value_counts().drop('', errors='ignore').to_dict()} "
            f"{allc['stage2'].value_counts().drop('', errors='ignore').to_dict()}")

    # ------------------------------------------------------------ labeling
    def set_label(self, stage1=None, stage2=None, clear=False):
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
        io.append_label(self.store, {
            "session_id": self.S.sid, "session_dir": self.S.dir, "roi_idx": i,
            "stage1": st1, "stage2": st2, "labeler": self.cfg.get("labeler", ""),
            "s2p_hash": self.S.plane["hash"]})
        if clear:
            del self.labels[i]
        cols = self.l_rois.edge_color
        cols[i] = EDGE[self._label_of(i)]
        self.l_rois.edge_color = cols
        advance = stage2 is not None or (stage1 is not None and stage1 != "SPN")
        if advance and self.auto_adv.isChecked():
            self.step(1)
        else:
            self._plot()

    def step(self, d):
        if self.queue:
            self.pos = int(np.clip(self.pos + d, 0, len(self.queue) - 1))
            self.select(self.queue[self.pos])

    def _bind_keys(self):
        keys = {"s": lambda: self.set_label(stage1="SPN"),
                "i": lambda: self.set_label(stage1="interneuron"),
                "j": lambda: self.set_label(stage1="junk"),
                "u": lambda: self.set_label(stage1="unsure"),
                "r": lambda: self.set_label(stage2="red+"),
                "g": lambda: self.set_label(stage2="red-"),
                "a": lambda: self.set_label(stage2="ambiguous"),
                "x": lambda: self.set_label(clear=True),
                "n": lambda: self.step(1), "b": lambda: self.step(-1)}
        for k, fn in keys.items():
            self.viewer.bind_key(k, (lambda f: lambda viewer: f())(fn), overwrite=True)

    def _on_click(self, viewer, event):
        y, x = (int(round(v)) for v in event.position[-2:])
        if 0 <= y < self.S.index_img.shape[0] and 0 <= x < self.S.index_img.shape[1]:
            i = self.S.index_img[y, x]
            if i >= 0:
                self.select(i)


def label_gui(sessions, cfg):
    import napari
    tool = LabelTool(sessions, cfg)
    napari.run()
    return tool
