# celltype: SPN / interneuron and dSPN / iSPN classification

Runs on suit2p output`:

1. **Features**: computes per-ROI morphology, activity and red-channel features. They are cached in `<session>\celltype\features.csv` (with `features.version` next to it, saying which model and settings they were made for). There is no feature version: `train` computes the features of the labeled sessions again, and `predict` computes the features of every session it predicts again unless they were already made for that model (a session this model has predicted is skipped before any feature work). The GUI and `--mode features` use any cache made with the current settings.
2. **Label, train or predict**, using two stages:
   - **Stage 1, SPN vs non-SPN:** uses morphology, activity and event-snapshot features. It is semi-supervised (self-training). With fewer than 20 non-SPN labels it falls back to an anomaly detector fit on SPNs. Size, shape and edge hard rules also apply.
   - **Stage 2, red+ vs red−:** uses red features only, with label spreading among SPNs. Red+ maps to dSPN for D1-Cre and to iSPN for D2-Cre.

## Setup

Install the dependencies:

```
pip install -r celltype\requirements.txt
```

## Usage

Run these from `Calcium_2p_preprocessing\python`. First edit `celltype\config.yaml`: set `inputs` (or `inputs_file`), `um_per_px`, `cre_line`, `label_store` and `model_dir`.

```
python -m celltype.pipeline --config celltype\config.yaml --mode label     # build a training set
python -m celltype.pipeline --config celltype\config.yaml --mode train     # fit + cross-validation report
python -m celltype.pipeline --config celltype\config.yaml --mode predict   # label unseen sessions
python -m celltype.pipeline --config celltype\config.yaml --mode auto      # whichever is needed
```

Many sessions (predict them all, then review them one after the other): `python -m celltype.batch_celltype`,
see "Batch prediction and review queue" below.

`--model-id 20261021-1432` (predict, auto, label, and `batch_celltype`) uses that saved model instead of the latest one; without it the latest model, i.e. the newest id in `model_dir`, is used (the config key `model_id` pins one for every command). `--refresh-features` recomputes the cached tables of `--mode features` / `label`. Suite2p must already have been run on each session (`<session>\suite2p\plane0`).

## Labeling GUI (napari)

| Key | Label |
|---|---|
| `s` | SPN |
| `i` | interneuron |
| `j` | junk (not a soma) |
| `u` | unsure |
| `r` | red+ |
| `g` | red− |
| `a` | ambiguous red |
| `y` | accept the predicted label (dSPN / iSPN / SPN_ambiguous; for nonSPN press `i` or `j`) |
| `x` | clear the label |
| `n` / `b` | next / previous ROI |
| `→` / `←` | next / previous ROI in the order of the *order* menu, without labeling anything; wraps around at the ends |

- **The keys work wherever you clicked** (image, trace plot, layer list, panels): they are Qt shortcuts, not napari key bindings. The exceptions: while you type in a number or text box the keys stay with the box, and while a manual ROI is being drawn the shape layer keeps its own letter keys (`d`, `z`, `s`, ...); `m`, `q`, `p`, `n`, `b` and the arrows still work then.
- **Selecting ROIs:** click an ROI in the image to jump to it.
- **ROI order:** the "ROI order" menu changes the order. Choose "largest first" or "highest baseline first" to find interneurons quickly. Once a model exists, choose "most uncertain first" for active learning.
- **Saving:** labels are appended to `label_store` as soon as you press a key. Each label is tied to the Suite2p run through the `s2p_hash` column. If you rerun Suite2p on a session, its old labels are ignored.

## Features added in v4 to v6, and the edge rule

- **Stage 1 (SPN vs non-SPN)** gets `green_blob` (image) and three trace features: `neuropil_corr` (the ROI's trace against
  its own neuropil), `common_mode_corr` (against the median trace of the cells) and `rise_decay_asym` (fast rise, slow
  decay), next to `skew` which was already there. On the labeled ROIs of one session `skew` (AUC 0.88) and `rise_decay_asym`
  (0.82) separated SPN from junk best, `neuropil_corr` and `common_mode_corr` about 0.70.
- **Event snapshot (v6), for stage 1.** For every ROI that is a cell and not an edge ROI, the movie around the ROI's own strongest events
  (the top 6 peaks of its dF/F above `event_z` noise SDs, at least 1 s apart, 3 frames averaged each; at least 3 events are needed) minus the
  movie at quiet frames, as a 32 x 32 px crop centred on the ROI, divided by its robust noise (noise units) and smoothed by 1 px. Six features
  come from it (distances in ROI radii; "near" = within 2 radii of the ROI centre): `snap_peak_z` and `snap_mean_z` (strongest and mean
  value inside the ROI), `snap_inside_frac` (share of the positive signal near the ROI that falls inside it), `snap_corr_mask` (correlation
  with the ROI mask), `snap_offset` (distance of the signal's centre of mass from the ROI centre) and `snap_spread` (its RMS width). A ROI
  with too few events, a non-cell and an edge ROI get NaN, which the models fill with the median. The snapshot images (average of the top events
  and the strongest event) are saved with the features in `<session>\celltype\snapshots.npz`, which the GUI reads. They need the movie of
  every session (`data.bin`: fast, the registered tif: slow); `snapshot_features: false` in the config turns them off (the `snap_*` columns stay
  NaN). Tests on the labeled sessions: with boosted trees the snapshot raised the held-out AUC of stage 1 from 0.958 to 0.976 (per session
  0.925 to 0.982) at the same number of confident mistakes; with logistic regression the gain was small.
- **Green mean and max projections (v5).** `green_z`, `green_max_z`, `green_max_rel`, `green_max_blob`, `max_mean_logratio` (from the mean image
  and `ConcatenatedMovie_MC_max_registered.tif`) are still computed and kept in the table, but they are no longer model inputs: the whole-frame
  max did not help reliably and the event snapshot does its job better.
- **Stage 2 (red+ vs red-)** gets `red_blob`. Earlier tests gave it little weight, so check it with the notebook comparison
  (with / without) when you retrain.
- Earlier red features (kept): `red_ring_ratio`, `red_z`, `red_fill`, `red_corr`, `red_mean_rel`, `z_core`, `z_coreB`.
  `s2p_redprob` is shown in the table and the CV report but is not a model input. Stage 2 is a class-balanced logistic
  regression (`model2: logreg`).
- **Edge ROIs are junk before any model is asked.** The red reference is a separate acquisition; mapped onto the green
  frame it can lack data at the borders (zero strips from cropping, or the rows lost by the shift). For both images a
  mask of where they hold data is made (`io.load_images_valid`: non-finite pixels, constant border rows / columns,
  border-touching zero regions of 20 pixels or more, the shift and resampling applied to the red mask, and for green also
  Suite2p's valid x / y range). Every ROI gets `red_oob_frac` and `green_oob_frac`, the share of its pixels outside
  the valid area (NaN without a red image). An ROI above `hard_rules.max_oob_frac` (default 0 = any pixel) is labeled
  nonSPN by the hard rule, is marked `edge_roi` in `celltype_model_<id>.csv`, and is left out of the unlabeled pool the
  models learn from. `hard_rules.max_oob_frac` is a fraction (0 = any pixel outside; 0.1 = only ROIs with more than 10 % of their pixels outside are discarded), it is one limit for both images, `edge_margin_px` (config, default 0) also counts that many pixels around the ROI. The messages
  of the feature step say how many ROIs were flagged. Because the rule is stored in the model, retrain after changing it.
- Retrain (`--mode train`) after these feature changes. Stage 1 is the mean of the self-trained logistic regression and the self-trained boosted trees (`model1: ensemble`; `selftrain_logreg` or `selftrain_hgb` alone are still options),
  stage 2 stays a logistic regression on its red features (boosted trees were worse there, and the red-green features added nothing). Each model stores
  the feature settings it was trained with (`snapshot_features`, `edge_margin_px`, `hard_rules.max_oob_frac`); predicting with it (also an older one chosen
  with `--model-id` / `model_id`) computes the features with those settings and says so when the config differs. Loading a model whose feature columns
  `features.py` no longer computes stops with "retrain".

## Browsing the ROIs: order menu and arrow keys

The **order** menu (Filter tab) decides the sequence of the ROIs; `→` / `←` walk through it
(no label is set, so you can just look at the predictions; click the image first so the keys reach
it). Besides the older orders there are:

- **p(red): most uncertain first** (p closest to 0.5) / **most certain first** / **high to low**
  (red+ to red−). ROIs the model calls SPN come first, because p(red) means nothing for the others;
  ROIs without a probability (manual ROIs) come last.
- **p(SPN): most uncertain first / most certain first.**
- **disagreements first (mine vs predicted).**

The probabilities are those of the loaded predictions (`celltype_model_<id>.csv`, or computed live from the
model); with neither, the status bar says so and the order is left as it was.

## Panel layout (right side)

Session menu, `Open...` and `Save` on top, then three small tabs instead of one long stack of
buttons: **Filter** (order, iscell and diameter limits, behavior, auto-advance), **ROIs** (draw /
compute / select, delete / undo) and **Predicted** (show menu, load / accept / accept all, retrain,
colour legend). Buttons carry short names; the full name and the key are in
the tooltip. Messages (manual ROI, saves, predictions) appear in a line under the tabs only while
there is one. In the plot the three crops (green / red / overlap) are squares as wide as the panel
allows, with the titles written inside the images, and the traces use the rest of the height; the
plot is redrawn when the panel is resized.

## Display controls (left dock "view display")

- **Follow ROI** (checkbox at the top of the panel, on by default; smallest view in the dock): when you
  step to another ROI (arrows, `n` / `b`, after a label) while zoomed in, the view centres on it. It
  never shows less than the *smallest view* (120 px on the shorter side; it zooms out to that), never
  goes past the image edge, and does nothing when you are zoomed out. Clicking an ROI never moves the view.

- **green + red layers show** (the max projection of the green layer is read from `ConcatenatedMovie_MC_max_registered.tif` in the session folder when it is there, otherwise computed from frames; the std projection is always computed): mean image, max projection or std projection (green: of the movie; red: of the frames of the red reference tif, if it has several; a single red image stays as it is) (computed from
  that many evenly spaced frames, once per setting). napari's own "projection mode" menu in the
  layer controls only works on 3-D data, so it does nothing on these 2-D images; use this menu.
- **crops under the view** (dock menu): the zoomed crops of the selected ROI can be `snapshot / red / overlap` (default: the snapshot replaces the green
  mean image), `green / red / overlap`, or `green / snapshot / red / overlap`. The snapshot is the event snapshot of the ROI (see above) with an automatic
  contrast (stretched between its 5th and 99.8th percentile) and gamma (the mean of the stretched image goes to about 0.3), in the same frame and with the same ROI
  outline as the other crops. "snapshot shows" chooses the strongest single event or the average of the top events (what the features use). Rolling through
  the ROIs is instant: the snapshots are read from `snapshots.npz` (made with the features). Without that file a snapshot is computed for the ROI you select
  from the movie, once (about 0.1 s with `data.bin`, much longer with the tif), and remembered; manual ROIs and ROIs with fewer than 3 events show "n/a".
- **auto-contrast view:** sets the contrast limits of the green and/or red layer from the pixels
  currently in view (low / high percentile boxes, 1 and 99.7 by default), so after zooming in the
  display is stretched to that region. "auto gamma" also sets gamma so the median of the view is
  mid-grey; the gamma boxes set it by hand. "follow zoom / pan" repeats the auto-contrast a moment
  after every zoom or pan.

- **red blob / green blob layers:** two extra image layers (hidden at first; tick their eyes in napari's layer
  list) showing where the red and the green image have soma-shaped bright spots: a Laplacian-of-Gaussian matched
  to `suite2p.diameter`, in noise units over the tissue (red: magma colours, green: viridis; contrast from 0).
  They are only for checking by eye that somata are the bright spots, and not a model input. Each is recomputed
  from whatever its layer shows, so they follow the max / std projection menu.

## Batch prediction and review queue

```
python -m celltype.batch_celltype predict --config celltype\config.yaml            # all sessions of the config
python -m celltype.batch_celltype predict --config celltype\config.yaml --force    # again, even if done
python -m celltype.batch_celltype review  --config celltype\config.yaml            # GUI on the sessions still to review
python -m celltype.batch_celltype status  --config celltype\config.yaml            # list by state, change nothing
```
All three take `--sessions <folder> ...` (a session, its suite2p folder, or a folder above many) instead
of the `inputs` of the config.

**Files per session** (in `<session>\celltype\`, nothing in the suite2p folder). `<id>` is the training
time of the model (`20261021-1432`), so it tells the models apart:

| file | what | written by | overwritten? |
|---|---|---|---|
| `celltype_model_<id>.csv` / `.mat` | what model `<id>` predicted: `roi_idx`, `roi_idx_matlab`, `iscell`, `p_spn`, `p_red`, `hard_rule`, `final_label`, `s2p_hash`, `model_id` (no traces) | `batch_celltype predict`, `--mode predict`, the GUI's `Retrain` | never by anything else; a new model writes new files |
| `celltype_model_<id>_qc.png` | QC figure of that prediction | same | with it |
| `celltype_curated_<id>.mat` / `.csv` | the edited version: the cell-type table of the kept ROIs, built on the prediction of model `<id>`, with the Fall.mat row of every ROI and the pixels of added ROIs; no traces | the GUI's `Save` | by the next `Save` of the same `<id>` only |
| `celltype_status.json` | summary: state, latest model id, edited ids, `fall_file` | predict and `Save` | every time (a summary; the files are the truth) |

- **Matching the Suite2p Fall.mat (no file here holds traces; they stay in `Fall.mat`):** every file carries `s2p_hash` (a hash
  of that run's `stat.npy`, the same value in the edited `.mat`) and the `.mat` files carry `fall_file`, the path of that run's
  `Fall.mat`. The prediction rows are in Suite2p order: row k (`roi_idx_matlab` = k) is `stat{k}` / row k of `F`, `Fneu`,
  `iscell` in `Fall.mat`, whether or not you ever opened the session in the GUI. In the edited file every row carries
  `fall_row` (that same 1-based row, empty for an ROI you added), `orig_index` (0-based, -1 for an added ROI) and `manual_id`;
  the ROIs you deleted are listed in `deleted_roi_idx`, and every added (manual) ROI has its pixels in `manual_stat`
  (`ypix`, `xpix` 0-based, `lam`), from which its trace can be extracted again from the movie. In MATLAB:
  `ct = load('celltype_curated_<id>.mat'); Fall = load('Fall.mat'); k = ct.celltype.fall_row(ct.celltype.manual == 0); F = Fall.F(k, :);`
- **The latest version** of a session is the file with the newest `<id>`: the edited one if it exists for that
  id, else the prediction. Older files are kept.
- **A new model:** retrain, then run `batch_celltype predict` again: every session gets the new model's files
  (the old ones stay), and a session with the files of the current model for its current Suite2p run is skipped,
  so rerunning is cheap. A Suite2p rerun changes `s2p_hash`, so that session is predicted again.
- **Your edits carry over:** hand labels (label store), deleted and manual ROIs live outside the files, so
  the edited file of the new model starts from the new prediction plus all of them. Use *disagreements only*
  to see what the new model changes.

**Session states** (`status` lists them):
- `no_prediction`: no prediction of the current Suite2p run yet (run `predict`)
- `pending`: predicted by the newest model, nothing saved yet
- `stale`: only edited under an older model (needs a look at the newest prediction)
- `reviewed`: an edited file exists for the newest model

**Review queue:** `batch_celltype review` opens the GUI with the `pending` sessions, then the `stale` ones, in the
session menu. The **Next >** button (top of the panel, next to `Open...`) loads the next session of the menu that
is still `pending` (then `stale`), wrapping around; a session you only looked at stays in the queue until you
`Save` it. After a `Save` the message line tells how many sessions are left. The GUI also works without the
queue: **Next >** goes through whatever is in the session menu. Without any session the command prints what is
missing (usually: run `predict` first).

## Regularization of the logistic regressions

Both logistic regressions (stage 1 only if `model1: selftrain_logreg`, which is no longer the default; stage 2 with `model2: logreg`) are class-balanced and
L2-penalised: the penalty shrinks the weights, and `C` is its inverse strength (smaller C = stronger shrinkage; 1.0 is
scikit-learn's default and what was used before this setting existed). With few labels per class (stage 1 has about 130
non-SPN labels) and 30 or so features, a stronger penalty can make the models more stable on a new session.
- `regularization.stage1_C`, `stage2_C` in the config set C for each stage.
- `regularization.tune: true` chooses C at every training from `C_grid` by cross-validation that holds out whole sessions
  (the strongest penalty whose held-out AUC is within 0.002 of the best); the choice and the AUC of each C are printed and the
  C used is stored in the model (`bundle["logreg_C"]`). The cross-validation report of `--mode train` re-tunes C inside every
  fold, so its numbers include the tuning.

## Reviewing and correcting a predicted session

Open a session that has been through the batch prediction (or `--mode predict`), or any session once a model exists.

- **Show** menu: *my labels* (outline colour, as before), *predicted* (the model's class),
  *both* (fill = predicted class, outline = my label) or *none*. *disagreements only* hides
  every ROI where my label and the prediction agree or one of them has no call, and the order
  menu can list disagreements first. A pink fill with a green outline is an ROI the model calls
  dSPN and I called red−.
- **Where the predictions come from:** the newest `<session>\celltype\celltype_model_<id>.csv` that belongs to
  this Suite2p run (it carries `s2p_hash`); else the `labels.csv` of the older scheme; otherwise they are
  computed live from the model in `model_dir`. `load predicted` re-reads them.
- **Correcting:** `y` accepts the predicted class of the current ROI as my label; typing a
  label replaces it; `accept all confident...` labels every shown, unlabeled dSPN / iSPN ROI at once.
- **Label sources** (new `source` column of the label store; older stores are upgraded once, with
  a `.bak_before_source` copy): `hand` (typed) and `accepted` (key `y`) are the training labels;
  `bulk` (accept all) is not used for training unless `train_on_bulk: true` is set in the config,
  because training on the model's own unreviewed calls only teaches it its own beliefs.
- **Adding the session to the training set:** nothing extra to do. The label store is the training
  set, so every label you set or accept here is already in it; the saved table of a session
  (`Save`, below) says which ROIs are in it (`used_in_training`).
- **`Retrain`:** trains on the whole store (all sessions, hand + accepted labels),
  writes this session's `celltype_model_<new id>.*` and QC figure with the new model and shows them; the
  older model's files stay. Other sessions get the new model's files when you run `batch_celltype predict`.
- **`Save`** (top of the panel; no retraining) writes two files into `<session>\celltype\` and nothing into the
  suite2p folder, named after the id of the prediction you are editing (the loaded `celltype_model_<id>`; the
  loaded model's id if the predictions were computed live). The next `Save` of the same id overwrites them;
  the model's own `celltype_model_<id>` files are never touched:
  - `celltype_curated_<id>.csv`: one row per kept ROI (deleted ROIs left out, manual ROIs last) with
    `row`, `roi_idx`, `roi_idx_matlab`, `orig_index` (-1 = manual), `fall_row`, `manual_id`, `manual`, `deleted`, my label
    (`hand_stage1/2`, `hand_source`), the prediction (`pred_label`, `p_spn`, `p_red`), `agreement`,
    `final_label` (mine where I gave one, else the model's; `final_source` says which) and
    `used_in_training`.
  - `celltype_curated_<id>.mat`: the struct `celltype` whose fields are exactly the csv columns (text columns are cell arrays)
    and `model_id`, `s2p_hash`, `fall_file`, plus `orig_index`, `is_manual`, `deleted_roi_idx` and `manual_stat` (see "Matching the
    Suite2p Fall.mat" above). No traces and no Suite2p `stat`: row k of the table is a ROI whose trace is `Fall.F(fall_row, :)`.
  Both files are written under temporary names and swapped in together, so a failed save leaves the
  previous pair as it was. The history is not touched: the label store, `roi_edits.json`,
  `manual_rois.pkl` and the `celltype_model_<id>` prediction files.

## Detailed feature calculation
# Features and calculations, by stage

Notation: F, Fneu = Suite2p raw and neuropil traces (ROI x frames); fs = frame rate; m = the ROI's
pixel mask; med = median; MAD = median absolute deviation; sigma_rob = 1.4826 * MAD (a robust SD).
Source: `features.py` (features), `models.py` (models, labels), `config.yaml` (numbers in brackets).

## Stage 1: SPN vs non-SPN   (inputs: MORPH + ACTIVITY)
### Morphology (from the ROI mask and the green mean image g = green - 1st percentile)
- area            A = number of pixels in m;   area_um2 = A * um_per_px^2
- equiv_diam_um   d = 2 * sqrt(A / pi) * um_per_px
- eccentricity, solidity (area / convex-hull area): `skimage.regionprops` of the mask
- aspect          = major axis length / minor axis length of the fitted ellipse
- circularity     = min(4*pi*A / P^2, 1),  P = mask perimeter
- nucleus_shadow  = mean g over the outer ring of the mask / mean g over its core,
                    with r = distance from the centroid / sqrt(A/pi);  core: r < 0.4, outer: r > 0.6
                    (GCaMP is cytosolic, so SPN somata have a dim centre, ratio > 1)
- green_rel       = mean g inside m / median over ROIs of that mean   (brightness relative to the field)
- snap_*          = the six event-snapshot features (see "Features added in v4 to v6"): snap_peak_z, snap_mean_z, snap_inside_frac,
                    snap_corr_mask, snap_offset, snap_spread
- green_blob      = strongest soma-shaped bright spot of the green image within half a diameter of the ROI centre
                    (Laplacian-of-Gaussian matched to `suite2p.diameter`, in noise units over the tissue; same map as the
                    'green blob' layer of the GUI)
- s2p_compact     = Suite2p's `compact`

### Activity (dF/F from the neuropil-corrected trace; frames flagged bad are dropped)
- Fc = F - c * Fneu                      c = neuropil_coef [0.7]
- F0c = rolling percentile of Fc   (1-s bins, window 60 s, percentile 8);  F0 = same for F
- dF/F = (Fc - F0c) / max(F0c, 0.1*|F0| + 1e-6)         (denominator floored)
- noise_sd = med|d - med(d)| / 0.6745 / sqrt(2),  d = frame-to-frame difference of dF/F
- z(t) = dF/F(t) / noise_sd
- events: runs of z > 3 [event_z] lasting >= 0.1 s [event_min_dur_s]; amplitude = max dF/F in the run
- event_rate      = events per minute;   event_amp_med = median amplitude (0 if none)
- event_amp_cv    = SD(amplitudes) / mean(amplitudes)   (NaN with < 2 events)
- frac_active     = fraction of frames with z > 2
- skew            = skewness of the dF/F trace
- baseline_rel    = median F0 of the ROI / median over ROIs of that value
- baseline_over_neuropil = median(F) / median(Fneu)
- acf_1s, acf_5s  = autocorrelation of dF/F at lag 1 s and 5 s:  sum x(t) x(t+L) / sum x(t)^2, x = dF/F - mean
- snr             = 95th percentile of dF/F / noise_sd
- common_mode_corr = correlation of the dF/F (drift slower than 2 s removed) with the median trace of the cells
                    (at most 300 of them): shared brain-wide activity or motion
- neuropil_corr   = the same with the ROI's own neuropil trace (F against Fneu): how much of the ROI is its surround
- rise_decay_asym = skewness of the frame-to-frame changes of the 0.2 s-smoothed dF/F: transients rise fast and decay
                    slowly (positive), noise is symmetric (about 0)

### Model and calls
- Missing values are replaced by the column median, then each column is standardised.
- With >= `min_labels_per_class` SPN and non-SPN labels: self-training logistic regression
  (or gradient boosting), class-balanced, adding unlabeled cells whose predicted probability is > 0.95.
  p_spn = P(SPN | features).
  Otherwise: Isolation Forest fit on SPNs; p_spn = fraction of training SPNs with a lower
  anomaly score than this ROI (a percentile).
- Extra positives: unlabeled cells with p_red > 0.97 (from stage 2) that pass the hard rules are added as SPNs.
- Hard rule (always non-SPN): d > 20 um, or solidity < 0.80, or aspect > 2.5, or an edge ROI (below).
- Call: p_spn < 0.2 or hard rule -> nonSPN;  0.2 <= p_spn < 0.8 -> uncertain;  else SPN. (iscell = 0 -> not_cell.)

## Stage 2: red+ vs red-   (inputs: RED only, so activity never enters the dSPN/iSPN decision)

Red image r = red reference - 1st percentile (optional green bleed-through term, off).
Ring for each ROI: pixels within R_out = max(6, round(0.6 * diameter)) of the mask, outside a
2-pixel dilation of it, and not belonging to another ROI when >= 10 such pixels exist.
vals = r inside m;  bg = r on the ring;  bg_med = med(bg);  bg_sd = 1.4826 * MAD(bg).

- red_ring_ratio = mean(vals) / mean(bg)
- red_z          = (mean(vals) - bg_med) / bg_sd            (ROI mean above its surround, in noise units)
- red_fill       = fraction of ROI pixels with r > bg_med + 2 * bg_sd
- red_corr       = Pearson correlation between the ROI weights (Suite2p `lam`) and r over the dilated region
- red_mean_rel   = mean(vals) / median over ROIs of mean(vals)
- s2p_redprob    = Suite2p's own red-cell probability (`redcell`): shown in the table / GUI and used as the baseline
                   in the cv report, but NOT a model input

- z_core         = (mean r in a disc of 0.5 * radius at the filled mask's centre - bg_med) / bg_sd
- z_coreB        = (mean r in that disc - med of a second ring) / its MAD-SD; second ring: 5 px gap, out to R_out + 4
- red_blob        = the same blob measure on the red image (the 'red blob' layer of the GUI)

### Model and calls
- Missing values -> column median, standardise, then class-balanced logistic regression on your red+/red- SPN
  labels (`model2: logreg`; `labelspreading` = kNN graph, 7 neighbours, alpha 0.2, is still available).
  p_red = P(red+ | features).
- Call (for cells that are SPN): p_red > 0.8 -> red+ (dSPN for D1-Cre, iSPN for D2-Cre);
  p_red < 0.2 -> red-; otherwise SPN_ambiguous.




## Outputs

**Per session, in `<session>\celltype\`** (full table of files, who writes them and what is overwritten:
"Batch prediction and review queue" above):
- `celltype_model_<id>.csv` / `.mat`: the prediction of model `<id>` (`roi_idx` 1-based in the .mat, `p_spn`,
  `p_red`, `final_label`, `s2p_hash`, `model_id`, `fall_file`; no traces), and `celltype_model_<id>_qc.png`.
  The possible values of `final_label` are dSPN, iSPN, SPN_ambiguous, nonSPN, uncertain and not_cell.
- `celltype_curated_<id>.mat` / `.csv`: the edited cell-type table with the Fall.mat row of every ROI (no traces), written by the GUI's `Save` only.
- `celltype_status.json`: a summary of the states above.
- The history: the label store, `roi_edits.json`, `manual_rois.pkl`, the feature cache `features.csv` and the event snapshots `snapshots.npz`.

**In `model_dir`** (every training keeps its model under its id; `<id>` is the same key as in the prediction files'
`celltype_model_<id>.csv` and in their `model_id` column, so a prediction can always be traced to its model):
- `celltype_model_<id>.joblib`: the model trained at that time, kept for good. **The latest model is the one with the newest id:**
  every command that needs a model (predict, auto, the GUI, `batch_celltype`) finds it from the file names, so there is no separate
  "latest" file. `--model-id <id>` (or the config key `model_id`) picks a specific one; `models.load_model(cfg, model_id="<id>")`
  loads it in a notebook and `models.list_models(cfg)` lists them.
- `cv_report_<id>.json` and `cv_predictions_<id>.csv`: its cross-validation report (with the Suite2p `redcell` baseline) and predictions.
- Two trainings within the same minute: the later one gets the next minute as its id, so ids never repeat.
- An unversioned `celltype_model.joblib`, `cv_report.json` or `cv_predictions.csv` in `model_dir` comes from an earlier version of this
  code. It is not written any more; the unversioned model is read only if there is no model with an id, and it is saved under its
  own id the next time you train. You can delete the unversioned files once their model exists under an id.

## CNN and stacking experiment (`model_test.py`)

Not part of the pipeline: a test of whether a small neural network, alone or combined with the feature model, beats the
logistic regression on the hand-built features. Needs PyTorch (CPU is enough: `mt.torch_status()` says what is installed).

- **Inputs per ROI:** a 32 x 32 crop with five channels (green and red brightness relative to the field, green blob, red
  blob, the ROI mask) and a 512-frame window of its dF/F in noise units, around its most active stretch
  (`build_inputs`).
- **CNN:** a 2-D CNN on the crop and a 1-D CNN on the trace, joined into one logit (class-balanced loss; flips, 90-degree
  turns and small brightness changes as augmentation).
- **Four methods compared** (`compare`, stage 1 or 2, leave-one-animal-out, leave-one-session-out with one animal):
  the feature model alone, the CNN alone, **stacking** (the feature model with the CNN's logit as one more feature; for the
  ROIs it trains on, that logit comes from CNNs trained without their session, so it learns how far the CNN can be trusted
  on an unseen session), and the mean of the two logits. `show` plots the ROC curves and the AUC of every held-out group.
- **Caution:** the same cells probably appear in several sessions of one animal, and a CNN can memorise how a cell looks, so
  its held-out score is optimistic. `fov_overlap` gives the peak correlation of the mean images of every pair of sessions
  (near 1 = the same field of view); `show_saliency` shows what the CNN reacts to.
- Cost: (inner folds + 1) CNNs per held-out group; `max_folds=2` gives a quick trial. A stage with a few hundred labels is
  small data for a CNN, so judge it by the held-out groups, not by the training loss.

## Notes

- **Which ROIs count as cells:** `iscell_prob_min` in the config. `null` (default) uses Suite2p's own binary
  `iscell` (column 0 of `iscell.npy`, which follows the classifier threshold or your manual curation in Suite2p).
  A number, e.g. `0.3`, makes a ROI a cell when its Suite2p probability (column 1) is >= that number. The
  ROIs that are not cells are left out of training and prediction and are labeled `not_cell`. It is applied
  after the cached feature table is loaded; with the event snapshots on (they exist for cell ROIs only) changing it rebuilds the table, then a
  retrain / re-predict is needed. The
  GUI's "iscell >=" box starts at this value (display only).

- **Red bleed-through:** the red image is `ref_field_MC_registeration.tif`. Bleed-through correction is off by default. Turn it on only if that image was acquired together with the green channel.
- **Stage separation:** stage 2 never sees activity features. That keeps later dSPN vs iSPN activity comparisons from being circular.
- **Features and models:** there is no feature version. A model remembers its feature columns and settings; features are computed for the model that predicts (once per model and session), so changing `features.py` needs a retrain only when a column the model uses is gone.
