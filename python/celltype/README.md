# celltype: SPN / interneuron and dSPN / iSPN classification

Runs on sessions motion-corrected by `batch_2p_Calcium.m`:

1. **Suite2p**: calls `run_suite2p_on_MC.py`. Sessions that already have output are skipped.
2. **Features**: computes per-ROI morphology, activity and red-channel features. They are cached in `<session>\celltype\features_v1.csv`.
3. **Label, train or predict**, using two stages:
   - **Stage 1, SPN vs non-SPN:** uses morphology and activity features. It is semi-supervised (self-training). With fewer than 20 non-SPN labels it falls back to an anomaly detector fit on SPNs. Size and shape hard rules also apply.
   - **Stage 2, red+ vs red−:** uses red features only, with label spreading among SPNs. Red+ maps to dSPN for D1-Cre and to iSPN for D2-Cre.

## Setup

In the environment that has Suite2p, install the other dependencies:

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

The `--sessions <folder> ...` option overrides the inputs in the config. `--skip-extract` skips the Suite2p step, and `--refresh-features` recomputes the cached feature tables.

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
| `x` | clear the label |
| `n` / `b` | next / previous ROI |

- **Selecting ROIs:** click an ROI in the image to jump to it.
- **ROI order:** the "ROI order" menu changes the order. Choose "largest first" or "highest baseline first" to find interneurons quickly. Once a model exists, choose "most uncertain first" for active learning.
- **Saving:** labels are appended to `label_store` as soon as you press a key. Each label is tied to the Suite2p run through the `s2p_hash` column. If you rerun Suite2p on a session, its old labels are ignored.

Aim for about 50 SPN labels with both red+ and red− (they are fast to label), plus at least 20 interneuron/junk labels across animals. Leave-one-animal-out cross-validation needs at least 2 animals.

## Outputs

**Per session, in `<session>\celltype\`:**
- `labels.csv` and `labels.mat`, with fields `roi_idx` (1-based in the .mat file), `p_spn`, `p_red` and `final_label`. The possible values of `final_label` are dSPN, iSPN, SPN_ambiguous, nonSPN, uncertain and not_cell.
- `labels_qc.png`

**In `model_dir`:**
- `celltype_model.joblib`
- `cv_report.json`, which includes the Suite2p `redcell` baseline for comparison
- `cv_predictions.csv`

## Notes

- **Red bleed-through:** the red image is `ref_field_MC_registeration.tif`. Bleed-through correction is off by default. Turn it on only if that image was acquired together with the green channel.
- **Stage separation:** stage 2 never sees activity features. That keeps later dSPN vs iSPN activity comparisons from being circular.
- **Feature versions:** bump `feature_version` in the config after changing `features.py`. Saved models refuse feature tables from a different version.
