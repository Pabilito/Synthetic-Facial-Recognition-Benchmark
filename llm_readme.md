# llm_readme.md — Synthetic Facial Recognition Benchmark

Compact reference for LLMs. Reproduction package: evaluates real- and synthetic-data-trained FR models on real + synthetic 1:1 verification benchmarks, then runs biometric, correlation, rank-consistency and demographic analyses.

## Directory layout

```
<root>/
├── data/
│   ├── real/                     # ann.txt files + pre-aligned 112x112 crops
│   │   ├── lfw_112x112/  calfw_112x112/  cplfw_112x112/  agedb_30_112x112/
│   │   ├── lfw_ann.txt  calfw_ann.txt  cplfw_ann.txt  agedb_30_ann.txt  cfp_fp_ann.txt
│   │   └── ijb/{IJBB,IJBC}/{loose_crop, meta/}
│   └── synth/
│       ├── <dataset>/            # per-dataset image folders (SFace, CemiFace, DCFace, ...)
│       ├── <dataset>_ann.txt     # produced by data/synth/generate_ann.py
│       └── generate_ann.py
├── Pretrained_Models/            # *.pt/*.pth + arch modules (iresnet, iresnet_attn, vit_face, vit_trans, swin_subnets, net)
├── Evaluation/                   # all output CSVs + PNGs land here
│   ├── tmp/<model_name>/         # per-worker scratch dirs from run_all.py
│   ├── FID/  Pearson/  Spearman/ # organisational subfolders for grouped outputs
│   ├── compute_correlations.py           # post-hoc; consumes biometric_metrics.csv
│   ├── compute_avg_metrics.py            # post-hoc; consumes recomputed correlations
│   ├── compute_comparison_table.py       # post-hoc; real-vs-real + focus-synth-vs-real
│   ├── compute_demographics_summary.py   # post-hoc; consumes demographics.csv + ann.txt
│   ├── compute_fid.py  compute_fid_iresnet.py  # FID (InceptionV3 / IResNet-R18)
│   ├── mae_analysis.py                   # MAE MorphFace/Vec2Face vs real
│   ├── eval_distributions.py             # mated vs non-mated cos-sim KDE grid
│   ├── plot_fid_heatmap.py  plot_spearman_heatmap.py
├── load_model.py
├── eval_utils.py
├── eval_real.py  eval_synth.py  eval_ijb.py
├── eval_biometric.py  eval_biometric_correlation.py
├── eval_correlation.py  eval_correlation_by_model_type.py  eval_correlation_ci.py
├── eval_rank_consistency.py
├── demographic_analysis.py
├── run_all.py
└── README.md
```

`.gitignore`: ignores `Ignore/`, `Pretrained_Models/*.pt`, `Pretrained_Models/*.pth`, standard python junk.

## Annotation file format

Every `*_ann.txt` (real, synth, and generated) uses one line per pair:
```
<label> <img1_relpath> <img2_relpath>     # label: 1=genuine, 0=impostor
```
Real: paths relative to `data/real/`. Synth: paths relative to `data/synth/<dataset>/`.

## Data flow (producer → consumer)

```
eval_real.py    ─► real_detail.csv,  real_summary.csv    ┐
eval_ijb.py     ─► real_detail.csv,  real_summary.csv    │─► eval_biometric.py ─► biometric_metrics.csv ─┐
eval_synth.py   ─► synth_detail.csv, synth_summary.csv   ┘                                              │
                                                                                                        ├─► eval_biometric_correlation.py ─► biometric_corr_<metric>.csv + heatmaps
real_summary.csv + synth_summary.csv ─► eval_correlation.py               ─► correlation_all_datasets.png, correlation_heatmap.png, dataset_correlations.csv
                                    ─► eval_correlation_by_model_type.py ─► correlation_*_{real,synth}.{png,csv}
                                    ─► eval_correlation_ci.py            ─► correlation_ci.csv, correlation_ci_heatmap.png, correlation_ci_forest.png
biometric_metrics.csv ─► eval_rank_consistency.py ─► rank_{spearman,displacement}[_ci]_<metric>.png, rank_consistency.csv
data/{real,synth}/*_ann.txt ─► demographic_analysis.py ─► demographics.csv
```

## CSV schemas

- **detail (`{real,synth}_detail.csv`)**: `model, dataset, img1, img2, label, cosine_sim, fold, correct`  (IJB rows: `fold=-1, correct=-1`).
- **summary (`{real,synth}_summary.csv`)**: `model, dataset, accuracy, std`  (percent; IJB `accuracy = TAR@FAR=1e-4 × 100`, `std=0`).
- **`biometric_metrics.csv`**: `split, model, dataset, n_genuine, n_impostor, EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01`  (percent; NIST FRVT conv.: `FMR=P(score≥t|impostor)`, `FNMR=P(score<t|genuine)`; lower better except TAR@…).
- **`demographics.csv`**: `dataset, image_path, identity, age, gender, gender_confidence`  (DeepFace, `detector_backend="skip"`).
- **correlation CSVs**: `dataset_x, dataset_y, r, p, n_models` (+ `fisher_ci_{lo,hi}, bootstrap_ci_{lo,hi}` for CI variants).
- **`rank_consistency.csv`**: tidy pairwise Spearman r + mean rank displacement, with bootstrap 95% CIs.

## Modules

### `load_model.py`
`ModelBundle(model, name, height, width, mean, std)` — uniform wrapper. Key functions:
- `list_model_specs(model_dir, include_library_models=True)` — enumerate without loading weights.
- `load_model_by_spec(spec, device) -> ModelBundle | None`
- `release_model(bundle)` — free GPU mem between models.
- `load_all_models(...)` — convenience.

Supported (name prefix → arch):
- File-based (`Pretrained_Models/`): `IResNet_R18_*`, `IResNet_R50_*`, `IDiffFace_CF50`, `SFace_IResNet50`, `IResNet_R100_*`, `VIT_*`, `TransFace_*` (.pt), `SwinFace` (.pt returns dict with key `"Recognition"`).
- Library: `edgeface_base` (torch.hub), `InceptionResnetV1_VGG`, `InceptionResnetV1_CASIA` (facenet_pytorch).
- HuggingFace (auto-download, Lightning ckpts, use `Pretrained_Models/net.py` AdaFace Backbone IR_50; strip `"model."` prefix; forward returns `(embedding, norm)`, wrapper unpacks and flips RGB→BGR):
  `hsface10k`, `hsface300k`, `hyperface_50k_stylegan`, `langevin_disco_30k`, `langevin_dispersion_10k`.

### `eval_utils.py` (shared building blocks)
- `BATCH_SIZE=64`, `NUM_WORKERS=4`.
- `parse_ann(path) -> [(p1, p2, label:int)]`
- `PairDataset(pairs, base_dir, transform)`
- `make_transform(h, w, mean, std)` — `Resize→ToTensor→Normalize`.
- `extract_embeddings(model, loader, device)` — handles dict output (`out["Recognition"]`, SwinFace) and tuple output (`out[0]`, TransFace/some ViTs); L2-normalises.
- `eval_10fold(emb1, emb2, labels, n_folds=10)` → `(mean_acc%, std%, cos_sim, fold_assignments, best_thresh_map)`. Threshold sweep in `np.arange(-1, 1, 0.005)` on train folds; sklearn `KFold(shuffle=False)`.
- `append_to_detail_csv`, `append_to_summary_csv` — auto-header on first write.

### `eval_real.py`
Runs each model on `DATASETS = {LFW, CALFW, CPLFW, AgeDB-30, CFP-FP}` via 10-fold protocol. One model loaded at a time; released after all datasets.
Flags: `--model-dir`, `--no-library-models`.

### `eval_synth.py`
Same protocol, but datasets are auto-discovered from subfolders under `data/synth/` with matching `<name>_ann.txt`. Skips if <10 pairs.
Flags: `--model-dir`, `--dataset NAME`, `--model NAME`, `--no-library-models`.

### `eval_ijb.py`
Template-based 1:1 verification on IJB-B / IJB-C. Appends into `real_detail.csv` / `real_summary.csv` (so `eval_biometric.py` picks it up transparently).
- Similarity-transform alignment from 5-pt landmarks (loose_crop, not pre-aligned).
- Media-aware mean pooling → L2-norm template embedding; horizontal-flip aug on by default.
- No 10-fold: `fold=-1, correct=-1`; `accuracy = TAR@FAR=1e-4 × 100`, `std=0`.
- Meta files expected under `data/real/ijb/<IJBB|IJBC>/meta/`: `*_face_tid_mid.txt`, `*_name_5pts_score.txt`, `*_template_pair_label.txt`.
Flags: `--target IJBB|IJBC`, `--no-flip`, `--no-detector-score`, `--no-library-models`, `--ijb-dir`.

### `eval_biometric.py`
Computes ISO/IEC 19795 metrics from `{real,synth}_detail.csv` grouped by (split, model, dataset). Metrics: `EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01`.
Flags: `--real-only`, `--synth-only`, `--detail-csv`, `--split`, `--datasets`, `--models`.

### `eval_biometric_correlation.py`
Per-metric pairwise Pearson r + Fisher CI + bootstrap CI (B=10000). **Y-axis: real datasets only; X-axis: real+synth; synth-vs-synth excluded.** Caches per-metric CSV; use `--recompute` to force.
Outputs: `biometric_corr_<metric>.csv`, `biometric_corr_heatmap_<metric>.png`, `biometric_corr_heatmap_<metric>_noci.png`.
Flags: `--metrics`, `--splits`, `--in`, `--bootstrap-b`, `--recompute`.

### `eval_correlation.py`
Cross-dataset accuracy correlation across all models. Full n×n scatter grid + heatmap + `dataset_correlations.csv`. Real datasets listed before synth; black separator lines in heatmap between them.

### `eval_correlation_by_model_type.py`
Runs `eval_correlation.py` logic twice, splitting models into real- vs synthetic-trained.
`SYNTH_MODELS = {IDiffFace_CF50, SFace_IResNet50, hsface10k, hsface300k, hyperface_50k_stylegan, langevin_disco_30k, langevin_dispersion_10k}`. Outputs suffixed `_real` / `_synth`.

### `eval_correlation_ci.py`
Adds Fisher-z + bootstrap (B=10000, seed=42) 95% CIs on accuracy correlations. Outputs: `correlation_ci.csv`, `correlation_ci_heatmap.png` (annotates `r\n[lo, hi]`), `correlation_ci_forest.png` (upper-triangle pairs sorted by r, red/green by sign, n annotated).

### `eval_rank_consistency.py`
From `biometric_metrics.csv`: Spearman r and mean |rank_i−rank_j| between every dataset pair, per metric; bootstrap 95% CI (B=10000, seed=42). Also generates the same four figures for recognition accuracy read from `{real,synth}_summary.csv` (metric name `recognition_accuracy`).
`HIGHER_IS_BETTER = {TAR@FAR1, TAR@FAR01}`.
Flags: `--metrics`, `--split`, `--no-ci`, `--no-plots`.

### `demographic_analysis.py`
DeepFace age (apparent-age regressor) + gender on every unique image referenced by any `*_ann.txt` under `data/real/` and `data/synth/`. `detector_backend="skip"` (images pre-cropped).
Identity heuristic: first path component unless it ends with `112x112|112|160|224|_crop|_aligned` or is purely numeric (then empty string, treated as flat/Convention B).
Flags: `--real-only`, `--synth-only`, `--datasets NAME...` (case-insens, searches both roots), `--resume` (skip rows already in CSV).

### `run_all.py`
Orchestrator. Stages:
1. **Parallel inference** (`--parallel N`, default 4; single GPU shared). Each worker runs `eval_real + eval_ijb + eval_synth` for its model, writing to `Evaluation/tmp/<model_name>/` (no CSV race). Merged after all finish. Skip-if-tmp-nonempty for resume.
2. `eval_biometric.py`
3. `eval_biometric_correlation.py`
4. `demographic_analysis.py` (opt-in: `--demographics`)
5. `eval_rank_consistency.py` (opt-in: `--rank-consistency`)

Flags: `--parallel`, `--force`, `--force-inference`, `--force-biometric`, `--force-correlation`, `--no-ijb`, `--no-synth`, `--no-library-models`, `--demographics`, `--rank-consistency`, `--dry-run`.

## Evaluation/ post-hoc scripts

These run **inside `Evaluation/`** and consume the CSVs the top-level pipeline produced. Paths default to script-directory-relative; override with `--input`/`--output` flags.

**Canonical dataset name sets** (declared in these scripts, keep synced):
- `REAL` = `{LFW, CALFW, CPLFW, AgeDB-30, CFP-FP}` (accuracy: 5) + `{IJBB, IJBC}` (biometric-only: 7 total)
- `SYNTH` = `{CemiFace, Control10k, DCFace, Digi2Real, HyperFace, IDiffFace, Langevin-DisCo, MorphFace, SFace2, SynFace, SynMulti-PIE, Vec2Face}` (12)
- `FOCUS_SYNTH` = `{MorphFace, Vec2Face}` (used by comparison_table / mae / spearman heatmap)
- Display-name remaps: `Control10k → ControlFace10k`, `IDiffFace → IDiff-Face`, `IJBB → IJB-B`, `IJBC → IJB-C` (defined per-script; internal CSVs still use the short names).

### `compute_correlations.py`
`biometric_metrics.csv` → `biometric_corr_all_metrics_recomputed.csv`. For each `metric ∈ {EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01}` and each `(real_ds, synth_ds)`: pearsonr + spearmanr with p-values on shared models. Skips n<3.

### `compute_avg_metrics.py`
Consumes the recomputed CSV above; averages `r, p_r, rho, p_rho` per `(metric, synthetic_dataset)` across the 7 real datasets. Prints 4 pivot tables (`METRIC_ORDER = [EER, FMR100, FMR1000, TAR@FAR01, TAR@FAR1, ZeroFMR]`).

### `compute_comparison_table.py`
Inputs: `biometric_metrics.csv` + `real_summary.csv` + `synth_summary.csv`. Output: `comparison_table.csv`. Columns per pair: `metric, dataset_x, dataset_y, n_models, r, r_ci_{lo,hi}, p_r, rho, rho_ci_{lo,hi}, p_rho`. Rows = (upper-triangle real×real) ∪ (FOCUS_SYNTH × real). Metrics = biometric + `accuracy`. Fisher-z CI hard-coded `z_crit=1.96`.

### `compute_demographics_summary.py`
Restricts `demographics.csv` to images actually appearing in any `*_ann.txt` (searches `--anndir` recursively; matches `dataset_stem.lower()`). Falls back to basename match if full-path match yields zero. Output: `demographics_summary.csv` with `n_images, ann_file, age_{mean,std,median,min,max}, age_missing_pct, n_<gender>, pct_<gender>`.

### `mae_analysis.py`
MAE of MorphFace/Vec2Face vs real benchmarks, computed as `mean(|synth_val − real_val|)` across shared models. Values divided by 100 → ratio scale (1pp = 0.01). Output: `mae_results.csv` (stacked pivot tables + summary block); paths hard-coded relative to CWD (`Evaluation/synth_summary.csv` etc.), so **run from repo root**.

### `compute_fid.py`
FID between every pair of dataset folders under `../data/real/` and `../data/synth/`. Extractor: **InceptionV3** (2048-d, ImageNet). Caches `_fid_stats.npz` (μ, Σ) inside each dataset folder; incremental CSV write; skips pairs already in output. Output: `fid_results.csv` (`dataset_a, dataset_b, fid`).
Flags: `--batch-size`, `--device`, `--output`, `--recompute-stats`, `--recompute-pairs`. Parses ann.txt with real/synth split heuristic (real: first path component == dataset folder; synth: first == identity).

### `compute_fid_iresnet.py`
Same as above but extractor = **IResNet-R18** loaded from `Pretrained_Models/IResNet_R18_MS1MV3.pt` (512-d, ArcFace/MS1MV3), input 112×112. Separate cache `_fid_stats_r18.npz`. Output: `fid_results_r18.csv`. Flag: `--model-path`, `--batch-size`, `--device`, `--recompute-{stats,pairs}`. No `--dims` (always 512).

### `plot_fid_heatmap.py`
Inputs: `fid_results.csv` + `fid_results_r18.csv`. Outputs: `fid_comparison_table.csv` (long format with both columns) and `fid_heatmap.png` (two log-scale heatmaps stacked, rows = 5 real, cols = 5 real + 12 synth, colormap `RdYlGn_r`, log norm, separator lines between real/synth).

### `plot_spearman_heatmap.py`
Reads `comparison_table.csv` → `spearman_heatmap.png`. One subplot per metric in `METRIC_ORDER = [accuracy, ZeroFMR, EER, FMR100, FMR1000]`; rows = `[MorphFace, Vec2Face]`, cols = 7 real (accuracy subplot drops IJB-B/IJB-C). `TwoSlopeNorm(vmin=0.5, vmax=1.0)`, cmap `RdBu_r`.

### `eval_distributions.py`
Reads `real_detail.csv` + `synth_detail.csv` **chunked** (memory-safe), groups by `(model, dataset)`, caches per-group cosine-sim arrays for label=0/1 to `distribution_data.npz`. Delete the cache to force recomputation. Outputs KDE distribution plots: `distribution_grid.png` (all model×dataset) and `distribution_grid_focused.png` (3 models × 7 datasets). **Skips `IJBB, IJBC`** (too large). Applies `DATASET_RENAME` (`Control10k → ControlFace10k`, etc.) after cache load so no rebuild needed on rename. Plot range `x ∈ [-0.5, 1.0]`, 120 bins, 400 KDE points, mated=blue `#2196F3`, non-mated=red `#F44336`.

## Post-hoc data-flow (Evaluation/ scripts)

```
biometric_metrics.csv ─► compute_correlations.py ─► biometric_corr_all_metrics_recomputed.csv ─► compute_avg_metrics.py (stdout tables)
biometric_metrics.csv + {real,synth}_summary.csv ─► compute_comparison_table.py ─► comparison_table.csv ─► plot_spearman_heatmap.py ─► spearman_heatmap.png
{real,synth}_summary.csv + biometric_metrics.csv ─► mae_analysis.py ─► mae_results.csv
demographics.csv + */*_ann.txt ─► compute_demographics_summary.py ─► demographics_summary.csv
../data/{real,synth}/*/ ─► compute_fid[_iresnet].py ─► fid_results[_r18].csv ─► plot_fid_heatmap.py ─► fid_comparison_table.csv, fid_heatmap.png
{real,synth}_detail.csv ─► eval_distributions.py ─► distribution_data.npz + distribution_grid[_focused].png
```

## Conventions to remember

- All accuracy/metric values in CSVs are **percent** (not fractions).
- Scores are **cosine similarity** ∈ [−1, 1]; higher = more similar.
- Real-model list is implicit (`all_models − SYNTH_MODELS`); update `SYNTH_MODELS` in `eval_correlation_by_model_type.py` when adding new synthetic-trained models.
- `SwinFace` forward → dict (`"Recognition"`); `TransFace` / some ViTs → tuple (index 0). Handled in `extract_embeddings`.
- IJB rows deliberately break the 10-fold assumption (`fold=-1`); only `model/dataset/label/cosine_sim` are used downstream, so it's safe.
- `real_summary.csv` and `real_detail.csv` are **shared** between `eval_real.py` and `eval_ijb.py` (append-only).
- Bootstrap default: `B=10000`, `seed=42`.
- Dataset-name inconsistency: internal CSVs use `Control10k`, `IDiffFace`, `IJBB`, `IJBC`; display layer (`plot_*` scripts) remaps to `ControlFace10k`, `IDiff-Face`, `IJB-B`, `IJB-C`. Keep both when adding new datasets.
- `mae_analysis.py` and post-hoc FID scripts use **CWD-relative** paths (`Evaluation/...` or `../data/...`) — must run from the correct directory (repo root or `Evaluation/` respectively).
- Accuracy is only defined on 5 real datasets (LFW/CALFW/CPLFW/AgeDB-30/CFP-FP); IJB-B/IJB-C accuracy rows in `real_summary.csv` are `TAR@FAR=1e-4×100` with `std=0` and are excluded from accuracy correlations/heatmaps.
- FID caches live *inside* each dataset folder (`_fid_stats.npz` / `_fid_stats_r18.npz`) — safe to delete for a fresh recompute of one dataset without touching others.
- `eval_distributions.py` skips IJB by design (dataset size); if you need IJB distributions you'll have to remove it from `SKIP_DATASETS` and expect a much larger cache.
