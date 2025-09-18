# qml-dna

Hybrid quantum-classical DNA classification pipeline that contrasts strong classical SVM baselines with quantum kernel and variational circuits. Every stage is notebook-driven and writes artefacts (metrics, figures, cached kernels, provenance logs) to `results/` for reproducible downstream analysis.

## Highlights
- End-to-end workflow from raw GenBank parsing to publishable reports.
- Comparable baselines (k-mer and one-hot SVMs) alongside QSVM and VQC models.
- Noise robustness sweeps plus cached ROC/PR data for probability-level diagnostics.
- Advanced reporting notebook that now emits composite scorecards, calibration analyses, DOCX/PDF exports, and ready-to-drop visuals.
- Reproducibility appendices capturing environment metadata, seeds, and dataset audits.

## Environment setup
1. Install a Conda implementation (Miniforge/Mambaforge recommended).
2. Create the environment: `conda env create -f environment.yml`.
3. Activate it: `conda activate qml-dna`.
4. (Optional) Register the kernel: `python -m ipykernel install --user --name qml-dna`.
5. Keep the environment in sync after dependency updates with `conda env update -f environment.yml --prune`.

The supplied `environment.yml` groups packages by purpose (core science, quantum backends, reporting, linting, testing) and installs a few pip-only helpers such as `papermill`, `notebook-as-pdf`, and `jupyterlab-git` for automation and collaboration.

## Workflow at a glance
Run the notebooks in ascending order; each notebook saves outputs that the later stages consume.

| Notebook | Purpose | Key outputs |
| --- | --- | --- |
| `01_data_preparation.ipynb` | Parse raw sequences, create features, emit splits/meta | `data/processed/`, `results/logs/` |
| `02_classical_baselines.ipynb` | Train SVM (k-mer & one-hot), export metrics | `results/metrics/svm_*`, ROC caches |
| `03_quantum_kernel.ipynb` | Build quantum kernels, persist Gram matrices | `results/kernels/`, QSVM metrics |
| `04_quantum_vqc.ipynb` | Optimise VQC, save weights and performance | `results/vqc_weights.npy`, `results/metrics/vqc_metrics.csv` |
| `05_noise_robustness.ipynb` | Sweep noise models for QSVM/VQC | `results/metrics/noise_sweep_*.csv`, heatmaps |
| `06_benchmark_analysis.ipynb` | Consolidate metrics, align labels, prep caches | `results/metrics/combined.csv`, ROC/PR caches |
| `07_reporting.ipynb` | Generate scorecards, calibration plots, DOCX/PDF report | `results/report/`, `results/figures/`, `results/tables/` |
| `08_appendix_reproducibility.ipynb` | Freeze environment + provenance bundles | `results/appendix/`, `results/metadata/` |

Each notebook is idempotent with respect to its outputs—re-running a notebook just refreshes the artefacts it owns.

## What is new in the reporting notebook?
`07_reporting.ipynb` now:
- Builds a composite model scorecard that ranks models using a weighted blend of F1, ROC-AUC, PR-AUC, and balanced accuracy, including validation-vs-test drift.
- Computes probability-level diagnostics (Brier score, log loss, expected calibration error) and emits calibration CSVs/plots for every model with cached probabilities.
- Adds radar charts, calibration overlays, and scatter plots (calibration vs discrimination) to `results/figures/` and automatically rolls them into DOCX/PDF exports.
- Generates `results/tables/scorecard.csv` and `probability_diagnostics.csv` for downstream tooling.
- Feeds bullet-point takeaways, scorecard tables, and calibration summaries directly into the Word/PDF report templates.

To refresh the report after upstream runs:
```
conda activate qml-dna
jupyter lab  # or jupyter notebook
# open 07_reporting.ipynb and run all cells
```
The notebook writes the compiled report to `results/report/DNA_QML_Results_Report.{docx,pdf}` and previews core figures inline.

## Artefact directories
- `data/raw/` original GenBank inputs (not versioned by default).
- `data/processed/` feature matrices, labels, splits, stats.
- `results/metrics/` CSV metrics, combined scorecards, probability diagnostics.
- `results/figures/` ROC/PR curves, confusion matrices, noise heatmaps, calibration plots, radar/composite charts.
- `results/tables/` LaTeX tables, CSV exports consumed by reports.
- `results/logs/` RunJournal JSON logs collected during sweeps.
- `results/report/` Final DOCX/PDF deliverables.
- `results/appendix/` Environment manifests and reproducibility bundles.

## Reproducibility tips
- Seed management lives in the modelling notebooks; reruns are deterministic when existing caches are removed.
- Use `papermill` (installed via pip) for headless runs: `papermill 07_reporting.ipynb 07_reporting.out.ipynb`.
- Capture the precise environment anytime via `conda env export --from-history > env.lock.yml`.
- Run `pytest` to exercise any auxiliary scripts or utilities before regenerating artefacts.

## Troubleshooting
- Missing metrics/artefacts: rerun the prerequisite notebook; helpers such as `safe_load_csv` surface missing paths in 07.
- Pennylane broadcast errors: the project now uses explicit entanglement loops compatible with current PennyLane releases.
- Report generation issues: ensure `python-docx` and `reportlab` are present (they ship with the environment), and delete `results/report/` to regenerate fresh outputs if a document gets corrupted.

Happy experimenting!
