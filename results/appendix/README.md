# Appendix — Reproducibility Bundle

This folder contains lightweight, human-readable artifacts to aid auditing and re-running.

## Files

- `environment.json` — Python/platform and key package versions.
- `dataset_audit.json` — counts, window config, and split sizes.
- `seeds.json` — random seeds used across notebooks.

## How to Reproduce (quick)

1. Recreate the Conda env; open notebooks in order: `01_data_preparation → 02_classical_baselines → 03_quantum_kernel → 04_quantum_vqc → 05_noise_robustness → 06_benchmark_analysis → 07_reporting → 08_appendix_reproducibility`.

2. Ensure `data/processed/encodings.npz` and `splits.json` are present from Step 01.

3. Metrics and figures will appear under `results/metrics/` and `results/figures/`.
