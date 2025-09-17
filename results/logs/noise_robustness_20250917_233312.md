| ts | step | status | message |
|---|---|---|---|
| 2025-09-17 23:18:31 | init | ok | Noise robustness sweep notebook initialized; directories ready. |
| 2025-09-17 23:18:31 | config | ok | Preset=turbo, D=6, MAX_TR=160, M_ANCHORS=96, sweep=2 |
| 2025-09-17 23:18:31 | load | ok | Loaded encodings from encodings_all.npz and splits from splits_pooled.json |
| 2025-09-17 23:18:31 | pca_scale | ok | PCA D=6 (explained_var=0.563); subset_train=(160, 6) |
| 2025-09-17 23:18:31 | device | ok | Using lightning.qubit for VQC (D=6) |
| 2025-09-17 23:18:31 | weights | ok | Loaded VQC weights with shape (2, 6) |
| 2025-09-17 23:28:24 | vqc_sweep | ok | Completed VQC sweep; rows=6 |
| 2025-09-17 23:28:24 | kernel_cache | ok | Loaded cached pure kernels pure_kernels_D6_N160_pca17_scaler.npz |
| 2025-09-17 23:28:24 | align | warn | pure_256_0p0_0p0:val y length 4112 > kernel rows 298; slicing to head 298. |
| 2025-09-17 23:28:24 | align | warn | pure_256_0p0_0p0:test y length 4112 > kernel rows 298; slicing to head 298. |
| 2025-09-17 23:28:24 | plot | warn | ROC skipped qsvm_pure_256_0p0_0p0: Found input variables with inconsistent numbers of samples: [4112, 298] |
| 2025-09-17 23:28:24 | plot | warn | PR skipped qsvm_pure_256_0p0_0p0: Found input variables with inconsistent numbers of samples: [4112, 298] |
| 2025-09-17 23:33:12 | qsvm_sweep | ok | Completed QSVM sweep; rows=6 |