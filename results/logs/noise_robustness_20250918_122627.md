| ts | step | status | message |
|---|---|---|---|
| 2025-09-18 12:11:29 | init | ok | Noise robustness sweep notebook initialized; directories ready. |
| 2025-09-18 12:11:29 | config | ok | Preset=turbo, D=6, MAX_TR=160, M_ANCHORS=96, sweep=2 |
| 2025-09-18 12:11:29 | load | ok | Loaded encodings from encodings_all.npz and splits from splits_pooled.json |
| 2025-09-18 12:11:29 | pca_scale | ok | PCA D=6 (explained_var=0.563); subset_train=(160, 6) |
| 2025-09-18 12:11:29 | device | ok | Using lightning.qubit for VQC (D=6) |
| 2025-09-18 12:11:29 | weights | ok | Loaded VQC weights with shape (2, 6) |
| 2025-09-18 12:21:23 | vqc_sweep | ok | Completed VQC sweep; rows=6 |
| 2025-09-18 12:21:23 | kernel_cache | warn | Pure kernels not in cache; will compute on demand. |
| 2025-09-18 12:21:23 | device | ok | Using lightning.qubit for pure-kernel states (D=6) |
| 2025-09-18 12:21:33 | kernel_cache | ok | Saved pure kernels -> pure_kernels_D6_N160_pca17_scaler.npz |
| 2025-09-18 12:26:27 | qsvm_sweep | ok | Completed QSVM sweep; rows=6 |