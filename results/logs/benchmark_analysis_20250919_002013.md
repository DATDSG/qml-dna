| ts | step | status | message |
|---|---|---|---|
| 2025-09-19 00:08:26 | load | ok | Loaded encodings_all.npz and splits_pooled.json |
| 2025-09-19 00:08:26 | splits | ok | train=12336, val=4112, test=4112, pos_rate=0.8654 |
| 2025-09-19 00:08:51 | svm_kmer | ok | Val-opt thr=0.40; test F1=0.954, AUC=0.802, PR-AUC=0.939 (elapsed=24.45s) |
| 2025-09-19 00:20:12 | svm_onehot | ok | Val-opt thr=0.30; test F1=0.954, AUC=0.842, PR-AUC=0.956 (elapsed=680.91s) |
| 2025-09-19 00:20:12 | qsvm_kernel | ok | Loaded Gram matrices: train=(300, 300), val=(4112, 300), test=(4112, 300) (elapsed=0.01s) |
| 2025-09-19 00:20:12 | qsvm_fit | ok | Val-opt thr=0.50; test F1=0.928, AUC=0.650, PR-AUC=0.907 (elapsed=0.16s) |
| 2025-09-19 00:20:12 | vqc_metrics | ok | Loaded stored VQC metrics from metrics/vqc_metrics.csv |
| 2025-09-19 00:20:12 | vqc_probs | warn | No VQC probability cache detected; calibration for VQC will be skipped |
| 2025-09-19 00:20:12 | roc_cache | ok | Saved probs_svm_kmer.npy |
| 2025-09-19 00:20:12 | roc_cache | ok | Saved probs_svm_onehot.npy |
| 2025-09-19 00:20:12 | roc_cache | ok | Saved probs_qsvm_kernel.npy |
| 2025-09-19 00:20:12 | roc_cache | warn | Skip vqc: probabilities are None. |
| 2025-09-19 00:20:12 | cm_cache | ok | Saved y_pred_svm_kmer.json (thr=0.400) |
| 2025-09-19 00:20:12 | cm_cache | ok | Saved y_pred_svm_onehot.json (thr=0.300) |
| 2025-09-19 00:20:12 | cm_cache | ok | Saved y_pred_qsvm_kernel.json (thr=0.500) |
| 2025-09-19 00:20:12 | cm_cache | warn | Skip y_pred_vqc: missing probs or threshold. |
| 2025-09-19 00:20:12 | reports | warn | Skip report for VQC: missing probs or threshold. |
| 2025-09-19 00:20:13 | summary | ok | Best test F1: SVM_onehot (0.954) |