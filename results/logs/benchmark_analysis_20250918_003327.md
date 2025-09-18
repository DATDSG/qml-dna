| ts | step | status | message |
|---|---|---|---|
| 2025-09-17 23:53:42 | load | ok | Loaded encodings_all.npz, splits_pooled.json |
| 2025-09-17 23:53:42 | splits | ok | train=12336, val=4112, test=4112, pos_rate=0.8654 |
| 2025-09-17 23:54:06 | svm_kmer | ok | Val-opt thr=0.40; test F1=0.954, AUC=0.802, PR-AUC=0.939 |
| 2025-09-18 00:02:45 | svm_onehot | ok | Val-opt thr=0.30; test F1=0.954, AUC=0.842, PR-AUC=0.956 |
| 2025-09-18 00:02:45 | qsvm_kernel | ok | Loaded Gram matrices: tr=(894, 894), va=(298, 894), te=(298, 894) | y: tr=894, va=298, te=298 |
| 2025-09-18 00:02:45 | qsvm_fit | ok | Val-opt thr=0.85; test F1=0.938, AUC=0.465, PR-AUC=0.898 |
| 2025-09-18 00:02:45 | qsvm_kernel | ok | Loaded precomputed Gram matrices: (894, 894), (298, 894), (298, 894) |
| 2025-09-18 00:14:14 | qsvm_kernel | ok | Loaded precomputed Gram matrices: train=(894, 894), val=(298, 894), test=(298, 894) |
| 2025-09-18 00:14:14 | qsvm_align | warn | Aligned labels to kernel rows: val 4112->298, test 4112->298 |
| 2025-09-18 00:14:14 | qsvm_fit | ok | Val-opt thr=0.85; test F1=0.938, AUC=0.465, PR-AUC=0.898 |
| 2025-09-18 00:18:23 | roc_cache | ok | Saved probs_svm_kmer.npy |
| 2025-09-18 00:18:23 | roc_cache | ok | Saved probs_svm_onehot.npy |
| 2025-09-18 00:18:23 | roc_cache | warn | Skip qsvm_kernel: length mismatch probs=298 vs y_test=4112. |
| 2025-09-18 00:18:23 | roc_cache | warn | Skip vqc: probabilities are None. |
| 2025-09-18 00:18:23 | cm_cache | ok | Saved y_pred_svm_kmer.json (thr=0.400) |
| 2025-09-18 00:18:23 | cm_cache | ok | Saved y_pred_svm_onehot.json (thr=0.300) |
| 2025-09-18 00:18:23 | cm_cache | warn | Skip y_pred_qsvm_kernel: length mismatch 298 vs 4112. |
| 2025-09-18 00:18:23 | cm_cache | warn | Skip y_pred_vqc: missing probs or threshold. |
| 2025-09-18 00:18:23 | reports | warn | Skip report for QSVM_kernel: length mismatch 298 vs 4112. |
| 2025-09-18 00:18:23 | reports | warn | Skip report for VQC: missing probs or threshold. |
| 2025-09-18 00:33:27 | align | warn | Aligned test length to n=298 to match shortest prob array (labels=4112, probs=[4112, 4112, 298]). |
| 2025-09-18 00:33:27 | summary | ok | Best test F1: SVM_onehot (0.954) |