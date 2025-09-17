| ts | step | status | message |
|---|---|---|---|
| 2025-09-17 10:49:12 | load | ok | Loaded encodings from encodings_all.npz and splits from splits_pooled.json |
| 2025-09-17 10:49:13 | datasets | ok | Detected 13 dataset(s) with mapping from dataset_index.csv |
| 2025-09-17 10:49:13 | splits | ok | train=12336, val=4112, test=4112, pos_rate=0.8654 |
| 2025-09-17 10:49:13 | fit | ok | SVM_kmer: fitting on train (k-mer, n=12336) |
| 2025-09-17 10:49:30 | threshold | ok | SVM_kmer: selected thr=0.40 (val F1=0.969) |
| 2025-09-17 10:49:39 | eval | ok | SVM_kmer: test F1=0.954, AUC=0.802, PR-AUC=0.939, thr=0.40 |
| 2025-09-17 10:49:39 | fit | ok | SVM_onehot: fitting on train (one-hot flat, n=12336, d=1024) |
| 2025-09-17 10:57:26 | threshold | ok | SVM_onehot: selected thr=0.30 (val F1=0.978) |
| 2025-09-17 10:59:47 | eval | ok | SVM_onehot: test F1=0.954, AUC=0.842, PR-AUC=0.956, thr=0.30 |
| 2025-09-17 10:59:48 | per-dataset | ok | SVM_kmer per-dataset metrics saved for 13 dataset(s). |
| 2025-09-17 11:00:08 | per-dataset | ok | SVM_onehot per-dataset metrics saved for 13 dataset(s). |