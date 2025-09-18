# Results Summary

- **Best test F1:** 0.954 (SVM_onehot) vs SVM_kmer DeltaF1=+0.001; Acc=0.918, Prec=0.917, Rec=0.994, ROC-AUC=nan, PR-AUC=0.956

## Dataset/Encoding

- Accession: `?`
- Window: 256  |  Stride: 128
- Total samples: 20560

## Noise & Shots - Top configs
**VQC (by mean test F1):**
- shots=256, pflip=0.0, pdepol=0.0 -> F1=0.778
- shots=256, pflip=0.01, pdepol=0.0 -> F1=0.777
**QSVM (by mean test F1):**
- shots=256, pflip=0.01, pdepol=0.0, anchors=96.0, S_NOISE=16.0 -> F1=0.921

## Limitations & Problems (from run journals)
- [WARN:kernel] Train truncated to 300 for speed. (source: qsvm_kernel_20250918_121103.json)
- [WARN:kernel_cache] Pure kernels not in cache; will compute on demand. (source: noise_robustness_20250918_122627.json)
- [WARN:kernel] Train truncated to 300 for speed. (source: qsvm_kernel_20250918_201239.json)
- [WARN:roc_cache] Skip vqc: probabilities are None. (source: benchmark_analysis_20250919_002013.json)
- [WARN:cm_cache] Skip y_pred_vqc: missing probs or threshold. (source: benchmark_analysis_20250919_002013.json)
- [WARN:vqc_probs] No VQC probability cache detected; calibration for VQC will be skipped (source: benchmark_analysis_20250919_002013.json)
- [WARN:reports] Skip report for VQC: missing probs or threshold. (source: benchmark_analysis_20250919_002013.json)

## Artifacts
- `results/metrics/summary_test.csv`
- `results/figures/roc_test.png`, `pr_test.png`, `bar_f1_test.png`, `cm_*.png`, `heat_*_shots*.png`
- `results/tables/summary_test.tex`
- `results/report/DNA_QML_Results_Report.docx` / `.pdf`