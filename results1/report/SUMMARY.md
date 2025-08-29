# Results Summary

- **Best test F1:** 0.960 (QSVM_kernel); Accuracy=0.923, Precision=0.923, Recall=1.000, ROC-AUC=0.788

## Noise & Shots (brief)
**Top VQC conditions (by mean test F1):**
- shots=512, pflip=0.0, pdepol=0.0 → F1=0.842
- shots=512, pflip=0.0, pdepol=0.01 → F1=0.842
- shots=2000, pflip=0.0, pdepol=0.01 → F1=0.842

## Environment
- Python: 3.10.13  on Windows 10
- Files: `results/metrics/summary_test.csv`, `results/figures/*`, `results/tables/summary_test.tex`