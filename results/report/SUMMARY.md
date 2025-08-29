# Results Summary

- **Best test F1:** 0.960 (QSVM_kernel) vs SVM_kmer ΔF1=+0.002; Acc=0.923, Prec=0.923, Rec=1.000, ROC-AUC=0.788

## Dataset/Encoding

- Accession: `KF986530.1`
- Window: 256  |  Stride: 128
- Total samples: 1490

## Noise & Shots — Top configs
**VQC (by mean test F1):**
- shots=256, pflip=0.01, pdepol=0.0 → F1=0.852
- shots=256, pflip=0.0, pdepol=0.0 → F1=0.847
**QSVM (by mean test F1):**
- shots=256, pflip=0.01, pdepol=0.0, anchors=96.0 → F1=0.958

## Artifacts
- `results/metrics/summary_test.csv`
- `results/figures/roc_test.png`, `bar_f1_test.png`, `cm_*.png`, `heat_*_shots*.png`
- `results/tables/summary_test.tex`
- `results/report/DNA_QML_Results_Report.docx` / `.pdf`