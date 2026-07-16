# AlphaGenome-GeneBayes Integration

## Overview
First systematic evaluation of AlphaGenome-derived sequence features for improving gene constraint estimation (shet) in data-sparse short genes, within the GeneBayes Bayesian framework.

## Background
Short genes with few expected loss-of-function variants have flat likelihoods in population genetics models — their constraint estimates are dominated by the prior. This project tests whether sequence-predicted functional features from AlphaGenome (DeepMind, Nature 2026) can improve the prior and thus improve shet estimates for these genes.

## Pipeline
1. Extract gene coordinates from GENCODE v46
2. Run AlphaGenome feature extraction across 11 modalities for 18,767 genes (Yale Bouchet HPC, H200 GPU)
3. Merge with existing GeneBayes feature table
4. Train 11 separate GeneBayes models (one per modality) to evaluate each modality's contribution
5. Train final combined model and compare against baseline

## Requirements
- Python 3.11+ (AlphaGenome extraction)
- Python 3.9 (GeneBayes environment)
- Yale Bouchet HPC access with H200 GPU
- alphagenome_research (Google DeepMind)
- GeneBayes

## Results
- Baseline val_loss: 143,401.75
- Full results: pending

## References
- Zeng et al. 2024 (GeneBayes)
- Avsec et al. 2026 (AlphaGenome, Nature)
