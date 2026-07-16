# AlphaGenome-GeneBayes Integration

## Overview
First systematic evaluation of AlphaGenome-derived sequence features for improving gene constraint estimation (shet) in data-sparse short genes within the GeneBayes Bayesian framework.

## Background
Short genes with few expected loss-of-function variants have flat likelihoods in population genetics models — their constraint estimates are dominated by the prior. This project tests whether sequence-predicted functional features from AlphaGenome (DeepMind, Nature 2026) can improve the prior and thus improve shet estimates for these genes.

## Pipeline
1. Extract gene coordinates from GENCODE v46
2. Run AlphaGenome feature extraction across 11 modalities for 18,767 human genes
3. Merge extracted features with the existing GeneBayes feature table
4. Train 11 separate GeneBayes models (one per modality) to evaluate each modality's contribution to constraint estimation
5. Train a final combined model and compare against baseline

## Requirements
- Python 3.11+ for AlphaGenome feature extraction
- Python 3.9 for GeneBayes
- GPU with sufficient VRAM (H200 recommended for full-scale extraction)
- [alphagenome_research](https://github.com/google-deepmind/alphagenome_research)
- [GeneBayes](https://github.com/tkzeng/GeneBayes)

## Usage

### Feature Extraction
```bash
python extract_alphagenome_bouchet.py
```

For HPC cluster submission (SLURM):
```bash
sbatch submit_job.sh
```

## Results
- Baseline val_loss: 143,401.75
- Full results: pending

## References
- Zeng, J. et al. (2024). GeneBayes: improving Bayesian estimation of gene properties with sequence information. [GitHub](https://github.com/tkzeng/GeneBayes)
- Avsec, Ž. et al. (2026). Advancing regulatory variant effect prediction with AlphaGenome. *Nature*, 649, 8099.
- Karczewski, K.J. et al. (2020). The mutational constraint spectrum quantified from variation in 141,456 humans. *Nature*, 581, 434–443.
