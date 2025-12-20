# Dataset Description

This project uses a publicly available **Kaggle News Headlines dataset**.

## Source
- Platform: Kaggle
- Dataset: News Headlines Dataset
- Content: News headlines with publication dates

(The full dataset URL is provided in the final report and slides.)

## Files in This Directory

### raw/
Contains the original Kaggle dataset.
- This folder is **not included** in the final submission due to file size constraints.

### sample/
Contains a sampled subset of the original dataset used for modeling.
- File: `news_headlines_sample_20k.csv`
- Number of records: ~20,000
- Sampling strategy:
  - Random sampling while preserving temporal coverage
  - Ensures diversity of topics and time periods

## Rationale for Sampling
The full Kaggle dataset exceeds the 10MB submission limit.
A representative subset was therefore used to:
- Enable reproducible experiments
- Maintain computational efficiency
- Preserve meaningful topic structure and temporal trends
