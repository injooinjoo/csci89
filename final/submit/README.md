# CSCI S-89B Final Project (InJoo Kim)

**Topic:** Structural Topic Modeling (STM) on news headlines

## Folder structure
- `data/`
  - `raw/` : original dataset (not committed if >10MB)
  - `sample/` : small sample CSV used by notebooks
  - `processed/` : cleaned outputs (optional)
- `notebooks/` : RMarkdown analysis notebooks
- `proposal/` : topic proposal / planning notes
- `report/` : exported figures/tables for the write-up
- `slides/` : slide assets
- `src/` : helper scripts (sampling, utilities)
- `submission/` : final PDF/PPT to submit
- `video/` : presentation recording (or link)

## Quick start
1) Put a small sample at `data/sample/news_headlines_sample.csv` with columns:
- `headline` (character)
- `publish_date` (YYYY-MM-DD)

2) Render the notebook:
```r
rmarkdown::render('notebooks/02_stm_topic_modeling.Rmd')
```
