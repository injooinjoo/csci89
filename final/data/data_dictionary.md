# Data Dictionary

## Dataset: News Headlines (Sample)

| Column Name    | Type    | Description |
|---------------|---------|-------------|
| headline      | string  | News headline text |
| publish_date  | date    | Publication date of the news article |

## Notes
- `headline` is used as the primary text input for topic modeling.
- `publish_date` is used as a **document-level covariate** in the STM prevalence model.
