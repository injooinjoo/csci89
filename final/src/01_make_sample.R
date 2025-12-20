# Create a random sample from the full ABC News headlines dataset
# Usage: Rscript src/01_make_sample.R [input_csv] [output_csv] [sample_size]

args <- commandArgs(trailingOnly = TRUE)

# Default paths
input_csv  <- if (length(args) >= 1) args[1] else "data/abcnews-date-text.csv"
output_csv <- if (length(args) >= 2) args[2] else "data/sample/news_headlines_sample.csv"
sample_size <- if (length(args) >= 3) as.integer(args[3]) else 20000

cat("Reading:", input_csv, "\n")
df <- read.csv(input_csv, stringsAsFactors = FALSE)
cat("Total rows:", nrow(df), "\n")

# Random sample
set.seed(42)
idx <- sample(seq_len(nrow(df)), size = min(sample_size, nrow(df)))
sample_df <- df[idx, ]

# Ensure output directory exists
output_dir <- dirname(output_csv)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(sample_df, output_csv, row.names = FALSE)
cat("Wrote", nrow(sample_df), "rows to:", output_csv, "\n")
