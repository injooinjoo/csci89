library(stm)
library(quanteda)
library(tidyverse)
library(lubridate)

# Load sample data
news <- read.csv("data/sample/news_headlines_sample.csv", stringsAsFactors = FALSE)

# Clean
news_clean <- news %>%
  transmute(
    headline = as.character(headline_text),
    publish_date_raw = as.character(publish_date)
  ) %>%
  filter(!is.na(headline), nchar(trimws(headline)) > 0) %>%
  mutate(
    publish_date = ymd(publish_date_raw),
    year = year(publish_date)
  ) %>%
  filter(!is.na(publish_date))

# Tokenize
text_clean <- tolower(news_clean$headline)
text_clean <- gsub("[^a-z ]", " ", text_clean)
text_clean <- gsub("\\s+", " ", trimws(text_clean))

toks <- tokens(text_clean, remove_punct = TRUE, remove_numbers = TRUE)
toks_clean <- tokens_remove(toks, stopwords("en"))
toks_clean <- tokens_keep(toks_clean, pattern = "^[a-z]{3,}$", valuetype = "regex")

dfm_mat <- dfm(toks_clean)
dfm_mat <- dfm_trim(dfm_mat, min_termfreq = 5)

non_empty <- rowSums(dfm_mat) > 0
dfm_mat <- dfm_mat[non_empty, ]
meta <- news_clean[non_empty, ]

stm_in <- convert(dfm_mat, to = "stm")
docs <- stm_in$documents
vocab <- stm_in$vocab

# Fit STM
set.seed(123)
K_final <- 10
cat("Fitting STM model...\n")
stm_model <- stm(documents = docs, vocab = vocab, data = meta, K = K_final, prevalence = ~ year, verbose = FALSE)

# Save high-res figures
path_figs <- "report/figures"

# Figure 1: Topic Summary
cat("Generating topic_summary_K10.png...\n")
png(file.path(path_figs, "topic_summary_K10.png"), width = 1600, height = 1000, res = 150)
plot(stm_model, type = "summary", main = "Topic Prevalence (K=10)", xlim = c(0, 0.2))
dev.off()

# Figure 2: Temporal
cat("Generating topic_prevalence_over_time_K10.png...\n")
eff <- estimateEffect(1:K_final ~ year, stmobj = stm_model, metadata = meta)
png(file.path(path_figs, "topic_prevalence_over_time_K10.png"), width = 1600, height = 1000, res = 150)
plot(eff, covariate = "year", method = "continuous", topics = c(1, 2, 3), model = stm_model, printlegend = TRUE, main = "Topic Prevalence Over Time")
dev.off()

# Figure 3: Word clouds (larger)
cat("Generating word clouds...\n")
library(wordcloud)
for (t in 1:3) {
  png(file.path(path_figs, paste0("topic_wordcloud_topic", t, "_K10.png")), width = 1200, height = 1200, res = 150)
  cloud(stm_model, topic = t, scale = c(5, 0.8), max.words = 50)
  dev.off()
}

cat("Figures regenerated successfully!\n")
