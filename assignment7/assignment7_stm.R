# =========================
# CSCI S-89B — A7 P1 Minimal & Stable STM (All assets)
# =========================

# 0) Local lib path (portable)
lib <- "r-lib"
if (!dir.exists(lib)) dir.create(lib, recursive = TRUE)
.libPaths(c(normalizePath(lib), .libPaths()))

# 1) Quiet install
qi <- function(pkg) if (!requireNamespace(pkg, quietly = TRUE)) {
  install.packages(pkg, lib = lib, repos = "https://cloud.r-project.org")
}
qi("stm"); qi("wordcloud"); qi("ggplot2"); qi("gridExtra"); qi("data.table")

suppressPackageStartupMessages({
  library(stm)
  library(wordcloud)
  library(ggplot2)
  library(gridExtra)
  library(data.table)
})

set.seed(89)

# ---------- helpers ----------
# Clean, prune, build stm inputs
build_inputs <- function(df, lower = 20) {
  stopifnot(all(c("Review","Rating") %in% names(df)))
  df$Rating <- as.numeric(df$Rating)
  proc <- textProcessor(
    documents         = df$Review,
    metadata          = data.frame(Rating = df$Rating),
    lowercase         = TRUE,
    removenumbers     = TRUE,
    removepunctuation = TRUE,
    removestopwords   = TRUE,
    stem              = TRUE
  )
  prep <- prepDocuments(proc$documents, proc$vocab, proc$meta,
                        lower.thresh = lower)
  if (length(prep$documents) == 0) stop("All docs removed; lower.thresh too high.")
  # align raw text for findThoughts
  prep$meta$.orig_text <- df$Review[as.integer(names(prep$documents))]
  list(docs = prep$documents, vocab = prep$vocab, meta = prep$meta)
}

# Robust fit with retries
fit_full_with_retry <- function(df_orig, K = 3, max_its = 100, seed = 89) {
  lowers <- c(20, 25, 30)
  inits  <- c("Spectral", "Spectral", "LDA")
  for (i in seq_along(lowers)) {
    cat(sprintf("\n[Retry %d] lower.thresh=%d, init=%s\n", i, lowers[i], inits[i]))
    inp <- try(build_inputs(df_orig, lower = lowers[i]), silent = TRUE)
    if (inherits(inp, "try-error")) next
    set.seed(seed)
    fit_try <- try(stm(documents = inp$docs, vocab = inp$vocab, K = K,
                       prevalence = ~ Rating, data = inp$meta,
                       init.type = inits[i], max.em.its = max_its,
                       verbose = FALSE),
                   silent = TRUE)
    if (!inherits(fit_try, "try-error")) {
      cat("  -> success\n")
      return(list(model = fit_try, docs = inp$docs, vocab = inp$vocab, meta = inp$meta))
    } else cat("  -> stm() failed; escalating...\n")
  }
  stop("All retries failed.")
}

# Safe topic table (Prob/FREX/Lift/Score)
extract_top_table <- function(lt, K) {
  pad <- function(v, n) c(v, if (length(v) < n) rep(NA_character_, n - length(v)) else NULL)
  tabs <- lapply(seq_len(K), function(k){
    P <- if (!is.null(lt$prob))  lt$prob[[k]]  else character()
    F <- if (!is.null(lt$frex))  lt$frex[[k]]  else character()
    L <- if (!is.null(lt$lift))  lt$lift[[k]]  else character()
    S <- if (!is.null(lt$score)) lt$score[[k]] else character()
    m <- max(length(P), length(F), length(L), length(S), 0L)
    data.frame(
      Topic = paste0("Topic", k),
      Rank  = seq_len(m),
      Prob  = pad(P, m),
      FREX  = pad(F, m),
      Lift  = pad(L, m),
      Score = pad(S, m),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, tabs)
}

# ---------- 2) Load data ----------
# Expect: hotel_reviews.csv with columns Review, Rating
df <- read.csv("hotel_reviews.csv", stringsAsFactors = FALSE)

# ---------- 3) Optional K search on subset (for report figure) ----------
subset_n <- min(3000, nrow(df))
df_s <- df[sample(nrow(df), subset_n), , drop = FALSE]
inp_s <- build_inputs(df_s, lower = 20)

set.seed(89)
sk <- try(searchK(
  documents = inp_s$docs, vocab = inp_s$vocab, K = c(3,4,5,6),
  prevalence = ~ Rating, data = inp_s$meta,
  init.type = "Spectral", max.em.its = 60, verbose = FALSE
), silent = TRUE)

if (!inherits(sk, "try-error")) {
  png("searchK_plot.png", width = 1200, height = 900, res = 140)
  plot(sk)  # uses stm's S3 method; handles list-columns safely
  dev.off()
}

# ---------- 4) Full STM fit (K=3 per assignment) ----------
fit <- fit_full_with_retry(df_orig = df, K = 3, max_its = 100, seed = 89)
stm_best <- fit$model; docs <- fit$docs; vocab <- fit$vocab; meta <- fit$meta
K <- nrow(stm_best$beta$logbeta[[1]])

# Save model for reproducibility
saveRDS(stm_best, file = "stm_model_K3.rds")

# ---------- 5) Topic words & labels for report ----------
# (A) exact top-7 FREX words per topic (report requirement)
lt7 <- labelTopics(stm_best, n = 7)
top7 <- data.frame(
  Topic = paste0("Topic", rep(1:K, each = 7)),
  Rank  = rep(1:7, times = K),
  FREX  = unlist(lt7$frex),
  stringsAsFactors = FALSE
)
write.csv(top7, "top7_frex_per_topic.csv", row.names = FALSE)

# (B) full metrics table (Prob/FREX/Lift/Score) with up to 10 words (optional but useful)
lt10 <- labelTopics(stm_best, n = 10)
top10 <- extract_top_table(lt10, K)
write.csv(top10, "top_words_by_metric.csv", row.names = FALSE)

# (C) simple auto label suggestion from FREX (top-3 joined) — you can edit in report
topic_labels_suggested <- data.frame(
  Topic = paste0("Topic", 1:K),
  Label = sapply(1:K, function(k) paste(head(lt7$frex[[k]], 3), collapse = ", "))
)
write.csv(topic_labels_suggested, "topic_labels_suggested.csv", row.names = FALSE)

# ---------- 6) Plots for report ----------
# 6.1 Wordclouds
png("wordcloud_topics.png", width = 1800, height = 600, res = 200)
par(mfrow = c(1, K), mar = c(1,1,2,1))
for (k in 1:K) { cloud(stm_best, topic = k, scale = c(2.5, 0.5)); title(paste("Topic", k)) }
par(mfrow = c(1,1)); dev.off()

# 6.2 Topic summary bar
png("summary_plot.png", width = 900, height = 600, res = 150)
plot(stm_best, type = "summary", n = 7)
dev.off()

# 6.3 Quality scatter (Exclusivity vs. Semantic Coherence)
sc <- semanticCoherence(stm_best, docs)
ex <- exclusivity(stm_best)
qdf <- data.frame(Topic = factor(1:K), Exclusivity = ex, Coherence = sc)
png("quality_scatter.png", width = 800, height = 650, res = 150)
print(
  ggplot(qdf, aes(x = Coherence, y = Exclusivity, label = paste0("T", Topic))) +
    geom_point() + geom_text(vjust = -0.7) +
    labs(title = "Topic Quality", x = "Semantic Coherence", y = "Exclusivity")
)
dev.off()
write.csv(qdf, "topic_quality_metrics.csv", row.names = FALSE)

# 7) Representative reviews (to file) — n = 5
rep_docs <- findThoughts(
  stm_best,
  texts  = meta$.orig_text,
  topics = 1:K,
  n      = 5
)$docs
sink("representative_reviews.txt")
for (k in 1:K) {
  cat("\n========== Topic", k, "==========\n")
  cat(paste0(rep_docs[[k]], collapse = "\n\n"), "\n")
}
sink()

# ---------- 8) Covariate effect (Rating) ----------
eff <- estimateEffect(1:K ~ Rating, stmobj = stm_best, metadata = meta, uncertainty = "Global")

# Keep the plot (works fine)
png("rating_effects.png", width = 1800, height = 600, res = 180)
par(mfrow = c(1, K), mar = c(4,4,3,1))
for (k in 1:K) {
  plot(eff, covariate = "Rating", topics = k, method = "continuous",
       xlab = "Rating", ylab = "Expected Topic Prevalence",
       main = paste("Topic", k), printlegend = FALSE)
}
par(mfrow = c(1,1))
dev.off()

# --- robust grid export for report (version-agnostic) ---
# Fallback to simple linear model on theta ~ Rating to create a 100-point grid
rating_vec <- as.numeric(meta$Rating)
theta <- stm_best$theta
colnames(theta) <- paste0("Topic", 1:K)

effect_grid_lm <- function(k, n = 100) {
  tp <- theta[, k]
  dfm <- data.frame(tp = tp, Rating = rating_vec)
  fit <- lm(tp ~ Rating, data = dfm)  # simple, stable
  r <- range(rating_vec, na.rm = TRUE)
  newd <- data.frame(Rating = seq(r[1], r[2], length.out = n))
  pr <- predict(fit, newd, se.fit = TRUE)
  ci <- 1.96 * pr$se.fit
  data.frame(
    Topic = k,
    Rating = newd$Rating,
    Mean = as.numeric(pr$fit),
    CI.L = as.numeric(pr$fit - ci),
    CI.U = as.numeric(pr$fit + ci)
  )
}

eff_grids <- data.table::rbindlist(lapply(1:K, effect_grid_lm))
data.table::fwrite(eff_grids, "rating_effects_grid.csv")
cat("Wrote rating_effects_grid.csv with", nrow(eff_grids), "rows\n")


# ---------- 9) Document-topic matrix for possible appendix ----------
theta <- stm_best$theta
colnames(theta) <- paste0("Topic", 1:K)
fwrite(as.data.table(theta), "doc_topic_matrix.csv")

cat("\nAssets written:\n",
    "- searchK_plot.png (if available)\n",
    "- wordcloud_topics.png\n",
    "- summary_plot.png\n",
    "- quality_scatter.png, topic_quality_metrics.csv\n",
    "- representative_reviews.txt\n",
    "- rating_effects.png, rating_effects_grid.csv\n",
    "- top_words_by_metric.csv, topic_labels_suggested.csv\n",
    "- doc_topic_matrix.csv\n",
    "- stm_model_K3.rds\n")
