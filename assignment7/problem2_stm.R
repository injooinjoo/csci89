# ================================================================
# CSCI S-89B — A7 Problem 2 (searchK + Composite + Final STM)
# Portable, minimal, and reproducible script with English comments
# ================================================================

# ---------- 0) Portable local library path ----------
lib <- "r-lib"
if (!dir.exists(lib)) dir.create(lib, recursive = TRUE)
.libPaths(c(normalizePath(lib), .libPaths()))

# ---------- 1) Quiet install & libraries ----------
qi <- function(pkg) if (!requireNamespace(pkg, quietly = TRUE)) {
  install.packages(pkg, lib = lib, repos = "https://cloud.r-project.org")
}
pkgs <- c("stm","ggplot2","wordcloud","data.table","scales")
invisible(lapply(pkgs, qi))

suppressPackageStartupMessages({
  library(stm)
  library(ggplot2)
  library(wordcloud)
  library(data.table)
  library(scales)
})

set.seed(89)

# ---------- 2) Helpers ----------
# Build STM-ready inputs with robust pruning; keep original text for findThoughts
build_inputs <- function(df, lower = 20) {
  stopifnot(all(c("Review","Rating") %in% names(df)))
  df$Rating <- suppressWarnings(as.numeric(df$Rating))
  tp <- textProcessor(
    documents          = df$Review,
    metadata           = data.frame(Rating = df$Rating),
    lowercase          = TRUE,
    removenumbers      = TRUE,
    removepunctuation  = TRUE,
    removestopwords    = TRUE,
    stem               = TRUE
  )
  prep <- prepDocuments(tp$documents, tp$vocab, tp$meta, lower.thresh = lower)
  if (length(prep$documents) == 0) stop("All docs removed; try lowering 'lower.thresh'.")
  # align original raw text by retained doc indices
  kept_idx <- as.integer(names(prep$documents))
  prep$meta$.orig_text <- df$Review[kept_idx]
  list(docs = prep$documents, vocab = prep$vocab, meta = prep$meta)
}

# Small safe wrapper to export labelTopics table (top-n words for multiple metrics)
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

# Fit STM with a couple of fallback configurations (more resilient on messy text)
fit_with_fallback <- function(inp, K, max_its = 150, seed = 89) {
  cfg <- list(
    list(init.type = "Spectral", max.em.its = max_its),
    list(init.type = "LDA",      max.em.its = max_its)
  )
  for (i in seq_along(cfg)) {
    set.seed(seed + i)
    cat(sprintf("[fit] attempt %d: init=%s\n", i, cfg[[i]]$init.type))
    fit_try <- try(
      stm(documents = inp$docs, vocab = inp$vocab, K = K,
          prevalence = ~ Rating, data = inp$meta,
          init.type = cfg[[i]]$init.type, max.em.its = cfg[[i]]$max.em.its,
          verbose = FALSE),
      silent = TRUE
    )
    if (!inherits(fit_try, "try-error")) return(fit_try)
  }
  stop("stm() failed for all fallback attempts.")
}

# ---------- 3) Load data ----------
# Expect: hotel_reviews.csv with columns: Review (text), Rating (numeric 1–5)
df <- read.csv("hotel_reviews.csv", stringsAsFactors = FALSE)

# ---------- 4) Build inputs for searchK (use a subset to speed up) ----------
subset_n <- min(3000, nrow(df))                  # adjustable budget
df_sub    <- df[sample(nrow(df), subset_n), , drop = FALSE]
inp_sub   <- build_inputs(df_sub, lower = 20)

# ---------- 5) searchK over a small grid; compute Composite score ----------
K_grid <- 2:6
cat("[searchK] K grid:", paste(K_grid, collapse = ","), "\n")
set.seed(901)
sk <- searchK(
  documents  = inp_sub$docs,
  vocab      = inp_sub$vocab,
  data       = inp_sub$meta,
  K          = K_grid,
  prevalence = ~ Rating,
  init.type  = "Spectral",
  max.em.its = 75,
  verbose    = FALSE
)

# --- FIX: list-columns -> numeric vectors (mean over topics per K) ---
coh <- vapply(sk$results$semcoh, function(x) mean(unlist(x)), numeric(1))
exc <- vapply(sk$results$exclus, function(x) mean(unlist(x)), numeric(1))

# Min–max rescale to [0,1]
coh_s <- scales::rescale(coh, to = c(0,1))
exc_s <- scales::rescale(exc, to = c(0,1))

# Composite = average of the two rescaled metrics
composite <- 0.5 * coh_s + 0.5 * exc_s

# Export metrics table
search_tbl <- data.frame(
  K = K_grid,
  coherence = coh,
  exclusivity = exc,
  coherence_s = coh_s,
  exclusivity_s = exc_s,
  composite = composite
)
write.csv(search_tbl, "p2_searchK_metrics.csv", row.names = FALSE)
print(search_tbl)

# Plots for report
p1 <- ggplot(search_tbl, aes(x = coherence, y = exclusivity, label = K)) +
  geom_point(size = 3) + geom_text(vjust = -0.9) +
  labs(title = "searchK: Exclusivity vs Semantic Coherence",
       x = "Semantic Coherence", y = "Exclusivity")
ggsave("p2_scatter_exclusivity_vs_coherence.png", p1, width = 7, height = 5, dpi = 160)

p2 <- ggplot(search_tbl, aes(x = K, y = composite)) +
  geom_line() + geom_point(size = 3) +
  scale_x_continuous(breaks = K_grid) +
  labs(title = "Composite Score by K",
       y = "Composite = mean(rescaled coherence, rescaled exclusivity)")
ggsave("p2_composite_by_k.png", p2, width = 7, height = 5, dpi = 160)

bestK <- search_tbl$K[ which.max(search_tbl$composite) ]
cat(sprintf("[searchK] best K by Composite = %d\n", bestK))

# ---------- 6) Rebuild full inputs (all data) & fit final STM at bestK ----------
inp_full <- build_inputs(df, lower = 20)
stm_fit  <- fit_with_fallback(inp_full, K = bestK, max_its = 150, seed = 89)

saveRDS(stm_fit, file = sprintf("p2_stm_model_K%d.rds", bestK))

# ---------- 7) Topic summaries: top-7 FREX, full table, wordclouds ----------
K <- nrow(stm_fit$beta$logbeta[[1]])

# (A) exact top-7 FREX words per topic (required)
lt7 <- labelTopics(stm_fit, n = 7)
top7 <- data.frame(
  Topic = paste0("Topic", rep(1:K, each = 7)),
  Rank  = rep(1:7, times = K),
  FREX  = unlist(lt7$frex),
  stringsAsFactors = FALSE
)
write.csv(top7, "p2_top7_frex_per_topic.csv", row.names = FALSE)

# (B) optional: rich table with Prob/FREX/Lift/Score (n=10)
lt10 <- labelTopics(stm_fit, n = 10)
top10 <- extract_top_table(lt10, K)
write.csv(top10, "p2_top_words_by_metric.csv", row.names = FALSE)

# (C) wordclouds per topic
dir.create("p2_topic_clouds", showWarnings = FALSE)
for (k in seq_len(K)) {
  png(sprintf("p2_topic_clouds/p2_cloud_topic_%02d.png", k), width = 900, height = 700)
  cloud(stm_fit, topic = k, max.words = 80)
  title(sprintf("Topic %d", k))
  dev.off()
}

# ---------- 8) Representative review (n = 1 per topic) ----------
# Use original kept texts aligned in 'meta'
rep1 <- findThoughts(stm_fit, texts = inp_full$meta$.orig_text,
                     topics = seq_len(K), n = 1)$docs

con <- file("p2_representative_reviews.txt", open = "w", encoding = "UTF-8")
for (k in seq_len(K)) {
  cat(sprintf("=== Topic %d: Representative Review ===\n", k), file = con)
  cat(rep1[[k]][1], "\n\n", file = con)
}
close(con)

# ---------- 9) Covariate effect of Rating (prevalence) ----------
eff <- estimateEffect(1:K ~ Rating, stmobj = stm_fit, metadata = inp_full$meta,
                      uncertainty = "Global")
png("p2_effect_rating.png", width = 1000, height = 750)
plot(eff, covariate = "Rating", method = "continuous",
     xlab = "Rating", main = "Effect of Rating on Topic Prevalence",
     topics = 1:K)
dev.off()

# ---------- 10) Export a quick quality scatter (coherence vs exclusivity) ----------
sc <- semanticCoherence(stm_fit, inp_full$docs)
ex <- exclusivity(stm_fit)
qdf <- data.frame(Topic = factor(1:K), Exclusivity = ex, Coherence = sc)
png("p2_quality_scatter.png", width = 850, height = 650)
print(
  ggplot(qdf, aes(x = Coherence, y = Exclusivity, label = paste0("T", Topic))) +
    geom_point() + geom_text(vjust = -0.7) +
    labs(title = "Topic Quality", x = "Semantic Coherence", y = "Exclusivity")
)
dev.off()
write.csv(qdf, "p2_topic_quality_metrics.csv", row.names = FALSE)

# ---------- 11) Final console summary ----------
cat("\n[Done]\n",
    "- p2_searchK_metrics.csv\n",
    "- p2_scatter_exclusivity_vs_coherence.png\n",
    "- p2_composite_by_k.png\n",
    sprintf("- p2_stm_model_K%d.rds\n", bestK),
    "- p2_top7_frex_per_topic.csv\n",
    "- p2_top_words_by_metric.csv\n",
    "- p2_topic_clouds/*.png\n",
    "- p2_representative_reviews.txt\n",
    "- p2_effect_rating.png\n",
    "- p2_quality_scatter.png, p2_topic_quality_metrics.csv\n", sep = "")
