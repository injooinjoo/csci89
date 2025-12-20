# Final Project Topic Proposal  
**CSCI S-89B: Introduction to Natural Language Processing**  
**Student:** InJoo Kim  

## 1. Project Title  
Case Study in Topic Modeling: Discovering Latent Themes in News Headlines  

## 2. Problem Statement & Motivation  
In this final project, I will explore how topic modeling can uncover latent themes from a large collection of news headlines.  
My main goal is to compare how different topic modeling approaches capture topic structure in real-world news data, with a focus on interpretability and stability of topics.

I chose this topic because news media produce enormous amounts of short, high-level text (headlines), and it is not feasible to manually read and categorize them. Topic modeling provides a scalable way to summarize and organize such collections, which is directly relevant to many real-world applications (news analytics, media monitoring, trend analysis).

## 3. Selected Technology  
I will implement topic modeling using **R’s STM (Structural Topic Modeling) framework**, as required by the course guidelines.  
Structural Topic Modeling is an extension of LDA that allows us to incorporate document-level metadata as covariates (for example, publication date) and study how topic prevalence changes along those covariates.

Planned steps include:
- Building a document–term matrix from news headlines.
- Fitting STM models with different numbers of topics \(K\).
- Comparing topic quality across different \(K\) values.
- Using metadata (e.g., publication date) as a prevalence covariate to analyze how topics evolve over time.
- Visualizing topic–word distributions and topic prevalence.

(If time permits, I may briefly contrast STM results with a baseline LDA model, but STM in R will be the primary technology used to satisfy the final project requirements. :contentReference[oaicite:0]{index=0})

## 4. Data Source  
I will use a publicly available **Kaggle News Headlines dataset**, which contains news headlines along with associated metadata such as publication date.

- **Data type:** Short text headlines (one-line summaries of news articles)  
- **Metadata:** Publication date (and possibly other fields depending on the specific Kaggle dataset variant)  
- **Planned usage:**  
  - Text content will be used as input for topic modeling.  
  - Publication date will be used as a covariate in STM to examine temporal dynamics of topics.

The final report and slides will include the full Kaggle URL. If the complete dataset exceeds 10 MB, I will include only a small sample in the submitted ZIP file and provide the original dataset URL as required. :contentReference[oaicite:1]{index=1}  

## 5. Inputs, Outputs, and Number of Observations  

- **Inputs**
  - Raw news headlines (text)
  - Document-level metadata such as:
    - Publication date (primary covariate)
  - Preprocessed representations:
    - Tokenized text
    - Document–term matrix suitable for STM

- **Outputs**
  - **Topic–word distributions:** Top words per topic and their probabilities
  - **Document–topic proportions:** Topic prevalence for each headline
  - **Visualizations:**
    - Bar plots of top words per topic
    - Word clouds for selected topics
    - Topic prevalence over time (using STM’s covariate functionality)
  - **Interpretation:**
    - Human-readable labels for each topic
    - Qualitative comparison of topics and their temporal trends

- **Number of Observations**
  - The Kaggle dataset typically contains approximately **150,000–200,000** headlines.
  - For computational efficiency, I may start with a subset (e.g., 20,000–50,000 headlines) and then scale up as needed, depending on performance on my hardware.

## 6. Hardware / Computing Environment  

I will run all experiments locally on:

- **Machine:** MacBook Pro 14"  
- **CPU/GPU:** Apple M3 Max  
- **Memory:** 16 GB RAM  
- **Software environment:**
  - R (latest stable version)
  - R packages: `stm`, `tm` or `quanteda`, and other basic utilities
  - (Optional) Jupyter Notebook or RMarkdown for literate programming and reproducible analysis

This hardware should be sufficient to:
- Load and preprocess a subset of 20k–50k headlines comfortably.
- Fit multiple STM models with different numbers of topics.
- Produce the required visualizations for the final project presentation and report.

## 7. Planned Deliverables (Aligned with Course Requirements)

- **Working code and demo:**
  - R scripts / RMarkdown notebook that load the data, preprocess text, run STM, and generate visualizations.
- **Visualization component:**
  - Graphical representation of topics and their evolution over time.
- **Slides (10–20):**
  - Problem statement, method overview (STM), demo snapshots, pros/cons, and a link to the presentation video.
- **Final report:**
  - Written as a tutorial-style document so that classmates can reproduce the entire pipeline (installation, configuration, data loading, modeling, and visualization).
- **Video presentation (7–15 minutes):**
  - Summary of topic modeling, explanation of STM, project walkthrough, results, and lessons learned.

This proposal summarizes my final project plan for a **case study in topic modeling using R’s STM on Kaggle News Headlines**.
