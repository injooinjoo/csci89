# =========================
# CSCI S-89B - Assignment 6
# Problem 1: LDA on a single long news article (Enhanced + Bigrams + TF-IDF)
# =========================

import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- NLP / Gensim ---
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
from gensim.models.phrases import Phrases, Phraser

# ---------- Robust NLTK bootstrapping ----------
def _safe_dl(pkg):
    try:
        nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False

_safe_dl('stopwords'); _safe_dl('wordnet'); _safe_dl('omw-1.4')
HAS_POS = _safe_dl('averaged_perceptron_tagger_eng') or _safe_dl('averaged_perceptron_tagger')

# ---------- (a) Article + manual K ----------
raw_article = """
Milei scores historic win in Argentina midterms, tightens grip on Congress

Argentine President Javier Milei secured a decisive victory Sunday in midterm elections, expanding his control of Congress and giving his government fresh momentum to push forward with deep spending cuts and sweeping free-market reforms.

The result gives Milei’s libertarian movement a boost and marks another sharp turn for one of Latin America’s largest and most volatile economies.

Milei’s party, La Libertad Avanza, won about 41.5% of the vote in Buenos Aires province, a historic upset in a region long dominated by the Peronist opposition. The rival coalition took 40.8%, according to Reuters and The Associated Press.

Nationwide, La Libertad Avanza increased its seats in the lower house from 37 to 64, positioning Milei to more easily defend his vetoes and executive decrees that have defined his economic agenda.

“The result is better than even the most optimistic Milei supporters were hoping for,” said Marcelo Garcia, director for the Americas at risk-analysis firm Horizon Engage. “With this result, Milei will be able to easily defend his decrees and vetoes in Congress.”

Political consultant Gustavo Cordoba told Reuters the outcome reflected a cautious optimism among voters who appear willing to give Milei’s economic policies more time.

“Many people were willing to give the government another chance,” Cordoba said. “The triumph is unquestionable.”

Reuters reported that inflation has fallen from 12.8% before Milei’s inauguration to 2.1% last month. His government has also posted a fiscal surplus and pushed through broad deregulation measures — a dramatic reversal after years of economic turbulence.

According to the Associated Press, the U.S. government under President Donald Trump offered Argentina a $40 billion aid package, including a $20 billion currency swap and a proposed $20 billion debt-investment facility, after tying future U.S. support to Milei’s performance in the midterms.

President Donald Trump congratulated Milei on Truth Social, writing: “Congratulations to President Javier Milei on his landslide victory in Argentina. He is doing a wonderful job! Our confidence in him was justified by the people of Argentina.”

Investors reacted positively to the results. Argentine bonds and stocks are expected to rally as Milei’s stronger hand in Congress gives him the political capital to accelerate his reforms.

Milei called the election “a turning point for Argentina,” according to AFP.
""".strip()

K = 3  # manual estimate

# ---------- (b) Split paragraphs & preprocessing ----------
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_article) if p.strip()]

base_stops = set(stopwords.words('english'))
custom_news_stops = {
    'according','said','reported','report','press','associated','reuters','ap','afp','news','told','via',
    'truth','social',
    'monday','tuesday','wednesday','thursday','friday','saturday','sunday','today','yesterday','tonight',
    'percent','billion','million',
    # optional de-noise of very generic article words
    'result','people'
}
# tiny bump to suppress residual generic verbs
custom_soft = {'give','another','willing','easily'}
stops = base_stops | custom_news_stops | custom_soft

lemmatizer = WordNetLemmatizer()

def _to_wn(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocess(text: str):
    toks = simple_preprocess(text, deacc=True, min_len=2)
    toks = [t for t in toks if t not in stops]
    if not toks: return []
    if HAS_POS:
        tagged = pos_tag(toks)
        lemmas = [lemmatizer.lemmatize(w, _to_wn(tag)) for w, tag in tagged]
    else:
        lemmas = [lemmatizer.lemmatize(w) for w in toks]
    lemmas = [w for w in lemmas if w not in stops and len(w) > 1]
    return lemmas

docs = [preprocess(p) for p in paragraphs]

print("=== Processed Paragraph Samples (enhanced) ===")
for i, d in enumerate(docs[:3], 1):
    print(f"Para {i}: {d[:20]} ...")
print()

# ---- Bigrams (phrase mining) ----
phrases = Phrases(docs, min_count=2, threshold=10)  # lenient for small corpus
bigram = Phraser(phrases)
docs_bi = [bigram[d] for d in docs]

# ---- Dictionary / BoW ----
dictionary = Dictionary(docs_bi)
dictionary.filter_extremes(no_below=1, no_above=0.9)
corpus = [dictionary.doc2bow(doc) for doc in docs_bi]

# ---- Light TF-IDF filtering to drop very low-information tokens ----
tfidf = TfidfModel(corpus, smartirs='ntc')
corpus_tfidf = []
for bow in corpus:
    tf = tfidf[bow]
    filtered = [(id_, w) for (id_, w) in tf if w > 0.10]
    corpus_tfidf.append(filtered if filtered else tf)

# ---------- (c) LDA ----------
lda = LdaModel(
    corpus=corpus_tfidf,
    id2word=dictionary,
    num_topics=K,
    random_state=42,
    passes=30,
    iterations=300,
    alpha='auto',
    eta='auto',
    minimum_probability=0.0
)

# ---------- (d) Results ----------
print("=== Top 10 words per topic (enhanced) ===")
topic_top_words = {}
for k in range(K):
    terms = lda.show_topic(k, topn=10)
    top_words = [w for w, p in terms]
    topic_top_words[k] = top_words
    print(f"Topic {k}: {top_words}")
print()

def topic_prob(bow, k):
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    return dict(dist).get(k, 0.0)

print("=== Top 2 paragraphs per topic (by topic probability) ===")
topic_top_docs = {}
for k in range(K):
    scored = [(i, topic_prob(corpus_tfidf[i], k)) for i in range(len(corpus_tfidf))]
    scored.sort(key=lambda x: x[1], reverse=True)
    top2 = scored[:2]
    topic_top_docs[k] = top2
    print(f"\n[Topic {k}] top2 docs (idx, prob): {top2}")
    for idx, prob in top2:
        snippet = paragraphs[idx]
        if len(snippet) > 240: snippet = snippet[:240] + " ..."
        print(f"  • Doc #{idx} (p={prob:.3f}): {snippet}")

label_block = {
    'milei','la','libertad','avanza','coalition','province','buenos','aires',
    'latin','america','election','midterm','win','vote','historic'
}
def auto_label(words):
    picked = [w for w in words if w not in label_block][:3]
    return " / ".join(picked) if picked else "topic"

print("\n=== Topic Labels (short) ===")
labels = {}
for k in range(K):
    labels[k] = auto_label(topic_top_words[k])
    print(f"Topic {k}: {labels[k]}")

print("\n=== Summary ===")
for k in range(K):
    top_words = ", ".join(topic_top_words[k][:10])
    reps = [idx for idx, _ in topic_top_docs[k]]
    print(f"- Topic {k} [{labels[k]}]")
    print(f"  Top words: {top_words}")
    print(f"  Representative paragraphs: {reps}")



# =========================
# Problem 2: sweep K, compute metrics, and SAVE plots as PNGs (Windows-safe)
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel

def topic_word_sets(lda_model, topn=15):
    return [set(w for w, _ in lda_model.show_topic(k, topn=topn))
            for k in range(lda_model.num_topics)]

def exclusivity_score(lda_model, topn=15):
    # 1 - average Jaccard overlap among topic pairs
    sets = topic_word_sets(lda_model, topn=topn)
    if len(sets) < 2:
        return 0.0
    pairs, jaccs = 0, 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            jacc = inter / union if union else 0.0
            jaccs += jacc
            pairs += 1
    avg_jacc = jaccs / pairs
    return 1.0 - avg_jacc

def rescale(x):
    x = np.array(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def save_lineplot(xs, ys, ylabel, outpath, title=None):
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Number of topics (K)')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Ensure figs directory exists
    os.makedirs("figs", exist_ok=True)

    Ks = list(range(2, 11))
    cohs, excls = [], []

    models = {}
    for K_ in Ks:
        # Train LDA for each K (reuse your dictionary/corpus_tfidf/docs_bi from Problem 1)
        ldaK = LdaModel(
            corpus=corpus_tfidf,
            id2word=dictionary,
            num_topics=K_,
            random_state=42,
            passes=20,
            iterations=200,
            alpha='auto',
            eta='auto'
        )
        models[K_] = ldaK

        # Coherence with single process to avoid Windows multiprocessing issues
        cm = CoherenceModel(model=ldaK, texts=docs_bi, dictionary=dictionary,
                            coherence='c_v', processes=1)
        cohs.append(cm.get_coherence())
        excls.append(exclusivity_score(ldaK, topn=15))

    # Rescale and composite
    cohs_r = rescale(cohs)
    excls_r = rescale(excls)
    comp = 0.5 * cohs_r + 0.5 * excls_r

    # Save PNGs
    save_lineplot(Ks, cohs, "Mean coherence (c_v)", "figs/coherence_vs_k.png",
                  title="Mean coherence (c_v) vs. K")
    save_lineplot(Ks, excls, "Topic exclusivity", "figs/exclusivity_vs_k.png",
                  title="Topic exclusivity vs. K")
    save_lineplot(Ks, comp, "Composite score", "figs/composite_vs_k.png",
                  title="Composite score vs. K")

    # (Optional) also save metrics as CSV for the table in your report
    with open("figs/problem2_metrics.csv", "w", encoding="utf-8") as f:
        f.write("K,coherence,exclusivity,coherence_rescaled,exclusivity_rescaled,composite\n")
        for k, c, e, cr, er, cp in zip(Ks, cohs, excls, cohs_r, excls_r, comp):
            f.write(f"{k},{c:.6f},{e:.6f},{cr:.6f},{er:.6f},{cp:.6f}\n")

    print("Saved: figs/coherence_vs_k.png, figs/exclusivity_vs_k.png, figs/composite_vs_k.png")
    print("Saved metrics CSV: figs/problem2_metrics.csv")
