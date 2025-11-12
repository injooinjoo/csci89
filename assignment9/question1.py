# assignment9_problem1_clean_tfidf_svm.py
# pip: pandas numpy scikit-learn nltk matplotlib

import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# -----------------------
# 1) Load
# -----------------------
df = pd.read_csv("news_data.csv")
# 예상 컬럼명: news_headline, news_article, news_category (이름이 다르면 아래 라인만 바꿔)
text = (df["news_headline"].astype(str) + " . " + df["news_article"].astype(str)).fillna("")
y_raw = df["news_category"].astype(str)

# -----------------------
# 2) Clean
# -----------------------
BOILER_PATTERNS = [
    r"\bgoogle\s*maps\b", r"\bclick here\b", r"\bread more\b", r"\bcopyright\s+\d{4}\b",
    r"\ballg\b", r"\bzeitung\b", r"\breuters\b", r"\bafp\b", r"\bap news\b",
]
URL = r"http[s]?://\S+|www\.\S+"
EMAIL = r"\S+@\S+"
NUM = r"\b\d+(?:[\.,]\d+)*\b"

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(URL, " ", s)
    s = re.sub(EMAIL, " ", s)
    # 보일러플레이트 제거
    for p in BOILER_PATTERNS:
        s = re.sub(p, " ", s, flags=re.IGNORECASE)
    # 괄호/구두점 등 정리
    s = re.sub(r"[\(\)\[\]\{\}<>]", " ", s)
    s = re.sub(NUM, " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

clean = text.apply(clean_text)

# 너무 짧은 문서 제거 (토큰 < 5)
mask = clean.str.split().apply(len) >= 5
clean = clean[mask]
y_raw = y_raw[mask]

# -----------------------
# 3) Vectorize (TF-IDF)
# -----------------------
# 경험상: sublinear_tf=True, max_df=0.9, min_df=3가 안정적
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    max_df=0.9,
    min_df=3,
    sublinear_tf=True,
    stop_words="english"  # 영문 기준
)
X = tfidf.fit_transform(clean)

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = list(le.classes_)

# -----------------------
# 4) Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 5) Train: Linear SVM (baseline strong)
# -----------------------
svm = LinearSVC(C=1.0)  # 필요시 C 조절
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
acc = accuracy_score(y_test, pred)

# (선택) 로지스틱 비교
# logreg = LogisticRegression(
#     penalty="l2", solver="saga", max_iter=3000, n_jobs=-1, C=2.0, class_weight="balanced"
# )
# logreg.fit(X_train, y_train)
# pred_lr = logreg.predict(X_test)
# acc_lr = accuracy_score(y_test, pred_lr)

# -----------------------
# 6) Report-friendly outputs
# -----------------------
print("\n=== Problem 1 — Clean TF-IDF + LinearSVC ===")
print(f"Documents: {X.shape[0]}  |  Features: {X.shape[1]}  |  Classes: {len(classes)}")
print(f"Test Accuracy (LinearSVC): {acc:.4f}")
print("Random Baseline:", f"{1.0/len(classes):.4f}", f"(K={len(classes)})\n")

print("Classification Report (SVM):")
print(classification_report(y_test, pred, target_names=classes, digits=3))

cm = confusion_matrix(y_test, pred)
np.set_printoptions(linewidth=200)
print("Confusion Matrix (rows=true, cols=pred):\n", cm)

# -----------------------
# 7) Top features per class (for report)
# -----------------------
def top_terms_per_class(clf, vectorizer, n=15):
    # LinearSVC has coef_ for OvR
    if not hasattr(clf, "coef_"):
        return {}
    terms = np.array(vectorizer.get_feature_names_out())
    tops = {}
    for i, cls in enumerate(classes):
        coef = clf.coef_[i]
        idx = np.argsort(coef)[-n:][::-1]
        tops[cls] = list(terms[idx])
    return tops

tops = top_terms_per_class(svm, tfidf, n=15)
print("\nTop 15 terms per class (SVM):")
for c, words in tops.items():
    print(f"- {c}: {', '.join(words)}")

# Save a tiny TSV for the report
with open("problem1_top_terms.tsv", "w", encoding="utf-8") as f:
    for c, words in tops.items():
        f.write(f"{c}\t{','.join(words)}\n")

print("\nSaved: problem1_top_terms.tsv")
