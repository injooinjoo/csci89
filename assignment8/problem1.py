# (a) Data preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("Spam_SMS.csv")[["Class","Message"]].dropna()

# Encode labels: ham->0, spam->1
df["Label"] = df["Class"].map({"ham":0, "spam":1})

# Train/test split (80/20, reproducible)
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["Message"], df["Label"], test_size=0.2, random_state=42, stratify=df["Label"]
)

# TF-IDF with English stop words; fit on train, transform both
tfidf = TfidfVectorizer(stop_words="english")
X_train = tfidf.fit_transform(X_train_txt)
X_test  = tfidf.transform(X_test_txt)

# (b) Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

def metrics(y_true, y_pred):
    # Compute accuracy, sensitivity (recall for class 1), specificity (recall for class 0)
    import numpy as np
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return acc, sens, spec, cm

nb = MultinomialNB()  # default params
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

acc_nb, sens_nb, spec_nb, cm_nb = metrics(y_test, y_pred_nb)
print("[NB] Accuracy:", acc_nb, "Sensitivity:", sens_nb, "Specificity:", spec_nb)
print("[NB] Confusion matrix:\n", cm_nb)



# (c) KNN: default, then tune n_neighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Default KNN
knn_def = KNeighborsClassifier()  # n_neighbors=5 by default
knn_def.fit(X_train, y_train)
y_pred_knn_def = knn_def.predict(X_test)
acc_kdef, sens_kdef, spec_kdef, cm_kdef = metrics(y_test, y_pred_knn_def)
print("[KNN default] Acc:", acc_kdef, "Sens:", sens_kdef, "Spec:", spec_kdef)

# Tune k = 1..15 and record test accuracy
k_values = range(1,16)
k_results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc, sens, spec, _ = metrics(y_test, y_pred)
    k_results.append((k, acc))
    print(f"[KNN k={k}] Test Acc: {acc}")

# Select best k by test accuracy
k_best, acc_best = max(k_results, key=lambda t: t[1])
knn_best = KNeighborsClassifier(n_neighbors=k_best)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)
acc_knn, sens_knn, spec_knn, cm_knn = metrics(y_test, y_pred_knn)
print(f"[KNN best k={k_best}] Acc:", acc_knn, "Sens:", sens_knn, "Spec:", spec_knn)
print("[KNN best] Confusion matrix:\n", cm_knn)


# (d) Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)  # default settings
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

acc_lr, sens_lr, spec_lr, cm_lr = metrics(y_test, y_pred_lr)
print("[LR] Accuracy:", acc_lr, "Sensitivity:", sens_lr, "Specificity:", spec_lr)
print("[LR] Confusion matrix:\n", cm_lr)


# (e) SVM: default, then tune (kernel, C)
from sklearn.svm import SVC

# Default SVM (kernel='rbf', C=1.0 by default)
svm_def = SVC()
svm_def.fit(X_train, y_train)
y_pred_svm_def = svm_def.predict(X_test)
acc_sdef, sens_sdef, spec_sdef, cm_sdef = metrics(y_test, y_pred_svm_def)
print("[SVM default] Acc:", acc_sdef, "Sens:", sens_sdef, "Spec:", spec_sdef)

# Grid over kernel and C
kernels = ["linear","rbf","poly"]
Cs = [0.5, 1.0, 2.0]
svm_grid = []
for ker in kernels:
    for C in Cs:
        clf = SVC(kernel=ker, C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc, sens, spec, _ = metrics(y_test, y_pred)
        svm_grid.append((ker, C, acc))
        print(f"[SVM {ker}, C={C}] Test Acc: {acc}")

# Select best combination by test accuracy
best_kernel, best_C, best_acc = max(svm_grid, key=lambda t: t[2])
svm_best = SVC(kernel=best_kernel, C=best_C)
svm_best.fit(X_train, y_train)
y_pred_svm = svm_best.predict(X_test)
acc_svm, sens_svm, spec_svm, cm_svm = metrics(y_test, y_pred_svm)
print(f"[SVM best {best_kernel}, C={best_C}] Acc:", acc_svm, "Sens:", sens_svm, "Spec:", spec_svm)
print("[SVM best] Confusion matrix:\n", cm_svm)


# (f) Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)  # defaults: n_estimators=100, max_depth=None, max_features='sqrt'
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf, sens_rf, spec_rf, cm_rf = metrics(y_test, y_pred_rf)
print("[RF] Accuracy:", acc_rf, "Sensitivity:", sens_rf, "Specificity:", spec_rf)
print("[RF] Confusion matrix:\n", cm_rf)


# (g) Plots: sorted accuracy bar, and sensitivity vs specificity scatter
import matplotlib.pyplot as plt
import pandas as pd

# Collect results into a table
results = pd.DataFrame([
    ["Naive Bayes", acc_nb, sens_nb, spec_nb],
    [f"KNN (k={k_best})", acc_knn, sens_knn, spec_knn],
    ["Logistic Regression", acc_lr, sens_lr, spec_lr],
    [f"SVM ({best_kernel}, C={best_C})", acc_svm, sens_svm, spec_svm],
    ["Random Forest", acc_rf, sens_rf, spec_rf],
], columns=["Model","Accuracy","Sensitivity","Specificity"])

# Sort by Accuracy (descending)
res_sorted = results.sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n=== Sorted by Test Accuracy ===\n", res_sorted)

# Bar chart of Accuracy
plt.figure()
plt.bar(res_sorted["Model"], res_sorted["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Test Accuracy")
plt.title("Model Test Accuracy (Descending)")
plt.tight_layout()
plt.show()

# Scatter: Sensitivity vs Specificity
plt.figure()
plt.scatter(results["Sensitivity"], results["Specificity"])
for i, row in results.iterrows():
    plt.annotate(row["Model"], (row["Sensitivity"], row["Specificity"]), xytext=(3,3), textcoords="offset points")
plt.xlabel("Test Sensitivity (Recall for spam=1)")
plt.ylabel("Test Specificity (Recall for ham=0)")
plt.title("Sensitivity vs Specificity (All Models)")
plt.tight_layout()
plt.show()

# Brief discussion (print)
best_model = res_sorted.iloc[0,0]
print(f"\nBest-performing model by accuracy: {best_model}")
print("Trade-off: Higher sensitivity catches more spam, while higher specificity reduces false alarms for ham.")