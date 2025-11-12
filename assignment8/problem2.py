# problem2.py
# Neural Network for SMS spam classification (Problem 2)
# - Prints test Accuracy/Sensitivity/Specificity
# - Saves: problem2_nn_results.csv, combined_results.csv (if provided),
#          Figure_1_NN.png (accuracy bar), Figure_2_NN.png (sens-spec scatter)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

# Reproducibility (best effort)
import os, random
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

# Keras imports
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# Helpers
# ------------------------
def compute_metrics(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) else 0.0  # recall for spam=1
    spec = tn / (tn + fp) if (tn + fp) else 0.0  # recall for ham=0
    return acc, sens, spec, (tn, fp, fn, tp)

def balanced_class_weights(y):
    # class weight to reduce class imbalance impact
    n = len(y)
    pos = y.sum()
    neg = n - pos
    w0 = n / (2.0 * neg) if neg > 0 else 1.0
    w1 = n / (2.0 * pos) if pos > 0 else 1.0
    return {0: w0, 1: w1}

def pick_threshold_from_val(y_val, p_val):
    # choose threshold that maximizes accuracy on validation set (not test)
    cand = np.arange(0.3, 0.71, 0.01)
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc, _, _, _ = compute_metrics(y_val, (p_val >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t

def maybe_load_problem1_csv(csv_path):
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        need_cols = {"Model","Accuracy","Sensitivity","Specificity"}
        if not need_cols.issubset(set(df.columns)):
            print("[WARN] problem1 CSV missing required columns; will ignore.")
            return None
        return df
    return None

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Spam_SMS.csv", help="Path to Spam_SMS.csv")
    ap.add_argument("--problem1_csv", default=None, help="Optional CSV with P1 results")
    ap.add_argument("--max_features", type=int, default=5000, help="TF-IDF vocabulary size")
    ap.add_argument("--epochs", type=int, default=6, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = ap.parse_args()

    # 1) Load data
    df = pd.read_csv(args.data)[["Class","Message"]].dropna().copy()
    df["Label"] = df["Class"].map({"ham":0, "spam":1}).astype(int)

    # 2) Split (same as Problem 1)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Message"], df["Label"], test_size=0.2, random_state=42, stratify=df["Label"]
    )

    # 3) TF-IDF features for NN (word-level, stop words removed)
    tfidf = TfidfVectorizer(stop_words="english", max_features=args.max_features, ngram_range=(1,2))
    X_train = tfidf.fit_transform(X_train_text).toarray()
    X_test  = tfidf.transform(X_test_text).toarray()

    # 4) Build NN (compact MLP with dropout + early stopping)
    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 5) Train with validation split and class weights
    cw = balanced_class_weights(y_train.values)
    es = EarlyStopping(monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train.values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=[es],
        class_weight=cw,
        verbose=1
    )

    # 6) Choose threshold from validation (to avoid test leakage)
    #    We'll re-run one forward pass on the last validation split to get p_val.
    #    Keras doesn't expose val indices directly; so we reproduce the split.
    #    Note: uses same random_state as Keras default shuffle for deterministic split? Not guaranteed.
    #    Simpler robust approach: do a small manual split for threshold selection.
    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train, y_train.values, test_size=0.1, random_state=42, stratify=y_train.values
    )
    # Refit briefly on sub-train to align with X_val distribution (fast, few epochs)
    model_short = Sequential([
        Dense(256, activation="relu", input_shape=(X_sub_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model_short.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_short.fit(
        X_sub_train, y_sub_train,
        epochs=2, batch_size=args.batch_size, verbose=0
    )
    p_val = model_short.predict(X_val, verbose=0).ravel()
    thr = pick_threshold_from_val(y_val, p_val)
    # Final predictions from main model on test
    p_test = model.predict(X_test, verbose=0).ravel()
    y_pred = (p_test >= thr).astype(int)

    # 7) Metrics on test
    acc, sens, spec, cm = compute_metrics(y_test.values, y_pred)
    tn, fp, fn, tp = cm
    print(f"[NN] Threshold={thr:.2f}  Accuracy={acc:.6f}  Sensitivity={sens:.6f}  Specificity={spec:.6f}")
    print(f"[NN] Confusion matrix: [[{tn}, {fp}], [{fn}, {tp}]]")

    # 8) Save NN results
    nn_row = pd.DataFrame([["Neural Network", acc, sens, spec]],
                          columns=["Model","Accuracy","Sensitivity","Specificity"])
    nn_row.to_csv("problem2_nn_results.csv", index=False)

    # 9) Combine with Problem 1 (if provided) and plot
    p1 = maybe_load_problem1_csv(args.problem1_csv)
    if p1 is not None:
        combined = pd.concat([p1, nn_row], ignore_index=True)
    else:
        combined = nn_row.copy()

    combined.to_csv("combined_results.csv", index=False)

    # Figure 1: Accuracy bar (descending)
    combined_sorted = combined.sort_values("Accuracy", ascending=False)
    plt.figure()
    plt.bar(combined_sorted["Model"], combined_sorted["Accuracy"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title("Model Test Accuracy (Including Neural Network)")
    plt.tight_layout()
    plt.savefig("Figure_1_NN.png", dpi=180)

    # Figure 2: Sensitivity vs Specificity scatter
    plt.figure()
    plt.scatter(combined["Sensitivity"], combined["Specificity"])
    for _, r in combined.iterrows():
        plt.annotate(r["Model"], (r["Sensitivity"], r["Specificity"]),
                     xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.xlabel("Test Sensitivity (Recall for spam=1)")
    plt.ylabel("Test Specificity (Recall for ham=0)")
    plt.title("Sensitivity vs Specificity (All Models + Neural Network)")
    plt.tight_layout()
    plt.savefig("Figure_2_NN.png", dpi=180)

    print("\nSaved files:")
    print(" - problem2_nn_results.csv")
    print(" - combined_results.csv")
    print(" - Figure_1_NN.png")
    print(" - Figure_2_NN.png")

if __name__ == "__main__":
    main()
