import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

import matplotlib.pyplot as plt
import spacy

tf.random.set_seed(42)
np.random.seed(42)

# data
df = pd.read_csv("news_data.csv")

df["text"] = (
    df["news_headline"].fillna("") + " " +
    df["news_article"].fillna("")
).str.strip()

le = LabelEncoder()
y = le.fit_transform(df["news_category"])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# tf-idf
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

num_classes = len(le.classes_)


def build_mlp(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# model 1: tf-idf only
model1 = build_mlp(X_train_tfidf.shape[1], num_classes)

es = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

history1 = model1.fit(
    X_train_tfidf,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    callbacks=[es],
    verbose=1
)

test_loss1, test_acc1 = model1.evaluate(X_test_tfidf, y_test, verbose=0)
print("[Model 1] TF-IDF only - Test accuracy:", round(test_acc1, 4))
print("[Random baseline]:", round(1.0 / num_classes, 4))

plt.figure()
plt.plot(history1.history["accuracy"], label="train")
plt.plot(history1.history["val_accuracy"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model 1: TF-IDF only")
plt.tight_layout()
plt.savefig("problem3_model1_accuracy.png")
plt.close()

# NER features (entity type presence)
nlp = spacy.load("en_core_web_sm")
entity_labels = ["PERSON", "ORG", "GPE", "NORP", "EVENT", "PRODUCT", "WORK_OF_ART"]


def build_ner_matrix(texts):
    m = np.zeros((len(texts), len(entity_labels)), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=16)):
        labels = {ent.label_ for ent in doc.ents}
        for j, lab in enumerate(entity_labels):
            if lab in labels:
                m[i, j] = 1.0
    return m


X_train_ner = build_ner_matrix(X_train_text.tolist())
X_test_ner = build_ner_matrix(X_test_text.tolist())

X_train_full = np.hstack([X_train_tfidf, X_train_ner])
X_test_full = np.hstack([X_test_tfidf, X_test_ner])

# model 2: tf-idf + NER
model2 = build_mlp(X_train_full.shape[1], num_classes)

history2 = model2.fit(
    X_train_full,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    callbacks=[es],
    verbose=1
)

test_loss2, test_acc2 = model2.evaluate(X_test_full, y_test, verbose=0)
print("[Model 2] TF-IDF + NER - Test accuracy:", round(test_acc2, 4))

plt.figure()
plt.plot(history2.history["accuracy"], label="train")
plt.plot(history2.history["val_accuracy"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model 2: TF-IDF + NER")
plt.tight_layout()
plt.savefig("problem3_model2_accuracy.png")
plt.close()
