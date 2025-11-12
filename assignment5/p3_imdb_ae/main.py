# p3_imdb_ae/main.py
# Text Autoencoder with TEACHER FORCING (IMDB)
# - Encoder: Embedding -> LSTM -> 32-dim code (state)
# - Decoder: Embedding (shared) -> LSTM (initial_state=encoder_state)
#            -> TimeDistributed Dense over vocab
# - Decoder inputs are the gold tokens shifted with <bos>,
#   targets are the original tokens

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# 1) Load & preprocess
num_words = 8000           
maxlen = 150

(X_train, _), (X_test, _) = imdb.load_data(num_words=num_words)
X_train = pad_sequences(X_train, maxlen=maxlen, padding="post", truncating="post")
X_test  = pad_sequences(X_test,  maxlen=maxlen, padding="post", truncating="post")
print("train/test:", X_train.shape, X_test.shape)

# Build index -> word mapping for later detokenization
word_index = imdb.get_word_index()
index_to_word = {idx + 3: w for w, idx in word_index.items()}
index_to_word[0] = "<pad>"
index_to_word[1] = "<bos>"
index_to_word[2] = "<unk>"

def detokenize(seq):
    out = []
    for t in seq:
        if t == 0:
            break
        out.append(index_to_word.get(t, "<unk>"))
    return " ".join(out)


# 2) Teacher forcing I/O
def make_decoder_io(X):
    bos = np.full((X.shape[0], 1), 1, dtype=np.int32)  # <bos>=1
    dec_in = np.concatenate([bos, X[:, :-1]], axis=1)
    y = X.copy()
    return dec_in, y

# Small/fast split for development
n_train = 15000
n_val   = 2000
Xtr, Xva = X_train[:n_train], X_train[n_train:n_train+n_val]
DecIn_tr, Ytr = make_decoder_io(Xtr)
DecIn_va, Yva = make_decoder_io(Xva)


# 3) Model (shared embedding + seq2seq)
embed_dim  = 128
enc_units  = 128
dec_units  = 128
latent_dim = 64

# Shared embedding
emb = layers.Embedding(num_words, embed_dim, mask_zero=True, name="shared_emb")

# Encoder
enc_inputs = layers.Input(shape=(maxlen,), name="enc_inputs")
x = emb(enc_inputs)
enc_out, state_h, state_c = layers.LSTM(
    enc_units, return_state=True, name="enc_lstm",
    dropout=0.1, recurrent_dropout=0.2
)(x)
code = layers.Dense(latent_dim, activation="linear", name="code")(state_h)  # optional probe

# Decoder
dec_inputs = layers.Input(shape=(maxlen,), name="dec_inputs")
y = emb(dec_inputs)  # reuse same weights
y = layers.LSTM(
    dec_units, return_sequences=True, name="dec_lstm",
    dropout=0.1, recurrent_dropout=0.2
)(y, initial_state=[state_h, state_c])
logits = layers.TimeDistributed(layers.Dense(num_words, activation="softmax"), name="softmax")(y)

ae = models.Model([enc_inputs, dec_inputs], logits, name="imdb_text_ae_tf")
ae.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
ae.summary()

# 4) Train
epochs = 10
batch_size = 64

hist = ae.fit(
    [Xtr, DecIn_tr], Ytr,
    validation_data=([Xva, DecIn_va], Yva),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)

# 5) Reconstructions
np.random.seed(0)
idx = np.random.choice(len(Xva), size=2, replace=False)
enc_in = Xva[idx]
dec_in = DecIn_va[idx]
pred = ae.predict([enc_in, dec_in], verbose=0)
y_hat = np.argmax(pred, axis=-1)

print("\n=== Reconstruction samples (teacher forcing) ===")
for k in range(len(idx)):
    orig = detokenize(enc_in[k])
    rec  = detokenize(y_hat[k])
    print(f"\n[Sample {k}]")
    print("Original   :", orig[:300])
    print("Reconstruct:", rec[:300])


# 6) Loss plot
plt.figure()
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (sparse CCE)")
plt.title("IMDB Text AE with Teacher Forcing (latent≈64)")
plt.legend()
plt.tight_layout()
plt.savefig("imdb_text_ae_tf_loss_lat64.png", dpi=150)
print("[save] imdb_text_ae_tf_loss.png")
