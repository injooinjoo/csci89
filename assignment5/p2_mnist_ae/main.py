# Step 1: load MNIST
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

np.random.seed(42)
tf.random.set_seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

print("train:", X_train.shape, X_train.min(), X_train.max())
print("test :", X_test.shape,  X_test.min(),  X_test.max())

# Step 2: build & train
from tensorflow.keras import layers, models

latent_dim = 8
X_train_flat = X_train.reshape((-1, 28 * 28))
X_test_flat  = X_test.reshape((-1, 28 * 28))

# Encoder
encoder_inputs = layers.Input(shape=(28 * 28,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
code = layers.Dense(latent_dim, activation="linear", name="code")(x)

# Decoder
x = layers.Dense(128, activation="relu")(code)
x = layers.Dense(256, activation="relu")(x)
decoder_outputs = layers.Dense(28 * 28, activation="sigmoid")(x)

autoencoder = models.Model(encoder_inputs, decoder_outputs, name="mnist_dense_ae")
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

history = autoencoder.fit(
    X_train_flat, X_train_flat,
    validation_split=0.1,
    epochs=10,
    batch_size=256,
    verbose=1,
)


# Step 3: visualize
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("MNIST Dense AE (latent=16)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_latent16.png", dpi=150)

encoder = models.Model(autoencoder.input, autoencoder.get_layer("code").output)
code_input = layers.Input(shape=(latent_dim,))
decoder_layers = autoencoder.layers[-3:] 
d = code_input
for layer in decoder_layers:
    d = layer(d)
decoder = models.Model(code_input, d)

# Pick 5 samples
idx = np.random.choice(len(X_test_flat), size=5, replace=False)
x = X_test_flat[idx]
recon = autoencoder.predict(x, verbose=0).reshape(-1, 28, 28)

plt.figure(figsize=(10, 4))
for i in range(5):
    # originals
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx[i]], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Original", fontsize=10)
    # reconstructions
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(recon[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Reconstruct", fontsize=10)

plt.title("MNIST Dense AE (latent=8)")
plt.suptitle("Original vs Reconstruction (latent=8)", y=1.02, fontsize=12)
plt.savefig("loss_curve_latent8.png", dpi=150)
plt.savefig("recon_panel_latent8.png", dpi=150)