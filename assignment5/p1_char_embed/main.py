import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

np.random.seed(42)
tf.random.set_seed(42)

#1 texts
large_text = (
    "'I wash my face and there's a burning sensation'\n"
    "By Orla Guerin\n"
    "BBC News, Basra\n\n"
    "Published 28 September 2018\n\n"
    "The southern Iraqi city of Basra is a place of protest and of pain. Tens of thousands of its people have been poisoned by polluted water. This is a city on the frontline of a global water crisis.\n\n"
    "The children of Basra are paying the price for the failures of the politicians. They are filling the hospital beds in a city let down by those who lead it.\n\n"
    "At the general hospital, a small boy with an Angry Birds t-shirt - five-year-old Ali - was limp in his mother's arms. His small body was battling a nasty rash, and a fever. His mother, Fatima, fanned him in the stifling heat, her face a mask of worry. \"He has been sick for a week,\" she said. \"He has diarrhoea and he has been vomiting.\"\n\n"
    "She is one of tens of thousands of mothers in Basra who have had to watch their children suffer after drinking the contaminated water that flows from the city's taps. More than 118,000 people have been hospitalised in recent months, according to the Iraqi High Commission for Human Rights.\n\n"
    "The city's main water source, the Shatt al-Arab waterway, is increasingly polluted by sewage, agricultural and industrial waste. Low rainfall has made the problem worse. The authorities have failed to provide a safe alternative.\n\n"
    "The crisis has sparked angry and sometimes violent protests. The demonstrators have torched government buildings and the offices of political parties and militias. They are demanding clean water, reliable electricity, and an end to corruption. They say the country's oil wealth has been squandered by a political elite that has failed to deliver basic services.\n\n"
    "\"When I wash my face there's a burning sensation,\" said Hassan, a 25-year-old protester with a black and white scarf wrapped around his face. \"The water is salty and dirty. It's not fit for animals to drink, let alone humans.\"\n\n"
    "He said he had been protesting for weeks. \"We are asking for our basic rights,\" he said. \"We want to live like human beings. We are not asking for palaces and villas. We are asking for water and electricity.\"\n\n"
    "The anger in Basra is a warning sign for the rest of Iraq, and for the world. The country is one of the most water-stressed in the world. The United Nations says that by 2025, water scarcity could be the cause of the next war in the Middle East.\n\n"
    "For now, the people of Basra are fighting their own battle, for their health and their dignity. \"We are drinking poison,\" said Fatima, as she held her son close. \"Our children are dying. What have they done to deserve this?\"\n\n"
    "The hospital is a chaotic and crowded place. The corridors are filled with patients and their families. Many are lying on the floor. There are not enough beds to go round. The doctors and nurses are overwhelmed.\n\n"
    "\"We are seeing cases of cholera, typhoid, and dysentery,\" said Dr. Riyad Abdel Amir, the head of the Basra health directorate. \"The situation is catastrophic.\" He said the hospitals were struggling to cope with the influx of patients. \"We are running out of medicine,\" he said. \"We are running out of everything.\"\n\n"
    "The water crisis has exposed the deep-seated problems in Iraq. The country is still reeling from years of war and conflict. The infrastructure is crumbling. The political system is paralysed by corruption and sectarianism.\n\n"
    "The people of Basra feel abandoned by their leaders. They say they have been promised much, but have received little. \"The politicians are thieves,\" said Hassan, the protester. \"They have stolen our money and our future.\"\n\n"
    "The anger in Basra is not just about water. It is about a sense of injustice and betrayal. It is about a generation of young people who have grown up with war and violence, and who now see no hope for the future.\n\n"
    "\"We have no jobs, no services, no life,\" said Hassan. \"We are living in a big prison.\" He said he was not afraid to die. \"We have nothing to lose,\" he said. \"We will continue to protest until we get our rights.\"\n\n"
    "The protests in Basra are a cry for help. They are a demand for a better life. They are a warning that if the politicians do not listen, the anger on the streets could boil over into something much more dangerous.\n"
)

small_text = (
    "Basra’s water still burns my skin — it feels like acid.\n"
    "People say the new filters cost €50, but we have no jobs.\n"
    "Sometimes I just smile and say “we’re fine” 🙂 though everyone knows we’re not.\n"
)

#2 vocabulary from large text
large_lc = large_text.lower()
chars = sorted(list(set(large_lc)))
char_to_int = {c: i + 1 for i, c in enumerate(chars)}
int_to_char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(char_to_int) + 1 

print(f"[info] vocab size (including OOV=0): {vocab_size}")

#3 Encode texts (OOV -> 0)
def encode_text(s: str, mapping: dict, oov_idx: int = 0):
    return [mapping.get(c, oov_idx) for c in s.lower()]

encoded_large = encode_text(large_text, char_to_int, 0)
encoded_small = encode_text(small_text, char_to_int, 0)

oov_count_small = sum(1 for i in encoded_small if i == 0)
print(f"[info] small_text length: {len(encoded_small)} | OOV chars: {oov_count_small}")

#4 Prepare for prediction
seq_len = 80
step = 4
sequences = []
targets = []

for i in range(0, len(encoded_large) - seq_len - 1, step):
    seq = encoded_large[i : i + seq_len]
    tgt = encoded_large[i + seq_len]  # predict next char
    sequences.append(seq)
    targets.append(tgt)

X = np.array(sequences, dtype=np.int32)
y = np.array(targets, dtype=np.int32)

print(f"[info] training samples: {X.shape[0]} | seq_len: {seq_len}")

#5 Small model: Embedding + LSTM + Dense over vocab
embedding_dim = 32
model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(64),
        Dense(vocab_size, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()


#6 Train briefly
history = model.fit(
    X,
    y,
    batch_size=512,
    epochs=5,
    validation_split=0.1,
    verbose=1,
)

# 7) Inspect embeddings for small_text
emb_layer = model.layers[0]
emb_matrix = emb_layer.get_weights()[0]

# Pick a few characters, including known + OOV cases
probe_chars = ["b", "a", " ", ".", "€", "—", "🙂"]  # some in-vocab, some OOV
print("\n[embeddings] sample rows:")
for ch in probe_chars:
    idx = char_to_int.get(ch.lower(), 0)
    vec = emb_matrix[idx]
    print(f"char={repr(ch):>4} | idx={idx:>3} | first5={np.round(vec[:5], 4)}")


print("\n[sample small_text indices]")
print(encoded_small[:120])

print("\n[notes]")
print("- OOV characters map to index 0.")
print("- Index 0 has its own embedding row; it represents any unseen character.")
print("- Known characters reuse the same learned weights when applied to small_text.")
