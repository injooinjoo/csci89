# Step 1 — Extract and preprocess text
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
with open("assignment4\The Complete Works of William Shakespeare.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "lxml")
    text = soup.get_text()

tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
sentences = [tokens[i:i+20] for i in range(0, len(tokens), 20)]

print("Number of tokens:", len(tokens))
print("Example tokens:", tokens[:30])


# Step 2 — Train Word2Vec
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    workers=4
)

print("Vocabulary size:", len(model.wv))

targets = ["king", "queen", "love", "death"]
for w in targets:
    if w in model.wv:
        print(f"\nTop-5 similar to '{w}':")
        for word, score in model.wv.most_similar(w, topn=5):
            print(f"  {word:12s}  {score:.3f}")
    else:
        print(f"\n'{w}' not in vocabulary.")

if all(w in model.wv for w in ["king", "queen", "boy"]):
    ans = model.wv.most_similar(positive=["boy","queen"], negative=["king"], topn=1)[0]
    print("\nAnalogy (king : queen = boy : ?):", ans)
else:
    print("\nAnalogy cannot be computed (missing one of: 'king','queen','boy').")
