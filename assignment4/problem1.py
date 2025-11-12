# Problem 1 — Tokenization, Lemmatization, TF-IDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt'); nltk.download('wordnet')

# ------------------------------------------------------------
# Step 1: Prepare texts
# ------------------------------------------------------------
large_text = """No survivors found after Tennessee explosives plant blast
Kwasi Gyamfi Asiedu & Blanca EstradaBucksnort, Tennessee
EPA/Shutterstock Police cars parked out front of the plantEPA/Shutterstock
Sixteen people missing after a major explosion at a Tennessee munitions factory on Friday are presumed to be dead, according to the sheriff.
Recovery teams had been clinging to hope of finding any of the missing alive, as of Saturday evening "it's safe" to assume they are deceased, said Humphreys County Sheriff Chris Davis.
Authorities had originally feared that 18 people had died, but two people believed present during the blast were actually not at the site and have been located, according to Davis.
It's still unclear what caused the explosion at the explosives plant in Bucksnort, Tennessee - roughly 56 miles (90km) south-west of Nashville.
Video footage taken on Friday showed fires still burning, charred vehicles, and smoke rising from the razed building at the facility, which specialises in the development and manufacture of explosives. Officials said debris was scattered for half a mile around where the building once stood.
Accurate Energetic Systems (AES), which runs the plant, has suspended its operations.
Aerial footage shows devastation after blast at explosives manufacturer in Tennessee
More than 300 state and local first responders have been searching the site since Friday morning, Sheriff Davis said on Saturday.
"The expectation of anyone who's inside of that building… we can assume that they are deceased," he told media.
"As we get into this, we find it even more devastating than we thought initially," he told a news conference.
By Saturday morning, the rescue mission had shifted to a recovery operation, said a visibly choked-up Davis.
The FBI is also at the scene conducting rapid DNA tests to identify victims and notify families.
"We're trying to focus as much attention as we can, on taking care of their families," Sheriff Jason Craft, from neighbouring Hickman County, told the BBC.
The Bureau of Alcohol, Tobacco, and Firearms is at the scene helping to investigate the incident.
There was a previous fatal explosion at a unit in the same location in 2014.
Ann Myers was woken from her sleep by the explosion on Friday morning and feared for the safety of her four children. She wondered if it was a tornado or a road accident.
The camper home they live in shook and the electricity went out briefly.
"It was the weirdest thing ever," she said. "So then I'm like, are we in a tornado? Is there, you know, a bomb somewhere?
"But then I thought maybe a tractor trailer, you know, hit somewhere off the interstate. So it's definitely very frightening."
Justin Stover, whose property borders the sprawling AES plant, told the BBC his house shook violently following the explosion.
At first, he thought it was a plane crash and he feared his house would collapse.
"Things fell off the wall, items fell off shelves," Mr Stover said. "It was very intimidating, like the loudest thunder you've ever heard in your life and rumble."
Then he saw "a large cloud of smoke coming from the area of AES", Stover added.
He is still assessing damage to his house and said the explosion may have affected his water well.
Stover, who has lived in Bucksnort for 20 years, said AES had about 80 workers.
"It's one of the only businesses in this area," he said. "So it's one of the only places for employment, for locals.
"There's a lot of people that we know that work there and that possibly lost their lives yesterday morning."
The tragedy will be devastating for the whole community, he said, as it has already been for families of the victims.
Residents farther away from the site also heard the blast, including some in a town about 15 miles (25km) away, Sheriff Davis said on Friday.
One local resident who lives about 20 minutes away from the facility told the BBC she was sitting at her daughter's dining table when she heard it.
"All of a sudden we just a heard a loud bang. We didn't know if it was a gun or what," she said.
Another local resident, Lucy Garton said her husband knows people who worked in the facility.
"I think it will definitely impact the area," she said. "It's a very close-knit community and everyone, they're just simple people, go to work every day, take care of their families and just real a family-oriented community."""

small_text = """Local volunteers used drones and blockchain-based sensors
to help rescue teams track survivors after the Bucksnort explosion.
Rumors spread across the metaverse bringing global attention to the tragedy."""

# 2) Preprocess (tokenize + lemmatize)
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
    return " ".join(lemmas)

processed_large = preprocess(large_text)
processed_small = preprocess(small_text)

# 3) Fit TF-IDF on large text, then transform small text
vectorizer = TfidfVectorizer()
X_large = vectorizer.fit_transform([processed_large])
X_small = vectorizer.transform([processed_small])

# 4) Inspect transferred terms and OOV
feature_names = vectorizer.get_feature_names_out()
small_vec = X_small.toarray().ravel()
nz_idx = np.where(small_vec > 0)[0]
print("Non-zero TF-IDF terms in small text:")
for i in nz_idx:
    print(feature_names[i], f"{small_vec[i]:.4f}")

small_tokens = processed_small.split()
oov = sorted(set(t for t in small_tokens if t not in set(feature_names)))
print("\nOOV tokens:", oov)