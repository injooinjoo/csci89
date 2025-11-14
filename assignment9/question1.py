# problem1

import re
import nltk

# Download 
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


# 1. Data
text = """
US to convene global AI safety summit in November
By David Shepardson
September 19, 2024 1:10 AM GMT+9 Updated September 19, 2024

AI for Good Global summit
Visitors take pictures of Captcha, a robot by Hidoba Research, during the AI for Good Global summit on artificial intelligence, organised by the International Telecommunication Union (ITU), in Geneva, Switzerland, May 30, 2024. REUTERS/Denis Balibouse/File photo Purchase Licensing Rights, opens new tab

WASHINGTON, Sept 18 (Reuters) - The Biden administration plans to convene a global safety summit on artificial intelligence, it said on Wednesday, as Congress continues to struggle with regulating the technology.
Commerce Secretary Gina Raimondo and Secretary of State Anthony Blinken will host on Nov. 20-21 the first meeting of the International Network of AI Safety Institutes in San Francisco to "advance global cooperation toward the safe, secure, and trustworthy development of artificial intelligence."

The Reuters Daily Briefing newsletter provides all the news you need to start your day. Sign up here.

The network members include Australia, Canada, the European Union, France, Japan, Kenya, South Korea, Singapore, Britain, and the United States.
Generative AI - which can create text, photos and videos in response to open-ended prompts - has spurred excitement as well as fears it could make some jobs obsolete, upend elections and potentially overpower humans and have catastrophic effects.

Raimondo in May announced the launch of the International Network of AI Safety Institutes during the AI Seoul Summit in May, where nations agreed to prioritize AI safety, innovation and inclusivity. The goal of the San Francisco meeting is to jumpstart technical collaboration before the AI Action Summit in Paris in February.

Raimondo said the aim is "close, thoughtful coordination with our allies and like-minded partners."
"We want the rules of the road on AI to be underpinned by safety, security, and trust," she added.

The San Francisco meeting will include technical experts from each member’s AI safety institute, or equivalent government-backed scientific office, to discuss priority work areas, and advance global collaboration and knowledge sharing on AI safety.

Last week, the Commerce Department said it was proposing to require detailed reporting requirements for advanced AI developers and cloud computing providers to ensure the technologies are safe and can withstand cyberattacks.
The regulatory push comes as legislative action in Congress on AI has stalled.

President Joe Biden in October 2023 signed an executive order requiring developers of AI systems posing risks to U.S. national security, the economy, public health or safety to share the results of safety tests with the U.S. government before they are publicly released.

How a Hungarian company aims to replace road stone with trash

Reporting by David Shepardson; editing by Miral Fahmy
Our Standards: The Thomson Reuters Trust Principles.
"""


# 2. Regex patterns

# Person names with titles
person_pattern = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
)

# Organization names
org_pattern = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
    r'(Inc|Ltd|Corp|Corporation|Company)\b'
)

# Expanded date pattern:
month_names = (
    "Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
    "Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December"
)

date_pattern = re.compile(
    rf'\b('
    rf'\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}'                 
    rf'|'
    rf'\d{{4}}-\d{{1,2}}-\d{{1,2}}'                         
    rf'|'
    rf'(?:{month_names})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}'   
    rf'|'
    rf'\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names}),?\s+\d{{4}}'  
    rf'|'
    rf'(?:{month_names})\s+\d{{4}}'      
    rf')\b'
)

# Time pattern
time_pattern = re.compile(
    r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(AM|PM|am|pm)?\b'
)

# Email pattern
email_pattern = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b'
)


# 3. NNP/NNPS-based location extraction 
def extract_locations_with_pos(text_str):
    locations = set()

    sentences = sent_tokenize(text_str)
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)

        current_seq = []
        for word, tag in tagged:
            if tag in ("NNP", "NNPS"):
                current_seq.append(word)
            else:
                if len(current_seq) > 0:
                    loc = " ".join(current_seq)
                    locations.add(loc)
                    current_seq = []
        if len(current_seq) > 0:
            loc = " ".join(current_seq)
            locations.add(loc)

    return locations


# 4. Main extraction
def extract_entities(text_str):
    
    persons = set(m.group(0) for m in person_pattern.finditer(text_str))
    orgs    = set(m.group(0) for m in org_pattern.finditer(text_str))
    dates   = set(m.group(0) for m in date_pattern.finditer(text_str))
    times   = set(m.group(0) for m in time_pattern.finditer(text_str))
    emails  = set(m.group(0) for m in email_pattern.finditer(text_str))

    locs = extract_locations_with_pos(text_str)

    # Remove overlaps
    locs = {l for l in locs if (l not in persons and l not in orgs)}

    result = {
        "PERSON": sorted(persons),
        "ORG": sorted(orgs),
        "LOC": sorted(locs),
        "DATE": sorted(dates),
        "TIME": sorted(times),
        "EMAIL": sorted(emails),
    }
    return result

if __name__ == "__main__":
    entities = extract_entities(text)

    for label, items in entities.items():
        print(f"\n{label}:")
        for ent in items:
            print("  -", ent)
