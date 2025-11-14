import spacy

# load model
nlp = spacy.load("en_core_web_sm")

# same article text as Problem 1
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

doc = nlp(text)

entities = {}
for ent in doc.ents:
    entities.setdefault(ent.label_, []).append(ent.text)

for label, items in entities.items():
    print(f"\n{label}:")
    for ent in sorted(set(items)):
        print("  -", ent)
