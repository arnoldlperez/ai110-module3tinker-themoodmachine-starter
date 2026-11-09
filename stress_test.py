"""
Stress tests ("breaker" sentences) for the rule based Mood Machine.

Kept separate from dataset.py so these do not become ML training data.
Run:  python stress_test.py
"""

from mood_analyzer import MoodAnalyzer

# (category, text, expected label)
BREAKERS = [
    ("sarcasm", "I love getting stuck in traffic", "negative"),
    ("sarcasm", "Oh great, another group project", "negative"),
    ("slang", "That concert was sick", "positive"),
    ("slang", "This beat is fire", "positive"),
    ("slang", "That was wicked fun", "positive"),
    ("emoji", "I'm fine 🙂", "neutral"),
    ("emoji", "Got the internship 😭😭", "positive"),
    ("emoji", "Great, my phone died 🙄", "negative"),
    ("mixed", "I'm exhausted but proud of myself", "mixed"),
    ("mixed", "Love my friends, hate this weather", "mixed"),
    ("negation", "I don't hate it", "positive"),
    ("negation", "Never been this happy", "positive"),
    ("idiom", "I can't wait for the weekend!", "positive"),
]

# Same meaning written in two styles: (standard, informal/non-standard, expected)
STYLE_PAIRS = [
    ("I am very happy today", "im hella happy today fr", "positive"),
    ("That was really good", "dat was bussin", "positive"),
    ("I am so tired of this", "i'm so done wit this smh", "negative"),
    ("This is boring", "this is mad boring ngl", "negative"),
    ("I am excited about the trip", "lowkey hyped for the trip", "positive"),
    ("That was terrible", "that was trash tbh", "negative"),
]


def show(analyzer: MoodAnalyzer, text: str, expected: str) -> bool:
    pred = analyzer.predict_label(text)
    mark = "OK " if pred == expected else "XX "
    print(f'  {mark}"{text}" -> {pred} (expected {expected}) | {analyzer.explain(text)}')
    return pred == expected


if __name__ == "__main__":
    analyzer = MoodAnalyzer()

    print("=== Breakers ===")
    hits = sum(show(analyzer, t, e) for _, t, e in BREAKERS)
    print(f"Breakers correct: {hits}/{len(BREAKERS)}\n")

    print("=== Style pairs (standard vs. informal) ===")
    std_ok = inf_ok = 0
    for standard, informal, expected in STYLE_PAIRS:
        std_ok += show(analyzer, standard, expected)
        inf_ok += show(analyzer, informal, expected)
        print()
    n = len(STYLE_PAIRS)
    print(f"Standard correct: {std_ok}/{n}   Informal correct: {inf_ok}/{n}")
