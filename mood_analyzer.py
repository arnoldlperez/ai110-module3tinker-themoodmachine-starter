# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# One token is either a text emoticon (":)", ":-(", ";p"), a single emoji
# character, or a word made of letters/digits/apostrophes ("can't").
TOKEN_PATTERN = re.compile(
    r"[:;]-?[()dp]"                       # text emoticons
    r"|[\U0001F300-\U0001FAFF☀-➿]"  # common emoji ranges
    r"|[a-z0-9']+"                        # words
)

# 3+ repeats of the same character collapse to 2 ("sooo" -> "soo").
REPEATED_CHARS = re.compile(r"(.)\1{2,}")

# Words that flip the sentiment of a word shortly after them.
NEGATORS = {"not", "no", "never"}
NEGATION_WINDOW = 2  # how many tokens after a negator are affected

# Emoji / emoticon signals. Only emojis with a fairly consistent meaning are
# scored. Deliberately left out because usage is split: 🙂 (sincere or
# passive-aggressive), 😭 (sad or overjoyed), 💀 (dread or laughing),
# 🥲 (sad or touched), 😂 (amused or mocking).
# Emojis are not affected by negation.
EMOJI_SCORES = {
    ":)": 1, ":d": 1, "😊": 1, "😄": 1, "😁": 1, "❤": 1, "🥰": 1, "🎉": 1,
    ":(": -1, "😞": -1, "😢": -1, "😡": -1, "😠": -1, "🙄": -1, "🙃": -1,
}

# Multi-word phrases whose meaning differs from their individual words.
# Matched before single words; their tokens are then skipped, and negation
# does not apply to them. Example: "passed away" (grief) vs. "passed my exam".
PHRASES = {
    ("passed", "away"): -1,
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Lowercase and normalize curly apostrophes (’ -> ')
          - Collapse 3+ repeated characters ("soooo" -> "soo")
          - Extract emoticons, individual emojis, and words; everything
            else (punctuation, extra spaces) is dropped
        """
        cleaned = text.strip().lower().replace("’", "'")
        cleaned = REPEATED_CHARS.sub(r"\1\1", cleaned)

        tokens = TOKEN_PATTERN.findall(cleaned)
        # Drop stray apostrophes left from quotes like 'hello'.
        tokens = [t.strip("'") for t in tokens]
        return [t for t in tokens if t]

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_negator(token: str) -> bool:
        return token in NEGATORS or token.endswith("n't")

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Shared scoring logic used by score_text() and explain().

        Returns (score, positive_hits, negative_hits). A sentiment word that
        appears within NEGATION_WINDOW tokens after a negator has its sign
        flipped, e.g. "not happy" counts as negative, "not bad" as positive.
        Negated hits are recorded with the negator, e.g. "not happy".
        PHRASES are matched first and score as a unit (no negation).
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        i = 0
        while i < len(tokens):
            phrase = next((p for p in PHRASES if tuple(tokens[i:i + len(p)]) == p), None)
            if phrase:
                polarity = PHRASES[phrase]
                score += polarity
                (positive_hits if polarity > 0 else negative_hits).append(" ".join(phrase))
                i += len(phrase)
                continue

            pos, token = i, tokens[i]
            i += 1
            if token in EMOJI_SCORES:
                polarity = EMOJI_SCORES[token]
                score += polarity
                (positive_hits if polarity > 0 else negative_hits).append(token)
                continue

            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1
            else:
                continue

            window = tokens[max(0, pos - NEGATION_WINDOW):pos]
            negator = next((t for t in reversed(window) if self._is_negator(t)), None)
            if negator:
                polarity = -polarity
                token = f"{negator} {token}"

            score += polarity
            (positive_hits if polarity > 0 else negative_hits).append(token)

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words add 1, negative words subtract 1, and a negator
        (not / no / never / *n't) shortly before a word flips its sign.
        Emojis in EMOJI_SCORES add their score directly (no negation).
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the score and matched words into a mood label:
          - both positive and negative signals that cancel out -> "mixed"
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 with no signals -> "neutral"
        """
        score, positive_hits, negative_hits = self._analyze(text)
        if score == 0 and positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label,
        e.g. 'Score = -1 (positive: [], negative: ['not happy'])'.
        """
        score, positive_hits, negative_hits = self._analyze(text)
        return (
            f"Score = {score} "
            f"(positive: {positive_hits}, "
            f"negative: {negative_hits})"
        )
