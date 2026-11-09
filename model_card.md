# Model Card: Mood Machine

This model card covers **two** versions of a mood classifier built on the same data:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

All numbers below come from running `python main.py`, `python ml_experiments.py`, and `python stress_test.py` on the final code and data.

## 1. Model Overview

**Model type:**
I compared both models. The rule based model is the one I designed and improved; the ML model was trained on the same posts and used for comparison.

**Intended purpose:**
Classify short, informal text posts (one sentence, social-media style) as **positive**, **negative**, **neutral**, or **mixed**. It is a learning tool for seeing how data, rules, and model choices shape predictions. It is **not** intended for real decisions about people.

**How it works (brief):**

- **Rule based:** the text is cleaned and split into tokens, each token is checked against word lists, an emoji table, and a phrase table, and the matches are added into a score. The score (plus which kinds of words matched) is mapped to a label. `explain()` prints the score and the exact words that counted, so every prediction can be traced.
- **ML:** `CountVectorizer` turns each post into word counts (bag of words), and `LogisticRegression` learns how strongly each word is associated with each label from the labeled examples. It predicts the label whose learned word weights best fit a new post.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **20 posts** with **20 matching labels** in `TRUE_LABELS` (lengths verified before every evaluation). The starter had 6 posts. I added 8 in Part 1 (slang, emojis, sarcasm, mixed feelings) and 6 more in Part 4 to test specific ML weaknesses. Separately, `stress_test.py` holds 25 "breaker" sentences (13 breakers + 6 standard/informal style pairs) that are **not** used for training.

**Labeling process:**
I chose and checked the labels myself; the AI assistant suggested candidate posts and draft labels, and I picked which to keep. Posts that were hard to label, where reasonable people could disagree:

| Post | Label | Why it is debatable |
|---|---|---|
| "This is fine" | neutral | Literally neutral, but the "this is fine" meme reads as resigned/negative |
| "Just got my exam grade back 💀💀" | negative | 💀 can mean "I'm dead (bad)" or "dying of laughter" |
| "Class got cancelled lol" | neutral | Could be happy (free time) or annoyed |
| "my dog passed away today, thank you all for the love 🥲" | mixed | Grief plus gratitude; some would say negative |
| "Feeling tired but kind of hopeful" | mixed | Some would say positive since it ends hopeful |
| "Honestly kind of proud of how my project turned out" | positive | "kind of" hedges it toward mixed |
| "Not sure how I feel about the new update" | neutral | Could be mixed |

**Important characteristics of the dataset:**

- Slang: "lowkey", "ugh", "lol"
- Emojis: 🙄 🙃 💀 🥲
- Sarcasm: "I absolutely love getting stuck in traffic 🙄", "can't wait for finals week 🙃"
- Negation: "I am not happy about this", "Not bad at all, actually"
- Mixed feelings: 3 posts labeled mixed
- Very short posts: "ugh mondays"

**Possible issues with the dataset:**

- **Tiny:** 20 posts across 4 labels; most words appear only once.
- **Imbalanced:** negative 7, positive 6, neutral 4, mixed 3.
- **Words seen in only one context:** "happy" first appeared only in "not happy", so the ML model learned "happy" as a negative cue.
- **One perspective:** all labels reflect one labeler's reading of tone, slang, and emojis.
- **Missing language:** no non-English, no regional dialect beyond a few slang words, no long posts.

## 3. How the Rule Based Model Works

**Scoring rules:**

1. **Preprocessing** (`preprocess`): lowercase; normalize curly apostrophes; shrink 3+ repeated letters ("sooo" → "soo"); keep text emoticons like `:)` and split each emoji into its own token ("💀💀" → `💀`, `💀`); drop other punctuation while keeping words like `can't`.
2. **Phrases** (`PHRASES`): multi-word phrases are matched first and scored as a unit. Currently only `"passed away"` = −1 (added in Part 4 so grief is recognized without making "I passed my exam" negative).
3. **Emojis** (`EMOJI_SCORES`, added in Part 3): e.g. `🙄 🙃 :( 😞` = −1, `:) 😊 ❤ 🎉` = +1. **Deliberately unscored** because usage is split: 🙂, 😭, 💀, 🥲, 😂.
4. **Words:** each word in `POSITIVE_WORDS` adds +1, each in `NEGATIVE_WORDS` subtracts 1.
5. **Negation** (Part 2 enhancement): if `not`, `no`, `never`, or any `*n't` word appears within 2 tokens before a sentiment word, its sign flips ("not happy" → −1, "not bad" → +1).
6. **Label mapping** (`predict_label`):
   - positive and negative signals both present and the score is 0 → **mixed**
   - score > 0 → **positive**
   - score < 0 → **negative**
   - score 0 with no signals → **neutral**

**Strengths:**

- **Transparent:** "I am not happy about this" → `Score = -1 (negative: ['not happy'])`. Every decision can be traced.
- **Predictable on clear sentiment words:** "Today was a terrible day" → negative, "So happy my sister is visiting this weekend" → positive.
- **Negation works for simple cases:** "Not bad at all, actually" → positive, "I don't hate it" → positive.
- **Stable when data changes:** adding posts did not change any existing prediction.

**Weaknesses:**

- **Missing words fall back to neutral:** "Finally passed my driving test!!" and "I'm so done with this week" score 0.
- **Sarcasm without an emoji:** "Oh great, another group project" → positive, because `great` is the only signal.
- **Mixed posts lose their positive half** when that word is not in the list: "Lowkey stressed but kind of proud of myself" → negative (`proud` unknown).
- **Some correct answers are accidental:**
  - "Never been this happy" is right only because `never` is 3 tokens before `happy`, outside the negation window.
  - "I'm fine 🙂" is right only because nothing matched.

## 4. How the ML Model Works

**Features used:**
Bag of words using `CountVectorizer` with default settings. With those defaults, **emojis and one-letter words are dropped entirely**, so 🙄, 🙃, 💀 and "i" never reach the model.

**Training data:**
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` (20 posts).

**Training behavior:**

- Training accuracy is **1.00** both before and after I added posts: it memorizes its training set.
- Held-out accuracy, measured with leave-one-out (train on 19 posts, predict the 20th), **dropped from 0.43 (at 14 posts) to 0.25 (at 20 posts)**.
- Adding six posts flipped held-out predictions on posts unrelated to them, from correct to wrong:
  - "Lowkey stressed but kind of proud of myself": mixed → positive
  - "Just got my exam grade back 💀💀": negative → positive

**Strengths and weaknesses:**

- *Strength:* it picks up multi-word context the rules cannot. "I love getting stuck in traffic" → negative, because `stuck`, `traffic`, and `getting` appeared in a negative training post.
- *Weakness, spurious cues* (verified by inspecting the learned weights):
  - `so` became a strong positive cue (+0.65) from "so much", "So excited", and "So happy". As a result, "I'm so done with this week" (held out) and "I am so tired of this" are predicted **positive**.
  - `kind` became a mixed cue.
- *Weakness, class imbalance:* the negative class has the largest bias term (+0.89), so unfamiliar posts default to **negative**. "I am very happy today" → negative and "I'm fine 🙂" → negative.

## 5. Evaluation

**How I evaluated:**

- Accuracy on the 20 labeled posts.
- Leave-one-out for the ML model, since training accuracy is misleading.
- The 25 held-out stress-test sentences.
- Reading every mismatch with `explain()` or the learned weights.

| Measure | Rule based | ML |
|---|---|---|
| Accuracy on the 20 labeled posts | **0.60** (12/20) | **1.00** (training accuracy, i.e. memorization) |
| Leave-one-out on the 20 posts | n/a (not trained) | **0.25** (5/20) |
| 13 breaker sentences | 5/13 | 4/13 |
| 6 standard-English style posts | 6/6 | 2/6 |
| 6 informal/slang style posts | 2/6 | 3/6 |
| All 25 stress sentences | **13/25** | **9/25** |

How the rule based accuracy changed over the parts: 0.00 (starter) → 0.50 (negation + preprocessing) → 0.57 (emoji scores) → 0.64 (`passed away` phrase) → 0.60 after adding 6 harder posts.

**Examples of correct predictions:**

- **"I am not happy about this" → negative.** Negation turns `happy` into `not happy` (−1). Before negation it would have been positive.
- **"can't wait for finals week 🙃" → negative.** This was originally neutral (no signals). After Part 3, 🙃 scores −1.
- **"my dog passed away today, thank you all for the love 🥲" → mixed.** This was originally positive (`love` was the only signal). After Part 4, the `passed away` phrase (−1) balances `love` (+1).

**Examples of incorrect predictions:**

- **"I absolutely love getting stuck in traffic 🙄" → mixed (true: negative).** `love` (+1) and 🙄 (−1) cancel out. The rules can't tell that the emoji is reversing the meaning of `love`; they just add the two. Before the emoji fix, this was confidently positive.
- **"Just got my exam grade back 💀💀" → neutral (true: negative).** 💀 is intentionally unscored because it also means laughing. The ML model got this wrong as well (positive in leave-one-out), but because of word overlap, not the emoji, which it never sees.
- **"I'm so done with this week" → rule: neutral, ML: positive (true: negative).**
  - The rules miss it because `done` isn't in the list.
  - The ML model gets it wrong for a different reason: the spurious `so` cue.
  - This shows the two models failing differently: the rules fail **silently to neutral**, while the ML model fails **confidently in the wrong direction**.

## 6. Limitations

- **Sarcasm needs context the rules don't have.** "Oh great, another group project" → positive. Sarcasm is caught only when a negative emoji is present, and even then only as far as mixed.
- **Vocabulary is closed.** Any sentiment word not in the lists scores 0:
  - "proud", "hopeful", "exhausted", "ugh", "done", "trash", "fire", "bussin"
  - This is why 5 of the 8 rule based misses on the dataset are predicted neutral.
- **Negation is word-level and limited to a 2-token window.**
  - Idioms break it: "I can't wait for the weekend!" scores 0 today. If `wait` were ever added to a list, the sincere version would become negative.
  - "Never been this happy" is right only by accident.
- **Phrase handling covers one phrase.** "I lost my grandma today" → neutral, because grief can be expressed in endless ways a list cannot cover.
- **Emoji meaning is assumed, not learned.** I scored 🙃 as −1 to match my label for the finals post. As a result, "lol ok 🙃" (possibly playful) → negative.
- **The ML model is unreliable at this data size.**
  - 1.00 training accuracy vs 0.25 held-out.
  - It learns spurious cues (`so`, `kind`) and defaults to the majority class.
- **There is no real test set.** The labeled posts double as the ML training data, so `ml_experiments.py` reports training accuracy only.

## 7. Ethical Considerations

**Overt bias: errors cluster on informal language.**
In the style-pair test, the rule based model got **6/6 standard-English posts right but only 2/6 informal ones** with the same meaning:

| Standard | Informal | Rule-based result for the informal post |
|---|---|---|
| "That was really good" → positive | "dat was bussin" | neutral (`bussin` unknown) |
| "I am excited about the trip" → positive | "lowkey hyped for the trip" | neutral |
| "That was terrible" → negative | "that was trash tbh" | neutral |
| "I am so tired of this" → negative | "i'm so done wit this smh" | neutral |

The two informal posts that passed ("im hella happy today fr", "this is mad boring ngl") passed only because they still contained a standard word.

This model **underserves people who write in slang-heavy or dialectal English**, including younger users and speakers of African American Vernacular English (e.g. "bussin"). This is not because their posts are less emotional. It is because the word lists were written in standard English. Their feelings are systematically read as "neutral", which in a real application could mean their distress or excitement goes unnoticed. I did not fix this; adding slang would help but has its own risk: "sick" would turn "I feel sick" positive.

The ML model did not show the same gap (2/6 standard vs 3/6 informal), but that is not evidence of fairness. It predicted **negative for 11 of the 12** style sentences, so its informal "wins" are just the three posts that happen to be negative.

**Other real-world risks:**

- **Missing distress:** a post like "I lost my grandma today" is predicted neutral. A system that triaged messages by mood could overlook someone who needs support.
- **Misreading emojis:** emoji meaning varies by age and community (💀, 😭). My scores encode one person's reading of them.
- **Privacy:** analyzing the mood of personal messages without consent is invasive, even when the model is accurate, and especially when it isn't.

**Covert compliance: did the AI assistant push back?**
I used Claude Code throughout. Where it did raise concerns:

- It flagged on its own that ML training accuracy (1.00) is misleading and ran a leave-one-out check.
- It pointed out the class imbalance and the ambiguous labels ("This is fine", 💀, "Class got cancelled lol").
- It warned that the "can't wait" idiom could break negation.
- It inspected the ML model's weights instead of guessing why the model failed.
- It designed the standard-vs-informal style test that revealed the bias above.
- It caught and fixed a negation-window bug in its own phrase-matching edit before running it.

Where it went along with my choices, or made choices I accepted without much challenge:

- **Labels it wrote itself.** It wrote the expected labels for all 25 stress sentences and the labels for the 6 posts added in Part 4, at my request. It flagged two as debatable but otherwise presented them as reasonable. The "correct" answers in that part of the evaluation are therefore partly the AI's reading of tone, not a human labeler's.
- **The emoji table.** It decided which emojis count as "unambiguous". It scored 🙃 as negative mainly to match my label for the finals post, which reinforces my interpretation instead of questioning it.
- **The starter word lists.** It did not question that the lists were standard-English only until the stress test exposed it. A reviewer could have flagged that coverage gap at the start of Part 2.
- **The mixed rule.** It described the mixed rule as applying when the score is "near 0" but implemented "exactly 0". It worked for this dataset, but the description and the code did not match.
- **A misreported result.** In its Part 4 summary it said "I am not happy about this" went from correct to neutral in the ML leave-one-out test. That post was already wrong at 14 posts. It caught the error only when rechecking the numbers for this card.

I think it should have raised the dialect-coverage issue before building the rules, not after. In general, the assistant's suggestions needed to be verified against the actual outputs, which is what caught most of these issues.

## 8. Ideas for Improvement

- **Get a real held-out test set** and more labels from several people, and measure how often labelers agree.
- **Balance the labels** (more mixed and neutral examples), and add posts that use words in several contexts ("happy", "so", "not").
- **Extend `PHRASES`** for idioms ("can't wait", "no cap"). Grief phrases would need much broader coverage.
- **Add slang to the word lists, with context checks** for words that have two meanings ("sick", "fire", "wicked").
- **Weight emojis more strongly** (±2) so sarcasm markers like 🙄 can outweigh a single literal word. Test it against sincere emoji use.
- **ML model:**
  - Configure `CountVectorizer` to keep emojis.
  - Try TF-IDF and word pairs (`ngram_range=(1, 2)`) so "not happy" becomes one feature.
  - Use a pretrained transformer that already knows slang and sarcasm better.
