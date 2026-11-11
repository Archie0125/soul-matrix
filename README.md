# Soul Matrix

**Persona management as linear algebra.** Every persona is a point in a correlated vector space. A seed gives you a new virtual persona; a few observations of a real person (given with consent) condition the vector toward them. The vector then steers how a language model writes, so a persona keeps the same voice across thousands of posts.

> **Prior work, stated up front.** I built the engine behind this repo between **Nov 6 and Nov 10, 2026**, inside a private Traditional Chinese forum codebase. That was **before** c0mpiled-18 Taipei. During the event (Nov 11) I only extracted its core into `soul_matrix.php`, checked it bit-for-bit against the original, and wrote this README and the demo video. Under the event rules this is prior work, and the judges should treat it that way.

Context: YC's Request for Startups *LLMs That Write Well* (Jared Friedman) argues that the reward signals behind strong coding models flatten voice and style. Soul Matrix does not train a model. It sits on top of one and supplies the voice, as an explicit vector you can inspect, edit, clone and blend.

## Run it

PHP 8.1+, no dependencies, no network.

```bash
php soul_matrix.php forge 42              # a virtual persona from seed 42
php soul_matrix.php forge 42 +-+-         # same seed, archetype signs set by hand
php soul_matrix.php clone 42 humor=1.8 swear=1.5 emoji=-1.2   # condition seed 42 on three observed axes
php soul_matrix.php blend 42 7 0.5        # halfway between two personas
```

## The math

```
seed ──sha256──▶ z ∈ ℝ¹⁷            independent normals, clipped at ±2.5
z ──────────────▶ x = A·z + b        A = Cholesky factor of the axis correlation R
                                     b = optional 4-axis archetype shift
clone:  z* = z₀ + A_oᵀ (A_o A_oᵀ + Σ)⁻¹ (x_o − A_o z₀ − b_o − ε)
```

- **Deterministic.** All randomness comes from sha256 (`unit()`, then an inverse-normal approximation). The same seed gives the same persona on any machine and any PHP version.
- **Correlated, not independent.** R has 18 non-zero pairs. For example, agreeableness vs. disagreeing is −0.40 and conscientiousness vs. typos is −0.35. Sampling through the Cholesky factor `L` makes every persona internally consistent.
- **Append-only axes.** Cholesky is prefix-stable: adding an 18th axis leaves the first 17 values of every existing persona untouched.
- **Anti-caricature bound.** A persona with ‖z‖² above the 97.5% quantile of χ²₁₇ (30.19) is rejected.
- **Clone = exact Gaussian conditioning** (Matheron's rule) with observation noise. Setting `var → 0` pins an observed axis. A large `var` falls back to the seed's prior. Unobserved axes move through R. For example, observing high swearing also lowers agreeableness a little.
- **Blend** two personas with `z_t = ((1−t)z_A + t·z_B) / √((1−t)² + t²)`. Without the divisor, a 50/50 blend has a standard deviation of 0.707 and reads as bland.

## Every dimension

### The 17 core axes (in this repo)

Each axis is a standardized latent value. The full engine turns it into a concrete writing parameter, shown in the third column.

| # | Axis | What it is | What it drives in the text | Correlated with |
|---|---|---|---|---|
| 1 | `E` | Extraversion | social energy, how often the persona joins threads | O +.20, emoji +.25, activity +.30, exclaim +.25 |
| 2 | `O` | Openness | range of topics and vocabulary | E +.20, english +.15, length +.20 |
| 3 | `A` | Agreeableness | warmth; how rarely the persona pushes back | C +.20, N −.20, disagree −.40, swear −.30 |
| 4 | `C` | Conscientiousness | care and tidiness of the writing | A +.20, N −.25, swear −.20, punct_heavy −.10, typo −.35 |
| 5 | `N` | Neuroticism | emotional volatility | C −.25, A −.20, punct_heavy +.25 |
| 6 | `humor` | Humor | jokes, irony, light teasing | slang +.30 |
| 7 | `slang` | Internet slang density | share of slang words | humor +.30, emoji +.30, particle +.35 |
| 8 | `emoji` | Emoji use | emoji rate = 0.30 · e^(0.5x), clamped to [0, 0.6] | E +.25, slang +.30 |
| 9 | `punct_heavy` | Punctuation intensity | 4 levels: none / sparse / normal / heavy (15/35/35/15%) | C −.10, N +.25 |
| 10 | `length` | Verbosity | median comment length | O +.20 |
| 11 | `english` | Code-switching | how often English words are mixed into Chinese | O +.15 |
| 12 | `typo` | Typo rate | 4 bands (40/30/20/10%) | C −.35 |
| 13 | `particle` | Sentence-final particles | density of Mandarin particles such as 啦 / 喔 / 欸 | slang +.35 |
| 14 | `activity` | Activity | posts per week | E +.30 |
| 15 | `disagree` | Pushback | share of replies that disagree | A −.40 |
| 16 | `exclaim` | Exclamation | exclamation-mark rate | E +.25 |
| 17 | `swear` | Vulgarity | mild-to-strong profanity | A −.30, C −.20 |

**Archetype shift (4 extra numbers, optional).** A coarse sign on each of E, O, A, C moves the persona by `r·s·d` with r = .70 / .70 / .45 / .50, where `d` is a per-seed magnitude. The directions follow McCrae & Costa (1989). The magnitudes are design values and I hold them with low confidence. Cells of R that touch a shifted axis are pre-divided by √(1−r²), so the population still has the target correlations.

### Layers in the full engine (not in this repo)

These exist in the private engine and are documented here so the design is complete. They are **not** included in this repo.

| Layer | Size | What it is |
|---|---|---|
| Typing-habit block | 9 axes | zhuyin (phonetic-symbol) typing, kaomoji, ellipsis style, intensifier budget, line breaks, typesetting (half-width punctuation, spacing), laughter forms, tilde, question-mark habits. Built as `u = σ·γ(age) + B·x + L_h·ε`, where `B = C·R⁻¹` ties habits to the core axes and γ(age) shifts how common each habit is by age group. |
| Identity chain | 14 fields | birth year, age, gender, home city, current city, life stage, job group, job, marital status, kids, home platform, era vocabulary, education, military service. Drawn as a conditional chain (each field depends only on earlier ones) and checked against Taiwanese census tables. |
| Life layer | ~20 fields | cohort and tech era, family structure, relatives, influences, memories, inner voice, convictions, hobbies, expertise, the 10 Schwartz values, the 3 self-determination needs, and debate style (reason / feelings / rules / group / own experience). |
| Writing pass | — | A persona card of at most 250 characters, plus the persona's own few-shot samples, goes to the model. A deterministic typing pass then enforces the vector's rates (emoji, particles, punctuation, typos), and quality gates reject generic "AI tone". |

## Verified tonight

`verify/compare_with_engine.php` runs this file against the private engine. Result (`verify/result.txt`):

```
seeds compared:                 2000
z vectors not identical:        0
max |x_engine - x_standalone|:  8.882e-16
clones compared (3 obs each):   1788 (both rejected as out-of-bounds: 212, disagreements: 0)
max |z*_engine - z*_standalone|: 0.000e+0
```

## What they write

The full engine generated these comments on **Nov 8, 2026**, before the event. It used an open 26B model (Gemma 4, run locally) and an earlier version of the habit layer. They are unedited. The English lines are my translations, not model output.

**Thread:** *"Company suddenly says everyone must come back to the office every Friday."* The comment being replied to: *"The worst part is the commute. Three hours round trip, who pays me back for that?"*

| Persona (vector highlights) | Comment | English |
|---|---|---|
| Reserved (E −0.40), heavy English mixing (0.59) | 超痛苦 通勤浪費生命 | Pure suffering. Commuting is a waste of life. |
| Very agreeable (A +2.34), heavy slang (0.82) | 超痛欸，單程一個半小時會累死，有誰有什麼躲避通勤的撇步嗎？ | Ugh, that hurts. An hour and a half each way would kill me. Anyone have tricks for dodging the commute? |
| Outgoing (E +0.65), exclaims a lot (0.34), no emoji | 蛤，這超痛苦 我住新北通勤起來也超累的 主要還是因為那些理由都瞎 什麼凝聚力什麼的。。。只是想叫大家回辦公室而已吧，那種通勤浪費的時間沒人補啦 我會想直接退休 | Huh? That's brutal. I live in New Taipei and my commute wears me out too. Mostly it's that the reasons are nonsense, "team cohesion" or whatever... they just want everyone back in the office, right? Nobody pays you back for commuting time. I'd just want to retire. |
| Slightly reserved, mildly disagreeable (A −0.27) | 真的超痛欸，那邊通勤單程又要一個多小時，回來的路上整個人都空了，沒辦法好好追劇，真的超不划算 | Seriously painful. Over an hour each way, and on the ride home you're totally drained, can't even binge your shows. Not worth it at all. |

Same model, same thread, same prompt template. Only the vector differs.

## What has been measured (full engine, before the event)

- **Personas are distinguishable from each other.** 48 personas × 32 topics, 1,533 comments. Character 1–3-gram TF-IDF with topic-controlled split halves gave a within-group AUC of **0.910** (95% CI 0.864–0.958). The shuffled-label control was 0.50, and no persona passed by repeating a catchphrase (n-gram leak check passed).
- **Known machine tics were reduced** after regenerating with the typing pass. For example, the filler 真的 ("really") went from 1.55 to 0.38 per comment, and sentence-final particles at the end of a comment went from 48% to 22%.

## What it does not do yet

- **No human blind test of "sounds like a real person."** The only blind test so far compared topic ideas, not voices. It got 25/40 against a preregistered pass bar of 27, so it failed.
- **Cloning has not been validated on real people.** The math is exact, but nobody has checked whether a clone writes like the person it was conditioned on.
- **English is untested.** The 17 core axes are language-agnostic. The habit layer and writing pass are Chinese-specific. Whether the vector carries over to English is a hypothesis, not a result.
- **Habits do not reveal personality.** Predicting the 4-letter archetype from typing habits scored 7%, worse than always guessing the most common class (12%).

## Responsible use

Clone a real person only with their consent. Label AI-generated text where people will read it. Do not use persona fleets to impersonate people or to simulate grassroots opinion.

## License

MIT
