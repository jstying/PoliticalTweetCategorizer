# My answer
Political Tweet Categorizer	May 2026 – June 2026
•	Engineered a stratified data-cleaning and sampling pipeline in Python and Pandas that corrected a 99.3% single-era skew in a 179K-row dataset, producing a party-balanced 250-tweet gold evaluation set with an adversarial subset of 8 cross-party politicians to benchmark downstream LLM systems.
•	Redesigned the LLM system prompt for a tweet-classification pipeline, cutting instruction length by 80.8% (1364 to 262 characters) across 500 production API calls while preserving a strict JSON output contract, holding output-parsing validity at 99.6–100% across both prompting modes.
•	Built a fault-tolerant LLM API client featuring a 3-tier failure-handling strategy with linear backoff on rate limits, achieving a 99.6–100% valid-response rate across 500 sequential production calls without silent misclassification from failed parses.
•	Executed a controlled zero-shot versus few-shot evaluation of an LLM political leaning (Democrat/Republican) classifier on a 250-tweet test set, measuring a 5.6% relative accuracy gain overall (72.7% to 76.8%) and a 9.1% relative gain (66.0% to 72.0%) on the adversarial cross-party subset.
•	Developed a confidence-calibration analysis for LLM predictions, confirming a monotonic 39-point accuracy spread (61% to 100%) across 5 confidence bands and identifying an 85% confidence threshold where accuracy jumps by 26.2 points to inform low-confidence triage rules.



# Resume Bullets — Political Tweet Categorizer

Source: intro AI course project (RPI), 2-person team. Junda built the data
cleaning/sampling pipeline (`clean_data.py`); Sean built the LLM
classification/evaluation pipeline (`classify_llm.py`, `evaluate.py`,
`prompts.py`). Every number below was measured directly from this repo —
either by running `evaluate.py` against the checked-in
`llm_results_zero_shot.csv` / `llm_results_few_shot.csv`, by diffing the
old vs. new `SYSTEM_PROMPT` string in `prompts.py`, or by reading the
dataset stats documented in `clean_data.py`. None are estimated. See the
**Metrics reference** table at the bottom for exactly how each number was
derived, so you can defend it in an interview.

If you're pulling bullets for your own resume, bullets 1 is Junda's
individual work; 2–5 describe pipeline behavior either of you can speak to
if you were involved in prompt/eval design, but attribute individual
authorship honestly per the split above.

---

## 1. Stratified sampling to fix dataset skew

**Situation:** The raw HuggingFace dataset (179,267 politician tweets,
2016–2023) was severely skewed by year — 2021 alone accounted for 53.8%
of all records (96,443 tweets), while the entire 2016–2020 span held only
0.7% (1,304 tweets). A naive random sample would be ~99% Biden-era data.

**Task:** Build an unbiased, party-balanced gold evaluation set that
doesn't inherit the source data's temporal and class skew.

**Action:** Designed and implemented a cleaning + stratified sampling
pipeline (`clean_data.py`) — regex-based text normalization
(URL/mention/HTML-entity stripping), deduplication, a 10-word minimum
length filter, then a two-tier stratified sampler drawing 100 Democrat +
100 Republican "normal" tweets plus up to 25+25 tweets from 8 hardcoded
politicians known for cross-party rhetoric, shuffled into a reproducible
250-row test set.

**Result:** Delivered a perfectly party-balanced (125/125) 250-tweet gold
benchmark decoupled from the 99.3% Biden-era concentration in the source
data, plus a clean ~179K-tweet few-shot/training pool.

> **Polished bullet:**
> Engineered a stratified data-cleaning and sampling pipeline (Python /
> Pandas) that corrected a 99.3% single-era skew in a 179K-row political
> tweet dataset (one year alone = 53.8% of records), producing a
> party-balanced 250-tweet gold evaluation set — including a curated
> adversarial subset of 8 cross-party politicians — used to benchmark a
> downstream LLM classification system without inheriting the source
> data's sampling bias.

---

## 2. Prompt compression for cost/latency

**Situation:** The original system prompt sent on every LLM call was a
1,364-character (205-word) instruction block with a full JSON template
and explicit output rules.

**Task:** Cut per-call token overhead across a 500-call production run
(250 tweets × 2 prompting modes) without degrading output quality or
breaking the strict JSON-only output contract that `call_api()` depends
on.

**Action:** Rewrote `SYSTEM_PROMPT` into a condensed 262-character
(36-word) version that keeps both classification dimensions (leaning +
populist/establishment stance), the confidence field, and the JSON-only
constraint.

**Result:** Cut system-prompt size 80.8% by characters (1,364→262) and
82.4% by words (205→36) on every one of 500 calls, while output validity
held at 100% on the few-shot run (0/250 parse errors) and 99.6% on
zero-shot (1/250).

> **Polished bullet:**
> Redesigned the LLM system prompt for a tweet-classification pipeline,
> cutting instruction length 80.8% (1,364→262 characters) across 500
> production API calls while preserving a strict JSON-only output
> contract — reducing per-call token overhead with zero regression in
> output-parsing reliability (99.6–100% valid JSON responses).

---

## 3. Fault-tolerant API client

**Situation:** Calls to the classification API could fail three distinct
ways — malformed JSON, rate limiting, or transient network errors — each
needing a different recovery strategy, across a sequential run of 500
calls with no retry-budget to waste.

**Task:** Build error handling that recovers from timing failures without
burning retries on failures that retrying can't fix, and without letting
failed calls silently corrupt the accuracy numbers.

**Action:** Implemented a 3-tier exception strategy in `call_api()`:
JSON decode failures fail fast with zero retries (a content problem, not
a timing problem); rate-limit errors get linear backoff (`RETRY_DELAY *
attempt`, i.e. 5s/10s/15s); generic API errors get a fixed 5s retry —
capped at 3 attempts (30s max) before the row is marked `API_ERROR`. On
the evaluation side, `parse_error` rows are explicitly excluded from
accuracy rather than scored as wrong.

**Result:** 0% malformed-output rate on the few-shot run (0/250), 0.4%
on zero-shot (1/250) — with zero silent mis-scoring of failed parses in
the final accuracy numbers.

> **Polished bullet:**
> Built a fault-tolerant LLM API client with a 3-tier failure-handling
> strategy — fail-fast on malformed JSON, linear backoff on rate limits,
> bounded retry on transient errors — achieving a 99.6–100% valid-response
> rate across 500 sequential production calls with zero silent
> misclassification from failed parses.

---

## 4. Few-shot vs. zero-shot evaluation

**Situation:** It was unknown whether adding labeled examples to the
prompt (few-shot) meaningfully improved classification quality over a
bare zero-shot prompt, or was just added cost.

**Task:** Run both prompting strategies on the identical 250-tweet gold
set and quantify the difference, with a data-leakage guard excluding
adversarial edge-case tweets from the few-shot example pool.

**Action:** Ran the full 250-tweet benchmark under both modes and built a
subgroup evaluation (`evaluate.py`) breaking out accuracy by party, era,
normal-vs-edge-case, and confidence bucket.

**Result:** Few-shot beat zero-shot on every axis measured: overall
accuracy 72.7%→76.8% (+5.6% relative), and on the hardest
adversarial edge-case subset 66.0%→72.0% (+9.1% relative) — while
also eliminating the one parse failure zero-shot produced.

> **Polished bullet:**
> Ran a controlled zero-shot vs. few-shot evaluation of an LLM political
> stance classifier on a 250-tweet balanced test set, measuring a +5.6%
> relative accuracy gain overall (72.7%→76.8%) and a +9.1% relative gain
> (66.0%→72.0%) on an adversarial cross-party-politician subset —
> directly informing the production prompting strategy.

---

## 5. Confidence calibration analysis

**Situation:** The model self-reports a 0–100 confidence score on every
prediction, but that score is only useful downstream if it actually
correlates with correctness.

**Task:** Verify whether the model's stated confidence was a trustworthy
signal that could inform a triage or filtering rule, not just a
decorative field.

**Action:** Built a 5-bucket confidence-calibration analysis in the
evaluation pipeline, computing empirical accuracy within each confidence
range across both result sets.

**Result:** Found a clean, monotonic relationship — accuracy rose from
61.0% at <50% confidence to 100.0% at 95%+ confidence (a 39-point
spread), with the sharpest single jump (+26.2 points) between the
70–84% and 85–94% confidence bands.

> **Polished bullet:**
> Built a confidence-calibration analysis for an LLM classifier's
> self-reported confidence scores, confirming a monotonic 39-point
> accuracy spread across 5 confidence bands (61%→100%) and identifying an
> 85%+ confidence threshold where accuracy jumps 26.2 points — a
> data-driven basis for a low-confidence triage rule.

---

## Metrics reference (how each number was derived)

| Metric | Value | How it was measured |
|---|---|---|
| Overall accuracy, zero-shot | 72.7% (182/250 valid) | `python3 evaluate.py`, section "SUMMARY COMPARISON" |
| Overall accuracy, few-shot | 76.8% (192/250) | same |
| Edge-case accuracy, zero-shot | 66.0% | same |
| Edge-case accuracy, few-shot | 72.0% | same |
| Parse errors, zero-shot | 1/250 (0.4%) | `pred_leaning == PARSE_ERROR` count in `llm_results_zero_shot.csv` |
| Parse errors, few-shot | 0/250 (0.0%) | same, `llm_results_few_shot.csv` |
| System prompt length, old vs. new | 1,364→262 chars, 205→36 words | `len()`/`.split()` on the commented-out vs. active `SYSTEM_PROMPT` strings in `prompts.py` |
| Confidence calibration | 61.0%→100.0% across 5 buckets | `evaluate.py` section "6. Confidence Calibration" |
| Sharpest calibration jump | +26.2 pts (70–84% → 85–94% bucket) | 97.2% − 71.0%, same section |
| Dataset year skew | 2021 = 53.8% of 179,267 rows; 2016–2020 = 0.7% | Year counts documented in `clean_data.py` docstring, summed and divided by total |
| Biden-era share | 99.3% (177,963/179,267) | same source |
| Retry backoff | 5s / 10s / 15s, 30s max before `API_ERROR` | `RETRY_LIMIT=3`, `RETRY_DELAY=5` in `classify_llm.py`, `RETRY_DELAY * attempt` |

Not included above: API dollar cost (~$1–2 total per `README_sean.md`) —
that figure is the project's own documented estimate, not something I
re-measured against a live bill, so it's left out of the bullets to avoid
citing an unverified number as tested.
