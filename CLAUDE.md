# CLAUDE.md

## 0. Update rule

Keep future updates short. Use this format when you change this file.

[Module name] Update reason: one short line. Changes: 1. 2. 3.

Do not leave old task logs or debug notes in this file. Delete them once the task is done.

## 1. What this project is

This is a school project for an intro AI course at RPI. The goal is to label tweets written by US politicians. Each tweet gets two labels. The first label is political leaning, either Democrat or Republican. The second label is stance, either Populist or Establishment. Only the leaning label has ground truth in the source dataset. The stance label is a qualitative output from the model, added to satisfy a rubric requirement about a semantic layer.

Two people worked on this. Junda built the data cleaning and sampling pipeline. Sean built the LLM classification and evaluation pipeline. The two halves connect through two files on disk: `balanced_test_sample.csv` and `cleaned_all_data.parquet`.

There is no server, no API layer, and no database in this project. It is a batch script that you run from the command line. It reads a dataset, calls the Anthropic API once per tweet, writes CSV files, and prints accuracy numbers.

整个共同项目的内容双方都可以在求职/面试中作为自己的项目经历来讲。

## 2. Repo file tree

```
PoliticalTweetCategorizer/
  README.md                  one line, project description
  README_sean.md             setup and usage guide for the LLM half
  clean_data.py               Junda's pipeline: load, clean, sample, split
  classify_llm.py              Sean's pipeline: call the API, save predictions
  evaluate.py                  Sean's pipeline: score predictions, save failures
  prompts.py                   all prompt text and prompt builder functions
  requirements_junda.txt      deps for the data cleaning half
  requirements_sean.txt       just the anthropic package, on top of requirements_junda.txt
  balanced_test_sample.csv    generated output, gold test set, gitignored
  llm_results_zero_shot.csv   generated output, checked in
  llm_results_few_shot.csv    generated output, checked in
  failure_analysis.csv        generated output, checked in
  .gitignore
```

Two files are not tracked in git: `balanced_test_sample.csv` and `cleaned_all_data.parquet`. Both come out of `clean_data.py`. If you clone this repo fresh, you must run `clean_data.py` before you can run anything else.

## 3. Full pipeline flow

The pipeline runs in one straight line. There is no branching, no parallel path, and no async step anywhere in this project.

```
HuggingFace dataset (raw)
  hf://datasets/Jacobvs/PoliticalTweets/formatted_data.parquet
        |
        v
clean_data.py
  1. load_and_clean_tweets()
     - drop rows missing text, party, labels, username, or date
     - drop duplicate tweet text
     - strip URLs, strip @mentions, unescape &amp;, trim whitespace
     - drop tweets under 10 words
     - assign a fresh sequential id
  2. engineer_features_and_flags()
     - pull year from date
     - bucket year into era: trump-era-early (2016-2017),
       trump-era-late (2018-2020), biden-era (2021-2023)
     - flag 8 hardcoded usernames as edge cases
  3. sample_balanced_test_set()
     - sample 100 Democrat + 100 Republican normal tweets
     - sample up to 25 Democrat + 25 Republican edge case tweets
     - shuffle and combine into one 250 row test set
     - everything not sampled becomes the remaining pool
        |
        +--> balanced_test_sample.csv   (250 rows, gold test set)
        +--> cleaned_all_data.parquet   (everything else, few-shot pool)
        |
        v
classify_llm.py
  - load both files above
  - for mode in [zero_shot, few_shot]:
      for each of the 250 test rows, in order:
        build a prompt with prompts.py
        call the Anthropic API, one tweet per request
        sleep 0.5 seconds
      write one results CSV for this mode
        |
        +--> llm_results_zero_shot.csv
        +--> llm_results_few_shot.csv
        |
        v
evaluate.py
  - drop rows where parse_error is True
  - print overall accuracy
  - print accuracy grouped by party, by era, by edge case flag,
    by individual edge case username, and by confidence bucket
  - print a zero-shot vs few-shot comparison table
  - collect every wrong prediction from both modes
        |
        v
failure_analysis.csv
```

There is also a small text-to-report step that is manual: whoever writes the final report reads `failure_analysis.csv` by hand and picks examples.

## 4. Data cleaning and sampling, `clean_data.py`

### Why regex cleaning instead of a tokenizer library

The text cleaning step strips URLs, strips @mentions, and unescapes `&amp;`, all with plain regex. There is no dedicated tokenizer or normalization library involved. The reason is that the LLM reads raw text directly. It does not need pre-tokenized input the way a BERT style model would. The cleaning here only removes noise that would waste tokens or confuse the classifier, like a raw URL or a mention handle that leaks the tweet's author.

### Why a 10 word minimum

Tweets under 10 words are dropped. Very short tweets are usually retweets with no added text, or a link with a short caption. They carry weak signal for a leaning classifier and would just add label noise to both the test set and the few-shot pool.

### Why stratified sampling instead of a random sample

The full dataset is badly skewed by year. Over 96,000 tweets come from 2021 alone, and the whole 2016 to 2020 range only has about 1,300 tweets combined. A plain random sample of 250 tweets would be almost entirely biden-era tweets. The stratified approach fixes the party balance directly, 100 Democrat and 100 Republican normal tweets, plus up to 25 Democrat and 25 Republican edge case tweets. This guarantees the test set accuracy number is not just measuring how well the model reads 2021 tweets.

### Why the edge case list is hardcoded

Eight usernames are hardcoded as edge cases: Romney, Collins, Murkowski, Hawley, Sherrod Brown, Manchin, Sanders, and Angus King. These are politicians known for crossing party lines on rhetoric or voting record. The list is hardcoded rather than computed because "crosses party lines" is not something you can derive from the columns in this dataset. It needed a human call, made once, and then applied consistently. The tradeoff is that this list needs manual updating if the politician roster in the source dataset changes.

### Why edge case senators are excluded from the few-shot pool

`prompts.py` filters out `is_edge_case == True` rows before sampling few-shot examples. This is a data leakage guard. An edge case tweet is deliberately not typical of its party. If one showed up as a labeled example in a few-shot prompt, it would teach the model the wrong pattern for what a "typical" Democrat or Republican tweet looks like, and that would hurt generalization on the rest of the test set. This rule is called out explicitly in the code comments and in `README_sean.md` as a warning to whoever edits the few-shot sampling logic.

### A real data quality note worth knowing

The docs and code comments describe the dataset as senator tweets, and `clean_data.py` even calls the id field a senator tracking id. But the underlying data is not strictly senators. `llm_results_few_shot.csv` has rows with usernames like `RepLoudermilk`, which is a House of Representatives handle, not a Senate one. The `Sen` prefix and `Rep` prefix both show up in the username column. If asked about this in a review, the honest answer is that the dataset was assumed to be senators only, and that assumption was never fully verified against the username column.

## 5. Prompt design, `prompts.py`

### Two dimensions, one has ground truth

The system prompt asks for two labels. Leaning (Democrat or Republican) has ground truth in the `party` column, so accuracy can be measured directly. Stance (Populist or Establishment) has no ground truth anywhere in the source dataset. It exists only because the course rubric asked for a semantic layer beyond a single binary label. Evaluation treats stance as a qualitative output. `evaluate.py` never scores it against anything.

### Why the prompt got shorter

`prompts.py` still has an old system prompt commented out at the top of the file. That older version spelled out both dimensions with a few sentences of definition each, plus explicit output rules and a copy paste JSON template. The commit history shows this was cut down in favor of a shorter prompt, in the same commit that switched the model, roughly halving the token count of the instructions. Fewer tokens means faster responses and a lower per call cost across 250 tweets times two modes times however many reruns happen while tuning the prompt. This is a real tradeoff: the shorter prompt is cheaper and faster, but it drops some of the explicit reasoning scaffolding the longer version had.

### Why the model must return only JSON

Every response is parsed with `json.loads()`. There is no fallback text parser, no regex extraction from free text. Forcing strict JSON output keeps `call_api()` simple. It also makes failures explicit and countable, since anything that fails to parse gets marked `parse_error = True` and excluded from accuracy, rather than silently mis-scored.

### Confidence field

The model self reports a confidence score from 0 to 100 on every call. This is not used to accept or reject a prediction. It is only used downstream in `evaluate.py`, bucketed into five ranges, to check whether the model's stated confidence actually correlates with being right. This is a calibration check, not a filtering mechanism.

## 6. API call reliability, `classify_llm.py`

### Exception handling order and why it is ordered this way

`call_api()` catches three kinds of failure, in this order.

`json.JSONDecodeError` is caught first and returns immediately with `parse_error = True`, no retry. If the model returned text that is not valid JSON, retrying the same call is not likely to fix it right away, and burning a retry budget on a formatting problem wastes API calls that cost money.

`anthropic.RateLimitError` is caught second and does retry, with a wait of `RETRY_DELAY * attempt` seconds, so the wait grows with each attempt. This is a linear backoff, not exponential. Rate limit errors are expected to clear once you wait long enough, so retrying makes sense here.

`anthropic.APIError` is caught last as the general case, covering network errors and other transient API problems, and retries with a fixed `RETRY_DELAY` wait.

If asked why JSON errors are not retried but rate limit errors are, the answer is that a JSON parse failure is a content problem, and a rate limit error is a timing problem. Retrying fixes timing problems. It does not fix content problems.

### Why a plain for loop instead of threads or async

All 250 API calls run one at a time in a plain Python for loop, with a `time.sleep(0.5)` between each call. There is no `ThreadPoolExecutor`, no `asyncio`, and no concurrent request batching anywhere in this codebase.

Three real reasons support this. First, this is a one shot batch script for a class assignment, not a live server handling concurrent user traffic. There is no latency requirement to hide behind concurrency. Second, the free tier of the Anthropic API has a request per minute limit, and firing requests concurrently would risk tripping that limit and wasting retries. A sequential loop with a fixed sleep is the simplest way to stay under a known rate limit. Third, the results list is built by appending in loop order and then concatenated back onto the test dataframe with `pd.concat` by position, not by an explicit key. That alignment only works because the loop processes rows in the same order every time. Adding concurrency here would require rewriting the result collection to track row identity explicitly, for a script that only needs to run once or twice total.

### The retry and cost math

`RETRY_LIMIT` is 3 and `RETRY_DELAY` is 5 seconds. A rate limit error on attempt 1 waits 5 seconds, attempt 2 waits 10 seconds, attempt 3 waits 15 seconds, before the call gives up and returns `API_ERROR`. Across a full run of 250 tweets times 2 modes, `README_sean.md` estimates total API cost at roughly 1 to 2 dollars, based on `claude-haiku-4-5` pricing.

## 7. Evaluation methodology, `evaluate.py`

### Parse errors are dropped, not scored as wrong

`accuracy()` filters out `parse_error == True` rows before computing anything. A row where the model's output could not be parsed is not evidence the model got the leaning wrong. It is evidence the output format broke. Counting it as a wrong answer would conflate two different failure modes into one number.

### Why subgroup accuracy uses `groupby` in pandas instead of separate queries

`accuracy_by_group()` takes a column name and returns accuracy per group with `groupby`. This one function covers accuracy by party, by era, and by individual edge case username, just by changing the column argument. There is no SQL here and no database, so this is really a question of pandas style, not query planning. The point worth knowing is that one small function replaced what would otherwise be three or four near duplicate blocks.

### Why the era comparison comes with a caveat printed alongside it

Biden era tweets make up about 99 percent of the dataset. `evaluate.py` prints era level accuracy anyway, but also prints a warning next to it that cross era comparisons have low statistical power. This is intentional. The number is worth showing, but showing it without the caveat would overstate what the comparison actually proves.

### Confidence calibration

Confidence scores are bucketed into five ranges: 0 to 49, 50 to 69, 70 to 84, 85 to 94, and 95 to 100. Accuracy is computed inside each bucket. This checks whether the model's confidence score tracks its actual correctness, not just whether the model was right overall.

### Edge case senators as an adversarial test set

The 8 hardcoded edge case usernames exist specifically because they are the hardest cases for a leaning classifier. Their rhetoric does not follow the typical pattern for their party. `evaluate.py` reports their accuracy separately from the normal cases, and this split is usually where a report finds its most interesting failure mode, since a model that is highly accurate overall can still fail badly on the tweets that were chosen specifically because they are hard.

## 8. Known limitations, worth knowing before a review

The test set is small, 250 tweets. Accuracy numbers at that sample size have real confidence intervals, and the code does not compute or report one.

The stance dimension, Populist versus Establishment, has no ground truth anywhere. Any claim about stance accuracy is not possible with this data. Only qualitative review of the reasoning field is possible.

`README_sean.md` mentions a BERT baseline as a possible comparison point. As of the current code, no BERT baseline exists in this repo. Only the LLM pipeline is implemented.

`requirements_junda.txt` has several duplicate lines, including `anthropic`, `annotated-types`, `distro`, `docstring_parser`, `jiter`, `pydantic`, `pydantic_core`, `sniffio`, and `typing-inspection`, each listed twice with the same pin. This does not break `pip install`, but it is a sign the file was regenerated by appending rather than by a clean freeze.

The edge case username list needs manual maintenance. If the source dataset adds or removes politicians, nothing in the code will tell you the list is stale.

## 9. Setup and running

Install both requirement files. `requirements_sean.txt` only adds the `anthropic` package on top of `requirements_junda.txt`, so both are needed.

```
pip install -r requirements_junda.txt
pip install -r requirements_sean.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

Run the pipeline in this order. Each step depends on the file the previous step wrote.

```
python clean_data.py
python classify_llm.py --dry-run
python classify_llm.py
python evaluate.py
```

The dry run flag limits classification to the first 10 tweets, so you can check the output format before spending API credits on the full 250 tweet run. `classify_llm.py` also takes a `--mode` flag, `zero_shot`, `few_shot`, or `both`, if you want to run one mode at a time.

## 10. Output column reference

Both `llm_results_zero_shot.csv` and `llm_results_few_shot.csv` share the same schema. They start with every column from `balanced_test_sample.csv`, which is `text`, `party`, `labels`, `username`, `date`, `year`, `era`, and `is_edge_case`. Six columns are appended by `classify_llm.py`.

`pred_leaning` holds the model's leaning prediction, Democrat or Republican, or PARSE_ERROR or API_ERROR on failure. `pred_stance` holds the model's stance prediction, Populist or Establishment, with the same failure values. `confidence` holds the model's self reported confidence, 0 to 100, or negative 1 on failure. `reasoning` holds the model's one sentence explanation. `raw_response` holds the full raw text returned by the API, kept for debugging bad parses. `parse_error` is a boolean, true if the response could not be parsed as JSON.

`failure_analysis.csv` has the same schema plus one more column, `mode`, marking whether each misclassified row came from the zero-shot or few-shot run.
