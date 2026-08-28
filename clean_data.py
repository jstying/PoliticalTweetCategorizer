"""
Dataset Notes for Sean

If you run the below code, it will produce two files:

1. Test set (balanced_test_sample.csv):
- 250 labeled tweets with columns: text, party(Republican/Democrat), labels(0/1), username, date, year(202x), era, is_edge_case(True/False)
- The 250 test tweets consist of 200 normal cases(100 each party) and 50 edge cases(25 each party), strictly balanced 50/50 between Democrats and Republicans.
- 'labels' is the ground truth from the original dataset - compare your LLM predictions directly against this column.
- is_edge_case=True marks the 50 edge case tweets (25 Democrat / 25 Republican) - focus your failure analysis here.

2. Full cleaned data (cleaned_all_data.parquet):
- Remaining cleaned tweets after test set removed. Use this for few-shot example selection or BERT baseline if needed.

[Few-Shot Sampling Warning]
If you select few-shot examples from 'cleaned_all_data.parquet', you MUST manually exclude 
the 8 edge-case senators (where is_edge_case=True). Feeding non-typical senator tweets as 
standard templates to the LLM will severely damage the model's generalization power.


[Note on "Populist vs. Establishment" Labels]
The original dataset has NO labels for "Populist vs. Establishment". 

-----------------------------------------------------------------------------------------

Dataset characteristics:
- 481 unique senators/users in the dataset
- Years covered: 2016-2023
- Year distribution is heavily skewed toward Biden era:

  2016:      18 tweets
  2017:     157 tweets
  2018:     126 tweets
  2019:     202 tweets
  2020:     801 tweets
  2021:  96,443 tweets
  2022:  52,336 tweets
  2023:  29,184 tweets

Era segmentation:
- "trump-era-early" (2016-2017)
- "trump-era-late" (2018-2020)
- "biden-era" (2021-2023) accounts for ~99% of the data, so cross-era analysis will have 
  limited statistical power and should be flagged as a limitation in the final report.

Quick Load Example:
import pandas as pd
test_df = pd.read_csv("balanced_test_sample.csv")
all_cleaned_df = pd.read_parquet("cleaned_all_data.parquet")  # requires 'pyarrow'
"""

import pandas as pd
import numpy as np

def load_and_clean_tweets(file_path: str) -> pd.DataFrame:
    """
    Loads the raw political tweets dataset, removes rows with missing values,
    eliminates duplicates, and applies text preprocessing filters.
    
    Parameters:
        file_path (str): The path or URI to the input parquet file.
        
    Returns:
        pd.DataFrame: A cleaned DataFrame with an assigned unique 'id' column.
    """
    # Load dataset
    df = pd.read_parquet(file_path)
    
    # Drop rows missing crucial fields and remove text duplicates
    df = df.dropna(subset=["text", "party", "labels", "username", "date"])
    df = df.drop_duplicates(subset=["text"])
    
    # Standardize and clean text data
    df["text"] = df["text"].str.replace(r"https?://\S+|www\.\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"&amp;", "&", regex=True)
    df["text"] = df["text"].str.replace(r"@\S+", "", regex=True)
    df["text"] = df["text"].str.strip()
    
    # Filter out short tweets to ensure sufficient textual content
    df = df[df["text"].str.split().str.len() >= 10]
    
    # Generate a globally unique ID column for tracking purposes
    df = df.reset_index(drop=True)
    df["id"] = df.index
    
    return df


def engineer_features_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features, categorizes tweets into political eras,
    and flags specific edge-case users.
    
    Parameters:
        df (pd.DataFrame): The cleaned tweet DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame augmented with 'year', 'era', and 'is_edge_case'.
    """
    # Extract year from date
    df["year"] = pd.to_datetime(df["date"]).dt.year
    
    # Segment data into specific political eras
    df["era"] = np.select(
        [
            df["year"].between(2016, 2017),
            df["year"].between(2018, 2020),
            df["year"].between(2021, 2023)
        ],
        [
            "trump-era-early",
            "trump-era-late",
            "biden-era"
        ],
        default="other-era" # This category should be empty given the dataset's year range, but it's included for robustness
    )
    
    print("Era distribution:")
    print(df["era"].value_counts())
    
    # Define and flag high-variance or cross-pressured political users, HARDCODED!
    confirmed_edge_users = {
        'SenatorRomney', 'SenatorCollins', 'lisamurkowski',
        'HawleyMO', 'SenSherrodBrown', 'Sen_JoeManchin',
        'SenSanders', 'SenAngusKing'
    }
    df["is_edge_case"] = df["username"].isin(confirmed_edge_users)
    
    return df


def _stratified_by_era(pool: pd.DataFrame, target: int, random_state: int) -> pd.DataFrame:
    """
    Samples `target` rows from `pool`, spreading the draw as evenly as
    possible across the political eras present in the pool.

    Each era gets an equal share of `target` (remainder distributed to the
    first eras alphabetically for determinism), capped by how many rows of
    that era actually exist. Any shortfall from a thin era (e.g. there are
    zero Democrat trump-era-early tweets in this dataset) is backfilled from
    whichever era has the most remaining capacity — in practice biden-era,
    since it dominates the source data. This is the same
    "min(target, available)" capping idiom used for edge-case sampling below,
    just applied per era instead of per party.
    """
    eras = sorted(pool["era"].unique())
    n_eras = len(eras)
    base = target // n_eras
    remainder = target % n_eras

    era_pools = {era: pool[pool["era"] == era] for era in eras}
    want = {era: base + (1 if i < remainder else 0) for i, era in enumerate(eras)}

    take = {era: min(want[era], len(era_pools[era])) for era in eras}
    shortfall = sum(want[era] - take[era] for era in eras)

    if shortfall > 0:
        # Backfill from the eras with the most leftover rows, largest first
        capacity_order = sorted(eras, key=lambda e: len(era_pools[e]) - take[e], reverse=True)
        for era in capacity_order:
            if shortfall <= 0:
                break
            room = len(era_pools[era]) - take[era]
            add = min(room, shortfall)
            take[era] += add
            shortfall -= add

    parts = [era_pools[era].sample(n, random_state=random_state) for era, n in take.items() if n > 0]
    return pd.concat(parts)


def sample_balanced_test_set(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a stratified gold test set consisting of 200 normal tweets
    (balanced by party, and spread across political eras where the source
    data allows it) and up to 50 edge-case tweets (balanced by party).

    Parameters:
        df (pd.DataFrame): The processed DataFrame.
        random_state (int): Seed used to ensure reproducible sampling.

    Returns:
        tuple: (balanced_test_sample, df_remaining) DataFrames with the temporary 'id' removed.
    """
    # Sample 100 non-edge tweets per major party, stratified across era
    normal_dem_pool = df[(df["party"] == "Democrat") & (~df["is_edge_case"])]
    normal_rep_pool = df[(df["party"] == "Republican") & (~df["is_edge_case"])]

    normal_dem = _stratified_by_era(normal_dem_pool, 100, random_state)
    normal_rep = _stratified_by_era(normal_rep_pool, 100, random_state)

    # Stratify edge cases evenly across parties (and, where the pool allows
    # it, across era — in this dataset every confirmed edge-case tweet that
    # survives cleaning happens to be biden-era, so this is a no-op today
    # but keeps the same guarantee if that ever changes)
    edge_pool = df[df["is_edge_case"]]
    edge_dem_pool = edge_pool[edge_pool["party"] == "Democrat"]
    edge_rep_pool = edge_pool[edge_pool["party"] == "Republican"]

    edge_dem_n = min(25, len(edge_dem_pool))
    edge_rep_n = min(25, len(edge_rep_pool))

    edge_dem_samples = _stratified_by_era(edge_dem_pool, edge_dem_n, random_state) if edge_dem_n > 0 else edge_dem_pool.iloc[0:0]
    edge_rep_samples = _stratified_by_era(edge_rep_pool, edge_rep_n, random_state) if edge_rep_n > 0 else edge_rep_pool.iloc[0:0]

    edge_samples = pd.concat([edge_dem_samples, edge_rep_samples])
    actual_edge = len(edge_samples)

    if actual_edge < 50:
        print(
            f"Warning: edge case stratified sampling yielded only {actual_edge} rows "
            f"(Democrat: {edge_dem_n} / Republican: {edge_rep_n}). Target of 50 not met."
        )

    # Combine normal and edge cases, then shuffle the final test set
    balanced_test_sample = pd.concat([normal_dem, normal_rep, edge_samples]).sample(frac=1, random_state=random_state)

    print(f"Test set total: {len(balanced_test_sample)} rows")
    print(f"  Edge cases - Democrat: {edge_dem_n} / Republican: {edge_rep_n}")
    era_counts = balanced_test_sample["era"].value_counts()
    biden_share = era_counts.get("biden-era", 0) / len(balanced_test_sample)
    print(f"  Era distribution: {era_counts.to_dict()} (biden-era share: {biden_share:.1%}, "
          f"vs. {177963/179267:.1%} in the source dataset)")

    # Isolate training/baseline data by excluding the sampled test records
    df_remaining = df[~df["id"].isin(balanced_test_sample["id"])]

    # Clean up tracking IDs before exporting
    balanced_test_sample = balanced_test_sample.drop(columns=["id"])
    df_remaining = df_remaining.drop(columns=["id"])

    return balanced_test_sample, df_remaining


def run_pipeline():
    """
    Executes the entire data preparation pipeline from loading to storage.
    """
    input_uri = "hf://datasets/Jacobvs/PoliticalTweets/formatted_data.parquet"
    
    df = load_and_clean_tweets(input_uri)
    df = engineer_features_and_flags(df)
    
    test_df, remaining_df = sample_balanced_test_set(df, random_state=42)
    
    # Save outputs to disk
    test_df.to_csv("balanced_test_sample.csv", index=False)
    remaining_df.to_parquet("cleaned_all_data.parquet")
    
    print("Data cleaning and gold test set split complete.")

if __name__ == "__main__":
    run_pipeline()








