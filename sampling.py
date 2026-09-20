"""Shared sampling protocol for the undersampling vs non-undersampling study.

The protocol is fixed so every model variant (RF / XGBoost x undersampled /
full) is trained and evaluated under identical conditions:

1. Clean the dataset (drop missing / duplicate URLs).
2. Common stratified 80/20 split (seed 42) -> train pool + common test set.
   The test set is shared by *all* variants; no model ever trains on it.
3. Variant "full": train on the entire train pool (imbalanced, ~76.8% legit).
4. Variant "under": undersample legitimate URLs in the train pool down to the
   phishing count (1:1), seed 42.

Used by ``benchmark_sampling.py``; available to any future training script.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
DEFAULT_DATASET = 'URL dataset.csv'


def load_clean_dataset(path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the URL dataset, clean it, and map the binary label column."""
    df = pd.read_csv(path)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    df = df.copy()
    df['label'] = df['type'].map({'phishing': 1, 'legitimate': 0})
    df = df.dropna(subset=['label'])
    return df.reset_index(drop=True)


def common_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Stratified train/test split shared by all sampling variants."""
    train_pool, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    return train_pool.reset_index(drop=True), test.reset_index(drop=True)


def undersample(train_pool: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Balance the train pool: sample legitimate URLs to the phishing count."""
    phish = train_pool[train_pool['label'] == 1]
    legit = train_pool[train_pool['label'] == 0].sample(n=len(phish), random_state=seed)
    return (
        pd.concat([phish, legit])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def describe(df: pd.DataFrame) -> dict:
    """Summary counts used in logs / reports."""
    n_legit = int((df['label'] == 0).sum())
    n_phish = int((df['label'] == 1).sum())
    total = n_legit + n_phish
    return {
        'total': total,
        'legit': n_legit,
        'phishing': n_phish,
        'legit_share': round(n_legit / total, 4) if total else 0.0,
        'imbalance': round(n_legit / n_phish, 2) if n_phish else None,
    }