"""Random Forest training (Gen 3) — TF-IDF + structural features.

Usage (run from the repo root):

    # Undersampled variant (default; the one saved as phishing_optimized.pkl)
    python hybrid/train_hybrid_model_v2.py

    # Non-undersampled variant (the one deployed in the extension)
    python hybrid/train_hybrid_model_v2.py --no-undersample --out rf_models/rf_full.pkl

See MODELS.md for model provenance.
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import StructuralFeatureExtractor, make_tokens  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hybrid RF phishing-URL model.")
    parser.add_argument('--data', default='URL dataset.csv',
                        help="Path to the CSV dataset (default: %(default)s)")
    parser.add_argument('--out', default=os.path.join('rf_models', 'phishing_optimized.pkl'),
                        help="Output .pkl path (default: %(default)s)")
    parser.add_argument('--no-undersample', dest='undersample', action='store_false',
                        help="Train on the full imbalanced dataset instead of undersampling")
    parser.set_defaults(undersample=True)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"Dataset not found: {args.data}")
        sys.exit(1)

    # --- 1. Data cleaning ---
    print(f"Dataset awal: {len(df)} baris")
    df = df.dropna(subset=['url', 'type'])            # drop rows with missing values
    df = df.drop_duplicates(subset=['url'])           # drop duplicate URLs
    print(f"Setelah cleaning: {len(df)} baris")

    # --- 2. Label encoding (Legitimate = 0, Phishing = 1) ---
    df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})

    # --- 3. Optional undersampling of the majority class (Proposal Bab 3.3.3) ---
    if args.undersample:
        df_phishing = df[df['label_binary'] == 1]
        df_legitimate = df[df['label_binary'] == 0]

        n_minority = len(df_phishing)
        print("\nDistribusi sebelum undersampling:")
        print(f"  Legitimate: {len(df_legitimate)}")
        print(f"  Phishing:   {len(df_phishing)}")

        df_legitimate_undersampled = df_legitimate.sample(n=n_minority, random_state=42)
        df_balanced = pd.concat([df_legitimate_undersampled, df_phishing])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nDistribusi setelah undersampling:")
        print(f"  Legitimate: {len(df_balanced[df_balanced['label_binary'] == 0])}")
        print(f"  Phishing:   {len(df_balanced[df_balanced['label_binary'] == 1])}")
        print(f"  Total:      {len(df_balanced)}")
    else:
        df_balanced = df
        print("\nTanpa undersampling (full dataset):")
        print(f"  Legitimate: {len(df_balanced[df_balanced['label_binary'] == 0])}")
        print(f"  Phishing:   {len(df_balanced[df_balanced['label_binary'] == 1])}")
        print(f"  Total:      {len(df_balanced)}")

    # --- 4. Train/test split (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced['url'], df_balanced['label_binary'],
        test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # --- Pipeline definition ---
    combined_features = FeatureUnion([
        # Size optimization: 5000 -> 1500 TF-IDF tokens; structural features
        # carry the rest of the signal.
        ('text_features', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None, max_features=1500)),
        ('structural_features', StructuralFeatureExtractor())
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        # Size optimization: bonsai trees (depth/leaf caps keep the ONNX small)
        ('clf', RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            # class_weight='balanced' removed: undersampled variant is balanced
        ))
    ])

    variant = "Undersampled" if args.undersample else "Full dataset"
    print(f"\nTraining Optimized Hybrid Model ({variant})...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Selesai: {(time.time()-start)/60:.2f} menit")

    print("\nEvaluasi:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Phishing']))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, args.out, compress=3)
    print(f"Model disimpan: '{args.out}' (compressed)")


if __name__ == "__main__":
    main()
