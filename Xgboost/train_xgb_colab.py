"""
XGBoost Phishing URL Detection Pipeline — Google Colab Version (Streamlined)
=============================================================================
Run on Google Colab with GPU runtime.

Setup:
1. Colab: Runtime -> Change runtime type -> GPU (T4)
2. Upload 'URL dataset.csv' to /content/
3. Upload this file and run: !python train_xgb_colab.py
"""

import pandas as pd
import numpy as np
import math
import time
import warnings
import os
import gc
from collections import Counter

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

# === GPU Detection ===
try:
    import subprocess
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
    if gpu_info.returncode == 0:
        print(f"GPU detected: {gpu_info.stdout.strip()}")
        USE_GPU = True
    else:
        USE_GPU = False
        print("No GPU detected, using CPU")
except:
    USE_GPU = False
    print("No GPU detected, using CPU")

XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
print(f"XGBoost device: {XGB_DEVICE}\n")

# ============================================================
# HELPER FUNCTIONS & CLASSES (identical to RF training)
# ============================================================

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
        for url in X:
            s_url = str(url).lower()
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            entropy = shannon_entropy(s_url)
            is_common_tld = 0
            for tld in common_tlds:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break
            try:
                clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
                subdomain_level = clean.count('.')
            except:
                subdomain_level = 0
            features.append([length, dot_count, slash_count, dash_count, at_count, 
                             digit_ratio, entropy, is_common_tld, subdomain_level])
        return np.array(features)

def make_tokens(f):
    tokens_by_slash = str(f).encode('utf-8').decode('utf-8').split('/')
    total_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_dot = []
        for j in range(0, len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_dot = tokens_dot + temp_tokens
        total_tokens = total_tokens + tokens + tokens_dot
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens: total_tokens.remove('com')
    if 'www' in total_tokens: total_tokens.remove('www')
    return total_tokens

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    total_start = time.time()
    
    # --- Find dataset ---
    dataset_paths = [
        'URL dataset.csv',
        '/content/URL dataset.csv',
        '/content/drive/MyDrive/URL dataset.csv',
    ]
    df = None
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            df = pd.read_csv(path)
            break
    if df is None:
        print("Dataset not found! Upload 'URL dataset.csv' to Colab.")
        exit()

    # ============================================================
    # STAGE 1: Preprocessing & Undersampling
    # ============================================================
    print("=" * 60)
    print("STAGE 1: Data Preprocessing & Undersampling")
    print("=" * 60)
    
    print(f"Dataset awal: {len(df)} baris")
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"Setelah cleaning: {len(df)} baris")
    
    df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})
    df_phishing = df[df['label_binary'] == 1]
    df_legitimate = df[df['label_binary'] == 0]
    n_minority = len(df_phishing)
    
    print(f"Sebelum: Legit={len(df_legitimate)}, Phishing={len(df_phishing)}")
    
    df_legitimate_under = df_legitimate.sample(n=n_minority, random_state=42)
    df_balanced = pd.concat([df_legitimate_under, df_phishing])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Setelah undersampling: {len(df_balanced)} total (1:1 ratio)")
    
    del df, df_phishing, df_legitimate, df_legitimate_under
    gc.collect()

    # ============================================================
    # STAGE 2: Feature Extraction
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 2: Feature Extraction (TF-IDF + Structural)")
    print("=" * 60)
    step_start = time.time()
    
    feature_extractor = FeatureUnion([
        ('text_features', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None, max_features=1500)),
        ('structural_features', StructuralFeatureExtractor())
    ])
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_balanced['url'], df_balanced['label_binary'],
        test_size=0.2, random_state=42, stratify=df_balanced['label_binary']
    )
    
    del df_balanced
    gc.collect()
    
    print(f"Extracting features from {len(X_train_raw)} train + {len(X_test_raw)} test URLs...")
    X_train = feature_extractor.fit_transform(X_train_raw, y_train)
    X_test = feature_extractor.transform(X_test_raw)
    
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray().astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray().astype(np.float32)
    else:
        X_test = X_test.astype(np.float32)
    
    n_original = X_train.shape[1]
    print(f"Feature matrix: {X_train.shape} ({X_train.dtype}), ~{X_train.nbytes/1024/1024:.0f}MB")
    print(f"Time: {time.time()-step_start:.1f}s")

    # ============================================================
    # STAGE 3: StandardScaler
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 3: StandardScaler")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()
    print("Done.")

    # ============================================================
    # STAGE 4: SelectKBest (ANOVA F-value)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 4: SelectKBest — ANOVA F-value")
    print("=" * 60)
    step_start = time.time()
    
    # Compute F-scores for all features at once (fast, no CV needed)
    selector_all = SelectKBest(score_func=f_classif, k='all')
    selector_all.fit(X_train_scaled, y_train)
    f_scores = selector_all.scores_
    
    # Find optimal k by testing a few candidates with a quick single train/eval
    k_candidates = [50, 100, 200, 300, 500, 750, 1000, n_original]
    print("Testing k candidates (single train/eval each):")
    
    best_k, best_acc = None, 0
    for k in k_candidates:
        sel = SelectKBest(score_func=f_classif, k=k)
        X_tr_k = sel.fit_transform(X_train_scaled, y_train)
        X_te_k = sel.transform(X_test_scaled)
        
        quick_xgb = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            tree_method='hist', device=XGB_DEVICE,
            random_state=42, verbosity=0
        )
        quick_xgb.fit(X_tr_k, y_train)
        acc = accuracy_score(y_test, quick_xgb.predict(X_te_k))
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_k = k
            marker = " <-- BEST"
        print(f"  k={k:>5}: accuracy={acc:.4f}{marker}")
    
    # Apply best k
    selector_kbest = SelectKBest(score_func=f_classif, k=best_k)
    X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
    X_test_kbest = selector_kbest.transform(X_test_scaled)
    
    n_after_kbest = X_train_kbest.shape[1]
    print(f"\n>> Best k={best_k}, Features: {n_original} -> {n_after_kbest}")
    print(f"Time: {time.time()-step_start:.1f}s")
    
    del X_train_scaled, X_test_scaled
    gc.collect()

    # ============================================================
    # STAGE 5: Feature Importance Pruning (replaces slow RFECV)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 5: XGBoost Feature Importance Pruning")
    print("=" * 60)
    step_start = time.time()
    
    # Train a single XGBoost to get feature importances
    importance_xgb = XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.1,
        tree_method='hist', device=XGB_DEVICE,
        random_state=42, verbosity=0
    )
    importance_xgb.fit(X_train_kbest, y_train)
    
    importances = importance_xgb.feature_importances_
    
    # Keep features with above-median importance
    threshold = np.median(importances)
    important_mask = importances > threshold
    n_important = important_mask.sum()
    
    # Don't prune if too few remain
    if n_important < 20:
        print(f"Only {n_important} features above median. Keeping top 50% instead.")
        top_indices = np.argsort(importances)[-max(20, n_after_kbest//2):]
        important_mask = np.zeros(len(importances), dtype=bool)
        important_mask[top_indices] = True
        n_important = important_mask.sum()
    
    X_train_imp = X_train_kbest[:, important_mask]
    X_test_imp = X_test_kbest[:, important_mask]
    
    n_after_importance = X_train_imp.shape[1]
    print(f"Feature importance threshold: {threshold:.6f}")
    print(f"Features: {n_after_kbest} -> {n_after_importance} (kept above-median importance)")
    print(f"Time: {time.time()-step_start:.1f}s")
    
    del X_train_kbest, X_test_kbest
    gc.collect()

    # ============================================================
    # STAGE 6: PCA (95% variance)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 6: PCA (95% explained variance)")
    print("=" * 60)
    step_start = time.time()
    
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_imp)
    X_test_pca = pca.transform(X_test_imp)
    
    n_after_pca = X_train_pca.shape[1]
    print(f"Features: {n_after_importance} -> {n_after_pca}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Time: {time.time()-step_start:.1f}s")
    
    del X_train_imp, X_test_imp
    gc.collect()

    # ============================================================
    # STAGE 7: XGBoost Hyperparameter Tuning (RandomizedSearchCV)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 7: XGBoost Hyperparameter Tuning (RandomizedSearchCV)")
    print("=" * 60)
    step_start = time.time()
    
    xgb_base = XGBClassifier(
        tree_method='hist', device=XGB_DEVICE,
        random_state=42, verbosity=0, eval_metric='logloss'
    )
    
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 300, 500, 700],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb_base, param_distributions,
        n_iter=30, cv=cv_5, scoring='accuracy',
        n_jobs=1, verbose=2, random_state=42
    )
    
    print(f"Searching 30 random combinations x 5-fold CV = 150 fits...")
    print(f"Device: {XGB_DEVICE}")
    random_search.fit(X_train_pca, y_train)
    
    best_params = random_search.best_params_
    print(f"\n>> Best accuracy: {random_search.best_score_:.4f}")
    print(f"Best params: {best_params}")
    print(f"Time: {time.time()-step_start:.1f}s")

    # ============================================================
    # STAGE 8: Final Evaluation
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 8: Final Evaluation on Test Set")
    print("=" * 60)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_pca)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Phishing']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature reduction summary
    print(f"\n{'=' * 60}")
    print("FEATURE REDUCTION SUMMARY")
    print("=" * 60)
    print(f"  Original:              {n_original}")
    print(f"  After SelectKBest:     {n_after_kbest}")
    print(f"  After Importance:      {n_after_importance}")
    print(f"  After PCA (95%):       {n_after_pca}")

    # Best hyperparameters
    print(f"\n{'=' * 60}")
    print("BEST HYPERPARAMETERS")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ============================================================
    # STAGE 9: Save
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STAGE 9: Saving")
    print("=" * 60)
    
    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'selector_kbest': selector_kbest,
        'important_mask': important_mask,
        'pca': pca,
        'feature_extractor': feature_extractor,
        'best_params': best_params,
        'feature_counts': {
            'original': n_original,
            'after_kbest': n_after_kbest,
            'after_importance': n_after_importance,
            'after_pca': n_after_pca,
        }
    }
    
    joblib.dump(artifacts, 'phishing_xgb_pipeline.pkl', compress=3)
    print(">> Saved: phishing_xgb_pipeline.pkl")
    
    joblib.dump(best_model, 'phishing_xgb_model.pkl', compress=3)
    print(">> Saved: phishing_xgb_model.pkl")
    
    total_time = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"TOTAL RUNTIME: {total_time:.1f} minutes")
    print("=" * 60)
    
    # Colab auto-download
    try:
        from google.colab import files
        print("\nDownloading files...")
        files.download('phishing_xgb_pipeline.pkl')
        files.download('phishing_xgb_model.pkl')
    except ImportError:
        print("\nFiles saved to current directory.")
