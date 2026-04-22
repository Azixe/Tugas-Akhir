"""
XGBoost Phishing URL Detection Pipeline
========================================
Pipeline: StandardScaler → SelectKBest (GridSearch) → RFECV → PCA(95%) → XGBoost (Tuned)
All transformers fit on train set only to prevent data leakage.
"""

import pandas as pd
import numpy as np
import math
import time
import warnings
from collections import Counter

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# 1. HELPER FUNCTIONS & CLASSES (identical to RF training)
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
# 2. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    total_start = time.time()
    
    # --- 2.1 Load & Undersample ---
    print("=" * 60)
    print("STAGE 1: Data Loading & Preprocessing")
    print("=" * 60)
    
    try:
        df = pd.read_csv('URL dataset.csv')
    except FileNotFoundError:
        print("Dataset not found! Run from project root directory.")
        exit()

    # Data Cleaning
    print(f"Dataset awal: {len(df)} baris")
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"Setelah cleaning: {len(df)} baris")
    
    # Label Encoding
    df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})
    
    # Undersampling (sesuai proposal Bab 3.3.3)
    df_phishing = df[df['label_binary'] == 1]
    df_legitimate = df[df['label_binary'] == 0]
    n_minority = len(df_phishing)
    
    print(f"\nSebelum undersampling: Legit={len(df_legitimate)}, Phishing={len(df_phishing)}")
    
    df_legitimate_undersampled = df_legitimate.sample(n=n_minority, random_state=42)
    df_balanced = pd.concat([df_legitimate_undersampled, df_phishing])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Setelah undersampling: Legit={len(df_balanced[df_balanced['label_binary']==0])}, "
          f"Phishing={len(df_balanced[df_balanced['label_binary']==1])}, Total={len(df_balanced)}")

    # --- 2.2 Feature Extraction ---
    print(f"\n{'=' * 60}")
    print("STAGE 2: Feature Extraction (TF-IDF + Structural)")
    print("=" * 60)
    
    step_start = time.time()
    
    feature_extractor = FeatureUnion([
        ('text_features', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None, max_features=1500)),
        ('structural_features', StructuralFeatureExtractor())
    ])
    
    X_all = df_balanced['url']
    y_all = df_balanced['label_binary']
    
    # Train/Test Split (80/20) — split BEFORE fitting feature extractor
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Fit feature extractor on train set only, transform both
    print(f"Extracting features from {len(X_train_raw)} train URLs...")
    X_train = feature_extractor.fit_transform(X_train_raw, y_train)
    print(f"Extracting features from {len(X_test_raw)} test URLs...")
    X_test = feature_extractor.transform(X_test_raw)
    
    # Convert sparse to dense if needed
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()
    
    n_original_features = X_train.shape[1]
    print(f"Feature matrix: {X_train.shape} (train), {X_test.shape} (test)")
    print(f"Feature extraction: {time.time()-step_start:.1f}s")

    # --- 2.3 StandardScaler ---
    print(f"\n{'=' * 60}")
    print("STAGE 3: StandardScaler Normalization")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled: mean~{X_train_scaled.mean():.4f}, std~{X_train_scaled.std():.4f}")

    # --- 2.4 SelectKBest with Grid Search ---
    print(f"\n{'=' * 60}")
    print("STAGE 4: SelectKBest (ANOVA F-value) — Grid Search for optimal k")
    print("=" * 60)
    
    step_start = time.time()
    
    # Candidate k values to search
    k_candidates = [50, 100, 200, 300, 500, 750, 1000, n_original_features]
    
    # Pipeline: SelectKBest → XGBoost (lightweight config for speed)
    skb_pipeline = Pipeline([
        ('selectkbest', SelectKBest(score_func=f_classif)),
        ('xgb', XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            tree_method='hist', nthread=-1, random_state=42, verbosity=0
        ))
    ])
    
    skb_param_grid = {'selectkbest__k': k_candidates}
    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Searching k in {k_candidates} with 5-fold CV...")
    skb_search = GridSearchCV(
        skb_pipeline, skb_param_grid, cv=cv_5, scoring='accuracy',
        n_jobs=4, verbose=1
    )
    skb_search.fit(X_train_scaled, y_train)
    
    best_k = skb_search.best_params_['selectkbest__k']
    print(f"\n✓ Best k = {best_k} (accuracy: {skb_search.best_score_:.4f})")
    
    # Print all k scores
    print("\nAll k scores:")
    for k, score in zip(k_candidates, 
                         [skb_search.cv_results_['mean_test_score'][i] for i in range(len(k_candidates))]):
        marker = " <-- BEST" if k == best_k else ""
        print(f"  k={k:>5}: accuracy={score:.4f}{marker}")
    
    # Apply SelectKBest with best k
    selector_kbest = SelectKBest(score_func=f_classif, k=best_k)
    X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
    X_test_kbest = selector_kbest.transform(X_test_scaled)
    
    n_after_kbest = X_train_kbest.shape[1]
    print(f"\nFeatures: {n_original_features} → {n_after_kbest} (post-SelectKBest)")
    print(f"SelectKBest search: {time.time()-step_start:.1f}s")

    # --- 2.5 RFECV ---
    print(f"\n{'=' * 60}")
    print("STAGE 5: RFECV (Recursive Feature Elimination with CV)")
    print("=" * 60)
    
    step_start = time.time()
    
    rfecv_estimator = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        tree_method='hist', nthread=-1, random_state=42, verbosity=0
    )
    
    # step=50 to reduce iterations significantly
    rfecv = RFECV(
        estimator=rfecv_estimator,
        step=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        min_features_to_select=10,
        n_jobs=4,
        verbose=1
    )
    
    print(f"Running RFECV on {n_after_kbest} features (step=50, 5-fold CV)...")
    print("(This may take a while...)")
    rfecv.fit(X_train_kbest, y_train)
    
    X_train_rfecv = rfecv.transform(X_train_kbest)
    X_test_rfecv = rfecv.transform(X_test_kbest)
    
    n_after_rfecv = X_train_rfecv.shape[1]
    print(f"\n✓ Optimal features: {rfecv.n_features_}")
    print(f"Features: {n_after_kbest} → {n_after_rfecv} (post-RFECV)")
    print(f"RFECV: {time.time()-step_start:.1f}s")

    # --- 2.6 PCA ---
    print(f"\n{'=' * 60}")
    print("STAGE 6: PCA (95% explained variance)")
    print("=" * 60)
    
    step_start = time.time()
    
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_rfecv)
    X_test_pca = pca.transform(X_test_rfecv)
    
    n_after_pca = X_train_pca.shape[1]
    print(f"Features: {n_after_rfecv} → {n_after_pca} (post-PCA)")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA: {time.time()-step_start:.1f}s")

    # --- 2.7 XGBoost Hyperparameter Tuning ---
    print(f"\n{'=' * 60}")
    print("STAGE 7A: XGBoost — RandomizedSearchCV (coarse search)")
    print("=" * 60)
    
    step_start = time.time()
    
    xgb_base = XGBClassifier(
        tree_method='hist', nthread=-1, random_state=42, verbosity=0,
        eval_metric='logloss'
    )
    
    # Full search space (as specified)
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 300, 500, 700],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Total possible combinations: 5*4*4*3*3*3*3 = 9720
    # RandomizedSearchCV samples n_iter=50 combinations
    random_search = RandomizedSearchCV(
        xgb_base, param_distributions, 
        n_iter=30, cv=cv_5, scoring='accuracy',
        n_jobs=4, verbose=1, random_state=42
    )
    
    print(f"Searching 50 random combinations with 10-fold CV...")
    random_search.fit(X_train_pca, y_train)
    
    best_random = random_search.best_params_
    print(f"\n✓ Best RandomizedSearch accuracy: {random_search.best_score_:.4f}")
    print(f"Best params: {best_random}")
    print(f"RandomizedSearchCV: {time.time()-step_start:.1f}s")
    
    # --- 2.7b GridSearchCV (fine-tuning around best params) ---
    print(f"\n{'=' * 60}")
    print("STAGE 7B: XGBoost — GridSearchCV (fine-tuning)")
    print("=" * 60)
    
    step_start = time.time()
    
    # Build refined grid around the best params from RandomizedSearch
    def refine_param(best_val, candidates):
        """Create a narrow grid around the best value."""
        idx = candidates.index(best_val) if best_val in candidates else 0
        start = max(0, idx - 1)
        end = min(len(candidates), idx + 2)
        return candidates[start:end]
    
    refined_grid = {
        'learning_rate': refine_param(best_random['learning_rate'], [0.01, 0.05, 0.1, 0.2, 0.3]),
        'max_depth': refine_param(best_random['max_depth'], [3, 5, 7, 9]),
        'n_estimators': refine_param(best_random['n_estimators'], [100, 300, 500, 700]),
        'subsample': refine_param(best_random['subsample'], [0.7, 0.8, 1.0]),
        'colsample_bytree': refine_param(best_random['colsample_bytree'], [0.7, 0.8, 1.0]),
        'reg_alpha': refine_param(best_random['reg_alpha'], [0, 0.1, 0.5]),
        'reg_lambda': refine_param(best_random['reg_lambda'], [1, 1.5, 2]),
    }
    
    n_combos = 1
    for v in refined_grid.values():
        n_combos *= len(v)
    print(f"Refined grid: {n_combos} combinations × 10 folds = {n_combos * 10} fits")
    print(f"Grid: {refined_grid}")
    
    grid_search = GridSearchCV(
        XGBClassifier(tree_method='hist', nthread=-1, random_state=42, verbosity=0, eval_metric='logloss'),
        refined_grid, cv=cv_5, scoring='accuracy',
        n_jobs=4, verbose=1
    )
    
    grid_search.fit(X_train_pca, y_train)
    
    best_params = grid_search.best_params_
    print(f"\n✓ Best GridSearch accuracy: {grid_search.best_score_:.4f}")
    print(f"Best params: {best_params}")
    print(f"GridSearchCV: {time.time()-step_start:.1f}s")

    # --- 2.8 Final Evaluation ---
    print(f"\n{'=' * 60}")
    print("STAGE 8: Final Evaluation on Test Set")
    print("=" * 60)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_pca)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Phishing']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # --- Feature count summary ---
    print(f"\n{'=' * 60}")
    print("FEATURE REDUCTION SUMMARY")
    print("=" * 60)
    print(f"  Original features:     {n_original_features}")
    print(f"  After SelectKBest:     {n_after_kbest}")
    print(f"  After RFECV:           {n_after_rfecv}")
    print(f"  After PCA (95% var):   {n_after_pca}")
    
    # --- Best hyperparameters ---
    print(f"\n{'=' * 60}")
    print("BEST HYPERPARAMETERS")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # --- 2.9 Save artifacts ---
    print(f"\n{'=' * 60}")
    print("STAGE 9: Saving Model & Preprocessing Artifacts")
    print("=" * 60)
    
    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'selector_kbest': selector_kbest,
        'rfecv': rfecv,
        'pca': pca,
        'feature_extractor': feature_extractor,
        'best_params': best_params,
        'feature_counts': {
            'original': n_original_features,
            'after_kbest': n_after_kbest,
            'after_rfecv': n_after_rfecv,
            'after_pca': n_after_pca,
        }
    }
    
    joblib.dump(artifacts, 'phishing_xgb_pipeline.pkl', compress=3)
    print("✓ Saved: phishing_xgb_pipeline.pkl")
    
    # Also save just the final model for ONNX export later
    joblib.dump(best_model, 'phishing_xgb_model.pkl', compress=3)
    print("✓ Saved: phishing_xgb_model.pkl")
    
    total_time = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"TOTAL RUNTIME: {total_time:.1f} minutes")
    print("=" * 60)

