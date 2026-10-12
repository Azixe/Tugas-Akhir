import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import joblib
import time

# --- 1. Definisi Class & Fungsi (Wajib ada di scope global) ---

# Helper function untuk deteksi IP
def user_is_ip(url):
    try:
        domain = url.split('/')[0] 
        parts = domain.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return True
        return False
    except:
        return False

# Class ekstraktor fitur manual (Panjang, titik, @, dll)
class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for url in X:
            s_url = str(url)
            
            # --- Logic Fitur Manual ---
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            # Menghitung rasio digit terhadap panjang total (indikator obfuscation)
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            
            has_ip = 1 if user_is_ip(s_url) else 0
            
            features.append([length, dot_count, slash_count, dash_count, at_count, digit_ratio, has_ip])
            
        return np.array(features)

# Tokenizer Teks
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
    if 'com' in total_tokens:
        total_tokens.remove('com')
    return total_tokens

# --- 2. Eksekusi Utama ---

if __name__ == "__main__":
    # Load Data
    try:
        print("Memuat dataset...")
        df = pd.read_csv('hybrid/URL dataset.csv') # Sesuaikan nama file
    except FileNotFoundError:
        print("Error: File CSV tidak ditemukan.")
        exit()

    # Label Mapping
    df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})
    X = df['url']
    y = df['label_binary']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Definisi Pipeline Hybrid ---
    # Ini bagian modifikasi utamanya: Menggabungkan Teks + Struktur
    combined_features = FeatureUnion([
        # Path 1: Analisis Teks (TF-IDF)
        # max_features=5000 agar RAM tidak meledak & training lebih cepat
        ('text_features', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None, max_features=5000)), 
        
        # Path 2: Analisis Struktur (Class buatan kita)
        ('structural_features', StructuralFeatureExtractor())
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
    ])

    # --- 4. Training ---
    print(f"Mulai training Hybrid Model pada {len(X_train)} data...")
    start_time = time.time()
    
    pipeline.fit(X_train, y_train)
    
    print(f"Selesai! Waktu: {(time.time() - start_time) / 60:.2f} menit.")

    # --- 5. Evaluasi ---
    print("\n--- Evaluasi ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # --- 6. Simpan Model ---
    joblib.dump(pipeline, 'phishing_hybrid_model.pkl')
    print("Model disimpan sebagai 'phishing_hybrid_model.pkl'")