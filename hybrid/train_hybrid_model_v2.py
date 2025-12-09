import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import joblib
import time

# --- 1. Definisi Helper & Class ---

def shannon_entropy(data):
    # Menghitung "ketidakteraturan" string.
    # URL legit biasanya terbaca (entropy rendah).
    # URL acak (dga/obfuscated) seperti 'x7z-9q.com' punya entropy tinggi.
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
        # TLD yang umum dan terpercaya (bisa ditambah)
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
        
        for url in X:
            s_url = str(url).lower()
            
            # 1. Fitur Dasar
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            
            # 2. Fitur Obfuscation
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            
            # 3. Shannon Entropy (PENTING untuk deteksi random chars)
            entropy = shannon_entropy(s_url)
            
            # 4. Deteksi TLD Aneh (xyz, top, club sering dipakai phishing)
            is_common_tld = 0
            for tld in common_tlds:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break
            
            # 5. Jumlah Subdomain (domain.com = 2 bagian, a.b.c.com = 4 bagian)
            # Phishing sering pakai: secure.login.apple.com.update.tk
            try:
                # Hapus protokol
                clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
                subdomain_level = clean.count('.')
            except:
                subdomain_level = 0
            
            features.append([length, dot_count, slash_count, dash_count, at_count, 
                             digit_ratio, entropy, is_common_tld, subdomain_level])
            
        return np.array(features)

def make_tokens(f):
    # Tokenizer sama seperti sebelumnya
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
    if 'www' in total_tokens: total_tokens.remove('www') # Hapus www juga
    return total_tokens

# --- 2. Eksekusi ---

if __name__ == "__main__":
    try:
        df = pd.read_csv('URL dataset.csv')
    except:
        print("Dataset not found")
        exit()

    df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})
    X_train, X_test, y_train, y_test = train_test_split(df['url'], df['label_binary'], test_size=0.2, random_state=42)

    combined_features = FeatureUnion([
        # OPTIMASI SIZE 1: Kurangi max_features drastis (5000 -> 1500)
        # Kita mengandalkan fitur struktur, jadi tidak butuh terlalu banyak kata.
        ('text_features', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None, max_features=1500)), 
        ('structural_features', StructuralFeatureExtractor())
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        # OPTIMASI SIZE 2: Bonsai Pohon Random Forest
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            n_jobs=-1,
            max_depth=15,          # Batasi kedalaman pohon (mencegah file bengkak)
            min_samples_split=10,  # Jangan split jika sampel < 10 (mencegah overfitting detail remeh)
            min_samples_leaf=4,    # Daun minimal 4 sampel
            class_weight='balanced'
        ))
    ])

    print("Training Optimized Hybrid Model...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Selesai: {(time.time()-start)/60:.2f} menit")

    print("\nEvaluasi:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Phishing']))

    # OPTIMASI SIZE 3: Kompresi saat save (level 3 cukup optimal speed/size)
    joblib.dump(pipeline, 'phishing_optimized.pkl', compress=3)
    print("Model disimpan: 'phishing_optimized.pkl' (Compressed)")