import joblib
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- Definisi Ulang Class & Fungsi (HARUS SAMA PERSIS DENGAN TRAINING) ---
def user_is_ip(url):
    try:
        domain = url.split('/')[0] 
        parts = domain.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return True
        return False
    except:
        return False

class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        for url in X:
            s_url = str(url)
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            has_ip = 1 if user_is_ip(s_url) else 0
            features.append([length, dot_count, slash_count, dash_count, at_count, digit_ratio, has_ip])
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
    if 'com' in total_tokens:
        total_tokens.remove('com')
    return total_tokens

# --- Load & Predict ---
try:
    print("Memuat Hybrid Model...")
    pipeline = joblib.load('phishing_hybrid_model.pkl')
    print("Siap.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

while True:
    url = input("\nMasukkan URL (atau 'exit'): ")
    if url == 'exit': break
    if not url: continue
    
    # Whitelist Manual (Pragmatis)
    whitelist = ['google.com', 'youtube.com', 'facebook.com', 'kompas.com']
    clean_url = url.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
    
    if clean_url in whitelist:
        print(f"\033[92mHasil: LEGITIMATE (Whitelisted)\033[0m")
        continue

    pred = pipeline.predict([url])[0]
    prob = max(pipeline.predict_proba([url])[0]) * 100
    
    if pred == 1:
        print(f"Hasil: \033[91mPHISHING\033[0m (Confidence: {prob:.2f}%)")
        print("-> Analisis: URL ini mencurigakan secara struktur atau teks.")
    else:
        print(f"Hasil: \033[92mLEGITIMATE\033[0m (Confidence: {prob:.2f}%)")