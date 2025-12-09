import joblib
import sys
import numpy as np
import math
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. Definisi Helper & Class (WAJIB SAMA PERSIS DENGAN TRAINING) ---

def shannon_entropy(data):
    # Menghitung ketidakteraturan string
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

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
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
        
        for url in X:
            s_url = str(url).lower()
            
            # --- Logic Fitur (Harus urut sesuai training) ---
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
    if 'www' in total_tokens: total_tokens.remove('www') # Update: Hapus www
    return total_tokens

# --- 2. Load & Predict ---
try:
    print("Memuat Model Optimized...")
    # Pastikan nama file sesuai dengan yang disimpan di training terakhir
    pipeline = joblib.load('phishing_optimized.pkl') 
    print("Siap digunakan.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

print("\n--- PHISHING DETECTOR (OPTIMIZED) ---")
print("Ketik 'exit' untuk keluar.")

while True:
    url = input("\nMasukkan URL: ")
    if url.strip().lower() == 'exit': break
    if not url.strip(): continue
    
    # Whitelist Manual (Filter lapis pertama)
    whitelist = ['google.com', 'youtube.com', 'facebook.com', 'kompas.com', 'whatsapp.com']
    clean_domain = url.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
    
    if clean_domain in whitelist:
        print(f"\033[92m[SAFE] {url} (Whitelisted)\033[0m")
        continue

    # Prediksi
    try:
        prob_phishing = pipeline.predict_proba([url])[0][1] * 100
        
        # Logika Threshold (Risk Levels)
        if prob_phishing > 80:
            print(f"\033[91m[DANGER] PHISHING DETECTED\033[0m")
            print(f"Confidence: {prob_phishing:.2f}%")
        elif prob_phishing > 50:
            print(f"\033[93m[WARNING] SUSPICIOUS URL\033[0m")
            print(f"Confidence: {prob_phishing:.2f}% (Perlu verifikasi manual)")
        else:
            print(f"\033[92m[SAFE] LEGITIMATE\033[0m")
            print(f"Confidence Aman: {100 - prob_phishing:.2f}%")
            
    except Exception as e:
        print(f"Gagal memproses URL: {e}")