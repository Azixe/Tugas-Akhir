import joblib
import json
import numpy as np
import sys
import math
from collections import Counter
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINISI HELPER (COPY-PASTE DARI TRAINING, JANGAN IMPORT) ---

# Fungsi entropy harus ada di sini karena dipanggil StructuralFeatureExtractor
def shannon_entropy(data):
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

# Class Struktural
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

# Fungsi Tokenizer
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

# --- 2. LOAD & EXPORT LOGIC ---

try:
    print("Memuat model pipeline...")
    # Karena semua fungsi di atas sudah didefinisikan, joblib akan mengenalnya
    pipeline = joblib.load('phishing_optimized.pkl')
    print("Model berhasil dimuat.")
except AttributeError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Pastikan nama fungsi SAMA PERSIS dengan file training.")
    sys.exit()
except FileNotFoundError:
    print("Error: File .pkl tidak ditemukan.")
    sys.exit()

# Ambil komponen
feature_union = pipeline.named_steps['features']
tfidf_model = feature_union.transformer_list[0][1] 
rf_model = pipeline.named_steps['clf']             

# --- Ekspor JSON (TF-IDF) ---
print("Mengekspor TF-IDF Vocabulary...")
vocab_raw = tfidf_model.vocabulary_
# Pakai int() untuk konversi numpy.int32 ke python int biasa
vocab_clean = {k: int(v) for k, v in vocab_raw.items()} 
idf_clean = tfidf_model.idf_.tolist()

tfidf_data = {
    "vocabulary": vocab_clean,
    "idf": idf_clean,
    "norm": tfidf_model.norm, 
    "use_idf": tfidf_model.use_idf,
    "smooth_idf": tfidf_model.smooth_idf,
    "sublinear_tf": tfidf_model.sublinear_tf
}

with open("tfidf_data.json", "w") as f:
    json.dump(tfidf_data, f)
print("-> tfidf_data.json OK.")

# --- Ekspor ONNX (Random Forest) ---
# Jumlah fitur = Vocab Size + 9 Fitur Manual
n_features = len(vocab_clean) + 9 
print(f"Jumlah fitur ONNX: {n_features}")

initial_type = [('float_input', FloatTensorType([None, n_features]))]

print("Mengonversi ke ONNX...")

# FIXED: Disable ZipMap output (tidak di-support browser)
onnx_model = convert_sklearn(
    rf_model, 
    initial_types=initial_type,
    target_opset=12,
    options={id(rf_model): {'zipmap': False}}  
)

with open("phishing_rf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("-> phishing_rf.onnx OK.")

print("\nSUKSES! File siap untuk Web Extension.")