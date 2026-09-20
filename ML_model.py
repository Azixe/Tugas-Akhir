import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import time

# --- 1. Definisi Fungsi (Harus didefinisikan sebelum dipanggil) ---
def make_tokens(f):
    # Tokenizer khusus untuk URL
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

# --- 2. Load & Persiapan Data ---
try:
    # Pastikan file berada di direktori yang sama dengan script ini
    df = pd.read_csv('URL dataset.csv')
    print(f"Dataset dimuat: {df.shape}")
except FileNotFoundError:
    print("CRITICAL ERROR: File csv tidak ditemukan.")
    exit()

# Mapping label: Phishing = 1, Legitimate = 0
# Pastikan nama kolom di CSV Anda benar ('type' dan 'url')
df['label_binary'] = df['type'].map({'phishing': 1, 'legitimate': 0})

X = df['url']
y = df['label_binary']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Membangun Arsitektur Model ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=make_tokens, token_pattern=None)), 
    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1)) 
])

# --- 4. Eksekusi Training (Benchmarking) ---
print(f"\nMulai training pada {len(X_train)} data... (Sambil ngopi dulu, Gan)")
start_time = time.time()

# Proses berat terjadi di sini
pipeline.fit(X_train, y_train)

end_time = time.time()
duration_minutes = (end_time - start_time) / 60
print(f"Selesai! Waktu training: {duration_minutes:.2f} menit.")

# --- 5. Evaluasi ---
print("\n--- Evaluasi Model ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# --- 6. Simpan Model ---
model_filename = 'phishing_detector_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"\nModel telah disimpan sebagai '{model_filename}'")