import joblib
import sys

# --- 1. Definisi Ulang Fungsi Tokenizer ---
# Fungsi ini HARUS ada dan identik dengan saat training
# agar pickle bisa memetakan referensi fungsi 'make_tokens'
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

# --- 2. Load Model ---
try:
    print("Memuat model...")
    pipeline = joblib.load('phishing_detector_model.pkl')
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("Error: File .pkl tidak ditemukan.")
    sys.exit()

# --- 3. Loop Pengujian ---
print("\n--- PHISHING DETECTOR CLI ---")
print("Ketik 'exit' untuk keluar.")

while True:
    url_input = input("\nMasukkan URL: ")
    
    if url_input.lower() == 'exit':
        break
        
    if not url_input:
        continue

    # Prediksi
    # Input harus dalam bentuk list/iterable
    prediction = pipeline.predict([url_input])[0]
    probability = pipeline.predict_proba([url_input])
    
    # Ambil probabilitas kelas prediksi (max probability)
    confidence = max(probability[0]) * 100
    
    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    
    # Output dengan warna (ANSI codes)
    color = "\033[91m" if prediction == 1 else "\033[92m" # Merah jika Phishing, Hijau jika Legit
    reset = "\033[0m"
    
    print(f"Hasil: {color}{label}{reset}")
    print(f"Confidence: {confidence:.2f}%")