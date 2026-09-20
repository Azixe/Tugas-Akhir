"""Interactive RF pipeline test — loads a trained .pkl and classifies URLs.

Usage (run from the repo root):

    python hybrid/test_hybrid.py
    python hybrid/test_hybrid.py --model rf_models/rf_full.pkl
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import alias_legacy_main  # noqa: E402

# Manual first-layer whitelist
WHITELIST = ['google.com', 'youtube.com', 'facebook.com', 'kompas.com', 'whatsapp.com']


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive test for a trained RF pipeline.")
    parser.add_argument('--model', default=os.path.join('rf_models', 'phishing_optimized.pkl'),
                        help="Path to the .pkl pipeline (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Legacy pickles reference __main__.make_tokens etc.; new ones reference features.*
    alias_legacy_main()

    try:
        print(f"Memuat model: {args.model}")
        pipeline = joblib.load(args.model)
        print("Siap digunakan.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- PHISHING DETECTOR (OPTIMIZED) ---")
    print("Ketik 'exit' untuk keluar.")

    while True:
        url = input("\nMasukkan URL: ")
        if url.strip().lower() == 'exit':
            break
        if not url.strip():
            continue

        clean_domain = url.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
        if clean_domain in WHITELIST:
            print(f"\033[92m[SAFE] {url} (Whitelisted)\033[0m")
            continue

        try:
            prob_phishing = pipeline.predict_proba([url])[0][1] * 100

            # Risk levels (same thresholds as the extension)
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


if __name__ == "__main__":
    main()
