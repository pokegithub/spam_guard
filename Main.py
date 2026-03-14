print("Booting up Spam & Phishing Detector...")
print("[1/3] Verifying NLTK local caches...")

import os
import re
import sys
import joblib
import nltk

def ensure_nltk_data():
    packages = {
        'corpora/stopwords':    'stopwords',
        'corpora/wordnet':      'wordnet',
        'tokenizers/punkt_tab': 'punkt_tab'
    }
    for path, package in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"   -> Downloading {package}...")
            nltk.download(package, quiet=True)

ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("[2/3] Loading machine learning models...")

# Pre-compile regex patterns for speed
url_pattern      = re.compile(r'(https?://\S+|www\.\S+)')
email_pattern    = re.compile(r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}')
num_pattern      = re.compile(r'\d+')
currency_pattern = re.compile(r'[$£€]')
punct_pattern    = re.compile(r'[^a-z\s]')

lemmatizer = WordNetLemmatizer()

# Build a local copy so we don't modify the shared NLTK set
stop_words = set(stopwords.words('english'))
stop_words.update(['http', 'https', 'www', 'com', 'org', 'net', 'co'])

def clean(text):
    text = text.lower()
    text = url_pattern.sub('webaddr', text)
    text = email_pattern.sub('emailaddr', text)
    text = num_pattern.sub('num', text)
    text = currency_pattern.sub('currency ', text)
    text = punct_pattern.sub('', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

# Lazy-load cv2 and pytesseract only if the user provides an image
cv2_engine      = None
pytesseract_eng = None

def extract_image_text(path):
    global cv2_engine, pytesseract_eng

    if not os.path.exists(path):
        print("   Warning: Image file not found.")
        return ""

    if cv2_engine is None:
        print("   -> Booting computer vision engine (first time only)...")
        import cv2
        import pytesseract
        cv2_engine      = cv2
        pytesseract_eng = pytesseract

    img = cv2_engine.imread(path)
    if img is None:
        print("   Warning: Could not read image file.")
        return ""

    gray = cv2_engine.cvtColor(img, cv2_engine.COLOR_BGR2GRAY)
    _, thresh = cv2_engine.threshold(gray, 0, 255, cv2_engine.THRESH_BINARY + cv2_engine.THRESH_OTSU)
    return pytesseract_eng.image_to_string(thresh).strip()

# Load trained model files
try:
    ensemble = joblib.load('model.pkl')
    vec      = joblib.load('tfidf_vectorizer.pkl')
    sel      = joblib.load('sel.pkl')
    print("[3/3] System online! ")
except Exception as e:
    print(f"Error: Could not load model files.\nDetails: {e}")
    sys.exit(1)

# Main loop
while True:
    text     = input("\nEmail Text  (Enter to skip | 'exit' to quit): ").strip()
    if text.lower() == 'exit':
        print("Shutting down.")
        break

    img_path    = input("Image Path  (Enter to skip): ").strip()
    vision_text = extract_image_text(img_path) if img_path else ""
    full_text   = (text + " " + vision_text).strip()

    if not full_text:
        print("No input provided. Please try again.")
        continue

    cleaned   = clean(full_text)
    vectorized = vec.transform([cleaned])
    features  = sel.transform(vectorized)

    pred  = ensemble.predict(features)[0]
    probs = ensemble.predict_proba(features)[0]

    if pred == 1:
        print(f"\nSPAM / PHISHING\n   Confidence: {probs[1] * 100:.2f}%")
    else:
        print(f"\nSAFE\n   Confidence: {probs[0] * 100:.2f}%")
        
