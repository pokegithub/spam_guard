"""
SpamGuard — Interactive Spam & Phishing Detector
Run this after the full training pipeline has produced model.pkl, tfidf_vectorizer.pkl, sel.pkl.
"""
print("Booting up Spam & Phishing Detector...")
print("[1/3] Verifying NLTK local caches...")

import os
import re
import sys
import logging
import joblib
import nltk

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── NLTK data check ──────────────────────────────────────────────────────────

def ensure_nltk_data() -> None:
    packages = {
        'corpora/stopwords':    'stopwords',
        'corpora/wordnet':      'wordnet',
        'tokenizers/punkt_tab': 'punkt_tab',
    }
    for path, package in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading missing NLTK package: %s", package)
            nltk.download(package, quiet=True)


ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("[2/3] Loading machine learning artefacts...")


# ── Text-cleaning pipeline ────────────────────────────────────────────────────
# These must exactly mirror data_preprocessing.py clean_text()

_URL_RE      = re.compile(r'(https?://\S+|www\.\S+)')
_EMAIL_RE    = re.compile(r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}')
_NUM_RE      = re.compile(r'\d+')
_CURRENCY_RE = re.compile(r'[$£€]')
_PUNCT_RE    = re.compile(r'[^a-z\s]')

_lemmatizer = WordNetLemmatizer()

# Build a local copy — never mutate the shared NLTK global set
_stop_words = set(stopwords.words('english'))
_stop_words.update(['http', 'https', 'www', 'com', 'org', 'net', 'co'])


def clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub('webaddr', text)
    text = _EMAIL_RE.sub('emailaddr', text)
    text = _NUM_RE.sub('num', text)
    text = _CURRENCY_RE.sub('currency ', text)
    text = _PUNCT_RE.sub('', text)
    tokens = [
        _lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in _stop_words and len(w) > 1
    ]
    return ' '.join(tokens)


# ── Optional computer-vision engine (lazy-loaded on first use) ────────────────

_cv2_engine       = None
_pytesseract_eng  = None


def extract_image_text(path: str) -> str:
    """OCR text from an image file. Returns empty string on any failure."""
    global _cv2_engine, _pytesseract_eng

    if not os.path.exists(path):
        log.warning("Image path not found: %s", path)
        return ""

    if _cv2_engine is None:
        log.info("Booting computer-vision engine (first time only)...")
        import cv2
        import pytesseract
        _cv2_engine      = cv2
        _pytesseract_eng = pytesseract

    img = _cv2_engine.imread(path)
    if img is None:
        log.warning("Could not read image (unsupported format or corrupt file): %s", path)
        return ""

    gray = _cv2_engine.cvtColor(img, _cv2_engine.COLOR_BGR2GRAY)
    _, thresh = _cv2_engine.threshold(gray, 0, 255,
                                      _cv2_engine.THRESH_BINARY + _cv2_engine.THRESH_OTSU)
    extracted = _pytesseract_eng.image_to_string(thresh)
    return extracted.strip()


# ── Load trained artefacts ────────────────────────────────────────────────────

def load_model():
    artefacts = {
        'model':     os.path.join(BASE_DIR, 'model.pkl'),
        'vectorizer': os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'),
        'selector':  os.path.join(BASE_DIR, 'sel.pkl'),
    }
    missing = [name for name, path in artefacts.items() if not os.path.exists(path)]
    if missing:
        log.error("Missing artefacts: %s. Run the training pipeline first.", missing)
        sys.exit(1)

    try:
        ensemble = joblib.load(artefacts['model'])
        vec      = joblib.load(artefacts['vectorizer'])
        sel      = joblib.load(artefacts['selector'])
    except Exception as e:
        log.error("Failed to load artefacts: %s", e)
        sys.exit(1)

    return ensemble, vec, sel


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(text: str, ensemble, vec, sel) -> dict:
    """Return a dict with prediction label, spam confidence, and ham confidence."""
    cleaned    = clean(text)
    vectorized = vec.transform([cleaned])
    features   = sel.transform(vectorized)

    pred  = ensemble.predict(features)[0]
    probs = ensemble.predict_proba(features)[0]

    return {
        'label':      int(pred),
        'spam_pct':   probs[1] * 100,
        'ham_pct':    probs[0] * 100,
    }


def format_verdict(result: dict) -> str:
    if result['label'] == 1:
        return (
            f"\n🚨 SPAM / PHISHING DETECTED\n"
            f"   Confidence: {result['spam_pct']:.2f}%"
        )
    return (
        f"\n✅ SAFE\n"
        f"   Confidence: {result['ham_pct']:.2f}%"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ensemble, vec, sel = load_model()
    print("[3/3] System online ✅\n")

    while True:
        text = input("Email text  (Enter to skip | 'exit' to quit): ").strip()

        if text.lower() == 'exit':
            print("Shutting down.")
            break

        img_path     = input("Image path  (Enter to skip): ").strip()
        vision_text  = extract_image_text(img_path) if img_path else ""
        full_content = (text + " " + vision_text).strip()

        if not full_content:
            print("No input provided. Please enter text or an image path.\n")
            continue

        result = predict(full_content, ensemble, vec, sel)
        print(format_verdict(result))
        print()


if __name__ == '__main__':
    main()
    vectorized = vec.transform([cleaned])
    features = sel.transform(vectorized)
    
    pred = ensemble.predict(features)[0]
    probs = ensemble.predict_proba(features)[0]
    
    spam_conf = probs[1] * 100
    safe_conf = probs[0] * 100
    
    # Conditional Heuristic Override (Slow Path ONLY if Spam is detected)
    if pred == 1:
        # VADER reads the RAW, uncleaned text to find emotional markers
        compound_score = sia.polarity_scores(full_content)['compound']
        
        if compound_score < -0.4:
            print(f"\n🛡️ SENTIMENT OVERRIDE TRIGGERED!")
            print(f"AI detected Spam ({spam_conf:.2f}%), but detected human distress.")
            print(f"✅ VERDICT: SAFE (Quarantined for manual review)")
        else:
            print(f"\n🚨 SPAM/PHISHING \nProbability: {spam_conf:.2f}%")
    else:
        print(f"\n✅ SAFE \nProbability: {safe_conf:.2f}%")
        
