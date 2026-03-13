# 1. IMMEDIATE UI RESPONSE (UX Trick)
print("Booting up Spam & Phishing Detector...")
print("[1/3] Verifying NLTK Local Caches...")

import os
import re
import joblib
import nltk

# 2. LOCAL NLTK CHECK (Saves network time)
def ensure_nltk_data():
    packages = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'sentiment/vader_lexicon.zip': 'vader_lexicon'
    }
    for path, package in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"      -> Downloading missing package: {package}...")
            nltk.download(package, quiet=True)

ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print("[2/3] Loading Machine Learning Matrices...")

# 3. PRE-COMPILED REGEX (Instant string matching)
url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
email_pattern = re.compile(r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}')
num_pattern = re.compile(r'\d+')
currency_pattern = re.compile(r'[$£€]')
punct_pattern = re.compile(r'[^a-z\s]')

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
stop_words.update(['http', 'https', 'www', 'com', 'org', 'net', 'co'])

def clean(text):
    text = text.lower()
    text = url_pattern.sub('webaddr', text)
    text = email_pattern.sub('emailaddr', text)
    text = num_pattern.sub('num', text)
    text = currency_pattern.sub('currency ', text)
    text = text.replace('escapenumber', 'num')
    text = punct_pattern.sub('', text)

    clean_words = [lemmatizer.lemmatize(i) for i in text.split() if i not in stop_words and len(i) > 1]
    return ' '.join(clean_words)

# 4. STATEFUL LAZY LOADING (Fixes the Hot Loop Bottleneck)
cv2_engine = None
pytesseract_engine = None

def extract_image_text(path):
    global cv2_engine, pytesseract_engine
    
    if not os.path.exists(path):
        return ""
    
    if cv2_engine is None or pytesseract_engine is None:
        print("   -> Booting Computer Vision Engine (First time only)...")
        import cv2
        import pytesseract
        cv2_engine = cv2
        pytesseract_engine = pytesseract
    
    img = cv2_engine.imread(path)
    gray = cv2_engine.cvtColor(img, cv2_engine.COLOR_BGR2GRAY)
    _, thresh = cv2_engine.threshold(gray, 0, 255, cv2_engine.THRESH_BINARY + cv2_engine.THRESH_OTSU)
    extracted = pytesseract_engine.image_to_string(thresh)
    
    return extracted if extracted.strip() else ""

# 5. LOAD THE TRAINED BRAIN
try:
    ensemble = joblib.load('model.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl') 
    sel = joblib.load('sel.pkl')
    print("[3/3] System Online! ✅")
except Exception as e:
    print(f"Error: Missing .pkl files. Please place them in this folder.\nDetails: {e}")
    exit()

# 6. INTERACTIVE SCANNER (Conditional VADER Execution)
while True:
    text = input("\nEmail Text (press Enter to skip, type 'exit' to quit): ")
    if text.lower() == 'exit':
        print("Systems shutting down. Good luck with the presentation!") 
        break
    
    img_path = input("Image Path (press Enter to skip): ")
    
    vision_text = extract_image_text(img_path) if img_path else ""
    full_content = text + " " + vision_text
    
    if not full_content.strip():
        print("No input detected. Please try again.")
        continue

    # ML Pipeline (Fast Path)
    cleaned = clean(full_content)
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
        
