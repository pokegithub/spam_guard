import os
import re
import joblib
import cv2
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = ['http', 'https', 'www', 'com', 'org', 'net', 'co']
stop_words.update(custom_stopwords)
def clean(text):
    text = text.lower()
    url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    email_regex = r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}'
    num_regex = r'\d+'
    currency_regex = r'[$£€]'
    text = re.sub(url_regex, 'webaddr', text)
    text = re.sub(email_regex, 'emailaddr', text)
    text = re.sub(num_regex, 'num', text)
    text = re.sub(currency_regex, 'currency ', text)
    text = text.replace('escapenumber', 'num')
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    clean_words = [lemmatizer.lemmatize(i) for i in words if i not in stop_words and len(i) > 1]
    return ' '.join(clean_words)

# checking for image text
def extract_image_text(path):
    if not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Extract the hidden text, they can't hide from us!!!!
    extracted = pytesseract.image_to_string(thresh)
    return extracted if extracted.strip() else ""
print("Initializing Spam & Phishing Detector...")

try:
    ensemble = joblib.load('model.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl')
    sel = joblib.load('sel.pkl')
except Exception as e:
    print(f"Error: Missing .pkl files.\nDetails: {e}")
    exit()

while True:
    text = input("\nEmail Text (press Enter to skip, type 'exit' to quit): ")
    if text.lower() == 'exit':
        print("Systems shutting down.") 
        break
    
    img_path = input("Image Path (press Enter to skip): ")
    vision_text = extract_image_text(img_path) if img_path else ""
    full_content = text + " " + vision_text
    if not full_content.strip():
        print("No input detected. Please try again.")
        continue
    cleaned = clean(full_content)
    vectorized = vec.transform([cleaned])
    features = sel.transform(vectorized)

    # Rhis predictts if the email is spam or safe
    pred = ensemble.predict(features)[0]

    # This checks how sure the model is about its prediction
    probs = ensemble.predict_proba(features)[0]
    spam_prob = probs[1] * 100
    safe_prob = probs[0] * 100
  
    if pred == 1:
        print(f"\nSPAM/PHISHING \nProbability: {spam_prob:.2f}%")  
    else:
        print(f"\nSAFE \nProbability: {safe_prob:.2f}%")
