# SpamGuard

A machine-learning pipeline that classifies emails as **spam / phishing** or **ham (safe)**, with optional OCR support for image-based content.

---

## Model

An ensemble (`VotingClassifier`, soft voting) of:
- Logistic Regression
- Support Vector Machine (linear / RBF kernel)
- Multinomial Naive Bayes

Features are extracted with TF-IDF (unigrams + bigrams, top 5 000 features) and reduced with Chi-Square selection (top 1 000).

---

## Project Structure

```
spam_guard/
├── datasets/                   # Raw source CSVs (see Datasets section)
│   ├── email.csv
│   ├── emails.csv
│   └── combined_data.csv
│
├── dataset_creation.py         # Step 1 — merge raw CSVs into data.csv
├── dataset_split.py            # Step 2 — stratified 70/30 train/test split
├── data_preprocessing.py       # Step 3 — clean, tokenize, lemmatize
├── exploratory_data_analysis.py # (optional) — plots saved to eda_output/
├── data_vectorizer.py          # Step 4 — TF-IDF vectorization
├── train.py                    # Step 5 — tune, train, evaluate, save model
├── Main.py                     # Step 6 — interactive inference CLI
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed on the system for image scanning:
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Running the Pipeline

Run each step in order from the project root:

```bash
python dataset_creation.py        # → data.csv
python dataset_split.py           # → training_data.csv, testing_data.csv
python data_preprocessing.py      # → preprocessed_training_data.csv, preprocessed_testing_data.csv
python exploratory_data_analysis.py  # (optional) → eda_output/
python data_vectorizer.py         # → *.pkl vector files
python train.py                   # → model.pkl, sel.pkl
python Main.py                    # interactive detector
```

---

## Datasets

Source CSVs are available here:  
https://drive.google.com/drive/folders/1xmLkKAGCCBAjuuBljmb_mwaD-Y9xZi1L?usp=drive_link

Download and place all three files inside a `datasets/` folder in the project root before running `dataset_creation.py`.

---

## Evaluation Metrics

The training script reports:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Log Loss | Calibration quality of predicted probabilities |
| ROC-AUC | Ranking quality across all thresholds |
| Precision (Spam) | Of emails flagged spam, how many actually are |
| Recall (Spam) | Of actual spam emails, how many were caught |
| F1 (Spam) | Harmonic mean of precision and recall |
| False Positive Rate | Legitimate emails incorrectly flagged as spam |
| False Negative Rate | Spam emails that slipped through |
