import os
import re
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

tqdm.pandas()

TRAIN_RAW_PATH  = os.path.join(os.path.dirname(__file__), 'training_data.csv')
TEST_RAW_PATH   = os.path.join(os.path.dirname(__file__), 'testing_data.csv')
TRAIN_OUT_PATH  = os.path.join(os.path.dirname(__file__), 'preprocessed_training_data.csv')
TEST_OUT_PATH   = os.path.join(os.path.dirname(__file__), 'preprocessed_testing_data.csv')

# Regex patterns compiled once at module level for performance
_URL_RE      = re.compile(r'(https?://\S+|www\.\S+)')
_EMAIL_RE    = re.compile(r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}')
_NUM_RE      = re.compile(r'\d+')
_CURRENCY_RE = re.compile(r'[$£€]')
_PUNCT_RE    = re.compile(r'[^a-z\s]')


def ensure_nltk_data() -> None:
    packages = {
        'corpora/stopwords':        'stopwords',
        'corpora/wordnet':          'wordnet',
        'corpora/omw-1.4':          'omw-1.4',
        # punkt_tab is required by NLTK >= 3.9; punkt alone will throw LookupError
        'tokenizers/punkt_tab':     'punkt_tab',
    }
    for path, package in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading missing NLTK package: %s", package)
            nltk.download(package, quiet=True)


def build_stop_words() -> set:
    """Return a stop-word set without mutating the shared NLTK global."""
    base = set(stopwords.words('english'))
    # Remove loose URL fragments that survive the regex substitution
    base.update(['http', 'https', 'www', 'com', 'org', 'net', 'co'])
    return base


def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    text = str(text).lower()
    text = _URL_RE.sub('webaddr', text)
    text = _EMAIL_RE.sub('emailaddr', text)
    text = _NUM_RE.sub('num', text)
    text = _CURRENCY_RE.sub('currency ', text)
    text = _PUNCT_RE.sub('', text)

    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(tokens)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight structural features.

    word_tokenize is intentionally NOT used here; for a simple word-count
    feature, str.split() is ~100x faster and the difference is negligible
    for what is ultimately just a numeric metadata column.
    """
    log.info("  Computing num_chars...")
    df['num_chars'] = df['text'].progress_apply(len)

    log.info("  Computing num_words...")
    df['num_words'] = df['text'].progress_apply(lambda x: len(str(x).split()))

    log.info("  Computing num_sentences...")
    df['num_sentences'] = df['text'].progress_apply(
        lambda x: len(nltk.sent_tokenize(str(x)))
    )
    return df


def preprocess(df: pd.DataFrame, lemmatizer: WordNetLemmatizer,
               stop_words: set, has_label: bool) -> pd.DataFrame:
    # Strip the 'Subject: ' prefix that some datasets include
    df['text'] = df['text'].str.replace('Subject: ', '', regex=False)

    df = add_features(df)

    log.info("  Dropping duplicates...")
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

    log.info("  Cleaning text...")
    df['clean_text'] = df['text'].progress_apply(
        lambda x: clean_text(x, lemmatizer, stop_words)
    )

    # Drop rows where cleaning left nothing useful
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

    null_count  = df['clean_text'].isna().sum()
    empty_count = (df['clean_text'].fillna('').str.strip() == '').sum()
    log.info("  Remaining NaN rows: %d | Empty strings: %d", null_count, empty_count)

    keep_cols = ['text', 'clean_text', 'num_chars', 'num_words', 'num_sentences']
    if has_label:
        # Normalise label: map string variants to 0/1 integers
        df['label'] = df['label'].replace({'ham': 0, 'spam': 1, '0': 0, '1': 1})
        df = df[df['label'].isin([0, 1])]
        keep_cols.insert(2, 'label')

    return df[keep_cols]


def main() -> None:
    ensure_nltk_data()

    lemmatizer = WordNetLemmatizer()
    stop_words = build_stop_words()

    for raw_path, out_path, has_label, split_name in [
        (TRAIN_RAW_PATH, TRAIN_OUT_PATH, True,  'Training'),
        (TEST_RAW_PATH,  TEST_OUT_PATH,  False, 'Testing'),
    ]:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"{split_name} CSV not found: {raw_path}")

        log.info("Preprocessing %s data...", split_name)
        df = pd.read_csv(raw_path)
        df = preprocess(df, lemmatizer, stop_words, has_label=has_label)

        log.info("%s set final shape: %s", split_name, df.shape)
        if has_label:
            log.info("Label distribution:\n%s", df['label'].value_counts())

        df.to_csv(out_path, index=False)
        log.info("Saved to: %s\n", out_path)

    log.info("Preprocessing complete.")


if __name__ == '__main__':
    main()
    
