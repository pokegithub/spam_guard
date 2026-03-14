import os
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

TRAIN_PATH   = os.path.join(os.path.dirname(__file__), 'preprocessed_training_data.csv')
TEST_PATH    = os.path.join(os.path.dirname(__file__), 'preprocessed_testing_data.csv')

X_TRAIN_PKL  = os.path.join(os.path.dirname(__file__), 'x_training_vector.pkl')
X_TEST_PKL   = os.path.join(os.path.dirname(__file__), 'x_testing_vector.pkl')
Y_TRAIN_PKL  = os.path.join(os.path.dirname(__file__), 'y_training_vector.pkl')
Y_TEST_PKL   = os.path.join(os.path.dirname(__file__), 'y_testing_vector.pkl')
TFIDF_PKL    = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed file not found: {path}. Run data_preprocessing.py first.")

    log.info("Loading preprocessed data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    # Defensive fill — should be empty after preprocessing, but guard anyway
    x_train = df_train['clean_text'].fillna('')
    x_test  = df_test['clean_text'].fillna('')
    y_train = df_train['label']

    # Testing data has no 'label' column — this is intentional.
    # y_test is loaded from the test CSV directly in train.py to evaluate
    # against raw labels that were held out before preprocessing.
    if 'label' not in df_test.columns:
        raise KeyError(
            "'label' column not found in testing data. "
            "The label column must be retained through dataset_split.py and data_preprocessing.py."
        )
    y_test = df_test['label']

    log.info("Fitting TF-IDF vectorizer on training data...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        ngram_range=(1, 2)
    )

    # CRITICAL: fit only on training data — transform on test.
    # Wrapping sklearn's sparse transform in tqdm does nothing useful;
    # the iteration completes before sklearn does any real work.
    x_train_vec = tfidf.fit_transform(x_train)
    log.info("Training matrix shape: %s", x_train_vec.shape)

    log.info("Transforming test data...")
    x_test_vec = tfidf.transform(x_test)
    log.info("Test matrix shape: %s", x_test_vec.shape)

    log.info("Saving vectors and vectorizer...")
    joblib.dump(x_train_vec, X_TRAIN_PKL)
    joblib.dump(x_test_vec,  X_TEST_PKL)
    joblib.dump(y_train,     Y_TRAIN_PKL)
    joblib.dump(y_test,      Y_TEST_PKL)
    joblib.dump(tfidf,       TFIDF_PKL)

    log.info("All artefacts saved. Vectorization complete.")


if __name__ == '__main__':
    main()
    
