import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATA_PATH     = os.path.join(os.path.dirname(__file__), 'data.csv')
TRAIN_PATH    = os.path.join(os.path.dirname(__file__), 'training_data.csv')
TEST_PATH     = os.path.join(os.path.dirname(__file__), 'testing_data.csv')

TEST_SIZE     = 0.30
RANDOM_STATE  = 36


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Merged dataset not found: {DATA_PATH}. Run dataset_creation.py first.")

    log.info("Loading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # stratify= ensures both splits preserve the original spam/ham ratio.
    # Without this, class imbalance means the splits can have very different
    # label distributions, making evaluation metrics unreliable.
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )

    log.info("Training set — shape: %s", df_train.shape)
    log.info("Training label distribution:\n%s", df_train['label'].value_counts())

    log.info("Test set — shape: %s", df_test.shape)
    log.info("Test label distribution:\n%s", df_test['label'].value_counts())

    df_train.to_csv(TRAIN_PATH, index=False)
    df_test.to_csv(TEST_PATH, index=False)

    log.info("Saved training data to: %s", TRAIN_PATH)
    log.info("Saved testing data to:  %s", TEST_PATH)


if __name__ == '__main__':
    main()
