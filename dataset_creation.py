import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), 'data.csv')


def load_email_csv(path: str) -> pd.DataFrame:
    """Load email.csv and normalise column names to (text, label)."""
    df = pd.read_csv(path)
    df = df.rename(columns={'Category': 'label', 'Message': 'text'})
    return df[['text', 'label']]


def load_emails_csv(path: str) -> pd.DataFrame:
    """Load emails.csv, drop all Unnamed columns, and keep (text, label)."""
    df = pd.read_csv(path)
    # Drop every auto-generated 'Unnamed: N' column regardless of how many there are
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    df = df.rename(columns={'spam': 'label'})
    return df[['text', 'label']]


def load_combined_csv(path: str) -> pd.DataFrame:
    """Load combined_data.csv — already has (text, label)."""
    df = pd.read_csv(path)
    return df[['text', 'label']]


def main() -> None:
    paths = {
        'email':    os.path.join(DATASETS_DIR, 'email.csv'),
        'emails':   os.path.join(DATASETS_DIR, 'emails.csv'),
        'combined': os.path.join(DATASETS_DIR, 'combined_data.csv'),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset not found: {path}")

    log.info("Loading source datasets...")
    df1 = load_email_csv(paths['email'])
    df2 = load_emails_csv(paths['emails'])
    df3 = load_combined_csv(paths['combined'])

    log.info("Shapes before merge — email: %s | emails: %s | combined: %s",
             df1.shape, df2.shape, df3.shape)

    combined = pd.concat([df1, df2, df3], ignore_index=True)

    log.info("Merged shape: %s", combined.shape)
    log.info("Label distribution:\n%s", combined['label'].value_counts())

    combined.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved merged dataset to: %s", OUTPUT_PATH)


if __name__ == '__main__':
    main()
    
