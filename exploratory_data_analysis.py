"""
Exploratory Data Analysis for SpamGuard.

Run this script after data_preprocessing.py. It produces six plots:
  1. Class distribution (pie chart)
  2. Character count distribution by class
  3. Word count distribution by class
  4. Sentence count distribution by class
  5. Feature correlation heatmap
  6. Spam word cloud
  7. Ham word cloud
"""
import os
import logging
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATA_PATH   = os.path.join(os.path.dirname(__file__), 'preprocessed_training_data.csv')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'eda_output')

# Tokens introduced by preprocessing that would dominate the word cloud
# without adding any linguistic signal
CLOUD_STOPWORDS = {'num', 'webaddr', 'emailaddr', 'currency'}


def plot_class_distribution(df: pd.DataFrame, out_dir: str) -> None:
    counts = df['label'].value_counts()
    labels = ['Ham (0)', 'Spam (1)']
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%0.2f%%', colors=['#4CAF50', '#F44336'])
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_class_distribution.png'), dpi=150)
    plt.close()
    log.info("Saved: 01_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame, out_dir: str) -> None:
    features = [
        ('num_chars',     'Character Count'),
        ('num_words',     'Word Count'),
        ('num_sentences', 'Sentence Count'),
    ]
    for i, (col, title) in enumerate(features, start=2):
        plt.figure(figsize=(10, 5))
        sns.histplot(df[df['label'] == 0][col], label='Ham',  color='#4CAF50', kde=True)
        sns.histplot(df[df['label'] == 1][col], label='Spam', color='#F44336', kde=True)
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.title(f'{title} Distribution by Class')
        plt.legend()
        plt.tight_layout()
        fname = f'0{i}_{col}_distribution.png'
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
        log.info("Saved: %s", fname)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df[['label', 'num_chars', 'num_words', 'num_sentences']].corr(),
        annot=True, fmt='.2f', cmap='coolwarm'
    )
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_correlation_heatmap.png'), dpi=150)
    plt.close()
    log.info("Saved: 05_correlation_heatmap.png")


def plot_word_cloud(df: pd.DataFrame, label: int, name: str,
                    colour: str, out_dir: str, file_num: int) -> None:
    corpus = df[df['label'] == label]['clean_text'].str.cat(sep=' ')
    wc = WordCloud(
        width=1200, height=600,
        background_color='black',
        colormap=colour,
        stopwords=CLOUD_STOPWORDS,
        max_words=100
    ).generate(corpus)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'{name} Emails — Top Words')
    plt.axis('off')
    plt.tight_layout()
    fname = f'0{file_num}_{name.lower()}_wordcloud.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    log.info("Saved: %s", fname)


def print_top_words(df: pd.DataFrame) -> None:
    for label, name in [(0, 'Ham'), (1, 'Spam')]:
        subset = df[df['label'] == label]['clean_text'].dropna()
        words = [w for text in subset for w in text.split()]
        log.info("%s — total emails: %d | total tokens: %d", name, len(subset), len(words))
        top = Counter(words).most_common(20)
        log.info("%s top 20 tokens:\n%s", name,
                 '\n'.join(f"  {w}: {c}" for w, c in top))


def print_descriptive_stats(df: pd.DataFrame) -> None:
    for label, name in [(0, 'Ham'), (1, 'Spam')]:
        log.info("%s descriptive stats:\n%s", name,
                 df[df['label'] == label][['num_chars', 'num_words', 'num_sentences']].describe())


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Preprocessed training data not found: {DATA_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    log.info("Dataset shape: %s", df.shape)
    log.info("Label distribution:\n%s", df['label'].value_counts())

    print_descriptive_stats(df)
    print_top_words(df)

    log.info("Generating plots...")
    plot_class_distribution(df, OUTPUT_DIR)
    plot_feature_distributions(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_word_cloud(df, label=1, name='Spam', colour='Reds',  out_dir=OUTPUT_DIR, file_num=6)
    plot_word_cloud(df, label=0, name='Ham',  colour='Greens', out_dir=OUTPUT_DIR, file_num=7)

    log.info("EDA complete. All plots saved to: %s", OUTPUT_DIR)


if __name__ == '__main__':
    main()
  
