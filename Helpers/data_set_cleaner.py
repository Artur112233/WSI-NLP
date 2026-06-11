import csv
import numpy as np
import pandas as pd
from pandas import Series
import json
import os


def import_amazon_reviews_csv_file(path: str, nrows: int = 1000000) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        nrows=nrows,
        header=None,
        names=['rating', 'review_title', 'review_text'],
        quotechar='"',
        doublequote=True)
    df['full_text'] = df['review_title'].fillna('') + " " + df['review_text'].fillna('')
    df.drop('review_title', axis=1, inplace=True)
    df.drop('review_text', axis=1, inplace=True)

    rating_map = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
    df['rating'] = df['rating'].map(rating_map)
    return df

def amazon_reviews_split(df: pd.DataFrame) -> tuple[Series, Series]:
    X = df['full_text']
    Y = df['rating']
    return X, Y

def import_financial_headlines_csv_file(path: str, nrows: int = 3000) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        nrows=nrows,
        header=None,
        names=['sentiment', 'headline'],
        quotechar='"',
        doublequote=True)

    rating_map = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}
    df['sentiment'] = df['sentiment'].map(rating_map)
    return df

def financial_headlines_split(df: pd.DataFrame) -> tuple[Series, Series]:
    X = df['headline']
    Y = df['sentiment']
    return X, Y

def import_newsmtsc_headlines_jsonl_file(path: str, nrows: int = 5000) -> pd.DataFrame:
    rows = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= nrows:
                break

            record = json.loads(line.strip())

            sentence = record.get('sentence_normalized', '')
            targets = record.get('targets', [])

            if targets and len(targets) > 0:
                polarity = targets[0].get('polarity', 4.0)
            else:
                polarity = 4.0  # default to neutral if no target

            rows.append({
                'headline': sentence,
                'sentiment': polarity
            })

    df = pd.DataFrame(rows)

    rating_map = {
        2.0: 'Negative',
        4.0: 'Neutral',
        6.0: 'Positive'
    }
    df['sentiment'] = df['sentiment'].map(rating_map)

    return df

def newsmtsc_headlines_split(df: pd.DataFrame) -> tuple[Series, Series]:
    X = df['headline']
    Y = df['sentiment']
    return X, Y


def import_imdb_dataset(directory_path: str, nrows: int = 20000) -> pd.DataFrame:
    rows = []
    half_rows = nrows // 2

    pos_dir = os.path.join(directory_path, 'pos')
    if os.path.isdir(pos_dir):
        pos_files = sorted(os.listdir(pos_dir))
        for i, filename in enumerate(pos_files):
            if i >= half_rows:
                break
            if filename.endswith('.txt'):
                filepath = os.path.join(pos_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                rows.append({
                    'headline': text,
                    'sentiment': 'Positive'
                })

    neg_dir = os.path.join(directory_path, 'neg')
    if os.path.isdir(neg_dir):
        neg_files = sorted(os.listdir(neg_dir))
        for i, filename in enumerate(neg_files):
            if i >= half_rows:
                break
            if filename.endswith('.txt'):
                filepath = os.path.join(neg_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                rows.append({
                    'headline': text,
                    'sentiment': 'Negative'
                })

    df = pd.DataFrame(rows)
    return df


def imdb_split(df: pd.DataFrame) -> tuple[Series, Series]:
    X = df['headline']
    Y = df['sentiment']
    return X, Y
