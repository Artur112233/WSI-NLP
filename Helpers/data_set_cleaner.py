import csv
import numpy as np
import pandas as pd
from pandas import Series


def import_amazon_reviews_csv_file(path: str, nrows: int) -> pd.DataFrame:
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