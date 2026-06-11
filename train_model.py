import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from Helpers.data_set_cleaner import import_amazon_reviews_csv_file, amazon_reviews_split, import_imdb_dataset, \
    import_newsmtsc_headlines_jsonl_file, import_financial_headlines_csv_file, imdb_split, newsmtsc_headlines_split, \
    financial_headlines_split
from Helpers.text_processor import TextProcessor
import pandas as pd

def train_model_amazon():
    print("[POCZATEK TWORZENIA MODELU]")
    data_path = r".\Data\train.csv"
    model_path = r".\Models\model.joblib"

    data_quantity = 10_000

    df = import_amazon_reviews_csv_file(data_path, data_quantity)

    X_raw, y = amazon_reviews_split(df)
    tp = TextProcessor()
    print("[PREPROCESSING (czasochlonne)]")
    X_processed = tp.preprocess_series(X_raw)

# wynik z search grida dla LogisticRegression:
# {'clf__C': 1.0, 'tfidf__max_df': 0.8, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=3)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C = 1.0))
    ])
    print("[Trenowanie model]")
    model_pipeline.fit(X_processed, y)
    print(f"[Zapisywanie modelu do {model_path}]")
    joblib.dump(model_pipeline, model_path)

    print("[KONIEC TWORZENIA MODELU]")

def train_model_full():
    print("[POCZATEK TWORZENIA MODELU]")
    amazon_data_path = r".\Data\amazon_train.csv"
    imdb_dir_path = r".\Data\imdb_train"
    newsmtsc_data_path = r".\Data\newsmtsc_train.jsonl"
    financial_data_path = r".\Data\financial_all.csv"
    model_path = r".\Models\model3.joblib"

    df_amazon = import_amazon_reviews_csv_file(amazon_data_path)
    df_imdb = import_imdb_dataset(imdb_dir_path)
    df_news = import_newsmtsc_headlines_jsonl_file(newsmtsc_data_path)
    df_fin = import_financial_headlines_csv_file(financial_data_path)

    X_amazon, y_amazon = amazon_reviews_split(df_amazon)
    X_imdb, y_imdb = imdb_split(df_imdb)
    X_news, y_news = newsmtsc_headlines_split(df_news)
    X_fin, y_fin = financial_headlines_split(df_fin)
    tp = TextProcessor()
    X_raw = X_amazon+X_imdb+X_news+X_fin
    y = y_amazon+y_imdb+y_news+y_fin
    mask = pd.notna(y)
    X_raw = X_raw[mask]
    y = y[mask]
    print("[PREPROCESSING (czasochlonne)]")
    X_processed = tp.preprocess_series(X_raw)

# wynik z search grida dla LogisticRegression:
# {'clf__C': 1.0, 'tfidf__max_df': 0.8, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=3)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C = 1.0))
    ])
    print("[Trenowanie model]")
    model_pipeline.fit(X_processed, y)
    print(f"[Zapisywanie modelu do {model_path}]")
    joblib.dump(model_pipeline, model_path)

    print("[KONIEC TWORZENIA MODELU]")


train_model_full()