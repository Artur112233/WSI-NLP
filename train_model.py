import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from Helpers.data_set_cleaner import import_amazon_reviews_csv_file, amazon_reviews_split
from Helpers.text_processor import TextProcessor

def train_model():
    print("[POCZATEK TWORZENIA MODELU]")
    data_path = r".\Data\train.csv"
    model_path = r".\Models\model.joblib"

    data_quantity = 10_000

    df = import_amazon_reviews_csv_file(data_path, data_quantity)

    X_raw, y = amazon_reviews_split(df)
    tp = TextProcessor()
    print("[PREPROCESSING (czasochlonne)]")
    X_processed = tp.preprocess_series(X_raw)
    print("[SPLIT DATA]")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

# wynik z search grida dla LogisticRegression:
# {'clf__C': 1.0, 'tfidf__max_df': 0.8, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 3)}
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=3)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C = 1.0))
    ])
    print("[Trenowanie model]")
    model_pipeline.fit(X_train, y_train)
    print(f"[Zapisywanie modelu do {model_path}]")
    joblib.dump(model_pipeline, model_path)

    print("[KONIEC TWORZENIA MODELU]")

train_model()