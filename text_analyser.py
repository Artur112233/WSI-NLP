from Helpers.text_processor import TextProcessor
import joblib

class SentimentAnalyzer:
    def __init__(self, model_path=r"Models/model.joblib"):
        self.text_processor = TextProcessor()
        self.model = joblib.load(model_path)

    def analyse_text(self, text):
        cleaned_text = self.text_processor.preprocess_single(text)
        return self.model.predict([cleaned_text])[0]

    def analyse_article_url(self, url):
        return "--"

