from Helpers.text_processor import TextProcessor
import joblib
import trafilatura
import requests

class SentimentAnalyzer:
    def __init__(self, model_path=r"Models/model.joblib"):
        self.text_processor = TextProcessor()
        self.model = joblib.load(model_path)

    def analyse_text(self, text):
        cleaned_text = self.text_processor.preprocess_single(text)
        return self.model.predict([cleaned_text])[0]

    def analyse_article_url(self, url):
        try:
            downloaded = trafilatura.fetch_url(url)

            if not downloaded:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                downloaded = response.text

            article_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )

            if not article_text:
                raise ValueError(f"Could not extract content from {url}")

            return self.analyse_text(article_text)

        except Exception as e:
            print(f"Error analyzing URL {url}: {e}")
            return None

