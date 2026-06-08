import string
import spacy
from pandas import Series
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        base_stop_words = set(stopwords.words('english'))
        sentiment_whitelist = {
            'no', 'not', 'nor', 'but', 'against', 'too', 'very',
            'don', "don't", 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
            'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
            'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }
        self.custom_stop_words = base_stop_words - sentiment_whitelist

    # maybe alternate stop words
    def preprocess_single(self, text: str) -> str:
        doc = self.nlp(text)
        # tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha] lepiej działa bez stop words
        #tokens = [token.lemma_.lower() for token in doc if token.is_alpha] lepiej działa bez sprawdzania alfy
        tokens = [token.lemma_.lower() for token in doc if token not in self.custom_stop_words]
        # tokens = [token.lemma_.lower() for token in doc] to jest takie samo jak z custom (wynikowo)
        return " ".join(tokens)

    # def preprocess(self, text_list: list[str]) -> list[str]:
    #     ret = []
    #     for text in text_list:
    #         ret.append(self.preprocess_single(text))
    #     return ret

    def preprocess(self, text_series: Series) -> Series:
        return text_series.apply(self.preprocess_single)
