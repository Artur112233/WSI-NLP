import string
import spacy

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def remove_stop_words(self, text: str) -> str:
        doc = self.nlp(text) # wymagające, w przyszłości działać na 1 metodzie
        clean_tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(clean_tokens)

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_.lower() for token in doc]
        return " ".join(lemmatized_tokens)

    def preprocess_single(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    def preprocess(self, text_list: list[str]) -> list[str]:
        ret = []
        for text in text_list:
            ret.append(self.preprocess_single(text))
        return ret

# nlp = spacy.load("pl_core_news_sm")
# print(nlp.Defaults.stop_words)
# print(len(nlp.Defaults.stop_words))