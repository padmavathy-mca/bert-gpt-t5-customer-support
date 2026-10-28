from transformers import pipeline

_model = None

def analyze_sentiment(text):
    global _model

    if _model is None:
        _model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    return _model(text)[0]