
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None


def load_t5_model():
    global _tokenizer, _model

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()

    return _tokenizer, _model


def summarize_ticket(query, response):
    tokenizer, model = load_t5_model()

    prompt = (
        "Summarize this customer support interaction "
        "in one short sentence. Do not invent facts.\n"
        f"Customer issue: {query}\n"
        f"Support response: {response}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()
