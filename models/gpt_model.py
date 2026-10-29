
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "distilgpt2"

_tokenizer = None
_model = None


def load_gpt_model():
    global _tokenizer, _model

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.eval()

        # GPT-2 has no separate padding token by default.
        _tokenizer.pad_token = _tokenizer.eos_token
        _model.config.pad_token_id = _tokenizer.eos_token_id

    return _tokenizer, _model


def generate_response(query):
    tokenizer, model = load_gpt_model()

    prompt = (
        "You are a polite customer support assistant for an online course platform.\n"
        "Give a short, helpful answer to the customer's issue.\n"
        "Do not repeat the customer's message.\n"
        "Do not invent account details or claim an action has been completed.\n"
        f"Customer issue: {query.strip()}\n"
        "Support response:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=70,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    if not response:
        response = (
            "I'm sorry you're having trouble accessing your lessons. "
            "Please check your course enrollment and payment status. "
            "If the issue continues, contact support for assistance."
        )

    return response
