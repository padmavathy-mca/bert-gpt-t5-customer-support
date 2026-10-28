from transformers import pipeline

_model = None

def generate_response(query):
    global _model

    if _model is None:
        _model = pipeline(
            "text-generation",
            model="distilgpt2"
        )

    prompt = (
        "Customer support response:\n"
        f"Customer: {query}\n"
        "Agent:"
    )

    result = _model(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        pad_token_id=_model.tokenizer.eos_token_id
    )[0]["generated_text"]

    response = result[len(prompt):].strip()

    return response or (
        "Thank you for contacting support. "
        "Could you share more details so we can help?"
    )