
import streamlit as st

from models.bert_model import analyze_sentiment
from models.gpt_model import generate_response
from models.t5_model import summarize_ticket

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Customer Support Assistant")
st.write("BERT + GPT + T5 in one pipeline")

query = st.text_area(
    "Enter your customer support query",
    placeholder="I paid for my course but cannot access it."
)

if st.button("Analyze Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a customer query.")
        st.stop()

    text = query.lower()

    sensitive_terms = [
        "refund", "billing", "charged", "payment",
        "password", "account hacked", "security"
    ]

    escalation = any(term in text for term in sensitive_terms)

    with st.spinner("Analyzing sentiment with BERT..."):
        sentiment = analyze_sentiment(query)

    st.subheader("1. BERT — Sentiment Analysis")
    st.write(f"Sentiment: **{sentiment['label']}**")
    st.write(f"Confidence: **{sentiment['score']:.2%}**")

    if escalation:
        st.warning("Human support review required.")

        response = (
            "Thank you for contacting us. Your request needs "
            "review by our support team. Please use the official "
            "support channel for assistance."
        )
    else:
        with st.spinner("Generating response with GPT..."):
            response = generate_response(query)

    st.subheader("2. GPT — Suggested Response")
    st.write(response)

    with st.spinner("Summarizing ticket with T5..."):
        summary = summarize_ticket(query, response)

    st.subheader("3. T5 — Support Ticket Summary")
    st.write(summary)

    st.success("Pipeline completed!")
    st.caption(
        "Prototype only. Validate model outputs and escalation "
        "rules before production use."
    )
