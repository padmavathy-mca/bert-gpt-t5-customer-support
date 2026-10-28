# 🤖 AI Customer Support Assistant — BERT + GPT + T5

An AI-powered customer support prototype that combines three Transformer architectures to analyze customer sentiment, generate suggested responses, and summarize support interactions.

Built with Python, Hugging Face Transformers, PyTorch, and Streamlit.

## 📌 Problem Statement

Customer support teams receive questions about course access, payments, account issues, and technical problems. Manually reviewing every request and preparing support notes can take time.

This project demonstrates how different Transformer architectures can work together in a single pipeline to assist with customer support workflows.

## 🎯 Project Objectives

- Analyze customer sentiment using BERT.
- Generate suggested customer responses using GPT.
- Summarize support interactions using FLAN-T5.
- Flag selected billing and account-security keywords for human review.
- Provide a simple interactive interface using Streamlit.

## 🧠 Transformer Architectures Used

| Model | Architecture | Responsibility |
|---|---|---|
| DistilBERT | Encoder-only | Sentiment analysis |
| DistilGPT-2 | Decoder-only | Response generation |
| FLAN-T5 Small | Encoder-decoder | Support-ticket summarization |

### 1. BERT — Sentiment Analysis

Model: `distilbert-base-uncased-finetuned-sst-2-english`

Analyzes the sentiment of a customer message and returns a positive or negative label with a confidence score.

Note: This model performs sentiment analysis, not customer-intent classification.

### 2. GPT — Response Generation

Model: `distilgpt2`

Generates a suggested text response based on the customer query. It is a lightweight demonstration model, so its responses may require review and improvement.

### 3. T5 — Ticket Summarization

Model: `google/flan-t5-small`

Transforms the customer query and suggested response into a concise support-ticket summary using the model's sequence-to-sequence generation capability.

## 🔄 Project Workflow

1. The customer enters a query in the Streamlit interface.
2. BERT analyzes the sentiment.
3. Simple keyword-based rules check whether human review is required.
4. GPT generates a suggested response for requests that do not trigger the escalation rule.
5. FLAN-T5 summarizes the customer interaction.
6. The interface displays the sentiment, response, and ticket summary.

```text
Customer Query
      |
      v
BERT — Sentiment Analysis
      |
      v
Keyword-Based Escalation Check
      |
      v
GPT — Response Generation
      |
      v
FLAN-T5 — Ticket Summarization
      |
      v
Streamlit Results
```

When a configured escalation keyword is detected, the prototype bypasses GPT and displays a predefined support message. T5 still summarizes the interaction.

## 🛠️ Technology Stack

- **Language:** Python
- **UI:** Streamlit
- **AI/ML:** Hugging Face Transformers
- **Deep Learning:** PyTorch
- **Models:** DistilBERT, DistilGPT-2, FLAN-T5 Small
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
BERT-GPT-T5-Support/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    ├── __init__.py
    ├── bert_model.py
    ├── gpt_model.py
    └── t5_model.py
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Internet connection for downloading pretrained models on the first run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd BERT-GPT-T5-Support
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The first launch may take several minutes because the pretrained models need to be downloaded and loaded.

## 🧪 Example Test Cases

**Example 1 — Course access**

Input: "I paid for my course but cannot access it."

Expected workflow: Sentiment analysis, suggested response generation, and ticket summarization.

**Example 2 — Refund request**

Input: "I want a refund for my payment."

Expected workflow: The keyword-based escalation rule flags the request for human review, and GPT response generation is bypassed.

**Example 3 — Account security**

Input: "My account has been hacked."

Expected workflow: The configured keyword rule flags the request for human review.

These examples demonstrate the intended workflow; model outputs and rule coverage may vary.

## 🌍 Real-World Applications

- EdTech student support assistants
- Customer feedback analysis
- Support-ticket summarization
- E-commerce customer service prototypes
- AI-assisted service desk workflows

## 📚 Key Learnings

- Understanding encoder-only, decoder-only, and encoder-decoder architectures.
- Integrating pretrained Transformer models into a single application.
- Separating model logic into modular Python files.
- Using PyTorch and Hugging Face Transformers for text generation.
- Building an interactive AI interface with Streamlit.
- Understanding the difference between sentiment analysis, text generation, and summarization.

## ⚠️ Limitations

- DistilBERT is a sentiment classifier, not a specialized support-intent classifier.
- DistilGPT-2 may generate irrelevant or incomplete responses.
- FLAN-T5 summaries may omit or distort details.
- Escalation is based on simple keyword matching and may miss important cases.
- This prototype does not verify payment records, authenticate users, or enforce production-grade security.
- Model outputs should be reviewed before use in real customer-support operations.

## 🚀 Future Enhancements

- Train a BERT-based customer-intent classifier.
- Replace DistilGPT-2 with a more capable instruction-tuned language model.
- Ground responses in approved support documents using Retrieval-Augmented Generation (RAG).
- Add English and Tamil support.
- Implement robust escalation policies and human handoff.
- Add automated tests, logging, and deployment support.

## 👩‍💻 Author

**Padmavathy**

GitHub: https://github.com/padmavathy-mca

## 📄 License

Choose an appropriate license before publishing this repository. Check the licenses of the pretrained models separately before redistributing or deploying them.
