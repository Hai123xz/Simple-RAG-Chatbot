# Banking RAG MVP (2026)

Quick local RAG chatbot for banking FAQs with guardrails.

## Features
- Hybrid retrieval (vector + BM25 keyword)
- PII redaction, source citations
- Refuses transactions (MVP is FAQ-only)
- Works with OpenAI or local Ollama

## Run
1. pip install -r requirements.txt
2. export OPENAI_API_KEY=sk-...  # or set USE_OLLAMA=1
3. streamlit run app.py

Test questions:
- "What is the overdraft fee?"
- "Phí thường niên thẻ tín dụng?"
