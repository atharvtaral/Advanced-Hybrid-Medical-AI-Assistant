# Advanced Hybrid Medical AI Assistant 🩺

A real-world, production-grade **Multimodal Retrieval-Augmented Generation (RAG)** Medical Assistant engineered with an intelligent **Conditional Model Routing** pipeline. This system leverages advanced vision capabilities to parse handwritten medical prescriptions and seamlessly transitions between high-speed open-source execution and sophisticated multi-lingual processing.

---

## 🚀 Architectural Overview & Core Workflow

The system is architected to optimize infrastructure costs, response latency, and translation quality by orchestrating open-source and proprietary foundation models through **LangChain Expression Language (LCEL)**.

---

## Output Image:


## 🔥 Key Technical Features

### 1. Dynamic Hybrid Model Routing & Cost Optimization

- **Contextual Routing:** Implemented dynamic execution trees via **LCEL**. English conversational inputs bypass commercial endpoints to run on high-performance infrastructure via **NVIDIA NIM** utilizing `meta/llama-3.1-8b-instruct`.
- **Advanced Cross-Lingual Nuance:** Automatically routes Marathi requests to OpenAI's `gpt-4o-mini`, resolving tokenization inefficiencies and preserving medical vocabulary syntax structures in regional semantics.

### 2. Multi-Modal Vision Processing Pipeline

- **Pillow Buffer Serialization:** Mitigated stream exceptions from non-standard desktop crop utilities (`image/x-png` format mutations) by implementing a strict **RGB-JPEG re-rendering engine** via `PIL.Image` and `io.BytesIO`.
- **Zero-Shot Clinical Classification:** Configured localized contextual prompts allowing the vision model to decipher non-standard Latin medical shorthand conventions (such as `G.`/`Gtt.` indicating _Guttae_ eye drops) to extract complex brand identities like _Azopt_, _Combigan_, and _Xalatan_.

### 3. Enterprise Cloud-Native RAG Pipeline

- **Deterministic Guardrails:** Prevented stochastically generated hallucinations by restricting domain response space strictly to an isolated knowledge corpus.
- **Vector Store Architecture:** Managed document embedding vectors through **Pinecone Serverless Vector Databases** synced using local HuggingFace feature-extraction pipelines (`all-MiniLM-L6-v2` generating **384-dimensional dense tensors** matching cosine similarity profiles).

### 4. Stateful Session Memory Framework

- **Thread Isolation:** Wrapped active prompt chains inside a state-aware **RunnableWithMessageHistory** block.
- **Streamlit Core Lifecycle Sync:** Bound temporary conversational historical states to Streamlit's structural execution stack via `st.session_state.chat_store`, ensuring transactional execution memory retention across asymmetric layout re-renders.

---

## 🛠️ Production Tech Stack

- **Core Framework:** Python 3.11+, LangChain Core, LCEL (LangChain Expression Language)
- **Orchestration & UI:** Streamlit (Stateful Session Cache Ecosystem)
- **Vector Database:** Pinecone Cloud (Serverless Architecture)
- **Models Utilized:** Meta Llama 3.1 8B (NVIDIA NIM Engine), OpenAI GPT-4o-mini (Text + Vision API)
- **Embeddings Configuration:** HuggingFace Transformers (`all-MiniLM-L6-v2`)
- **Data Processing & Utilities:** Pillow (PIL), Python-Dotenv, Urllib3 / Requests Optimization

---

## ⚡ Production Deployment & Local Setup

### 1. Environment Configuration

Create a `.env` file within the root directory of your workspace:

```bash
NVIDIA_API_KEY="nvapi-your-secure-nvidia-nim-token"
PINECONE_API_KEY="your-serverless-pinecone-cloud-token"
OPENAI_API_KEY="sk-proj-your-commercial-openai-token"
```
