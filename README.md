# 🛡️ Enterprise Claims Operations Assistant

An advanced, multi-document Retrieval-Augmented Generation (RAG) application built to assist Insurance Underwriters and Claims Agents. This tool ingests complex, often conflicting insurance policies (Auto, Homeowners, Commercial) and synthesizes highly accurate coverage assessments using Gemini 2.5 Flash.

Unlike basic chatbots, this application is designed for enterprise trust: it employs hybrid search, handles cross-policy contradictions, refuses to guess without context, and explicitly cites the exact document and clause used to make its determination.

## ✨ Key Features

* **Hybrid Search Retrieval:** Combines Dense Vector Search (ChromaDB) for semantic understanding with Sparse Lexical Search (BM25) to capture exact alphanumeric policy codes and hyper-specific insurance jargon.
* **Cross-Encoder Reranking:** Utilizes the `BAAI/bge-reranker-base` model to dynamically score and filter a wide candidate pool, ensuring only the most logically relevant clauses reach the LLM context window.
* **Multi-Document Synthesis:** Designed to evaluate overlapping or conflicting coverage rules simultaneously (e.g., distinguishing between a vehicle covered under an Auto policy vs. a laptop denied under a Homeowners policy in the same incident).
* **Source Transparency:** Surfaces the exact chunk of text and the source filename used by the LLM, allowing human agents to audit the AI's logic with zero hallucinations on verified context.
* **Streamlit UI:** A clean, chat-based interface with live status spinners mapping the backend RAG pipeline.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Google Gemini 2.5 Flash
* **Dense Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) & ChromaDB
* **Sparse Retrieval:** BM25 (`rank_bm25`)
* **Reranker:** HuggingFace Cross-Encoder (`BAAI/bge-reranker-base`)
* **Document Processing:** PyPDFLoader

## 📁 Project Structure

```text
claims-ops-assistant/
|-- data/
|   |-- policies/ (Raw PDF insurance policies: Auto, Home, BOP)
|-- db/
|   |-- chroma_db/ (Local SQLite vector database)
|-- notebooks/
|   |-- 1_ingestion_test.ipynb (Builds the vector database)
|   |-- 2_retrieval_test.ipynb (Tests similarity search)
|   |-- 3_generation_test.ipynb (Tests end-to-end RAG without UI)
|-- src/
|   |-- retrieval.py (Logic for connecting to Chroma and fetching chunks)
|   |-- generation.py (Logic for formatting context and calling Gemini)
|-- .env (Environment variables / API Keys)
|-- .gitignore (Git ignore rules)
|-- app.py (Main Streamlit application)
```

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/claims-ops-assistant.git](https://github.com/yourusername/claims-ops-assistant.git)
cd claims-ops-assistant
```

### 2. Create a virtual environment
```bash
conda create -n claims-ops-assistant python=3.10
conda activate claims-ops-assistant
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a .env file in the root directory and add your Google Gemini API key:
```bash
GOOGLE_API_KEY="your_api_key_here"
```

## 🧠 Usage

### Step 1: Ingest the Knowledge Base
Before running the app, you must chunk and embed the PDF policies into the local vector database.
1. Open `notebooks/0_ingestion_test.ipynb`.
2. Run all cells to populate the `db/chroma_db/` folder.

### Step 2: Run the Application
Launch the Streamlit frontend from the root directory:
```bash
streamlit run app.py
```