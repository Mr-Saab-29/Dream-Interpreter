# Dream-Interpreter

#  Dream Interpreter — RAG + Groq + Text‑to‑Video (Notebook Edition)

This repo contains a **notebook‑first** dream‑interpreter built with **LangChain**, **hybrid retrieval (BM25 + FAISS)**, **BAAI/BGE embeddings**, and **Groq LLMs**. It ships with a polished **Gradio Blocks** UI (dark mode + custom CSS) and an optional **text‑to‑video** step via **fal.ai** (fast‑SVD), which turns the interpretation into a short clip.

> This README is tailored to the uploaded notebook (`Dream Interpreter code.ipynb`). If you later refactor into a package, ping me and I’ll convert this to a standard multi‑module README.

---

## What it does

* **Interprets a pasted dream** using a **RAG pipeline** grounded in your **PDF/CSV** sources.
* **Loads sources from** `/content/Dream data` (Colab‑style path) using `PyMuPDFLoader` (PDF) and `CSVLoader`.
* **Chunks** with `RecursiveCharacterTextSplitter(chunk_size=256, overlap=50)`.
* **Hybrid retrieval**: `BM25Retriever(k=5)` + `FAISS` vector store using `HuggingFaceBgeEmbeddings(model="BAAI/bge-small-en", normalize=True)`.
* **(Optional) Cross‑encoder reranking**: `BAAI/bge-reranker-base` via `CrossEncoderReranker`.
* **LLM**: `ChatGroq(model_name="llama3-8b-8192", temperature=0)`.
* **UI**: Gradio Blocks with history, citations, and 1‑click **Generate Video**.
* **Video**: summarises the interpretation to a **concise scene prompt** and calls `fal-ai/fast-svd/text-to-video` to render a short clip.

---

## Architecture (Notebook)

```
initialize_dream_interpreter()
  ├─ Load PDFs/CSVs from /content/Dream data
  ├─ Split → Embed (BGE small) → FAISS
  ├─ Build BM25 + FAISS → EnsembleRetriever (k=5 each)
  ├─ (Optional) Cross‑encode rerank
  └─ ChatGroq(llama3-8b-8192)

interpret_dream(dream_text)
  ├─ Retrieve top passages
  ├─ Synthesize grounded answer (+citations)
  └─ Create short video scene prompt

generate_video_sync()
  └─ fal-ai/fast-svd/text-to-video (returns video URL)

launch_app()
  └─ Gradio Blocks UI (dream input → interpretation → video)
```

---

## Files

* `Dream Interpreter code.ipynb` — complete pipeline (data loading → retrieval → LLM → UI → video).
* `/content/Dream data/` — **required in Colab**. Put your Freud/Jung PDFs and any CSV notes here.

 If you’re running **locally**, create a folder (e.g., `data/`) and update the variable `path_data` in the notebook from `"/content/Dream data"` to your local path.

---

## Secrets & Environment

The notebook currently reads secrets from **Colab** `userdata`:

* **Groq**: `userdata.get("GROQ_API_KEy")` → sets `os.environ["GROQ_API_KEY"]`
* **fal.ai**: `userdata.get("FAL_AI_KEY")` → sets `os.environ["FAL_KEY"]`

> Note: there’s a **case typo** in the Groq key name (`GROQ_API_KEy`). Keep it as‑is in Colab, or fix the code to `GROQ_API_KEY`.

**Local .env alternative (recommended if you export to .py):**

```bash
GROQ_API_KEY=your_groq_key
FAL_KEY=your_fal_key
```

Load with `python-dotenv` or set in your shell before running.

---

## Getting Started

### A) Run in Google Colab (quickest)

1. Open the notebook and **upload** your sources into `/content/Dream data` (create the folder if missing).
2. In **Colab → Settings → Secrets**, add:

   * `GROQ_API_KEy` → your Groq API key (matches the notebook’s current spelling)
   * `FAL_AI_KEY` → your fal.ai API key
3. Run cells in order until **`launch_app()`** displays the Gradio URL.
4. Paste a dream → **Interpret** → optionally **Generate Video**.

### B) Run locally (Jupyter / VS Code)

```bash
# 1) Create and activate a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U pip
pip install -U langchain langchain-community gradio sentence-transformers \
    faiss-cpu rank-bm25 pymupdf torch requests fal-client python-dotenv

# 3) Put sources in ./data (or similar) and edit `path_data` in the notebook
# 4) Run all cells → call launch_app()
```

---

## UI Overview

* **Dream input** (`Textbox`) → **Interpret** (`Button`)
* **Output**: Markdown interpretation with **citations** (document index + snippet)
* **Video**: click **🎥 Generate Video** to create a short clip from the scene prompt
* **Clear**: resets input / output / video status

---

## Key Functions (as implemented)

* `initialize_dream_interpreter()` — loads files, builds BM25 + FAISS, sets `ChatGroq`
* `interpret_dream(dream_text)` — runs retrieval + LLM to produce (interpretation, video\_prompt)
* `generate_video_sync()` — synchronous wrapper around `fal_client.run_async("fal-ai/fast-svd/text-to-video")`
* `launch_app()` — Gradio Blocks UI (dark theme, custom CSS, sections for input/output/video)

---

## Configuration Tips

* **Chunking**: `chunk_size=256`, `chunk_overlap=50` (token‑light, good recall); raise size for longer quotes.
* **Retriever k**: `bm25_retriever.k = 5` (match FAISS top‑k for balanced fusion).
* **Embeddings**: `BAAI/bge-small-en` (fast CPU). For multilingual, try `bge-m3`.
* **Reranking**: enable `CrossEncoderReranker` with `BAAI/bge-reranker-base` to improve top‑k quality.
* **LLM**: `llama3-8b-8192` is fast/cheap; swap to `mixtral-8x7b` for richer prose.

---

##  Evaluate (lightweight)

Add a small list of (query, expected source ids) and verify:

* **Coverage**: cited passages include expected doc pages/rows
* **Relevance\@k**: top‑k contains the right concepts (manual MRR/nDCG is fine for a small set)

---

##  Suggested Requirements (pin as needed)

```txt
langchain>=0.2.0
langchain-community>=0.2.0
gradio>=4.40.0
sentence-transformers>=2.6.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2
pymupdf>=1.24.0
fal-client>=0.5.0
requests>=2.32.0
torch>=2.2.0
python-dotenv>=1.0.1
```

---

##  Troubleshooting

* **No documents found**: ensure your files are inside `/content/Dream data` (or update `path_data`).
* **Groq key not picked up**: in Colab secrets, the notebook expects `GROQ_API_KEy` (note the lowercase `y`).
* **PyMuPDF errors**: make sure `pymupdf` is installed; restart the kernel after install.
* **FAISS on macOS**: use `faiss-cpu`; if import fails, reinstall with matching Python version.
* **Slow/empty results**: increase `k` to 8–10, or reduce `chunk_size`/overlap; enable reranker.

---

##  License

MIT (or your preferred license).

---

##  Acknowledgments

* BAAI **BGE** family for embeddings & reranking
* **Groq** for low‑latency LLM inference
* **fal.ai** for fast SVD text‑to‑video
* LangChain, FAISS, rank‑bm25, PyMuPDF
