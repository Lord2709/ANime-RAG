## Anime RAG

Local RAG app for anime recommendations and factual lookup using a CSV dataset, chunked ingestion, FAISS vector search, and a Streamlit UI with cinematic styling.

**Highlights**
- Chunked ingestion for better retrieval on long synopses
- FAISS vector search with optional BM25 tie‑break rerank
- Metadata stored alongside embeddings (genres, studio, score, etc.)
- Optional image rendering from `Image URL`
- Local LLM via Ollama (no hosted API required)

## Project Structure
- `app/ingestion.py` builds documents, embeddings, and FAISS index
- `app/retrieval.py` handles vector search + metadata filtering + dedupe
- `app/routing.py` wires retrieval to the LLM
- `app/streamlit_app.py` Streamlit UI
- `data/raw/anime_data23.csv` source dataset
- `data/processed/documents.json` generated documents
- `embeddings/faiss_index/` FAISS index + embeddings

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv anime
source anime/bin/activate
pip install -r requirements.txt
```

Install and run Ollama locally if you want LLM responses. The models currently used in the UI are:

- `phi3:mini`
- `deepseek-llm:7b-chat`
- `qwen:7b`
- `mistral:7b-instruct-v0.3-q4_0`
- `llama3.1:8b`

```bash
ollama pull llama3.1:8b
ollama serve
```

## Ingest Data (Build Index)
This reads the CSV, chunks synopsis text, builds embeddings, and writes the FAISS index.

```bash
python app/ingestion.py
```

## Run the App
```bash
streamlit run app/streamlit_app.py
```

## How Retrieval Works
- Each anime becomes one or more **chunked documents**.
- Embeddings are built from `content` only.
- Metadata is stored separately and can be used for filtering.
- Results are deduped by title to avoid multiple chunks of the same anime.
- A lightweight BM25 scorer is used as a tie‑breaker.
- If the top score is too low, the app returns a “no confident match” response.

## Chunking Settings
Edit these in `app/ingestion.py` if you want to change chunk size:

```
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
```

Re‑run ingestion after changing these values.

## Image Support
If your CSV has an `Image URL` column, the UI can display posters for retrieved titles.
Toggle this in the sidebar.

## Metadata Filtering
`app/retrieval.py` supports filters like:

```
filters = {
  "genres": ["Action", "Adventure"],
  "studio": "Bones",
  "type": "TV",
  "min_score": 7.5,
  "min_episodes": 12,
  "max_episodes": 26,
}
```

You can wire these into the UI if needed.

## Notes
- If you update `data/raw/anime_data23.csv`, re‑run `python app/ingestion.py`.
- Large models will require more RAM/VRAM depending on your machine.
