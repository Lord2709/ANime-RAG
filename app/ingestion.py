import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR/ "data" / "raw" / "anime_data23.csv"
DATA_PROCESSED = BASE_DIR/ "data" / "processed" / "documents.json"
FAISS_DIR = BASE_DIR/ "embeddings" / "faiss_index"

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5" # For Generating embeddings

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text

def load_and_clean_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {path}")

    df = pd.read_csv(path)

    # Drop unwanted columns if they exist
    for col in ["Unnamed: 0", "anime_id", "Other name"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    df = df.replace(
        to_replace=["UNKNOWN", "unknown", "Unknown"],
        value=0
    )
    
    # Ensure numeric types where needed (will raise if non‑numeric garbage)
    for col in ["Score", "Episodes", "Rank", "Scored By"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text cleaning for key text columns
    for col in ["Name", "English name", "Genres", "Synopsis", "Studios"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    return df

def build_documents(df: pd.DataFrame) -> List[Dict]:
    """
    Convert the anime csv into a list of RAG documents with 'content' + 'metadata', similar to the exploration.ipynb structure
    """
    
    required_cols = [
        "Name",
        "English name",
        "Genres",
        "Type",
        "Episodes",
        "Studios",
        "Synopsis",
        "Score",
        "Duration",
        "Aired",
        "Producers",
        "Rank",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in CSV")
    
    df = df.fillna({
        "English name": "",
        "Name": "",
        "Genres": "",
        "Type": "",
        "Episodes": "",
        "Studios": "",
        "Synopsis": "",
        "Duration": "",
        "Aired": "",
        "Producers": "",
        "Rank": "",
        "Score": 0.0,
    })
    
    documents: List[Dict] = []
    
    for _, row in df.iterrows():
        title = row["English name"] if str(row["English name"]).strip() != "" else row["Name"]
        doc_text = f"""
        Title: {title}
        Genres: {row['Genres']}
        Type: {row['Type']}
        Episodes: {row['Episodes']}
        Studio: {row['Studios']}
        Synopsis: {row['Synopsis']}
        """.strip()
        
        doc = {
            "content": doc_text,
            "metadata": {
                "title": title,
                "score": float(row["Score"]) if not pd.isna(row["Score"]) else None,
                "genres": row["Genres"],
                "type": row["Type"],
                "episodes": row["Episodes"],
                "duration": row["Duration"],
                "aired": row["Aired"],
                "producer": row["Producers"],
                "studio": row["Studios"],
                "rank": row["Rank"],
            },
        }
        documents.append(doc)

    return documents

def save_documents(documents: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(documents)} documents to {path}")
    
# Now begins embeddings and Faiss index part

def build_embeddings(
    documents: List[Dict],
    model_name: str = EMBED_MODEL_NAME,
) -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [d["content"] for d in documents]
    print(f"Encoding {len(texts)} documents... ")
    
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True, # For BGE only 
    ).astype("float32")

    return embs

def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    dim = embs.shape[1]
    print(f"Building FAISS index with dim={dim}, n={embs.shape[0]}")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

def save_index_and_embeddings(index: faiss.IndexFlatIP, embs: np.ndarray, dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    index_path = dir_path / "anime.index"
    embs_path = dir_path / "embeddings.npy"
    
    faiss.write_index(index, str(index_path))
    np.save(embs_path, embs)
    
    print(f"Saved Faiss index to {index_path}")
    print(f"Saved Embeddings to {embs_path}")
    
def main():
    print(f"Loading and Cleaning Raw Data from {DATA_RAW}")
    df = load_and_clean_csv(DATA_RAW)
    
    print("Building Documents...")
    documents = build_documents(df)
    
    print(f"Saving documents JSON to {DATA_PROCESSED}")
    save_documents(documents, DATA_PROCESSED)
    
    print("Building Embeddings...")
    embs = build_embeddings(documents, EMBED_MODEL_NAME)
    
    print("Building FAISS Index...")
    index = build_faiss_index(embs)
    
    print("Saving index and embeddings...")
    save_index_and_embeddings(index, embs, FAISS_DIR)
    
    print("Ingestion pipeline completed.")
    
if __name__ == "__main__":
    main()
    
