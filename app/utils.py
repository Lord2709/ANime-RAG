from pathlib import Path
import json
import time
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR/ "data" / "raw" / "anime_data23.csv"
DATA_PROCESSED = BASE_DIR/ "data" / "processed" / "documents.json"
FAISS_DIR = BASE_DIR/ "embeddings" / "faiss_index"

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5" # For Generating embeddings
CHAT_MODEL_NAME = "" # For text generation
# Retrieve top 50 similar docs from Faiss and return top 5 to send to llm
TOP_K_RETRIEVE = "50"
TOP_K_RETURN = "5"

# Safety can be removed
FILTER_NSFW = True
NSFW_KEYWORDS = ["Hentai", "Ecchi", "Rx - Hentai"]

# This will load the json file and parse it into Python objects(dict/list) 
def load_json(path: Path) -> Any:
    with path.open("r", encoding = "utf-8") as f:
        return json.load(f)

# Autocreates directories if required and saves python objects as json(formatted)
def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        
# Just to understand basic runtime of a function for either optimization further or understanding the load.        
def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{fn.__name__} took {elapsed:.2f}s")
        return res
    return wrapper        