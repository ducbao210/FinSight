"""Download Hugging Face models into the image cache during Docker build."""

from sentence_transformers import CrossEncoder, SentenceTransformer
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.configs import settings


def main() -> None:
    print(f"Downloading embedding model: {settings.embedding_model}", flush=True)
    SentenceTransformer(settings.embedding_model, token=settings.hf_token or None)
    if settings.use_reranker:
        print(f"Downloading reranker model: {settings.reranker_model}", flush=True)
        CrossEncoder(settings.reranker_model, token=settings.hf_token or None)
    else:
        print("Reranker disabled; skipping reranker download", flush=True)
    print("Model download complete", flush=True)


if __name__ == "__main__":
    main()
