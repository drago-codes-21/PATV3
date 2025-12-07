# Embedding Model

Place offline-compatible SentenceTransformer or Hugging Face model weights
inside this directory (e.g. `pat/embeddings/model/`). The application will
load the model from the path defined in `pat.config.settings`. No downloads are
performed at runtime, so ensure the full model folder is available locally
before invoking the pipeline. For best results, drop the `all-mpnet-base`
SentenceTransformer into `pat/embeddings/model/`, which the pipeline loads
automatically when computing semantic similarity or span-level features.
