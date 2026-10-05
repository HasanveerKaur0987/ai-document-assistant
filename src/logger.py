import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Silence noisy libraries
for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name):
    return logging.getLogger(name)