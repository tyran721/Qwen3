import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "/media/disk2/ztr/model/Qwen3-Embedding-4B"

model = SentenceTransformer(
    MODEL_PATH,
    model_kwargs={
        "torch_dtype": torch.float16,
        "device_map": "cuda",
        "attn_implementation": "sdpa",
        "local_files_only": True,
    },
    tokenizer_kwargs={
        "padding_side": "left",
        "local_files_only": True,
    },
)

sentences = ["你好", "今天天气不错"]

embeddings = model.encode(
    sentences,
    batch_size=1,
    normalize_embeddings=True,
)

print(embeddings.shape)