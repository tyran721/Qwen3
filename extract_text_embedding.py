import os
import re
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# =======================
# 路径配置
# =======================
ALL_CSV = "all.csv"

MODEL_PATH = "/media/disk2/ztr/model/Qwen3-Embedding-4B"

SAVE_DIR = "features/qwen3_text_all"
SAVE_PT = os.path.join(SAVE_DIR, "all_qwen3_text_embeddings.pt")
SAVE_NPY = os.path.join(SAVE_DIR, "all_qwen3_text_embeddings.npy")
SAVE_SLICE_IDS = os.path.join(SAVE_DIR, "slice_ids.txt")
SAVE_META = os.path.join(SAVE_DIR, "meta.json")

# 是否为每个切片单独保存一个 .pt 文件
SAVE_EACH_SLICE_PT = True
EACH_SLICE_DIR = os.path.join(SAVE_DIR, "per_slice_pt")

# 4090 + Qwen3-Embedding-4B，建议先 batch_size=1，跑通后可改成 2 或 4
BATCH_SIZE = 2

# 病理报告一般不长，1024 通常够用；如果报告特别长可以改 2048
MAX_SEQ_LENGTH = 1024

# 是否归一化 embedding，建议 True，方便后续 cosine / contrastive learning
NORMALIZE_EMBEDDINGS = True

# 是否加 instruction
USE_INSTRUCTION = True


# =======================
# 文本 instruction
# =======================
INSTRUCTION = (
    "Represent the Chinese pathology diagnosis report for histopathology-text "
    "multimodal learning. Focus on anatomical site, specimen source, epithelial "
    "morphology, invasion-related description, carcinoma-related description, "
    "dysplasia, inflammation, keratinization, and diagnostic uncertainty."
)


# =======================
# 工具函数
# =======================
def read_csv_safely(path):
    """
    兼容 utf-8-sig / gbk 编码
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def clean_text(x):
    """
    清洗病理报告文本
    """
    if pd.isna(x):
        return ""

    x = str(x)
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = x.replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_slice_id(x):
    """
    清洗切片号
    """
    if pd.isna(x):
        return ""

    x = str(x).strip()
    x = re.sub(r"\s+", "", x)

    if x.endswith(".0"):
        x = x[:-2]

    return x


def build_model_input(text):
    """
    构建 Qwen3 输入文本
    """
    if USE_INSTRUCTION:
        return INSTRUCTION + "\n" + text
    return text


def safe_filename(x):
    """
    避免文件名中出现特殊字符
    """
    x = str(x)
    x = x.replace("/", "_")
    x = x.replace("\\", "_")
    x = x.replace(":", "_")
    x = x.replace("*", "_")
    x = x.replace("?", "_")
    x = x.replace('"', "_")
    x = x.replace("<", "_")
    x = x.replace(">", "_")
    x = x.replace("|", "_")
    return x


# =======================
# 主函数
# =======================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if SAVE_EACH_SLICE_PT:
        os.makedirs(EACH_SLICE_DIR, exist_ok=True)

    # 读取 all.csv
    all_df = read_csv_safely(ALL_CSV)

    print("all.csv shape:", all_df.shape)
    print("columns:", list(all_df.columns))

    # 按你定义的 all.csv：
    # 第一列：病理切片号
    # 第六列：病理诊断报告
    slice_ids = all_df.iloc[:, 0].apply(clean_slice_id).tolist()
    report_texts = all_df.iloc[:, 5].apply(clean_text).tolist()

    assert len(slice_ids) == len(report_texts)

    model_inputs = [build_model_input(x) for x in report_texts]

    print("Total slices:", len(slice_ids))
    print("Example slice_id:", slice_ids[0])
    print("Example report:", report_texts[0][:200])

    # 加载 Qwen3-Embedding-4B
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

    model.max_seq_length = MAX_SEQ_LENGTH

    all_embeddings = []

    # 批量提取
    for start in tqdm(range(0, len(model_inputs), BATCH_SIZE), desc="Extracting embeddings"):
        end = min(start + BATCH_SIZE, len(model_inputs))
        batch_texts = model_inputs[start:end]

        with torch.no_grad():
            emb = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        emb = emb.detach().cpu().float()
        all_embeddings.append(emb)

    embeddings = torch.cat(all_embeddings, dim=0)

    print("Final embedding shape:", embeddings.shape)
    print("Number of slice ids:", len(slice_ids))

    assert embeddings.shape[0] == len(slice_ids), "embedding 数量和切片数量不一致"

    # 保存总文件
    save_obj = {
        "slice_ids": slice_ids,
        "texts": report_texts,
        "embeddings": embeddings,
        "model_path": MODEL_PATH,
        "embedding_dim": embeddings.shape[1],
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "use_instruction": USE_INSTRUCTION,
        "instruction": INSTRUCTION if USE_INSTRUCTION else "",
    }

    torch.save(save_obj, SAVE_PT)
    np.save(SAVE_NPY, embeddings.numpy())

    with open(SAVE_SLICE_IDS, "w", encoding="utf-8") as f:
        for sid in slice_ids:
            f.write(sid + "\n")

    meta = {
        "input_csv": ALL_CSV,
        "num_slices": len(slice_ids),
        "embedding_shape": list(embeddings.shape),
        "embedding_dim": int(embeddings.shape[1]),
        "model_path": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "use_instruction": USE_INSTRUCTION,
        "instruction": INSTRUCTION if USE_INSTRUCTION else "",
        "columns": list(all_df.columns),
    }

    with open(SAVE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 每个切片单独保存一个 .pt，方便后面 Dataset 直接按切片号读取
    if SAVE_EACH_SLICE_PT:
        for i, sid in enumerate(tqdm(slice_ids, desc="Saving per-slice pt")):
            out_path = os.path.join(EACH_SLICE_DIR, safe_filename(sid) + ".pt")
            torch.save(
                {
                    "slice_id": sid,
                    "text": report_texts[i],
                    "embedding": embeddings[i],
                },
                out_path,
            )

    print("\nSaved files:")
    print(SAVE_PT)
    print(SAVE_NPY)
    print(SAVE_SLICE_IDS)
    print(SAVE_META)

    if SAVE_EACH_SLICE_PT:
        print(EACH_SLICE_DIR)


if __name__ == "__main__":
    main()