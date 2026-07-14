import os
import re
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =======================
# 路径配置
# =======================
INPUT_XLSX = r"D:\mine\code\MSRL\gastric_data_report_generation.xlsx"

# 已经生成好的总 embedding 文件
EMBEDDING_PT = r"D:\mine\code\MSRL\features\qwen3_text_all\all_qwen3_text_embeddings.pt"

OUTPUT_DIR = r"D:\mine\code\MSRL\features\qwen3_text_all\classifier"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_classifier.pt")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "standard_scaler.npz")
SPLIT_SAVE_PATH = os.path.join(OUTPUT_DIR, "train_val_split.csv")
PREDICTION_SAVE_PATH = os.path.join(OUTPUT_DIR, "val_predictions.csv")
METRICS_SAVE_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CONFUSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")


# =======================
# 训练配置
# =======================
RANDOM_SEED = 42
TEST_SIZE = 0.30
BATCH_SIZE = 64
HIDDEN_DIM = 512
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12

CLASS_NAMES = {
    0: "非肿瘤性胃病变",
    1: "低级别上皮内瘤变",
    2: "高级别上皮内瘤变",
    3: "粘液腺癌",
    4: "腺癌",
    5: "印戒细胞癌",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_slice_id(x):
    if pd.isna(x):
        return ""
    value = re.sub(r"\s+", "", str(x).strip())
    if value.endswith(".0"):
        value = value[:-2]
    return value


def clean_raw_label(x):
    if pd.isna(x):
        return ""
    value = str(x).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def map_label(x):
    raw = clean_raw_label(x)
    if raw in {"0-0", "0-1", "0-2", "0-3", "0-4"}:
        return 0
    if raw in {"1", "2", "3", "4", "5"}:
        return int(raw)
    raise ValueError(f"无法识别标签: {x!r}")


def load_embedding_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 embedding 文件: {path}")

    # PyTorch 2.6+ 默认 weights_only=True，字典中含普通对象时需要显式关闭。
    data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise TypeError("embedding .pt 文件应当是一个字典")
    if "slice_ids" not in data or "embeddings" not in data:
        raise KeyError("embedding .pt 中必须包含 slice_ids 和 embeddings")

    slice_ids = [clean_slice_id(x) for x in data["slice_ids"]]
    embeddings = data["embeddings"]

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if not torch.is_tensor(embeddings):
        raise TypeError("embeddings 必须是 Tensor 或 NumPy 数组")

    embeddings = embeddings.detach().cpu().float()

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 应为二维矩阵，实际形状: {tuple(embeddings.shape)}")
    if len(slice_ids) != embeddings.shape[0]:
        raise ValueError(
            f"SlideId 数量 {len(slice_ids)} 与 embedding 数量 "
            f"{embeddings.shape[0]} 不一致"
        )

    texts = data.get("texts", [""] * len(slice_ids))
    if len(texts) != len(slice_ids):
        texts = [""] * len(slice_ids)

    return slice_ids, list(texts), embeddings


def load_and_align_labels(xlsx_path, embedding_slice_ids):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"找不到 Excel: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    if df.shape[1] < 4:
        raise ValueError(f"Excel 至少需要4列，当前只有 {df.shape[1]} 列")

    excel_ids = df.iloc[:, 0].apply(clean_slice_id).tolist()
    raw_labels = df.iloc[:, 3].apply(clean_raw_label).tolist()

    # 最可靠情况：数量和顺序都与提取 embedding 时一致。
    if len(excel_ids) == len(embedding_slice_ids) and excel_ids == embedding_slice_ids:
        print("Excel 与 embedding 的 SlideId 数量、顺序完全一致。")
        return raw_labels

    # 顺序不一致时，尝试按唯一 SlideId 对齐。
    excel_series = pd.Series(excel_ids)
    emb_series = pd.Series(embedding_slice_ids)

    duplicate_excel = excel_series[excel_series.duplicated(keep=False)].unique().tolist()
    duplicate_emb = emb_series[emb_series.duplicated(keep=False)].unique().tolist()

    if duplicate_excel or duplicate_emb:
        raise ValueError(
            "Excel 与 embedding 顺序不一致，且存在重复 SlideId，无法安全自动对齐。\n"
            f"Excel 重复示例: {duplicate_excel[:10]}\n"
            f"Embedding 重复示例: {duplicate_emb[:10]}"
        )

    label_by_id = dict(zip(excel_ids, raw_labels))
    missing = [sid for sid in embedding_slice_ids if sid not in label_by_id]
    if missing:
        raise ValueError(f"有 {len(missing)} 个 embedding SlideId 在 Excel 中找不到，例如: {missing[:10]}")

    print("Excel 与 embedding 顺序不同，已根据唯一 SlideId 自动对齐。")
    return [label_by_id[sid] for sid in embedding_slice_ids]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        targets.append(y.cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    predictions = probabilities.argmax(axis=1)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return accuracy, macro_f1, targets, predictions, probabilities


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    slice_ids, texts, embeddings = load_embedding_pt(EMBEDDING_PT)
    raw_labels = load_and_align_labels(INPUT_XLSX, slice_ids)

    valid_indices = []
    mapped_labels = []
    invalid_rows = []

    for i, raw_label in enumerate(raw_labels):
        try:
            if not raw_label:
                raise ValueError("空标签")
            mapped = map_label(raw_label)
            valid_indices.append(i)
            mapped_labels.append(mapped)
        except ValueError:
            invalid_rows.append((i, slice_ids[i], raw_label))

    if invalid_rows:
        print(f"跳过无效或空标签样本: {len(invalid_rows)} 条")
        print("示例:", invalid_rows[:10])

    if not valid_indices:
        raise ValueError("没有可用于训练的有效标签")

    index_tensor = torch.tensor(valid_indices, dtype=torch.long)
    embeddings = embeddings[index_tensor]
    slice_ids = [slice_ids[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]
    raw_labels = [raw_labels[i] for i in valid_indices]
    labels = np.asarray(mapped_labels, dtype=np.int64)

    print(f"\n有效样本数: {len(labels)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print("合并后的标签分布:")
    for label in sorted(CLASS_NAMES):
        count = int((labels == label).sum())
        print(f"  {label} - {CLASS_NAMES[label]}: {count}")

    present_classes = sorted(np.unique(labels).tolist())
    missing_classes = sorted(set(CLASS_NAMES) - set(present_classes))
    if missing_classes:
        raise ValueError(f"数据中缺少类别: {missing_classes}")

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # 只用训练集拟合标准化参数，避免验证集信息泄漏。
    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_idx].numpy()).astype(np.float32)
    x_val = scaler.transform(embeddings[val_idx].numpy()).astype(np.float32)
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    np.savez(
        SCALER_SAVE_PATH,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 使用训练集类别权重，缓解小类别样本不足。
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    class_weights = len(y_train) / (len(CLASS_NAMES) * counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n分类器训练设备: {device}")

    model = MLPClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    patience = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_examples += x.size(0)

        train_loss = total_loss / max(total_examples, 1)
        val_accuracy, val_macro_f1, _, _, _ = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
            "val_macro_f1": float(val_macro_f1),
        })

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("训练过程中没有获得有效模型")

    model.load_state_dict(best_state)
    val_accuracy, val_macro_f1, y_true, y_pred, probabilities = evaluate(
        model, val_loader, device
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES.keys()),
        target_names=[CLASS_NAMES[i] for i in CLASS_NAMES],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES.keys()))

    print("\n最佳 epoch:", best_epoch)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro-F1: {val_macro_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES.keys()),
        target_names=[CLASS_NAMES[i] for i in CLASS_NAMES],
        zero_division=0,
        digits=4,
    ))

    torch.save({
        "model_state_dict": best_state,
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "num_classes": len(CLASS_NAMES),
        "dropout": DROPOUT,
        "class_names": CLASS_NAMES,
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(val_accuracy),
        "best_val_macro_f1": float(val_macro_f1),
        "embedding_pt": EMBEDDING_PT,
    }, MODEL_SAVE_PATH)

    split_df = pd.DataFrame({
        "slice_id": slice_ids,
        "raw_label": raw_labels,
        "label": labels,
        "label_name": [CLASS_NAMES[x] for x in labels],
        "split": "train",
    })
    split_df.loc[val_idx, "split"] = "val"
    split_df.to_csv(SPLIT_SAVE_PATH, index=False, encoding="utf-8-sig")

    prediction_data = {
        "slice_id": [slice_ids[i] for i in val_idx],
        "text": [texts[i] for i in val_idx],
        "raw_label": [raw_labels[i] for i in val_idx],
        "true_label": y_true,
        "true_label_name": [CLASS_NAMES[int(x)] for x in y_true],
        "pred_label": y_pred,
        "pred_label_name": [CLASS_NAMES[int(x)] for x in y_pred],
        "correct": y_true == y_pred,
    }
    for class_id in CLASS_NAMES:
        prediction_data[f"prob_{class_id}"] = probabilities[:, class_id]

    pd.DataFrame(prediction_data).to_csv(
        PREDICTION_SAVE_PATH, index=False, encoding="utf-8-sig"
    )

    class_labels = [f"{i}_{CLASS_NAMES[i]}" for i in CLASS_NAMES]
    pd.DataFrame(
        matrix,
        index=[f"true_{x}" for x in class_labels],
        columns=[f"pred_{x}" for x in class_labels],
    ).to_csv(CONFUSION_SAVE_PATH, encoding="utf-8-sig")

    metrics = {
        "embedding_pt": EMBEDDING_PT,
        "num_samples": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "train_ratio": float(1.0 - TEST_SIZE),
        "val_ratio": float(TEST_SIZE),
        "best_epoch": int(best_epoch),
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(val_macro_f1),
        "class_names": CLASS_NAMES,
        "class_counts_all": {
            str(i): int((labels == i).sum()) for i in CLASS_NAMES
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "history": history,
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n结果已保存:")
    print(MODEL_SAVE_PATH)
    print(SCALER_SAVE_PATH)
    print(SPLIT_SAVE_PATH)
    print(PREDICTION_SAVE_PATH)
    print(METRICS_SAVE_PATH)
    print(CONFUSION_SAVE_PATH)


if __name__ == "__main__":
    main()