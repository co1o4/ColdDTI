"""
extract_llm_features.py

使用预训练语言模型批量提取药物（SMILES）和蛋白质（氨基酸序列）的特征向量，
并将结果保存为 .pt 文件，供后续模型训练使用。

依赖：torch, pandas, tqdm, transformers
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from transformers import PreTrainedModel
# 动态注入一个空壳方法，骗过 MoLFormer 的强行调用，保护底层环境不被破坏
if not hasattr(PreTrainedModel, "warn_if_padding_and_no_attention_mask"):
    PreTrainedModel.warn_if_padding_and_no_attention_mask = lambda *args, **kwargs: None


# ────────────────────────────────────────────────────────────────────────────────
# 核心推理函数
# ────────────────────────────────────────────────────────────────────────────────

def extract_features(model, tokenizer, id_seq_dict, batch_size, max_length, device,
                     is_prot_t5=False):
    """
    通用批量推理函数：对一组序列进行编码并返回平均池化后的特征向量。

    参数:
        model       : 已加载并已调用 eval() 的 HuggingFace 模型
        tokenizer   : 与 model 配套的分词器
        id_seq_dict : dict，{id: sequence_str} 的映射
        batch_size  : 每批处理的序列数量
        max_length  : tokenizer 截断/填充的最大 token 长度
        device      : torch.device，模型所在设备
        is_prot_t5  : bool，是否为 ProtT5 模型（需要氨基酸间加空格，过滤 token_type_ids）

    返回:
        features_dict : dict，{id: 1D feature Tensor (CPU)}
    """
    ids = list(id_seq_dict.keys())
    seqs = list(id_seq_dict.values())
    features_dict = {}

    for batch_start in tqdm(range(0, len(ids), batch_size), desc="推理进度", leave=False):
        batch_ids = ids[batch_start: batch_start + batch_size]
        batch_seqs = seqs[batch_start: batch_start + batch_size]

        # ProtT5 要求氨基酸序列中每个残基间加空格，并将非标准氨基酸替换为 X
        if is_prot_t5:
            batch_seqs = [" ".join(re.sub(r"[UZOB]", "X", seq)) for seq in batch_seqs]

        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        # 过滤掉模型不接受的多余键（如 token_type_ids 对 T5/MoLFormer 无效）
        import inspect
        forward_params = inspect.signature(model.forward).parameters
        encoded = {k: v.to(device) for k, v in encoded.items() if k in forward_params}

        input_ids     = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model(**encoded)

        # last_hidden_state 形状: (batch, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # 带掩码的平均池化
        mask_expanded = attention_mask.unsqueeze(-1).float()          # (batch, seq_len, 1)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)   # (batch, hidden_size)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)       # (batch, 1)
        pooled = sum_hidden / token_counts                             # (batch, hidden_size)

        for i, seq_id in enumerate(batch_ids):
            features_dict[seq_id] = pooled[i].cpu()

    return features_dict


# ────────────────────────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # ── 设备配置 ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")
    if device.type == "cuda":
        print(f"[设备] GPU 型号: {torch.cuda.get_device_name(0)}")

    # ── 路径配置 ──────────────────────────────────────────────────────────────
    project_root = os.path.dirname(os.path.abspath(__file__))
    drug_csv_path    = os.path.join(project_root, "data", "DrugKB", "drug.csv")
    protein_csv_path = os.path.join(project_root, "data", "DrugKB", "protein.csv")
    drug_feat_out    = os.path.join(project_root, "data", "drug_molformer_features.pt")
    protein_feat_out = os.path.join(project_root, "data", "protein_prott5_features.pt")

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    print("\n[数据] 正在读取 drug.csv ...")
    # drug.csv 列名：drugid1, drug_smile1
    drug_df = pd.read_csv(drug_csv_path)
    drug_id_col    = drug_df.columns[0]   # drugid1
    drug_smiles_col = drug_df.columns[1]  # drug_smile1
    drug_df = drug_df.dropna(subset=[drug_id_col, drug_smiles_col])
    id_smiles_dict = dict(zip(drug_df[drug_id_col].astype(str),
                              drug_df[drug_smiles_col].astype(str)))
    print(f"[数据] 药物数量: {len(id_smiles_dict)}")

    print("\n[数据] 正在读取 protein.csv ...")
    # protein.csv 列名：protein_name, protein_seq
    protein_df = pd.read_csv(protein_csv_path)
    protein_id_col  = protein_df.columns[0]  # protein_name
    protein_seq_col = protein_df.columns[1]  # protein_seq
    protein_df = protein_df.dropna(subset=[protein_id_col, protein_seq_col])
    id_seq_dict = dict(zip(protein_df[protein_id_col].astype(str),
                           protein_df[protein_seq_col].astype(str)))
    print(f"[数据] 靶点数量: {len(id_seq_dict)}")

    # ── 模型加载：药物（MoLFormer-XL） ───────────────────────────────────────
    drug_model_name = "ibm-research/MoLFormer-XL-both-10pct"
    print(f"\n[模型] 正在加载药物模型: {drug_model_name}")
    drug_tokenizer = AutoTokenizer.from_pretrained(drug_model_name, trust_remote_code=True)
    drug_model = AutoModel.from_pretrained(drug_model_name, trust_remote_code=True)
    drug_model = drug_model.to(device)
    drug_model.eval()
    print("[模型] 药物模型加载完毕")

    # ── 模型加载：蛋白质（ProtT5-XL） ────────────────────────────────────────
    protein_model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"\n[模型] 正在加载蛋白质模型: {protein_model_name}")
    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name, do_lower_case=False)
    protein_model = T5EncoderModel.from_pretrained(protein_model_name)
    protein_model = protein_model.to(device)
    protein_model.eval()
    print("[模型] 蛋白质模型加载完毕")

    # ── 批量推理：药物特征 ────────────────────────────────────────────────────
    print("\n[推理] 开始提取药物特征 ...")
    drug_features_dict = extract_features(
        model=drug_model,
        tokenizer=drug_tokenizer,
        id_seq_dict=id_smiles_dict,
        batch_size=32,
        max_length=512,
        device=device,
        is_prot_t5=False,
    )
    print(f"[推理] 药物特征提取完毕，共 {len(drug_features_dict)} 条")
    # 打印第一个特征的维度以供核对
    sample_drug_feat = next(iter(drug_features_dict.values()))
    print(f"[维度] 单个药物特征维度: {sample_drug_feat.shape}")

    # ── 批量推理：蛋白质特征 ──────────────────────────────────────────────────
    print("\n[推理] 开始提取蛋白质特征 ...")
    protein_features_dict = extract_features(
        model=protein_model,
        tokenizer=protein_tokenizer,
        id_seq_dict=id_seq_dict,
        batch_size=4,
        max_length=1024,
        device=device,
        is_prot_t5=True,
    )
    print(f"[推理] 蛋白质特征提取完毕，共 {len(protein_features_dict)} 条")
    sample_protein_feat = next(iter(protein_features_dict.values()))
    print(f"[维度] 单个蛋白质特征维度: {sample_protein_feat.shape}")

    # ── 保存离线特征文件 ──────────────────────────────────────────────────────
    print(f"\n[保存] 药物特征 → {drug_feat_out}")
    torch.save(drug_features_dict, drug_feat_out)

    print(f"[保存] 蛋白质特征 → {protein_feat_out}")
    torch.save(protein_features_dict, protein_feat_out)

    # ── 汇总信息 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("特征提取完成，汇总如下：")
    print(f"  药物数量      : {len(drug_features_dict)}")
    print(f"  药物特征维度  : {sample_drug_feat.shape[0]}")
    print(f"  蛋白质数量    : {len(protein_features_dict)}")
    print(f"  蛋白质特征维度: {sample_protein_feat.shape[0]}")
    print(f"  药物特征文件  : {drug_feat_out}")
    print(f"  蛋白质特征文件: {protein_feat_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
