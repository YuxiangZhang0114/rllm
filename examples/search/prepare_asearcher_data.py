"""准备 asearcher 数据集"""

import os

import pandas as pd

from rllm.data.dataset import DatasetRegistry


def prepare_asearcher_data(train_path=None, val_path=None, train_size=None, val_size=None):
    """
    加载 asearcher 数据集并注册到 DatasetRegistry。
    
    Args:
        train_path: 训练数据路径（默认从项目根目录查找）
        val_path: 验证数据路径（可选）
        train_size: 训练集最大样本数（可选）
        val_size: 验证集最大样本数（可选）
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # 默认路径
    if train_path is None:
        # 尝试从多个可能的位置查找数据
        possible_paths = [
            "data/asearcher_train/train.parquet",
            "../data/asearcher_train/train.parquet",
            "../../data/asearcher_train/train.parquet",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                train_path = path
                break
        
        if train_path is None:
            raise FileNotFoundError(
                "Cannot find asearcher training data. Please specify train_path or ensure "
                "data/asearcher_train/train.parquet exists."
            )
    
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    
    # 限制训练集大小
    if train_size is not None and train_size < len(train_df):
        train_df = train_df.head(train_size)
    
    # 转换为 rllm 格式
    train_data = []
    for _, row in train_df.iterrows():
        # 尝试多个可能的列名
        question = row.get("question") or row.get("query") or row.get("prompt")
        ground_truth = row.get("answer") or row.get("ground_truth") or row.get("response")
        
        if question is None:
            raise ValueError(f"Cannot find question field in data. Available columns: {list(row.keys())}")
        
        train_data.append({
            "question": question,
            "ground_truth": ground_truth,
            "data_source": "asearcher"
        })
    
    print(f"Processed {len(train_data)} training examples")
    
    # 注册训练数据集
    train_dataset = DatasetRegistry.register_dataset("asearcher", train_data, "train")
    
    # 处理验证数据集
    val_dataset = None
    if val_path is not None and os.path.exists(val_path):
        print(f"Loading validation data from: {val_path}")
        val_df = pd.read_parquet(val_path)
        
        if val_size is not None and val_size < len(val_df):
            val_df = val_df.head(val_size)
        
        val_data = []
        for _, row in val_df.iterrows():
            question = row.get("question") or row.get("query") or row.get("prompt")
            ground_truth = row.get("answer") or row.get("ground_truth") or row.get("response")
            
            val_data.append({
                "question": question,
                "ground_truth": ground_truth,
                "data_source": "asearcher"
            })
        
        print(f"Processed {len(val_data)} validation examples")
        val_dataset = DatasetRegistry.register_dataset("asearcher", val_data, "val")
    else:
        # 使用 hotpotqa 作为验证集
        print("No validation data provided, will use hotpotqa validation set")
        try:
            val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
            if val_dataset is None:
                print("hotpotqa validation set not found, will need to prepare it separately")
        except Exception as e:
            print(f"Failed to load hotpotqa validation set: {e}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = prepare_asearcher_data()
    
    if train_dataset:
        print(f"\nTrain dataset registered with {len(train_dataset)} examples")
        print(f"Sample: {train_dataset[0]}")
    
    if val_dataset:
        print(f"\nValidation dataset: {len(val_dataset)} examples")
        print(f"Sample: {val_dataset[0]}")
