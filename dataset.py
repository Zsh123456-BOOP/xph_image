import torch
from torch.utils.data import Dataset

class CDDataset(Dataset):
    def __init__(self, data_df, concept_offset=0):
        """
        认知诊断数据集
        
        Args:
            data_df: 包含stu_id, exer_id, cpt_seq, label的DataFrame
            concept_offset: (已废弃) 保留此参数仅为向后兼容，实际不再使用
                          原因：习题GCN和知识点GCN使用独立的embedding，无需offset区分
        """
        self.data = data_df
        # concept_offset已优化移除，保留参数仅为API兼容性

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        stu_id = int(row['stu_id'])
        exer_id = int(row['exer_id'])
        cpt_seq = row['cpt_seq']
        if isinstance(cpt_seq, str):
            # 优化：不加 concept_offset，直接使用映射后的概念ID
            # 原因：习题GCN和知识点GCN是独立的，不需要通过offset区分索引空间
            # 优化前：cpts = [int(c) + self.concept_offset for c in ...]
            # 优化后：直接使用映射后的ID（已在main.py中完成映射）
            cpts = [int(c) for c in cpt_seq.split(',')]
        else:
            cpts = [int(cpt_seq)]
        label = int(row['label'])
        return stu_id, exer_id, cpts, label


def collate_fn(batch):
    """
    优化后的 collate_fn：在此处完成 padding，避免在 model.forward 中使用 Python 循环
    
    Returns:
        stu_ids: LongTensor [B]
        exer_ids: LongTensor [B]
        cpt_ids_padded: LongTensor [B, L] - L 为 batch 内最大概念数
        cpt_mask: BoolTensor [B, L] - True 表示有效位置，False 表示 padding
        labels: FloatTensor [B]
    """
    stu_ids, exer_ids, cpts_list, labels = zip(*batch)
    
    # 计算 batch 内最大概念数量
    max_len = max(len(cpts) for cpts in cpts_list)
    batch_size = len(cpts_list)
    
    # 预分配张量（避免循环中创建）
    cpt_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    cpt_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # 填充
    for i, cpts in enumerate(cpts_list):
        length = len(cpts)
        cpt_ids_padded[i, :length] = torch.LongTensor(cpts)
        cpt_mask[i, :length] = True
    
    return (
        torch.LongTensor(stu_ids),
        torch.LongTensor(exer_ids),
        cpt_ids_padded,
        cpt_mask,
        torch.FloatTensor(labels)
    )
