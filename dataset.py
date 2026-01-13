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
    stu_ids, exer_ids, cpts_list, labels = zip(*batch)
    return torch.LongTensor(stu_ids), torch.LongTensor(exer_ids), list(cpts_list), torch.FloatTensor(labels)
