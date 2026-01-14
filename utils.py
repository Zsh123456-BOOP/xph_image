import os

import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import copy
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error


def build_graph(data_df, num_students, num_items, correct=True, item_type='exercise', concept_offset=0,
                graph_dir='./graphs'):
    """
    构建学生-习题/概念二分图的归一化邻接矩阵。
    优化：使用纯稀疏归一化，避免 adj.toarray() 导致的内存爆炸。
    """
    graph_filename = f"graph_{item_type}_{correct}_s{num_students}_i{num_items}.pkl"
    graph_path = os.path.join(graph_dir, graph_filename)

    # 检查图是否已存在
    if os.path.exists(graph_path):
        print(f"Loading pre-built graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            adj_data = pickle.load(f)
        # 支持新格式 (indices, values, shape) 或旧格式 coo_matrix
        if isinstance(adj_data, tuple) and len(adj_data) == 3:
            indices, values, shape = adj_data
            return torch.sparse_coo_tensor(indices, values, shape).coalesce()
        elif isinstance(adj_data, coo_matrix):
            indices = torch.LongTensor(np.vstack([adj_data.row, adj_data.col]))
            values = torch.FloatTensor(adj_data.data)
            shape = torch.Size(adj_data.shape)
            return torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return adj_data

    print(f"Graph not found, building graph...")
    print(f"  Dataset: {num_students} students, {num_items} items, type={item_type}, correct={correct}")
    filtered_df = data_df[data_df['label'] == (1 if correct else 0)]
    rows = []
    cols = []

    if item_type == 'exercise':
        for _, row in filtered_df.iterrows():
            stu_id = int(row['stu_id'])
            exer_id = int(row['exer_id'])
            rows.append(stu_id)
            cols.append(exer_id)
    else:
        for _, row in filtered_df.iterrows():
            stu_id = int(row['stu_id'])
            cpt_seq = row['cpt_seq']
            if isinstance(cpt_seq, str):
                cpts = [int(c) for c in cpt_seq.split(',')]
            else:
                cpts = [int(cpt_seq)]
            for cpt_id in cpts:
                rows.append(stu_id)
                cols.append(cpt_id)

    data = np.ones(len(rows), dtype=np.float32)
    rows_np = np.array(rows, dtype=np.int64)
    cols_np = np.array(cols, dtype=np.int64)

    # 构建二分图邻接矩阵索引
    n_total = num_students + num_items
    row_stu_item = rows_np
    col_stu_item = cols_np + num_students
    row_item_stu = cols_np + num_students
    col_item_stu = rows_np

    adj_rows = np.concatenate([row_stu_item, row_item_stu])
    adj_cols = np.concatenate([col_stu_item, col_item_stu])
    adj_data = np.concatenate([data, data])

    # === 纯稀疏归一化：D^{-1/2} A D^{-1/2} ===
    adj_coo = coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(n_total, n_total))
    
    # 计算度数（稀疏 sum）
    rowsum = np.array(adj_coo.sum(axis=1)).flatten()
    epsilon = 1e-12
    d_inv_sqrt = np.power(rowsum + epsilon, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # 对 COO values 做归一化：val * d_inv_sqrt[row] * d_inv_sqrt[col]
    norm_values = adj_coo.data * d_inv_sqrt[adj_coo.row] * d_inv_sqrt[adj_coo.col]
    
    # 构造归一化后的稀疏张量
    indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(norm_values)
    shape = torch.Size([n_total, n_total])
    
    # 保存为新格式 (indices, values, shape)
    os.makedirs(graph_dir, exist_ok=True)
    with open(graph_path, 'wb') as f:
        pickle.dump((indices, values, shape), f)
    print(f"Graph saved to {graph_path}")

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_min = 0
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_auc, model):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_min:.6f} --> {val_auc:.6f}). Saving model state...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_auc_min = val_auc


def train_epoch(model, train_loader, optimizer, device, adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc, args, epoch, verbose=True):
    """
    训练一个 epoch。
    保持原始训练语义：每 batch 计算 GCN + Fusion 并参与梯度更新。
    优化：使用 collate_fn 预处理的 padded tensor。
    """
    model.train()
    total_loss = 0
    total_aux_losses = {'fusion_se': 0, 'fusion_sc': 0, 'contrastive_exer': 0, 'contrastive_cpt': 0}
    
    pbar = tqdm(train_loader, desc='Training', disable=not verbose)
    for batch in pbar:
        # 新版 collate_fn 返回 5 元素：(stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels)
        stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels = batch
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        cpt_ids_padded = cpt_ids_padded.to(device)
        cpt_mask = cpt_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 原始训练逻辑：每 batch 调用 forward，GCN + Fusion 参与梯度更新
        predictions, aux_losses, _ = model(
            stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels,
            adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc
        )
        
        main_loss = F.binary_cross_entropy(predictions, labels)
        fusion_weight = args.lambda_fusion * min(1.0, epoch / args.fusion_warmup_epochs)
        contrastive_weight = args.lambda_contrastive * max(args.contrastive_min_weight, 1.0 - epoch / args.contrastive_decay_epochs)
        aux_loss = (fusion_weight * (aux_losses['fusion_se'] + aux_losses['fusion_sc']) +
                    contrastive_weight * (aux_losses['contrastive_exer'] + aux_losses['contrastive_cpt']))
        loss = main_loss + aux_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        for key in total_aux_losses:
            total_aux_losses[key] += aux_losses[key].item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    avg_aux_losses = {k: v / len(train_loader) for k, v in total_aux_losses.items()}
    return avg_loss, avg_aux_losses


def evaluate(model, data_loader, device, adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc):
    """
    评估模型。
    保持原始逻辑。
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_knowledge_states = []
    
    with torch.no_grad():
        for batch in data_loader:
            stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            cpt_ids_padded = cpt_ids_padded.to(device)
            cpt_mask = cpt_mask.to(device)
            labels = labels.to(device)
            
            predictions, aux_losses, knowledge_states = model(
                stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels,
                adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc
            )
            
            loss = F.binary_cross_entropy(predictions, labels)
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_knowledge_states.extend(knowledge_states.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    pred_labels = (all_predictions >= 0.5).astype(int)
    accuracy = (pred_labels == all_labels).mean()
    
    try:
        auc = roc_auc_score(all_labels, all_predictions)
        precision = precision_score(all_labels, pred_labels, zero_division=0)
        recall = recall_score(all_labels, pred_labels, zero_division=0)
        f1 = f1_score(all_labels, pred_labels, zero_division=0)
        RMSE = np.sqrt(mean_squared_error(all_labels, all_predictions))
        conf_matrix = confusion_matrix(all_labels, pred_labels)
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        print("Using default values. This may happen with very small batches or imbalanced data.")
        auc = 0.5; precision = 0.5; recall = 0.5; f1 = 0.5; RMSE = 0.5
        conf_matrix = np.array([[0, 0], [0, 0]])
    
    avg_loss = total_loss / len(data_loader)
    metrics = {
        'accuracy': accuracy, 'auc': auc, 'precision': precision, 'recall': recall,
        'f1': f1, 'rmse': RMSE, 'confusion_matrix': conf_matrix
    }
    return avg_loss, metrics, all_knowledge_states
