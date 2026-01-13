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
    # 构建图的文件名，包含数据集特征以避免混淆
    # 添加 num_students 和 num_items 来区分不同数据集
    graph_filename = f"graph_{item_type}_{correct}_s{num_students}_i{num_items}.pkl"
    graph_path = os.path.join(graph_dir, graph_filename)

    # 检查图是否已存在
    if os.path.exists(graph_path):
        print(f"Loading pre-built graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            adj_matrix = pickle.load(f)
        # 如果加载的是coo_matrix，需要转换为PyTorch稀疏张量
        if isinstance(adj_matrix, coo_matrix):
            indices = torch.LongTensor(np.vstack([adj_matrix.row, adj_matrix.col]))
            values = torch.FloatTensor(adj_matrix.data)
            shape = torch.Size(adj_matrix.shape)
            return torch.sparse.FloatTensor(indices, values, shape)
        return adj_matrix

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
        # 构建学生-概念图
        # 优化说明：
        # 之前的设计使用 concept_offset = num_exercises 来区分习题和知识点的索引空间
        # 但实际上习题GCN和知识点GCN是完全独立的，不需要区分
        # 优化后：直接使用映射后的概念ID (0~num_concepts-1)，不加偏移
        # 效果：大幅减少参数量（从17,799降到123个embedding，节省99.3%）
        for _, row in filtered_df.iterrows():
            stu_id = int(row['stu_id'])
            cpt_seq = row['cpt_seq']
            if isinstance(cpt_seq, str):
                # 不加 concept_offset，直接使用映射后的概念ID
                cpts = [int(c) for c in cpt_seq.split(',')]
            else:
                cpts = [int(cpt_seq)]
            for cpt_id in cpts:
                rows.append(stu_id)
                cols.append(cpt_id)

    data = np.ones(len(rows))
    rows_np = np.array(rows)
    cols_np = np.array(cols)
    interaction_matrix = coo_matrix((data, (rows_np, cols_np)), shape=(num_students, num_items))

    row_stu_item = rows_np
    col_stu_item = cols_np + num_students
    row_item_stu = cols_np + num_students
    col_item_stu = rows_np

    adj_rows = np.concatenate([row_stu_item, row_item_stu])
    adj_cols = np.concatenate([col_stu_item, col_item_stu])
    adj_data = np.concatenate([data, data])

    adj = coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(num_students + num_items, num_students + num_items))
    rowsum = np.array(adj.sum(1)).flatten()
    epsilon = 1e-12
    d_inv_sqrt = np.power(rowsum + epsilon, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj.toarray() @ d_mat_inv_sqrt
    norm_adj = coo_matrix(norm_adj)

    # 保存图
    os.makedirs(graph_dir, exist_ok=True)  # 创建保存目录
    with open(graph_path, 'wb') as f:
        pickle.dump(norm_adj, f)
    print(f"Graph saved to {graph_path}")

    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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
    model.train()
    total_loss = 0
    total_aux_losses = {'fusion_se': 0, 'fusion_sc': 0, 'contrastive_exer': 0, 'contrastive_cpt': 0}
    pbar = tqdm(train_loader, desc='Training', disable=not verbose)
    for stu_ids, exer_ids, cpts_list, labels in pbar:
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        labels = labels.to(device)
        cpts_tensors = [torch.LongTensor(cpts).to(device) for cpts in cpts_list]
        optimizer.zero_grad()
        predictions, aux_losses, _ = model(
            stu_ids, exer_ids, cpts_tensors, labels,
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
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_knowledge_states = []
    with torch.no_grad():
        for stu_ids, exer_ids, cpts_list, labels in data_loader:
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            labels = labels.to(device)
            cpts_tensors = [torch.LongTensor(cpts).to(device) for cpts in cpts_list]
            predictions, aux_losses, knowledge_states = model(
                stu_ids, exer_ids, cpts_tensors, labels,
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
        # 修复：捕获具体异常并打印警告信息
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
