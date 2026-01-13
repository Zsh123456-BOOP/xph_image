import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph, train_epoch, evaluate, EarlyStopping


def get_args():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--train_file', type=str, default='assist-09/train.csv')
    parser.add_argument('--valid_file', type=str, default='assist-09/valid.csv')
    parser.add_argument('--test_file', type=str, default='assist-09/test.csv')
    parser.add_argument('--graph_dir', type=str, default='./graphs', 
                        help="Directory to save/load graph files. Graphs are dataset-specific (named by dimensions).")

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--fusion_type', type=str, default='enhanced_gated',
                        choices=['enhanced_gated', 'concat_gate', 'cross_attention', 'residual', 'multi_scale', 
                                 'soft_orthogonal', 'gru_knowledge'])
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--use_supervised_contrastive', action='store_true', default=True)
    parser.add_argument('--gated_num_gates', type=int, default=3)
    parser.add_argument('--ortho_weight', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.3)


    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_fusion', type=float, default=0.7)
    parser.add_argument('--lambda_contrastive', type=float, default=0.4)
    parser.add_argument('--fusion_warmup_epochs', type=int, default=1)
    parser.add_argument('--contrastive_decay_epochs', type=int, default=18)
    parser.add_argument('--contrastive_min_weight', type=float, default=0.12)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=7)

    # 调度器参数
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--T_0', type=int, default=8)
    parser.add_argument('--T_mult', type=int, default=2)

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='saved_models')

    args = parser.parse_args()
    return args


def prepare_data(args, device):
    """
    负责读取数据、ID映射、构建图和DataLoader，并将所有需要的数据打包返回
    """
    print('Loading data...')
    train_df = pd.read_csv(args.train_file)
    valid_df = pd.read_csv(args.valid_file)
    test_df = pd.read_csv(args.test_file)
    all_df = pd.concat([train_df, valid_df, test_df])

    # 1. 收集所有唯一ID (原始数据中的ID可能不连续)
    print("Collecting unique IDs...")
    student_ids = sorted(all_df['stu_id'].unique())
    exercise_ids = sorted(all_df['exer_id'].unique())
    
    # 收集所有知识点ID
    all_cpts = set()
    for cpt_seq in all_df['cpt_seq']:
        if isinstance(cpt_seq, str):
            cpts = [int(c) for c in cpt_seq.split(',')]
        else:
            cpts = [int(cpt_seq)]
        all_cpts.update(cpts)
    concept_ids = sorted(all_cpts)

    # 2. 创建ID映射，将原始ID重新映射为从0开始的连续整数
    print("Creating ID mappings (original -> continuous)...")
    student_id_map = {old_id: new_id for new_id, old_id in enumerate(student_ids)}
    exercise_id_map = {old_id: new_id for new_id, old_id in enumerate(exercise_ids)}
    concept_id_map = {old_id: new_id for new_id, old_id in enumerate(concept_ids)}
    
    # 3. 统计映射后的数量（这些数量对应连续ID的范围）
    num_students = len(student_id_map)  # 映射后: 0 ~ num_students-1
    num_exercises = len(exercise_id_map)  # 映射后: 0 ~ num_exercises-1
    num_concepts = len(concept_id_map)  # 映射后: 0 ~ num_concepts-1
    
    # 优化：不使用偏移
    # 原设计: concept_offset = num_exercises (浪费99.3%参数)
    # 原因：虽然知识点ID与习题ID有重叠，但使用的是两个独立的GCN，无需区分索引空间
    # 优化后：直接使用0，大幅减少参数量 (从17,799降到123个embedding)
    concept_offset = 0
    concept_total = num_concepts  # 优化：不加offset，直接使用概念数量

    print(f'Students: {num_students}, Exercises: {num_exercises}, Concepts: {num_concepts}')
    print(f'Concept offset: {concept_offset} (Optimized: no offset used)')
    print(f'Concept embedding size: {concept_total} (was {num_concepts + num_exercises} before optimization)')
    print(f'Parameter reduction: {((num_concepts + num_exercises - concept_total) / (num_concepts + num_exercises) * 100):.1f}%')
    print(f'ID mapping completed: all IDs are now continuous starting from 0')

    # 4. 应用ID映射到各个数据集
    print("Applying ID mappings to datasets...")
    def map_concepts(cpt_seq):
        """将概念ID序列映射为连续ID"""
        if isinstance(cpt_seq, str):
            cpts = [int(c) for c in cpt_seq.split(',')]
            mapped_cpts = [concept_id_map[c] for c in cpts]
            return ','.join(str(c) for c in mapped_cpts)
        else:
            return str(concept_id_map[int(cpt_seq)])

    # 对train, valid, test三个数据集应用映射
    for df in [train_df, valid_df, test_df]:
        df['stu_id'] = df['stu_id'].map(student_id_map)
        df['exer_id'] = df['exer_id'].map(exercise_id_map)
        df['cpt_seq'] = df['cpt_seq'].apply(map_concepts)

    # 5. 构建图 (Build Graphs)
    print('Building graphs or loading from file...')
    print(f'Graph directory: {args.graph_dir}')
    print(f'Graph naming: graph_{{type}}_{{correct}}_s{num_students}_i{{num_items}}.pkl')
    print(f'Note: Graph files are dataset-specific and will not conflict between datasets.')
    
    # 学生-习题图：使用习题数量
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True, item_type='exercise',
                                 graph_dir=args.graph_dir).to(device)
    adj_wrong_se = build_graph(train_df, num_students, num_exercises, correct=False, item_type='exercise',
                               graph_dir=args.graph_dir).to(device)

    # 学生-知识点图：直接使用知识点数量（不需要offset）
    # 优化说明：concept_total已在前面定义为 num_concepts（不加offset）
    adj_correct_sc = build_graph(train_df, num_students, concept_total, correct=True, item_type='concept',
                                 concept_offset=concept_offset, graph_dir=args.graph_dir).to(device)
    adj_wrong_sc = build_graph(train_df, num_students, concept_total, correct=False, item_type='concept',
                               concept_offset=concept_offset, graph_dir=args.graph_dir).to(device)

    # 6. 创建 DataLoader
    train_dataset = CDDataset(train_df, concept_offset=concept_offset)
    valid_dataset = CDDataset(valid_df, concept_offset=concept_offset)
    test_dataset = CDDataset(test_df, concept_offset=concept_offset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 打包所有数据
    data_bundle = {
        'num_students': num_students,
        'num_exercises': num_exercises,
        'num_concepts': num_concepts,
        'concept_offset': concept_offset,
        'adj_graphs': (adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc),
        'loaders': (train_loader, valid_loader, test_loader)
    }

    return data_bundle


def train(args, data_bundle, device):
    """
    训练流程，返回最佳模型保存路径
    """
    print('\n=== Start Training ===')

    # 解包数据
    num_students = data_bundle['num_students']
    num_exercises = data_bundle['num_exercises']
    num_concepts = data_bundle['num_concepts']
    concept_offset = data_bundle['concept_offset']
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = data_bundle['adj_graphs']
    train_loader, valid_loader, _ = data_bundle['loaders']

    # 初始化模型
    model = CognitiveDiagnosisModel(
        num_students=num_students, num_exercises=num_exercises, num_concepts=num_concepts,
        embedding_dim=args.embedding_dim, num_layers=args.num_layers, concept_offset=concept_offset,
        fusion_type=args.fusion_type, temperature=args.temperature,
        num_heads=args.num_heads, use_supervised_contrastive=args.use_supervised_contrastive,
        gated_num_gates=args.gated_num_gates, ortho_weight=args.ortho_weight, dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    elif args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=5)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_aux_losses = train_epoch(
            model, train_loader, optimizer, device,
            adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc, args, epoch
        )

        valid_loss, valid_metrics, _ = evaluate(
            model, valid_loader, device,
            adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc
        )

        # 调度器步进
        if args.scheduler_type == 'plateau':
            scheduler.step(valid_metrics["auc"])
        else:
            scheduler.step()

        print(f'Train Loss: {train_loss:.4f} | Aux Losses: {train_aux_losses}')
        print(f'Valid AUC: {valid_metrics["auc"]:.4f} | Acc: {valid_metrics["accuracy"]:.4f}')

        early_stopping(valid_metrics["auc"], model)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # 保存最佳模型到文件（确保 test 阶段是全新加载）
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    torch.save(early_stopping.best_model_state, best_model_path)
    print(f"Best model saved to {best_model_path}")

    return best_model_path


def test(args, data_bundle, device, model_path):
    """
    测试流程，加载指定路径的模型并评估
    """
    print('\n=== Start Testing ===')

    # 解包数据
    num_students = data_bundle['num_students']
    num_exercises = data_bundle['num_exercises']
    num_concepts = data_bundle['num_concepts']
    concept_offset = data_bundle['concept_offset']
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = data_bundle['adj_graphs']
    _, _, test_loader = data_bundle['loaders']

    # 重新初始化一个干净的模型
    model = CognitiveDiagnosisModel(
        num_students=num_students, num_exercises=num_exercises, num_concepts=num_concepts,
        embedding_dim=args.embedding_dim, num_layers=args.num_layers, concept_offset=concept_offset,
        fusion_type=args.fusion_type, temperature=args.temperature,
        num_heads=args.num_heads, use_supervised_contrastive=args.use_supervised_contrastive,
        gated_num_gates=args.gated_num_gates, ortho_weight=args.ortho_weight, dropout=args.dropout
    ).to(device)

    # 加载权重
    print(f'Loading model from {model_path}...')
    model.load_state_dict(torch.load(model_path, map_location=device))

    _, test_metrics, _ = evaluate(
        model, test_loader, device,
        adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc
    )

    print(f'\nFinal Test Results:')
    print(f'AUC: {test_metrics["auc"]:.4f}')
    print(f'Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Precision: {test_metrics["precision"]:.4f}')
    print(f'Recall: {test_metrics["recall"]:.4f}')
    print(f'F1: {test_metrics["f1"]:.4f}')
    print(f'RMSE: {test_metrics["rmse"]:.4f}')
    print(f'Confusion Matrix:\n{test_metrics["confusion_matrix"]}')

    return test_metrics


if __name__ == '__main__':
    # 1. 获取参数
    args = get_args()

    # 设置种子和设备
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 准备数据 (Train 和 Test 共用)
    data_bundle = prepare_data(args, device)

    # 3. 训练并保存最佳模型
    best_model_path = train(args, data_bundle, device)

    # 4. 加载最佳模型进行测试
    test(args, data_bundle, device, best_model_path)
