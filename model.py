import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    LightGCN, EnhancedGatedFusion, ConcatGateFusion,
    EnhancedContrastiveLearning, FCDiagnosisHead
)

class CognitiveDiagnosisModel(nn.Module):
    def __init__(self, num_students, num_exercises, num_concepts, embedding_dim,
                 num_layers, concept_offset=0, fusion_type='enhanced_gated',
                 temperature=0.1, num_heads=4, use_supervised_contrastive=True,
                 gated_num_gates=3, ortho_weight=0.1, dropout=0.1):
        super(CognitiveDiagnosisModel, self).__init__()

        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.concept_offset = concept_offset
        self.use_supervised_contrastive = use_supervised_contrastive
        self.fusion_type = fusion_type

        # GCNs
        self.gcn_correct_se = LightGCN(num_students, num_exercises, embedding_dim, num_layers)
        self.gcn_wrong_se = LightGCN(num_students, num_exercises, embedding_dim, num_layers)
        # 优化：concept_total不再加offset，直接使用num_concepts
        # 这与main.py中的优化保持一致
        concept_total = num_concepts  # 优化后不加offset
        self.gcn_correct_sc = LightGCN(num_students, concept_total, embedding_dim, num_layers)
        self.gcn_wrong_sc = LightGCN(num_students, concept_total, embedding_dim, num_layers)

        # Fusion Modules
        fusion_kwargs = {'dropout': dropout}
        if fusion_type == 'enhanced_gated':
            self.fusion_se = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, **fusion_kwargs))
            self.fusion_sc = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, **fusion_kwargs))
            # Gated Fusion for different views
            self.gated_fusion_student = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))
            self.gated_fusion_exercise = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))
            self.gated_fusion_concept = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))
        elif fusion_type == 'concat_gate':
            self.fusion_se = self._wrap_fusion_module(ConcatGateFusion(embedding_dim, **fusion_kwargs))
            self.fusion_sc = self._wrap_fusion_module(ConcatGateFusion(embedding_dim, **fusion_kwargs))
            # Gated Fusion for different views
            self.gated_fusion_student = self._wrap_fusion_module(ConcatGateFusion(embedding_dim, dropout=dropout))
            self.gated_fusion_exercise = self._wrap_fusion_module(ConcatGateFusion(embedding_dim, dropout=dropout))
            self.gated_fusion_concept = self._wrap_fusion_module(ConcatGateFusion(embedding_dim, dropout=dropout))
        else:
            # 默认使用enhanced_gated
            self.fusion_se = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, **fusion_kwargs))
            self.fusion_sc = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, **fusion_kwargs))
            # Gated Fusion for different views
            self.gated_fusion_student = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))
            self.gated_fusion_exercise = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))
            self.gated_fusion_concept = self._wrap_fusion_module(EnhancedGatedFusion(embedding_dim, num_gates=gated_num_gates, dropout=dropout))

        # Contrastive Learning
        self.contrastive_exercise = EnhancedContrastiveLearning(temperature)
        self.contrastive_concept = EnhancedContrastiveLearning(temperature)

        # Diagnosis Head
        diagnosis_kwargs = {'dropout': dropout}
        self.diagnosis_head = FCDiagnosisHead(embedding_dim, **diagnosis_kwargs)

        # Knowledge State Diagnosis
        self.knowledge_diagnosis = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_concepts),
            nn.Sigmoid()
        )
        self._init_weights()

    def _wrap_fusion_module(self, module):
        class WrappedModule(nn.Module):
            def __init__(self, module):
                super(WrappedModule, self).__init__()
                self.module = module
            def forward(self, emb1, emb2):
                result = self.module(emb1, emb2)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return result, torch.tensor(0.0).to(emb1.device)
        return WrappedModule(module)

    def _init_weights(self):
        """
        初始化模型权重
        注意：Embedding层已在LightGCN的__init__中单独初始化，此处不重复初始化
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, stu_ids, exer_ids, cpt_ids_list, labels,
                adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc):
        stu_emb_correct_se, exer_emb_correct = self.gcn_correct_se(adj_correct_se)
        stu_emb_wrong_se, exer_emb_wrong = self.gcn_wrong_se(adj_wrong_se)
        stu_emb_correct_sc, cpt_emb_correct = self.gcn_correct_sc(adj_correct_sc)
        stu_emb_wrong_sc, cpt_emb_wrong = self.gcn_wrong_sc(adj_wrong_sc)


        stu_emb_se, fusion_loss_se = self.fusion_se(stu_emb_correct_se, stu_emb_wrong_se)
        stu_emb_sc, fusion_loss_sc = self.fusion_sc(stu_emb_correct_sc, stu_emb_wrong_sc)

        stu_emb_final, _ = self.gated_fusion_student(stu_emb_se, stu_emb_sc)
        exer_emb_final, _ = self.gated_fusion_exercise(exer_emb_correct, exer_emb_wrong)
        cpt_emb_final, _ = self.gated_fusion_concept(cpt_emb_correct, cpt_emb_wrong)

        batch_size = stu_ids.size(0)
        if self.use_supervised_contrastive and labels is not None and batch_size > 1:
            contrastive_loss_exer = self.contrastive_exercise(exer_emb_correct[exer_ids], exer_emb_wrong[exer_ids], labels)
        else:
            contrastive_loss_exer = self.contrastive_exercise(exer_emb_correct[exer_ids], exer_emb_wrong[exer_ids])

        # 收集批次中所有唯一的概念ID用于对比学习
        batch_cpt_ids = []
        for cpts in cpt_ids_list:
            batch_cpt_ids.extend(cpts)
        batch_cpt_ids = list(set(batch_cpt_ids))

        # 修复：确保有足够的概念进行对比学习（至少需要2个）
        if len(batch_cpt_ids) > 1:
            batch_cpt_ids_tensor = torch.LongTensor(batch_cpt_ids).to(stu_ids.device)
            if self.use_supervised_contrastive and labels is not None:
                cpt_labels = torch.ones(len(batch_cpt_ids)).to(stu_ids.device)
                contrastive_loss_cpt = self.contrastive_concept(cpt_emb_correct[batch_cpt_ids_tensor], cpt_emb_wrong[batch_cpt_ids_tensor], cpt_labels)
            else:
                contrastive_loss_cpt = self.contrastive_concept(cpt_emb_correct[batch_cpt_ids_tensor], cpt_emb_wrong[batch_cpt_ids_tensor])
        else:
            contrastive_loss_cpt = torch.tensor(0.0).to(stu_ids.device)

        batch_stu_emb = stu_emb_final[stu_ids]
        batch_exer_emb = exer_emb_final[exer_ids]

        batch_cpt_embs = []
        batch_cpt_masks = []  # 添加掩码列表
        max_cpts = max(len(cpts) for cpts in cpt_ids_list)
        for cpts in cpt_ids_list:
            cpt_embs = cpt_emb_final[cpts]
            actual_len = len(cpts)
            
            # 创建掩码：True表示真实位置，False表示填充位置
            mask = torch.ones(actual_len, dtype=torch.bool).to(cpt_embs.device)
            
            if actual_len < max_cpts:
                padding = torch.zeros(max_cpts - actual_len, self.embedding_dim).to(cpt_embs.device)
                cpt_embs = torch.cat([cpt_embs, padding], dim=0)
                # 填充掩码
                padding_mask = torch.zeros(max_cpts - actual_len, dtype=torch.bool).to(cpt_embs.device)
                mask = torch.cat([mask, padding_mask], dim=0)
            
            batch_cpt_embs.append(cpt_embs)
            batch_cpt_masks.append(mask)
        
        batch_cpt_embs = torch.stack(batch_cpt_embs)  # [batch_size, max_cpts, embedding_dim]
        batch_cpt_masks = torch.stack(batch_cpt_masks)  # [batch_size, max_cpts]

        predictions = self.diagnosis_head(batch_stu_emb, batch_exer_emb, batch_cpt_embs, batch_cpt_masks)
        knowledge_state = self.knowledge_diagnosis(batch_stu_emb)

        auxiliary_losses = {
            'fusion_se': fusion_loss_se, 'fusion_sc': fusion_loss_sc,
            'contrastive_exer': contrastive_loss_exer, 'contrastive_cpt': contrastive_loss_cpt
        }
        return predictions, auxiliary_losses, knowledge_state
