import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class EnhancedGatedFusion(nn.Module):
    def __init__(self, embedding_dim, num_gates=3, dropout=0.1):
        super(EnhancedGatedFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_gates = num_gates
        self.input_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()) for _ in range(num_gates)
        ])
        self.forget_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()) for _ in range(num_gates)
        ])
        self.update_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Tanh()) for _ in range(num_gates)
        ])
        self.output_gate = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.attention_weights = nn.Parameter(torch.ones(num_gates) / num_gates)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb1, emb2):
        concat = torch.cat([emb1, emb2], dim=-1)
        gate_outputs = []
        for i in range(self.num_gates):
            forget = self.forget_gates[i](concat)
            inp = self.input_gates[i](concat)
            update = self.update_gates[i](concat)
            gate_output = forget * emb1 + inp * update
            gate_outputs.append(gate_output)
        weights = F.softmax(self.attention_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, gate_outputs))
        output = self.output_gate(concat) * torch.tanh(fused)
        output = self.layer_norm(emb1 + self.dropout(output))
        return output


class ConcatGateFusion(nn.Module):
    """
    concat_gate融合方式：
    1. 将两个嵌入进行拼接（Concat）
    2. 通过线性层和Sigmoid函数计算门控系数 g_sc
    3. 用门控系数与特征向量相乘进行加权筛选（保留关键信息并抑制噪声）
    4. 通过Tanh激活函数非线性变换后输出更新后的认知状态向量
    """
    def __init__(self, embedding_dim, dropout=0.1):
        super(ConcatGateFusion, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 门控系数计算：拼接后通过线性层和Sigmoid得到门控权重
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # 特征变换：将拼接的特征映射到embedding_dim维度
        self.feature_network = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm用于稳定训练
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, emb1, emb2):
        """
        Args:
            emb1: 第一个嵌入 [batch_size, embedding_dim] 或 [num_nodes, embedding_dim]
            emb2: 第二个嵌入 [batch_size, embedding_dim] 或 [num_nodes, embedding_dim]
        
        Returns:
            output: 融合后的认知状态向量 [batch_size/num_nodes, embedding_dim]
        """
        # 步骤1: 拼接两个嵌入
        concat = torch.cat([emb1, emb2], dim=-1)  # [*, embedding_dim * 2]
        
        # 步骤2: 计算门控系数 g_sc (通过线性层 + Sigmoid)
        gate = self.gate_network(concat)  # [*, embedding_dim]
        
        # 步骤3: 计算特征向量
        feature = self.feature_network(concat)  # [*, embedding_dim]
        
        # 步骤4: 门控加权筛选 (保留关键信息，抑制噪声)
        gated_feature = gate * feature  # [*, embedding_dim]
        
        # 步骤5: Tanh非线性变换 + Dropout
        output = torch.tanh(gated_feature)  # [*, embedding_dim]
        output = self.dropout(output)
        
        # 步骤6: 残差连接 + LayerNorm (增强稳定性)
        output = self.layer_norm(emb1 + output)  # [*, embedding_dim]
        
        return output




class EnhancedContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.1, margin=1.0, contrastive_type='supervised'):
        super(EnhancedContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.contrastive_type = contrastive_type

    def forward(self, emb1, emb2, labels=None):
        batch_size = emb1.size(0)
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        if self.contrastive_type == 'supervised' and labels is not None and batch_size > 1:
            # 修复：正确使用label mask进行监督对比学习
            labels = labels.unsqueeze(1)
            # mask[i,j]=1 表示样本i和j属于同一类（正样本对）
            mask = torch.eq(labels, labels.T).float().to(emb1.device)
            similarity = torch.matmul(emb1, emb2.T) / self.temperature
            
            # 对角线位置：锚点与自身的相似度
            pos_similarity = torch.diag(similarity)
            
            # 计算对比损失：使用mask区分正负样本对
            # 排除自身（对角线）
            mask_without_diag = mask * (1 - torch.eye(batch_size).to(emb1.device))
            
            # InfoNCE风格的对比损失
            exp_sim = torch.exp(similarity)
            # 正样本：标签相同的样本（包括自身）
            pos_sum = (exp_sim * mask).sum(dim=1)
            # 所有样本
            all_sum = exp_sim.sum(dim=1)
            
            pos_loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
            
            # 负样本损失：不同标签的样本应该距离更远
            neg_mask = 1 - mask
            neg_similarity = similarity * neg_mask
            neg_loss = F.relu(neg_similarity - self.margin).mean()
            
            loss = pos_loss.mean() + neg_loss
        else:
            similarity = torch.matmul(emb1, emb2.T) / self.temperature
            labels = torch.arange(batch_size).to(emb1.device)
            loss = F.cross_entropy(similarity, labels)
        return loss

class PosLinear(nn.Linear):
    """
    单调性约束的线性层：确保权重非负
    用于认知诊断中的单调性假设：学生能力越强，答对概率越高
    """
    def forward(self, input):
        # 修复：使用ReLU确保权重严格非负
        # 原实现: 2 * F.relu(-w) + w 当w<0时结果为-w（仍为负）
        # 正确实现: 直接对权重应用ReLU
        weight = F.relu(self.weight)  # 确保所有权重 >= 0
        return F.linear(input, weight, self.bias)

class FCDiagnosisHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dims=[128, 64], dropout=0.4):
        super(FCDiagnosisHead, self).__init__()
        layers = []
        input_dim = embedding_dim
        self.prednet_stu = PosLinear(embedding_dim, embedding_dim)
        self.prednet_exer = PosLinear(embedding_dim, embedding_dim)
        self.prednet_kc = PosLinear(embedding_dim, embedding_dim)
        self.stu_state_pred = nn.Linear(embedding_dim*2, embedding_dim)
        self.exer_state_pred = nn.Linear(embedding_dim*2, embedding_dim)
        # 修复：在__init__中定义dropout层，而不是在forward中创建
        self.dropout_layer = nn.Dropout(0.1)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, stu_emb, exer_emb, cpt_embs, cpt_mask=None):
        # 计算注意力分数
        attention_scores = torch.matmul(cpt_embs, stu_emb.unsqueeze(-1))  # [batch_size, max_cpts, 1]
        
        # 应用掩码：将填充位置设置为-inf，使其在softmax后权重为0
        if cpt_mask is not None:
            attention_scores = attention_scores.masked_fill(~cpt_mask.unsqueeze(-1), float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        cpt_agg = torch.sum(cpt_embs * attention_weights, dim=1)
        cpt_agg = self.prednet_kc(cpt_agg)
        cpt_agg = self.prednet_exer(cpt_agg)
        stu_emb = self.prednet_stu(stu_emb)
        # 修复：使用定义好的dropout层
        stu_emb = self.dropout_layer(stu_emb)
        stu_cpt_state = torch.cat([stu_emb, cpt_agg], dim=-1)
        stu_cpt_state = self.stu_state_pred(stu_cpt_state)
        exer_emb = self.prednet_exer(exer_emb)
        # 修复：使用定义好的dropout层
        exer_emb = self.dropout_layer(exer_emb)
        exer_cpt_state = torch.cat([exer_emb, cpt_agg], dim=-1)
        exer_cpt_state = self.exer_state_pred(exer_cpt_state)
        prediction = stu_cpt_state - exer_cpt_state
        prediction = self.network(prediction)
        prediction = torch.sigmoid(prediction)
        return prediction.squeeze(-1)
