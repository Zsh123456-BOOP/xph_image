import torch
import torch.nn as nn


class NonNegativeClipper:
    def __call__(self, module):
        if hasattr(module, "weight"):
            weight = module.weight.data
            weight.add_(torch.relu(-weight))


class NeuralCDModel(nn.Module):
    """
    Minimal NeuralCD baseline used for analysis-time loading and inference.
    """

    def __init__(
        self,
        num_students,
        num_exercises,
        num_concepts,
        q_matrix,
        dropout=0.5,
        prednet_len1=512,
        prednet_len2=256,
        discrimination_scale=10.0,
        use_clipper=False,
    ):
        super().__init__()
        if int(num_concepts) <= 0:
            raise ValueError("NeuralCD requires at least one concept.")

        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.discrimination_scale = float(discrimination_scale)
        self.use_clipper = bool(use_clipper)

        self.student_emb = nn.Embedding(self.num_students, self.num_concepts)
        self.k_difficulty = nn.Embedding(self.num_exercises, self.num_concepts)
        self.e_discrimination = nn.Embedding(self.num_exercises, 1)

        self.prednet_full1 = nn.Linear(self.num_concepts, int(prednet_len1))
        self.drop_1 = nn.Dropout(p=float(dropout))
        self.prednet_full2 = nn.Linear(int(prednet_len1), int(prednet_len2))
        self.drop_2 = nn.Dropout(p=float(dropout))
        self.prednet_full3 = nn.Linear(int(prednet_len2), 1)

        self.register_buffer("q_matrix", q_matrix.float())
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * self.discrimination_scale
        kn_emb = self.q_matrix[exer_id]

        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x)).view(-1)
        return output

    def apply_clipper(self):
        if not self.use_clipper:
            return
        clipper = NonNegativeClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        return torch.sigmoid(self.student_emb(stu_id))
