import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes_taskA, num_classes_taskB):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Classification head for Task A
        self.classification_head = nn.Linear(hidden_size, num_classes_taskA)

        # Another head for Task B (could be sentiment or NER, etc.)
        self.taskB_head = nn.Linear(hidden_size, num_classes_taskB)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Example: mean pool
        last_hidden = outputs.last_hidden_state

        # Simple mean-pooling (you can choose another pooling strategy)
        pooled_output = last_hidden.mean(dim=1)

        logits_taskA = self.classification_head(pooled_output)
        logits_taskB = self.taskB_head(pooled_output)

        return logits_taskA, logits_taskB
