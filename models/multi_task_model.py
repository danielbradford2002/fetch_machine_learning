import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    """
    MultiTaskModel wraps a pretrained transformer (DistilBERT) with two task-specific heads:
      - classification_head: For sentence classification (Task A).
      - taskB_head: For a second NLP task (e.g., sentiment analysis, Task B).
    """
    def __init__(self, model_name, num_classes_taskA, num_classes_taskB):
        super().__init__()
        # Load the pretrained transformer model
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Linear layer for Task A (Classification)
        self.classification_head = nn.Linear(hidden_size, num_classes_taskA)
        # Linear layer for Task B (e.g., Sentiment Analysis)
        self.taskB_head = nn.Linear(hidden_size, num_classes_taskB)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass: compute transformer embeddings, apply mean pooling,
        then pass through both task-specific heads.
        """
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Mean pooling: average over the sequence length dimension
        pooled_output = last_hidden.mean(dim=1)

        # Obtain logits for each task
        logits_taskA = self.classification_head(pooled_output)
        logits_taskB = self.taskB_head(pooled_output)
        return logits_taskA, logits_taskB
