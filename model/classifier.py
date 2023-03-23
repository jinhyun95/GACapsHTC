import torch
import torch.nn as nn
import torch.nn.functional as F

from model import text_feature_propagation


class TextClassifier(nn.Module):
    def __init__(self, config, label_ids):
        super(TextClassifier, self).__init__()
        self.config = config
        self.label_ids = label_ids

        # TEXT ENCODER (BERT-BASED LANGUAGE MODEL)
        self.text_embedder = torch.hub.load('huggingface/pytorch-transformers', 'model', self.config.model.embedding.type)
        for p in self.text_embedder.parameters():
            p.requires_grad = False
        self.register_buffer('is_finetune', torch.tensor([-1.]))
        self.embedding_dropout = nn.Dropout(self.config.model.embedding.dropout)

        self.text_encoder_cnn = torch.nn.Conv1d(
            in_channels=config.model.embedding.dimension,
            out_channels=self.config.model.structure_encoder.dimension * len(self.label_ids),
            kernel_size=3,
            padding=1
        )
        # FEATURE AGGREGATION
        self.information_aggregation = text_feature_propagation.TextFeaturePropagation(self.config, self.label_ids)
        
    def forward(self, batch):
        if self.is_finetune.item() < 0:
            if not hasattr(self.config.training, "finetune") or not self.config.training.finetune.tune:
                self.text_embedder.eval()
            with torch.no_grad():
                text_embedding = self.text_embedder(batch[0], batch[1], batch[2])['last_hidden_state']
        else:
            text_embedding = self.text_embedder(batch[0], batch[1], batch[2])['last_hidden_state']
        embedding = self.embedding_dropout(text_embedding)
        
        out = self.text_encoder_cnn(embedding.transpose(1, 2))
        out = torch.max(out, dim=-1)[0]
        out = out.view(out.size(0), len(self.label_ids), -1)
        out = F.relu(out)
        # FEATURE AGGREGATION (LOGIT CALCULATION)
        preds, p = self.information_aggregation(out)
        
        return preds, p
