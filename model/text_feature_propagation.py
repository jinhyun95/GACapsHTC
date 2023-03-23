from torch import nn

from model import capsnet, graph_network


class TextFeaturePropagation(nn.Module):
    def __init__(self, config, label_ids):
        super(TextFeaturePropagation, self).__init__()
        self.config = config
        self.label_ids = label_ids
        structure_encoder = []
        for _ in range(self.config.model.structure_encoder.layers):
                    structure_encoder.append(
                        graph_network.GraphAttentionNetwork(
                            config=self.config,
                            label_ids=self.label_ids
                        )
                    )
        self.structure_encoder = nn.Sequential(*structure_encoder)
        self.dropout = nn.Dropout(p=self.config.model.feature_aggregation.dropout)

        if self.config.model.capsule_setting.type == 'original':
            self.classifier = capsnet.CapsuleLayer(self.config, label_ids)
        elif self.config.model.capsule_setting.type == 'kde':
            self.classifier = capsnet.KDECapsuleLayer(self.config, label_ids)
        else:
            raise NotImplementedError
    
    def forward(self, inputs):
        node_inputs = self.dropout(inputs)
        labelwise_text_feature = self.structure_encoder(node_inputs)
        preds = self.classifier(labelwise_text_feature)
        return preds
