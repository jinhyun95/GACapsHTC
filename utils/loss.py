import torch


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = 2.
        self.bce = torch.nn.BCELoss(reduction='none')

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        weight = torch.pow(1. - preds, self.gamma) * targets + torch.pow(preds, self.gamma) * (1. - targets)

        return torch.mean(bce * weight) / (0.5 ** self.gamma)


class LabelContradictionPenalty(torch.nn.Module):
    def __init__(self, config, hierarchy):
        super(LabelContradictionPenalty, self).__init__()
        self.penalty_weight = config.training.label_contradiction_penalty.weight
        self.is_absolute = config.training.label_contradiction_penalty.absolute
        self.margin = config.training.label_contradiction_penalty.margin
        self.hierarchy = hierarchy
    
    def forward(self, preds):
        contradictions = 0.
        for parent in self.hierarchy.keys():
            children = self.hierarchy[parent]
            contradiction = preds[:, parent] - torch.max(preds[:, children], dim=1)[0]
            
            if self.is_absolute:
                contradiction = torch.abs(contradiction)
            if self.margin > 0:
                contradiction = contradiction[torch.logical_or(contradiction > self.margin, contradiction < - self.margin)]
            if contradiction.size(0) > 0:
                contradictions = contradictions + torch.sum(contradiction)

        return self.penalty_weight * contradictions / preds.size(0)
