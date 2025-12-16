from torch import nn
from .GraphTransformer.graphformer import Classifier


class GraphTransformerNet(nn.Module):
    def __init__(self, n_class=7, n_features=512, loss=None):
        super(GraphTransformerNet, self).__init__()
        if loss == "bce":
            loss_func = nn.BCEWithLogitsLoss()
        elif loss == "mls":
            loss_func = nn.MultiLabelSoftMarginLoss()
        elif loss == "ce":
            loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid Loss Type!")
        self.classifier = Classifier(n_class, n_features, loss_func)

    def forward(self, x):
        node_feat, labels, adj, mask = x
        return self.classifier(node_feat, labels, adj, mask)
