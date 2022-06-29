import torch
from torch.nn import functional as F


class IBGNN(torch.nn.Module):
    def __init__(self, gnn, mlp, discriminator=lambda x, y: x @ y.t(), pooling='concat'):
        super(IBGNN, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.pooling = pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr, batch, edge_flag = data.x, data.edge_index, data.edge_attr, data.batch, data.edge_flag
        g = self.gnn(x, edge_index, edge_attr, edge_flag, batch)
        if self.pooling == 'concat':
            _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits
        return g
