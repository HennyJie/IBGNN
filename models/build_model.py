import torch
from models.models_common import IBGConv, MLP
from models.models_mp import IBGNN


def build_model(args, device, num_features):
    model = IBGNN(IBGConv(num_features, args, num_classes=2),
                  MLP(args.hidden_dim, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                  pooling=args.pooling).to(device)

    return model
