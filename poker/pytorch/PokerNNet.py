import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        return F.relu(x + out)          # residual add

class PokerNNet(nn.Module):
    def __init__(self, input_size: int, action_size: int, args):
        super().__init__()
        self.fc_in = nn.Linear(input_size, args.block_width)
        self.bn_in = nn.BatchNorm1d(args.block_width)
        self.blocks = nn.Sequential(
            *[ResBlock(args.block_width, args.dropout) for _ in range(args.n_blocks)]
        )
        # policy head
        self.pi_h = nn.Linear(args.block_width, 64)
        self.pi_out = nn.Linear(64, action_size)
        # value head
        self.v_h = nn.Linear(args.block_width, 64)
        self.v_out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.blocks(x)

        pi = F.relu(self.pi_h(x))
        pi = F.log_softmax(self.pi_out(pi), dim=-1)

        v  = F.relu(self.v_h(x))
        v  = self.v_out(v)   

        return pi, v