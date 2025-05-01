import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        # zero-init the second Weight so residual starts as identity
        nn.init.zeros_(self.fc2.weight)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.act(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return self.act(out + x)
# Put back in batchnorm
class PokerNNet(nn.Module):
    def __init__(self, input_size: int, action_size: int, args=None):
        super().__init__()
        if hasattr(args, 'dim'):
            hidden_dim = args.dim  # maybe do between this and 98 for main training, maybe like 80
        else:
            hidden_dim = 80
        self.act = nn.SiLU()

        # input embedding
        self.fc_in = nn.Linear(input_size, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim) # commented out
        # self.bn_in = nn.BatchNorm1d(hidden_dim)

        # deeper with residuals
        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        # policy head
        self.pi_h   = nn.Linear(hidden_dim, 64)
        self.pi_out = nn.Linear(64, action_size)

        # value head
        self.v_h   = nn.Linear(hidden_dim, 64)
        self.v_out = nn.Linear(64, 1)

    def forward(self, x):
        # input → hidden_dim
        x = self.act(self.ln_in(self.fc_in(x))) # commented out
        # x = self.act(self.bn_in(self.fc_in(x)))

        # depth via residuals
        x = self.blocks(x)

        # policy
        pi = self.act(self.pi_h(x))
        pi = self.pi_out(pi)
        pi = F.log_softmax(pi, dim=-1)

        # value
        v = self.act(self.v_h(x))
        v = self.v_out(v)

        return pi, v