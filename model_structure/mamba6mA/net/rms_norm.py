import torch
import torch.nn as nn

from net.mamba_block import MambaBlock1,MambaBlock2,MambaBlock3


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class ResidualBlockBase(nn.Module):
    def __init__(self, args, mixer_class):
        super().__init__()
        self.args = args
        self.mixer = mixer_class(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class ResidualBlock1(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, MambaBlock1)

class ResidualBlock2(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, MambaBlock2)

class ResidualBlock3(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, MambaBlock3)