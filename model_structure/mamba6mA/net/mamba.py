import torch
import torch.nn as nn
from .model_args import ModelArgs
from .rms_norm import RMSNorm ,ResidualBlock1, ResidualBlock2, ResidualBlock3
from .fusion import MultiRepresentationFusion
class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # padding_idx makes embedding vectors at PAD positions always zero.
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=args.pad_idx)
        self.layers1 = nn.ModuleList([ResidualBlock1(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock2(args) for _ in range(args.n_layer)])
        self.layers3 = nn.ModuleList([ResidualBlock3(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.fusion_module = MultiRepresentationFusion()
        self.linear_layer = nn.Linear(args.seq_len, 1)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.LastDense = nn.Sequential(
            nn.Linear(64 * args.seq_len, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.mapping = {0: 3, 1: 2, 2: 1, 3: 0}

    def forward(self, input_ids, padding_mask=None):
        input_ids = input_ids.long()
        if padding_mask is None:
            padding_mask = (input_ids == self.args.pad_idx)
        padding_mask = padding_mask.to(input_ids.device)
        embedded = self.embedding(input_ids)
        x = embedded.clone()
        for layer in self.layers1:
            x_1 = layer(x)
        x_1 = self.norm_f(x_1)
        for layer in self.layers2:
            x_2 = layer(x)
        x_2 = self.norm_f(x_2)
        for layer in self.layers3:
            x_3 = layer(x)
        x_3 = self.norm_f(x_3)
        x = self.fusion_module(x_1, x_2, x_3)
        # Ensure PAD tokens do not contribute to the classifier.
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = torch.flatten(x, start_dim=1)
        logits = self.LastDense(x)
        logits = logits.squeeze(dim=1)
        return logits
