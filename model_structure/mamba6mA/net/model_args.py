from __future__ import annotations
import math

class ModelArgs:
    d_model: int = 64
    n_layer: int = 6
    # 0/1/2/3: A/C/G/T, 4: N(padding)
    vocab_size: int = 5
    pad_idx: int = 4
    # Your CSV windows are fixed length 51.
    seq_len: int = 51
    d_state: int = 16
    expand: int = 4
    dt_rank: str = 'auto'
    d_conv: int = 4
    # Keep vocab_size as-is (so pad_idx stays 4).
    pad_vocab_size_multiple: int = 1
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)