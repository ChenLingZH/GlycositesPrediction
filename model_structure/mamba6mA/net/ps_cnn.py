import torch
import torch.nn as nn
import torch.nn.functional as F

class PSConvNetBase(nn.Module):
    def __init__(self, window_size, filter_num, feature, seq_len):
        super().__init__()
        self.filter_num = filter_num
        self.feature = feature
        self.window_size = window_size
        self.seq_len = seq_len
        self.pad_len = int(self.window_size / 2)

        self.dense_layer_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size * self.feature, filter_num),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(self.seq_len)
        ])

    def forward(self, input_mat):
        h_d = F.pad(input_mat, (self.pad_len, self.pad_len), "constant", 0)
        h_d = torch.transpose(h_d, 1, 2)

        h_d_temp = []
        for i in range(self.pad_len, self.seq_len + self.pad_len):
            segment = h_d[:, i - self.pad_len: i + self.pad_len + 1, :]
            segment_flat = segment.reshape(-1, self.window_size * self.feature)
            h_d_temp.append(self.dense_layer_net[i - self.pad_len](segment_flat))

        h_d = torch.stack(h_d_temp)
        h_d = torch.transpose(h_d, 0, 1)

        return h_d

class PSConvNet1(PSConvNetBase):
    def __init__(self, window_size=3, filter_num=256, feature=256, seq_len=41):
        super().__init__(window_size, filter_num, feature, seq_len)

class PSConvNet2(PSConvNetBase):
    def __init__(self, window_size=5, filter_num=256, feature=256, seq_len=41):
        super().__init__(window_size, filter_num, feature, seq_len)

class PSConvNet3(PSConvNetBase):
    def __init__(self, window_size=7, filter_num=256, feature=256, seq_len=41):
        super().__init__(window_size, filter_num, feature, seq_len)