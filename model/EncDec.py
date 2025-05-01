import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, 2*N),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.BatchNorm1d(2*N),
        )
    def forward(self, m):
        # m: [batch_size, K] (0/1 bits)
        # returns c: [batch_size, 2*N] (real/imag parts)
        return self.net(m)

class Decoder(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*N, 2*N),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, K*2),      # 2-class output for each bit
        )
        self.ce  = nn.CrossEntropyLoss()  # applied per bit

    def forward(self, c_noisy, m):
        # c_noisy: [batch_size, 2*N]
        # m:        [batch_size, K] (ground-truth bits)
        logits = self.net(c_noisy)                # [B, K*2]
        logits = logits.view(-1, 2)               # [(B*K), 2]
        target = m.view(-1).long()                # [(B*K)], values 0 or 1
        loss   = self.ce(logits, target)          # cross-entropy loss
        return loss