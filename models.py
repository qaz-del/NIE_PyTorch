import os
import torch
from torch import nn

class defineModel(nn.Module):
    def __init__(self, sample_size, dimsize, n_cls):
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv1d(sample_size, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512, 1e-3, 0.99)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512, 1e-3, 0.99)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 1e-3, 0.99)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 1e-3, 0.99)
        )

        self.MaxPool = nn.AdaptiveMaxPool1d(1)
        # Pool Output (bs, 1, feature)

        self.Dense1 = nn.Sequential(
            nn.Linear(dimsize, 1024),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.pred = nn.Sequential(
            nn.Linear(128, n_cls),
            # nn.Softmax(dim=0)
        )

    def forward(self, x):
        # (bs, channel, feature)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv4(x)

        x = x.permute(0, 2, 1)
        x = self.MaxPool(x)
        x = x.permute(0, 2, 1)

        # (bs, 1, dimsize)
        x = self.Dense1(x)

        # classification
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)

        x = self.pred(x)
        return x

class defineModelPN(nn.Module):
    def __init__(self, sample_size, dimsize, n_cls):
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv1d(sample_size + 6, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512, 1e-3, 0.99)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512, 1e-3, 0.99)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 1e-3, 0.99)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 1e-3, 0.99)
        )

        self.MaxPool = nn.AdaptiveMaxPool1d(1)
        # Pool Output (bs, 1, feature)

        self.Dense1 = nn.Sequential(
            nn.Linear(dimsize, 1024),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.Dense4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(1, 1e-3, 0.99),
            nn.Dropout(p=0.4)
        )
        self.pred = nn.Sequential(
            nn.Linear(128, n_cls),
            # nn.Softmax(dim=0)
        )

    def forward(self, x):
        # (bs, channel, dim+6)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv4(x)

        x = x.permute(0, 2, 1)
        x = self.MaxPool(x)
        x = x.permute(0, 2, 1)

        # (bs, 1, dimsize)
        x = self.Dense1(x)

        # classification
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)

        x = self.pred(x)
        return x



if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (bs, channel, feature)
    data = torch.randn(10,1024,256).to(device)
    mlp = defineModel(1024, 256, 10).to(device)

    out = mlp.forward(data)
    print(out)