"""试井曲线网络。卷积块顺序采用 Dong 等人的图 4，层宽为本实验设置。"""

import torch
from torch import nn


class WellTestCNN(nn.Module):
    """输入形状为（批大小，通道数，时间点数），输出类别分数或归一化参数。"""

    def __init__(
        self, in_channels: int, out_dim: int, length: int = 128, batch_norm: bool = True
    ):
        super().__init__()
        layers = []
        for width in [32, 64, 128]:
            layers.append(
                nn.Conv1d(in_channels, width, kernel_size=5, stride=2, padding=2)
            )
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers += [nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)]
            in_channels = width
        self.conv = nn.Sequential(*layers)
        # 用虚拟曲线确定全连接层输入宽度。eval 避免改变 BN 的运行统计量。
        self.conv.eval()
        with torch.no_grad():
            probe = torch.zeros(1, layers[0].in_channels, length)
            flat = self.conv(probe).numel()
        self.conv.train()
        head = []
        for hidden in [512, 256]:
            head.append(nn.Linear(flat, hidden))
            if batch_norm:
                head.append(nn.BatchNorm1d(hidden))
            head.append(nn.ReLU())
            flat = hidden
        # 分类器直接输出 logits；反演器使用相同结构，最后一层默认输出 k 和 S。
        head.append(nn.Linear(flat, out_dim))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


class WellTestANN(nn.Module):
    """将曲线展平后拟合的全连接对照模型。"""

    def __init__(
        self, in_channels: int, out_dim: int, length: int = 128, batch_norm: bool = True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * length, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
