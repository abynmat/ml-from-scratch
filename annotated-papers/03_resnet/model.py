"""Plain vs residual CIFAR-style networks (depth 6n+2), from He et al. section 4.2.

Paper: He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition",
       CVPR 2016 (arXiv:1512.03385). Equation numbers refer to that paper.
Author: Abyn Scaria.

The plain and residual nets share every layer; the ONLY difference is whether
the block output adds the shortcut x. That makes the comparison exactly the
paper's: "plain" and "residual" nets with the same depth, width and parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Two 3x3 conv-BN layers, i.e. the residual function F(x, {W_i}) of eq. (1)."""

    def __init__(self, c_in, c_out, stride, residual, shortcut="A"):
        super().__init__()
        self.residual, self.shortcut = residual, shortcut
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.needs_proj = stride != 1 or c_in != c_out
        self.c_pad = c_out - c_in
        if residual and self.needs_proj and shortcut == "B":
            # eq. (2): y = F(x, {W_i}) + W_s x, a 1x1 projection to match dimensions
            self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                      nn.BatchNorm2d(c_out))

    def forward(self, x):
        f = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))   # F(x, {W_i})
        if not self.residual:
            return F.relu(f)                                        # plain: y = F(x)
        if not self.needs_proj:
            s = x                                                   # identity shortcut
        elif self.shortcut == "A":
            # Option A (used for the paper's CIFAR nets): identity with stride-2
            # subsampling and zero-padded extra channels, adds no parameters.
            s = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, self.c_pad))
        else:
            s = self.proj(x)
        return F.relu(f + s)                                        # eq. (1): y = F(x) + x, then ReLU


class Net(nn.Module):
    """3x3 conv (16) -> 3 stages of n blocks (16, 32, 64 filters; feature maps
    28, 14, 7 on Fashion-MNIST) -> global average pool -> fc.  Depth = 6n + 2."""

    def __init__(self, depth, residual, n_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        blocks, c = [], 16
        for stage, c_out in enumerate((16, 32, 64)):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                blocks.append(Block(c, c_out, stride, residual))
                c = c_out
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # He init

    def forward(self, x):
        return self.fc(self.blocks(self.stem(x)).mean(dim=(2, 3)))
