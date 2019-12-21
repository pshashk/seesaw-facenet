import torch
from torch import nn
import torch.nn.functional as F
from math import pi


def hard_sigmoid(input):
    return F.relu6(input + 3.0) / 6.0


class HSigmoid(nn.Module):
    def forward(self, input):
        return hard_sigmoid(input)


class HSwish(nn.Module):
    def forward(self, input):
        return input * hard_sigmoid(input)


class SELayer(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            HSigmoid(),
        )

    def forward(self, input):
        output = input.mean((2, 3))
        output = self.layers(output)[:, :, None, None]
        return input * output


class PermutationBlock(nn.Module):
    def forward(self, input):
        n, c, h, w = input.size()
        output = input.view(n, 2, c // 2, h, w).transpose(1, 2)
        output = output.contiguous().view_as(input)
        return output


def Conv(c_in, c_out, ks=1, stride=1, pad=True, dwise=False):
    return nn.intrinsic.ConvBn2d(
        nn.Conv2d(
            c_in,
            c_out,
            kernel_size=ks,
            groups=c_in if dwise else 1,
            stride=stride,
            padding=(ks - 1) // 2 if pad else 0,
            bias=False,
        ),
        nn.BatchNorm2d(c_out),
    )


class UnevenGroupConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.small = Conv(c_in // 4, c_out // 4)
        self.big = Conv(c_in * 3 // 4, c_out * 3 // 4)
        self.split_at = c_in // 4

    def forward(self, input):
        split_at = self.split_at
        small = self.small(input[:, :split_at, ...])
        big = self.big(input[:, split_at:, ...])
        return torch.cat((small, big), 1)


class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input) + input


def Block(c_in, groups, c_out, stride=1, repeat=1):
    if repeat == 1:
        layers = [
            UnevenGroupConv(c_in, groups),
            HSwish(),
            PermutationBlock(),
            Conv(groups, groups, dwise=True, ks=3, stride=stride),
            HSwish(),
            SELayer(groups),
            UnevenGroupConv(groups, c_out),
        ]
        if stride == 1:
            for module in layers[-1].modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.data.zero_()
        container = nn.Sequential if (stride == 2) else Residual
        return container(*layers)
    else:
        blocks = []
        for _ in range(repeat):
            blocks.append(Block(c_in, groups, c_out, stride=stride))
        return nn.Sequential(*blocks)


def FaceNet():
    return nn.Sequential(
        Conv(3, 64, ks=3, stride=2),
        nn.ReLU(inplace=True),
        Conv(64, 64, ks=3, dwise=True),
        nn.ReLU(inplace=True),
        Block(64, 128, 64, stride=2),
        Block(64, 128, 64, repeat=4),
        Block(64, 256, 128, stride=2),
        Block(128, 256, 128, repeat=6),
        Block(128, 512, 128, stride=2),
        Block(128, 256, 128, repeat=2),
        Conv(128, 512),
        HSwish(),
        Conv(512, 512, ks=7, dwise=True, pad=False),
        nn.Flatten(1),
        nn.Linear(512, 512, bias=False),
        nn.BatchNorm1d(512),
    )


def fuse_modules(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.intrinsic.ConvBn2d):
            torch.quantization.fuse_modules(child, ["0", "1"], inplace=True)
        else:
            fuse_modules(child)


def release(model):
    model.eval()
    fuse_modules(model)
    model.requires_grad_(False)
    model = torch.jit.script(model)

    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert not any(m.training for m in model.modules())
    assert all(p.is_floating_point() for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model


class LiArcFace(nn.Module):
    def __init__(self, num_classes, emb_size=512, m=0.45, s=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_size))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s

    def forward(self, input, label):
        W = F.normalize(self.weight)
        input = F.normalize(input)
        cosine = input @ W.t()
        theta = torch.acos(cosine)
        m = torch.zeros_like(theta)
        m.scatter_(1, label.view(-1, 1), self.m)
        logits = self.s * (pi - 2 * (theta + m)) / pi
        return logits
