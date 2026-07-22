import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            affine_shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*affine_shape) + self.beta.view(*affine_shape)
        return x


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride=1,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super().__init__()

        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

        if norm == "batch":
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == "inst":
            self.norm = nn.InstanceNorm2d(output_dim, track_running_stats=False)
        elif norm == "ln":
            self.norm = LayerNorm(output_dim)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, bias=True)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        modules = []
        if scale == 1:
            pass
        elif (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError(f"Unsupported upsample scale: {scale}")

        super().__init__(*modules)

