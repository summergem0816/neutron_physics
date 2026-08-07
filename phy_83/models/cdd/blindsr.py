from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from . import common


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super().__init__()
        hidden = max(channels_in // reduction, 1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, hidden, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.conv_du(x[1][:, :, None, None])
        return x[0] * att


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, channels_in * kernel_size * kernel_size, bias=False),
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        batch, channels, height, width = x[0].size()
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(
            F.conv2d(
                x[0].view(1, -1, height, width),
                kernel,
                groups=batch * channels,
                padding=(self.kernel_size - 1) // 2,
            )
        )
        out = self.conv(out.view(batch, -1, height, width))
        out = out + self.ca(x)
        return out


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super().__init__()
        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        modules_body = [DAB(conv, n_feat, kernel_size, reduction) for _ in range(n_blocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = x[0]
        if len(x) > 2 and x[2] is not None:
            if x[2].shape[0] > 1:
                res = res + x[2][0].unsqueeze(0)
            else:
                res = res + x[2]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        return res + x[0]


class LPE(nn.Module):
    def __init__(self, dim=256, out_dim=256, k=13):
        super().__init__()

        lpe_p = [
            common.Conv2dBlock(3, dim // 4, k, int((k - 1) / 2), padding=0, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 4, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 2, dim, 3, 1, padding=1, norm="batch", activation="lrelu"),
        ]

        lpe_l = [
            common.Conv2dBlock(3, dim // 4, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 4, 1, 1, padding=0, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 2, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 2, dim // 2, 1, 1, padding=0, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 2, dim, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim, dim, 1, 1, padding=0, norm="batch", activation="lrelu"),
        ]

        self.lpe_p = nn.Sequential(*lpe_p)
        self.lpe_l = nn.Sequential(*lpe_l)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.last_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        x_p = self.lpe_p(x)
        x_l = self.lpe_l(x)
        x_p = self.avg(x_p).squeeze(-1).squeeze(-1)
        x_l = self.avg(x_l).squeeze(-1).squeeze(-1)
        feat = torch.cat((x_p, x_l), dim=1)
        out = self.last_mlp(feat)
        return out, out

