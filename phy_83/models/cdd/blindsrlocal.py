import torch
from torch import nn

from . import common
from .blindsr import DA_conv


MAP_RATIO = 2


class IDADepthwiseConv(nn.Module):
    def __init__(self, channels_in_x, channels_in_map, channels_mid, channels_out, ds_factor=2):
        super().__init__()
        self.transform_map = nn.Sequential(
            nn.ConvTranspose2d(
                channels_in_map,
                channels_mid,
                kernel_size=MAP_RATIO,
                stride=MAP_RATIO,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_mid, channels_mid, kernel_size=3, stride=ds_factor, padding=1, bias=False),
        )
        self.transform_x = nn.Conv2d(
            channels_in_x,
            channels_mid,
            kernel_size=3,
            stride=ds_factor,
            padding=1,
            bias=False,
        )
        self.deconv = nn.ConvTranspose2d(
            channels_mid,
            channels_out,
            kernel_size=2,
            stride=ds_factor,
            padding=0,
            output_padding=0,
        )

    def forward(self, x):
        res = self.transform_x(x[0]) * self.transform_map(x[1])
        return self.deconv(res)


class IDAChannelwiseConv(nn.Module):
    def __init__(self, channels_in, channels_mid, channels_out):
        super().__init__()
        self.transform_map = nn.Sequential(
            nn.ConvTranspose2d(
                channels_in,
                channels_mid,
                kernel_size=MAP_RATIO,
                stride=MAP_RATIO,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_mid, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x[0] * self.transform_map(x[1])


class IDA_conv(nn.Module):
    def __init__(self, channels_in_x, channels_in_map, channels_out, ds_factor=2):
        super().__init__()
        channels_mid = max((channels_in_map + channels_out) // 2, 1)
        self.depthwise = IDADepthwiseConv(
            channels_in_x,
            channels_in_map,
            channels_mid,
            channels_out,
            ds_factor,
        )
        self.channelwise = IDAChannelwiseConv(channels_in_map, channels_mid, channels_out)

    def forward(self, x):
        return self.depthwise(x) + self.channelwise(x)


class IDAB(nn.Module):
    def __init__(self, n_feat, n_feat_map, kernel_size, reduction):
        super().__init__()
        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.ida_conv1 = IDA_conv(n_feat, n_feat_map, n_feat)
        self.ida_conv2 = IDA_conv(n_feat, n_feat_map, n_feat)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.relu(self.da_conv1([x[0], x[1]]))
        out = self.relu(self.ida_conv1([out, x[2]]))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.ida_conv2([out, x[2]]) + x[0]
        return out


class IDAG(nn.Module):
    def __init__(self, conv, n_feat, n_feat_map, kernel_size, reduction, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        modules_body = [IDAB(n_feat, n_feat_map, kernel_size, reduction) for _ in range(n_blocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = x[0]
        if len(x) > 3 and x[2] is not None:
            if x[2].shape[0] > 1:
                res = res + x[2][0].unsqueeze(0)
            else:
                res = res + x[2]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[3]])
        res = self.body[-1](res)
        return res + x[0]


class IDASR(nn.Module):
    def __init__(
        self,
        in_channels=3,
        map_in_channels=4,
        global_dim=256,
        scale=1,
        kernel_size=3,
        hf=False,
        n_feats=64,
        n_groups=5,
        n_blocks=5,
        reduction=8,
        n_feats_map=16,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.hf_enabled = hf

        self.head = nn.Sequential(common.default_conv(in_channels, n_feats, kernel_size))
        self.head_map = nn.Sequential(common.default_conv(map_in_channels, n_feats_map, kernel_size))
        self.compress = nn.Sequential(
            nn.Linear(global_dim, n_feats, bias=False),
            nn.LeakyReLU(0.1, True),
        )

        modules_body = [
            IDAG(common.default_conv, n_feats, n_feats_map, kernel_size, reduction, n_blocks)
            for _ in range(n_groups)
        ]
        modules_body.append(common.default_conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        tail = []
        if scale != 1:
            tail.append(common.Upsampler(common.default_conv, scale, n_feats, act=False))
        tail.append(common.default_conv(n_feats, in_channels, kernel_size))
        self.tail = nn.Sequential(*tail)

        if self.hf_enabled:
            self.hf = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_feats, 16),
                        nn.LeakyReLU(0.1, True),
                        nn.Linear(16, 2),
                    )
                    for _ in range(n_groups)
                ]
            )

    def forward(self, x, k_v, dmap, pos_emb=None):
        k_v = self.compress(k_v)
        x = self.head(x)
        dmap = self.head_map(dmap)

        res = x
        for i in range(self.n_groups):
            if self.hf_enabled:
                hf_param = self.hf[i](k_v).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                noise = hf_param[:, 0] + torch.randn_like(res) * hf_param[:, 1]
                res = res + noise.to(res.device)
            res = self.body[i]([res, k_v, pos_emb, dmap])
        res = self.body[-1](res)
        res = res + x
        return self.tail(res)


class LPEL(nn.Module):
    def __init__(self, dim=256, out_dim=256, k=13):
        super().__init__()
        if k % 2 != 1:
            raise ValueError("k should be odd")

        stride = (k - 1) // 2
        padding = stride // 2
        self._stride = stride

        lpe_p = [
            common.Conv2dBlock(3, dim // 4, k, stride, padding=padding, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 4, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 4, dim // 2, 3, 1, padding=1, norm="batch", activation="lrelu"),
            common.Conv2dBlock(dim // 2, dim // 2, 3, 1, padding=1, norm="batch", activation="lrelu"),
            nn.ConvTranspose2d(dim // 2, dim, k, stride=stride, padding=0),
            nn.AvgPool2d(kernel_size=2),
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
        self.last_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_p = self.lpe_p(x)
        x_l = self.lpe_l(x)
        if x_p.shape[-1] != x_l.shape[-1] or x_p.shape[-2] != x_l.shape[-2]:
            if not (0 < x_p.shape[-1] - x_l.shape[-1] < self._stride and 0 < x_p.shape[-2] - x_l.shape[-2] < self._stride):
                raise RuntimeError(f"Unexpected LPEL shape mismatch: {x_p.shape} vs {x_l.shape}")
            x_p = x_p[:, :, : x_l.shape[-2], : x_l.shape[-1]]
        feat = torch.cat((x_p, x_l), dim=1)
        out = self.last_mlp(feat)
        return out, out

