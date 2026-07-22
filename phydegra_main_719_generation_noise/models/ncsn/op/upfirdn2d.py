import os
import warnings

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
# Original code for Linux / full CUDA toolkit environments:
# upfirdn2d_op = load(
#     "upfirdn2d",
#     sources=[
#         os.path.join(module_path, "upfirdn2d.cpp"),
#         os.path.join(module_path, "upfirdn2d_kernel.cu"),
#     ],
# )
#
# Temporary Windows fallback patch:
# Keep the original block above commented out, and use lazy loading below so
# machines without CUDA_HOME / cl.exe can still run by falling back to the
# native PyTorch implementation.
upfirdn2d_op = None
_upfirdn2d_load_attempted = False


def _get_upfirdn2d_op():
    global upfirdn2d_op, _upfirdn2d_load_attempted
    if upfirdn2d_op is not None:
        return upfirdn2d_op
    if _upfirdn2d_load_attempted:
        return None

    _upfirdn2d_load_attempted = True
    try:
        upfirdn2d_op = load(
            "upfirdn2d",
            sources=[
                os.path.join(module_path, "upfirdn2d.cpp"),
                os.path.join(module_path, "upfirdn2d_kernel.cu"),
            ],
        )
    except (OSError, RuntimeError) as error:
        warnings.warn(
            "Failed to build/load CUDA upfirdn2d extension; falling back to "
            f"the native PyTorch implementation. Original error: {error}"
        )
        upfirdn2d_op = None

    return upfirdn2d_op


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        # Original code:
        # up_x, up_y = up
        # down_x, down_y = down
        # g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        #
        # grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        #
        # grad_input = upfirdn2d_op.upfirdn2d(
        #     grad_output,
        #     grad_kernel,
        #     down_x,
        #     down_y,
        #     up_x,
        #     up_y,
        #     g_pad_x0,
        #     g_pad_x1,
        #     g_pad_y0,
        #     g_pad_y1,
        # )
        op = _get_upfirdn2d_op()
        if op is None:
            raise RuntimeError("upfirdn2d CUDA extension is unavailable")

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        # Original code:
        # gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        #
        # gradgrad_out = upfirdn2d_op.upfirdn2d(
        #     gradgrad_input,
        #     kernel,
        #     ctx.up_x,
        #     ctx.up_y,
        #     ctx.down_x,
        #     ctx.down_y,
        #     ctx.pad_x0,
        #     ctx.pad_x1,
        #     ctx.pad_y0,
        #     ctx.pad_y1,
        # )
        op = _get_upfirdn2d_op()
        if op is None:
            raise RuntimeError("upfirdn2d CUDA extension is unavailable")

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        # Original code:
        # up_x, up_y = up
        # down_x, down_y = down
        # pad_x0, pad_x1, pad_y0, pad_y1 = pad
        # ...
        # out = upfirdn2d_op.upfirdn2d(
        #     input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        # )
        op = _get_upfirdn2d_op()
        if op is None:
            raise RuntimeError("upfirdn2d CUDA extension is unavailable")

        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == "cpu":
        out = upfirdn2d_native(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )

    else:
        # Original code:
        # out = UpFirDn2d.apply(
        #     input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        # )
        #
        # Temporary Windows fallback:
        # If CUDA extension build/load fails, fall back to native PyTorch op
        # instead of aborting import/training.
        op = _get_upfirdn2d_op()
        if op is None:
            out = upfirdn2d_native(
                input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
            )
        else:
            out = UpFirDn2d.apply(
                input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
            )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
