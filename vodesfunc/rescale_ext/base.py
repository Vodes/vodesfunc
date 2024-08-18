from vstools import FunctionUtil, KwargsT, vs, FieldBasedT, core, expect_bits, depth
from vskernels import Kernel, Bilinear, Bicubic, Lanczos

__all__ = ["RescaleBase", "RescaleNumbers", "descale_rescale"]


class RescaleNumbers:
    height: float | int
    width: float | int
    base_height: int | None
    base_width: int | None
    border_handling: int = 0


class RescaleBase(RescaleNumbers):
    funcutil: FunctionUtil
    kernel: Kernel
    post_crop: KwargsT = KwargsT()
    rescale_args: KwargsT = KwargsT()
    descale_func_args: KwargsT = KwargsT()
    field_based: FieldBasedT | None = None

    descaled: vs.VideoNode
    rescaled: vs.VideoNode
    upscaled: vs.VideoNode | None = None
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None


def descale_rescale(builder: RescaleBase, clip: vs.VideoNode, **kwargs: KwargsT) -> vs.VideoNode:
    kernel_args = KwargsT(border_handling=builder.border_handling)
    if isinstance(builder.kernel, Bilinear):
        kernel_function = core.descale.Bilinear
    elif isinstance(builder.kernel, Bicubic) or issubclass(builder.kernel.__class__, Bicubic):
        kernel_function = core.descale.Bicubic
        kernel_args.update({"b": builder.kernel.b, "c": builder.kernel.c})
    elif isinstance(builder.kernel, Lanczos):
        kernel_function = core.descale.Lanczos
        kernel_args.update({"taps": builder.kernel.taps})
    else:
        # I'm just lazy idk
        raise ValueError(f"{builder.kernel.__class__} is not supported for rescaling!")

    kernel_args.update(kwargs)
    clip, bits = expect_bits(clip, 32)
    return depth(kernel_function(clip, **kernel_args), bits)
