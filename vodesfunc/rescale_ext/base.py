from vstools import FunctionUtil, KwargsT, vs, FieldBasedT, core, expect_bits, depth, vs_object
from vskernels import Kernel, Bilinear, Bicubic, Lanczos
from typing import Self, MutableMapping, TYPE_CHECKING
from abc import abstractmethod

__all__ = ["RescaleBase", "RescaleNumbers", "descale_rescale"]


class RescaleNumbers:
    height: float | int
    width: float | int
    base_height: int | None
    base_width: int | None
    border_handling: int = 0


class RescaleBase(RescaleNumbers, vs_object):
    funcutil: FunctionUtil
    kernel: Kernel
    post_crop: KwargsT
    rescale_args: KwargsT
    descale_func_args: KwargsT
    field_based: FieldBasedT | None = None

    descaled: vs.VideoNode
    rescaled: vs.VideoNode
    upscaled: vs.VideoNode | None = None
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None

    @abstractmethod
    def final(self) -> tuple[Self, vs.VideoNode]: ...

    def _return_creditmask(self) -> vs.VideoNode:
        return self.errormask_clip if isinstance(self.errormask_clip, vs.VideoNode) else core.std.BlankClip(self.funcutil.work_clip)

    def _return_linemask(self) -> vs.VideoNode:
        return self.linemask_clip if isinstance(self.linemask_clip, vs.VideoNode) else core.std.BlankClip(self.funcutil.work_clip)

    def __vs_del__(self, core_id: int) -> None:
        if not TYPE_CHECKING:
            self.descaled = None
            self.rescaled = None
        self.upscaled = None
        self.doubled = None
        self.linemask_clip = None
        self.errormask_clip = None
        for v in self.__dict__.values():
            if not isinstance(v, MutableMapping):
                continue

            for k2, v2 in v.items():
                if isinstance(v2, vs.VideoNode):
                    v[k2] = None


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
