from vstools import FunctionUtil, KwargsT, vs, FieldBasedT, core, vs_object
from vskernels import Kernel, Bilinear, Bicubic, Lanczos, BorderHandling
from vsscale import ScalingArgs
from typing import Self, MutableMapping, TYPE_CHECKING, Callable
from abc import abstractmethod

__all__ = ["RescaleBase", "RescaleNumbers", "descale_rescale", "Ignore_Mask_Func"]

Ignore_Mask_Func = Callable[[vs.VideoNode, ScalingArgs, Kernel, BorderHandling], vs.VideoNode]


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
    ignore_mask: bool | vs.VideoNode | Ignore_Mask_Func

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
        self.ignore_mask = False
        for v in self.__dict__.values():
            if not isinstance(v, MutableMapping):
                continue

            for k2, v2 in v.items():
                if isinstance(v2, vs.VideoNode):
                    v[k2] = None


def descale_rescale(clip: vs.VideoNode, kernel: Kernel, **kwargs: KwargsT) -> vs.VideoNode:
    kernel_args = KwargsT(border_handling=kwargs.pop("border_handling", 0))
    if isinstance(kernel, Bilinear):
        kernel_function = core.descale.Bilinear
    elif isinstance(kernel, Bicubic) or issubclass(kernel.__class__, Bicubic):
        kernel_function = core.descale.Bicubic
        kernel_args.update({"b": kernel.b, "c": kernel.c})
    elif isinstance(kernel, Lanczos):
        kernel_function = core.descale.Lanczos
        kernel_args.update({"taps": kernel.taps})
    else:
        # I'm just lazy idk
        raise ValueError(f"{kernel.__class__} is not supported for rescaling!")

    kernel_args.update(kwargs)
    return kernel_function(clip, **kernel_args)
