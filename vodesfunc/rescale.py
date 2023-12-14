from vstools import vs, core, split, get_y, depth, get_depth, FunctionUtil, to_arr, KwargsT, get_w, GenericVSFunction, ColorRange, iterate
from vskernels import KernelT, Kernel, ScalerT, Scaler, Bilinear, Hermite
from vsscale import fdescale_args, descale_args
from vsmasktools import EdgeDetectT, EdgeDetect, KirschTCanny
from typing import Sequence, Self
from dataclasses import dataclass

from .scale import Doubler

__all__ = ["RescaleBuilder"]


class RescaleBuilder:
    """
    Proof of concept Builder approach to rescaling.

    In no way, shape or form done.
    Linemask needs borderhandling
    Descale doesn't handle fields yet
    """

    funcutil: FunctionUtil
    kernel: Kernel
    descaled: vs.VideoNode
    rescaled: vs.VideoNode
    upscaled: vs.VideoNode
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None
    post_crop: KwargsT | None = None
    shift: tuple[float, float] = (0, 0)

    def __init__(self, clip: vs.VideoNode):
        self.funcutil = FunctionUtil(clip, self.__class__.__name__, planes=0, color_family=(vs.YUV, vs.GRAY), bitdepth=(16, 32))

    def descale(
        self,
        kernel: KernelT,
        height: float,
        width: float | None = None,
        base_height: int | None = None,
        base_width: int | None = None,
        shift: tuple[float, float] = (0, 0),
        mode: str = "hw",
    ) -> Self:
        clip = self.funcutil.work_clip
        self.kernel = Kernel.ensure_obj(kernel)
        self.shift = shift
        if float(height).is_integer():
            if not width:
                width = get_w(height, clip)
            self.descaled = self.kernel.descale(
                clip,
                int(width) if "w" in mode else clip.width,
                int(height) if "h" in mode else clip.height,
                shift,
            )
            self.rescaled = self.kernel.scale(self.descaled, clip.width, clip.height, shift)
        else:
            args, self.post_crop = fdescale_args(clip, height, base_height, base_width, shift[0], shift[1], width, mode)
            _, rescale_args = fdescale_args(clip, height, base_height, base_width, shift[0], shift[1], width, mode, up_rate=1)
            self.descaled = self.kernel.descale(clip, **args)
            self.rescaled = self.kernel.scale(self.descaled, clip.width, clip.height, **rescale_args)
        return self

    def post_descale(self, func: GenericVSFunction) -> Self:
        self.descaled = func(self.descaled)
        return self

    def linemask(
        self,
        mask: vs.VideoNode | EdgeDetectT | None = None,
        downscaler: ScalerT | None = None,
        maximum_iter: int = 0,
        inflate_iter: int = 0,
        **kwargs,
    ) -> Self:
        if isinstance(mask, vs.VideoNode):
            self.linemask_clip = mask
            return self
        edgemaskFunc = KirschTCanny.ensure_obj(mask)

        # Perform on doubled clip if exists and downscale
        if self.doubled:
            scaler = Bilinear.ensure_obj(downscaler)
            self.linemask_clip = edgemaskFunc.edgemask(self.doubled, **kwargs)
            self.linemask_clip = scaler.scale(self.linemask_clip, self.funcutil.work_clip.width, self.funcutil.work_clip.height)
        else:
            self.linemask_clip = edgemaskFunc.edgemask(self.funcutil.work_clip, **kwargs)

        if maximum_iter:
            self.linemask_clip = iterate(self.linemask_clip, core.std.Maximum, maximum_iter)
        if inflate_iter:
            self.linemask_clip = iterate(self.linemask_clip, core.std.Inflate, inflate_iter)
        return self

    def errormask(self, mask: vs.VideoNode | float = 0.05, maximum_iter: int = 2, inflate_iter: int = 3) -> Self:
        if isinstance(mask, vs.VideoNode):
            self.errormask_clip = mask
            return self

        err_mask = core.std.Expr([depth(self.funcutil.work_clip, 32), depth(self.rescaled, 32)], f"x y - abs {mask} < 0 1 ?")
        err_mask = depth(err_mask, 16, range_out=ColorRange.FULL, range_in=ColorRange.FULL)
        err_mask = err_mask.rgvs.RemoveGrain(mode=6)
        err_mask = iterate(err_mask, core.std.Maximum, maximum_iter)
        err_mask = iterate(err_mask, core.std.Inflate, inflate_iter)
        self.errormask_clip = depth(err_mask, get_depth(self.funcutil.work_clip))
        return self

    def double(self, upscaler: Doubler | ScalerT | None = None) -> Self:
        if isinstance(upscaler, Doubler):
            self.doubled = upscaler.double(self.descaled)
        else:
            from vsscale import Waifu2x

            scaler = Waifu2x.ensure_obj(upscaler)
            self.doubled = scaler.multi(self.descaled)
        return self

    def post_double(self, func: GenericVSFunction) -> Self:
        self.doubled = func(self.doubled)
        return self

    def downscale(self, downscaler: ScalerT | None = None) -> Self:
        scaler = Hermite(linear=True).ensure_obj(downscaler)
        if not self.doubled:
            raise SyntaxError("Downscale/Final is the last one that should be called in a chain!")
        wclip = self.funcutil.work_clip
        if self.post_crop:
            self.upscaled = scaler.scale(self.doubled, wclip.width, wclip.height, **self.post_crop)
        else:
            self.upscaled = scaler.scale(self.doubled, wclip.width, wclip.height, (self.shift[0] * 2, self.shift[1] * 2))

        if isinstance(self.errormask_clip, vs.VideoNode) and isinstance(self.linemask_clip, vs.VideoNode):
            self.final_mask = core.std.Expr([self.linemask_clip, self.errormask_clip], "x y -")
            self.upscaled = self.upscaled.std.MaskedMerge(self.upscaled, self.final_mask.std.Limiter())
        elif isinstance(self.errormask_clip, vs.VideoNode):
            self.upscaled = self.upscaled.std.MaskedMerge(self.upscaled, self.errormask_clip.std.Limiter())
        elif isinstance(self.linemask_clip, vs.VideoNode):
            self.upscaled = self.upscaled.std.MaskedMerge(self.upscaled, self.linemask_clip.std.Limiter())

        return self

    def final(self) -> tuple[Self, vs.VideoNode]:
        if not self.upscaled:
            self.downscale()
        return (self, self.funcutil.return_clip(self.upscaled))
