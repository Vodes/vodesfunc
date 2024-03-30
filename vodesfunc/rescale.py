from vstools import (
    vs,
    core,
    depth,
    get_depth,
    FunctionUtil,
    get_h,
    get_lowest_value,
    get_peak_value,
    padder,
    KwargsT,
    get_w,
    GenericVSFunction,
    ColorRange,
    iterate,
)
from vskernels import KernelT, Kernel, ScalerT, Scaler, Bilinear, Hermite
from vsscale import fdescale_args, descale_args
from vsmasktools import EdgeDetectT, EdgeDetect, KirschTCanny
from typing import Sequence, Self, Any
from math import ceil
from dataclasses import dataclass

from .scale import Doubler

__all__ = ["RescaleBuilder"]


class RescaleNumbers:
    height: float | int
    width: float | int
    base_height: int | None
    base_width: int | None


class RescaleClips:
    descaled: vs.VideoNode
    rescaled: vs.VideoNode
    upscaled: vs.VideoNode
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None


class RescaleBuilder(RescaleClips, RescaleNumbers):
    """
    Proof of concept Builder approach to rescaling.

    In no way, shape or form done.
    Linemask needs borderhandling
    Descale doesn't handle fields yet
    """

    funcutil: FunctionUtil
    kernel: Kernel
    post_crop: KwargsT | None = None
    rescale_args: KwargsT = KwargsT()
    shift: tuple[float, float] = (0, 0)
    border_handling: int = 0
    border_radius: int | None = None

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
        border_handling: int = 0,
        border_radius: int | None = None,
    ) -> Self:
        clip = self.funcutil.work_clip
        self.kernel = Kernel.ensure_obj(kernel)
        self.shift = shift
        self.border_handling = self.kernel.kwargs.pop("border_handling", border_handling)
        self.border_radius = border_radius

        if float(height).is_integer():
            if not width:
                width = get_w(height, clip)
            self.width = int(width) if "w" in mode else clip.width
            self.height = int(height) if "h" in mode else clip.height
            self.descaled = self.kernel.descale(
                clip,
                self.width,
                self.height,
                shift=shift,
                border_handling=self.border_handling,
            )
            self.rescaled = perform_rescale(self)
        else:
            args, self.post_crop = fdescale_args(clip, height, base_height, base_width, shift[0], shift[1], width, mode)
            _, self.rescale_args = fdescale_args(clip, height, base_height, base_width, shift[0], shift[1], width, mode, up_rate=1)
            self.height = args.get("src_height", clip.height)
            self.width = args.get("src_width", clip.width)
            self.base_height = base_height
            self.base_width = base_width

            self.descaled = self.kernel.descale(clip, border_handling=self.border_handling, **args)
            self.rescaled = perform_rescale(self, **self.rescale_args)
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
        expand: int | tuple[int, int | None] = 0,
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

        if expand:
            if isinstance(expand, int):
                expand = (expand, expand)
            from vsmasktools import Morpho, XxpandMode

            self.linemask_clip = Morpho.expand(self.linemask_clip, expand[0], expand[1], XxpandMode.ELLIPSE)

        if self.border_handling:
            self.linemask_clip = self._crop_mask_bord(self.linemask_clip)

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

        if self.border_handling:
            self.errormask_clip = self._crop_mask_bord(self.errormask_clip)

        return self

    def double(self, upscaler: Doubler | ScalerT | None = None) -> Self:
        if isinstance(upscaler, Doubler):
            self.doubled = upscaler.double(self.descaled)
        else:
            from vsscale import Waifu2x

            scaler = Waifu2x.ensure_obj(upscaler)  # type: ignore
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
            self.final_mask = core.std.Expr([self.linemask_clip.std.Limiter(), self.errormask_clip], "x y -")
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

    def _crop_mask_bord(self, mask: vs.VideoNode, color: float = 0.0) -> vs.VideoNode:
        if not hasattr(self, "_bord_crop_args"):
            self._bord_crop_args = get_border_crop(self)

        return mask.std.Crop(*self._bord_crop_args).std.AddBorders(*self._bord_crop_args, color=color)

    @property
    def _kernel_window(self) -> int:
        if (bord_rad := self.border_radius) is None:
            try:
                bord_rad = self.kernel.kernel_radius
            except (AttributeError, NotImplementedError):
                bord_rad = 2

        return bord_rad


def perform_rescale(builder: RescaleBuilder, **kwargs: Any) -> vs.VideoNode:
    input_clip = builder.funcutil.work_clip
    clip = builder.descaled
    match int(builder.border_handling):
        case 1:
            clip = clip.std.AddBorders(
                *((0, 0) if builder.width == input_clip.width else (10, 10)),
                *((0, 0) if builder.height == input_clip.height else (10, 10)),
                get_lowest_value(clip, False, ColorRange.from_video(clip)),
            )
        case 2:
            clip = padder(
                clip,
                *((0, 0) if builder.width == input_clip.width else (10, 10)),
                *((0, 0) if builder.height == input_clip.height else (10, 10)),
                reflect=False,
            )
        case _:
            pass

    shift_top = kwargs.pop("src_top", False) or builder.shift[0]
    shift_left = kwargs.pop("src_left", False) or builder.shift[1]

    shift = [
        shift_top + (builder.height != input_clip.height and builder.border_handling) * 10,
        shift_left + (builder.width != input_clip.width and builder.border_handling) * 10,
    ]

    src_width = kwargs.pop("src_width", clip.width)
    src_height = kwargs.pop("src_height", clip.height)

    return builder.kernel.scale(
        clip,
        input_clip.width,
        input_clip.height,
        shift,  # type: ignore # Why?
        src_width=src_width - ((clip.width - builder.width) if float(builder.width).is_integer() else 0),
        src_height=src_height - ((clip.height - builder.height) if float(builder.width).is_integer() else 0),
    )


def get_border_crop(builder: RescaleBuilder) -> tuple:
    input_clip = builder.funcutil.work_clip
    # fmt: off
    if builder.height == input_clip.height:
        vertical_crop = (0, 0)
    else:
        base_height = builder.base_height or get_h(builder.base_width, builder.descaled) if builder.base_width else builder.height
        src_top = builder.rescale_args.get("src_top", False) or builder.shift[0]
        top = max(ceil(
            (-(builder.height - 1) / 2 + builder._kernel_window - src_top - 1)
            * input_clip.height / builder.height + (input_clip.height - 1) / 2
        ), 0)

        bottom = max(ceil(
            (-(builder.height - 1) / 2 + builder._kernel_window - (base_height - builder.height - src_top) - 1)
            * input_clip.height / builder.height + (input_clip.height - 1) / 2
        ), 0)

        vertical_crop = (top, bottom)

    if builder.width == input_clip.width:
        horizontal_crop = (0, 0)
    else:
        base_width = builder.base_width or get_w(builder.base_height, builder.descaled) if builder.base_height else builder.width
        src_left = builder.rescale_args.get("src_left", False) or builder.shift[1]

        left = max(ceil(
            (-(builder.width - 1) / 2 + builder._kernel_window - src_left - 1)
            * input_clip.width / builder.width + (input_clip.width - 1) / 2
        ), 0)

        right = max(ceil(
            (-(builder.width - 1) / 2 + builder._kernel_window - (base_width - builder.width - src_left) - 1)
            * input_clip.width / builder.width + (input_clip.width - 1) / 2
        ), 0)

        horizontal_crop = (left, right)
    # fmt: on
    return horizontal_crop + vertical_crop
