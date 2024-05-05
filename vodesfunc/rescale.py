from vstools import (
    vs,
    core,
    depth,
    get_depth,
    FunctionUtil,
    get_h,
    get_lowest_value,
    padder,
    KwargsT,
    get_w,
    GenericVSFunction,
    ColorRange,
    iterate,
)
from vskernels import KernelT, Kernel, ScalerT, Bilinear, Hermite
from vsscale import fdescale_args
from vsmasktools import EdgeDetectT, KirschTCanny
from typing import Self, Any
from math import ceil

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
    upscaled: vs.VideoNode | None = None
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None


class RescaleBuilder(RescaleClips, RescaleNumbers):
    """
    Proof of concept Builder approach to rescaling.\n
    Mostly ready for single rescale use. Not entirely sure how to handle multiple properly yet.

    Doesn't handle FieldBased yet.\n
    (Do I even have to do anything? Pretty sure vskernels does most of the work nowadays)

    Example usage:
    ```py
    builder, rescaled = (
        RescaleBuilder(clip)
        .descale(Bilinear, 843.75, base_height=846)
        .double()
        .errormask(0.0975)
        .linemask()
        .post_double(lambda x: aa_dehalo(x)) # Or a function like post_double(aa_dehalo)
        .downscale(Hermite(linear=True))
        .final()
    )
    ```
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
        """
        Performs descale and rescale (with the same kernel).

        :param kernel:              Kernel to descale with
        :param height:              Height to descale to
        :param width:               Width to descale to
        :param base_height:         Padded height used in a "fractional" descale
        :param base_width:          Padded width used in a "fractional" descale
                                    Both of these are technically optional but highly recommended to have set for float width/height.

        :param shift:               A custom shift to be applied
        :param mode:                Whether to descale only height, only width, or both.
                                    "h" or "w" respectively for the former two.

        :param border_handling:     Adjust the way the clip is padded internally during the scaling process. Accepted values are:\n
                                    0: Assume the image was resized with mirror padding.\n
                                    1: Assume the image was resized with zero padding.\n
                                    2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far.\n
                                    Defaults to 0.

        :param border_radius:       Radius for the border mask. Only used when border_handling is set to 1 or 2.
                                    Defaults to kernel radius if possible, else 2.
        """
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
            sanitized_shift = (shift[0] if shift[0] else None, shift[1] if shift[1] else None)
            args, self.post_crop = fdescale_args(clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode)
            _, self.rescale_args = fdescale_args(
                clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode, up_rate=1
            )
            self.height = args.get("src_height", clip.height)
            self.width = args.get("src_width", clip.width)
            self.base_height = base_height
            self.base_width = base_width

            self.descaled = self.kernel.descale(clip, border_handling=self.border_handling, **args)
            self.rescaled = perform_rescale(self, **self.rescale_args)
        return self

    def post_descale(self, func: GenericVSFunction) -> Self:
        """
        A function to apply any arbitrary function on the descaled clip.\n
        I can't think of a good usecase/example for this but I was asked to add this before.

        :param func:    This can be any function that takes a videonode input and returns a videonode.
                        You are responsible for keeping the format the same.
        """
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
        """
        A function to apply a linemask to the final output.

        :param mask:            This can be a masking function like `KirschTCanny` (also the default if `None`) or a clip.
        :param downscaler:      Downscaler to use if creating a linemask on the doubled clip. Defaults to `Bilinear` if `None`.
        :param maximum_iter:    Apply std.Maximum x amount of times
        :param inflate_iter:    Apply std.inflate x amount of times
        :param expand:          Apply an ellipse morpho expand with the passed amount.
                                Can be a tuple of (horizontal, vertical) or a single value for both.
        :param **kwargs:        Any other params to pass to the edgemask creation. For example `lthr` or `hthr`.
        """
        if isinstance(mask, vs.VideoNode):
            self.linemask_clip = mask
            if self.border_handling:
                self.linemask_clip = self._crop_mask_bord(self.linemask_clip)
            return self
        edgemaskFunc = KirschTCanny.ensure_obj(mask)

        # Perform on doubled clip if exists and downscale
        if self.doubled:
            scaler = Bilinear.ensure_obj(downscaler)
            self.linemask_clip = edgemaskFunc.edgemask(self.doubled, **kwargs)
            self.linemask_clip = scaler.scale(self.linemask_clip, self.funcutil.work_clip.width, self.funcutil.work_clip.height)
        else:
            self.linemask_clip = edgemaskFunc.edgemask(self.funcutil.work_clip, **kwargs)

        self.linemask_clip = self._process_mask(self.linemask_clip, maximum_iter, inflate_iter, expand)

        if self.border_handling:
            self.linemask_clip = self._crop_mask_bord(self.linemask_clip)

        return self

    def errormask(
        self, mask: vs.VideoNode | float = 0.05, maximum_iter: int = 2, inflate_iter: int = 3, expand: int | tuple[int, int | None] = 0
    ) -> Self:
        """
        A function to apply a basic error mask to the final output.

        :param mask:            With a float, and by default, will be created internally. Could also pass a clip.
        :param maximum_iter:    Apply std.Maximum x amount of times
        :param inflate_iter:    Apply std.inflate x amount of times
        :param expand:          Apply an ellipse morpho expand with the passed amount.
                                Can be a tuple of (horizontal, vertical) or a single value for both.
        """
        if isinstance(mask, vs.VideoNode):
            self.errormask_clip = mask
            return self

        err_mask = core.std.Expr([depth(self.funcutil.work_clip, 32), depth(self.rescaled, 32)], f"x y - abs {mask} < 0 1 ?")
        err_mask = depth(err_mask, 16, range_out=ColorRange.FULL, range_in=ColorRange.FULL)
        err_mask = err_mask.rgvs.RemoveGrain(mode=6)
        err_mask = self._process_mask(err_mask, maximum_iter, inflate_iter, expand)
        self.errormask_clip = depth(err_mask, get_depth(self.funcutil.work_clip))

        if self.border_handling:
            self.errormask_clip = self._crop_mask_bord(self.errormask_clip)

        return self

    def double(self, upscaler: Doubler | ScalerT | None = None) -> Self:
        """
        Upscales the descaled clip by 2x

        :param upscaler:        Any kind of vsscale scaler. Defaults to Waifu2x.
        """
        if isinstance(upscaler, Doubler):
            self.doubled = upscaler.double(self.descaled)
        else:
            from vsscale import Waifu2x

            scaler = Waifu2x.ensure_obj(upscaler)  # type: ignore
            self.doubled = scaler.multi(self.descaled)
        return self

    def post_double(self, func: GenericVSFunction) -> Self:
        """
        A function to apply any arbitrary function on the doubled clip.

        :param func:    This can be any function that takes a videonode input and returns a videonode.
                        You are responsible for keeping the format the same.
        """
        if not self.doubled:
            raise SyntaxError("post_double: Doubled clip has not been generated yet. Please call this after double().")
        self.doubled = func(self.doubled)
        return self

    def downscale(self, downscaler: ScalerT | None = None) -> Self:
        """
        Downscales the clip back the size of the original input clip and applies the masks, if any.

        :param downscaler:      Any vsscale scaler to use. Defaults to Linear Hermite.
        """
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
            self.upscaled = wclip.std.MaskedMerge(self.upscaled, self.final_mask.std.Limiter())
        elif isinstance(self.errormask_clip, vs.VideoNode):
            self.upscaled = self.upscaled.std.MaskedMerge(wclip, self.errormask_clip.std.Limiter())
        elif isinstance(self.linemask_clip, vs.VideoNode):
            self.upscaled = wclip.std.MaskedMerge(self.upscaled, self.linemask_clip.std.Limiter())

        return self

    def final(self) -> tuple[Self, vs.VideoNode]:
        """
        This is the last function in the chain that also returns the final clip.
        It internally calls `downscale` if you haven't done so before and then merges the resulting clip with the input chroma, if any.

        :return: A tuple of this class and the resulting final rescale.
        """
        if not self.upscaled:
            self.downscale()
        if not self.upscaled:
            raise TypeError("No downscaled clip has been generated yet!")

        return (self, self.funcutil.return_clip(self.upscaled))

    def _process_mask(
        self, mask: vs.VideoNode, maximum_iter: int = 0, inflate_iter: int = 0, expand: int | tuple[int, int | None] = 0
    ) -> vs.VideoNode:
        if maximum_iter:
            mask = iterate(mask, core.std.Maximum, maximum_iter)

        if inflate_iter:
            mask = iterate(mask, core.std.Inflate, inflate_iter)

        if expand:
            if isinstance(expand, int):
                expand = (expand, expand)
            from vsmasktools import Morpho, XxpandMode

            mask = Morpho.expand(mask, expand[0], expand[1], XxpandMode.ELLIPSE)

        return mask

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
