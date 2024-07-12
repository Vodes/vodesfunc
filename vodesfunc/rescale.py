from vstools import (
    vs,
    core,
    depth,
    get_depth,
    FunctionUtil,
    KwargsT,
    get_w,
    GenericVSFunction,
    ColorRange,
    iterate,
    expect_bits,
    replace_ranges,
    FrameRangesN,
)
from vskernels import KernelT, Kernel, ScalerT, Bilinear, Hermite, Bicubic, Lanczos
from vsscale import fdescale_args
from vsmasktools import EdgeDetectT, KirschTCanny
from typing import Self

from .scale import Doubler

__all__ = ["RescaleBuilder"]


class RescaleNumbers:
    height: float | int
    width: float | int
    base_height: int | None
    base_width: int | None


class RescaleBase:
    funcutil: FunctionUtil
    kernel: Kernel
    post_crop: KwargsT = KwargsT()
    rescale_args: KwargsT = KwargsT()

    descaled: vs.VideoNode
    rescaled: vs.VideoNode
    upscaled: vs.VideoNode | None = None
    doubled: vs.VideoNode | None = None
    linemask_clip: vs.VideoNode | None = None
    errormask_clip: vs.VideoNode | None = None


class RescaleBuilder(RescaleBase, RescaleNumbers):
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

    def __init__(self, clip: vs.VideoNode):
        self.funcutil = FunctionUtil(clip, self.__class__.__name__, planes=0, color_family=(vs.YUV, vs.GRAY), bitdepth=(16, 32))

    def descale(
        self,
        kernel: KernelT,
        width: int | float,
        height: int | float,
        base_height: int | None = None,
        base_width: int | None = None,
        shift: tuple[float, float] = (0, 0),
        mode: str = "hw",
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

        sanitized_shift = (shift[0] if shift[0] else None, shift[1] if shift[1] else None)
        args, self.post_crop = fdescale_args(clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode)
        _, self.rescale_args = fdescale_args(clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode, up_rate=1)
        print("Descale Args:", args, "Rescale Args:", self.rescale_args)
        self.height = args.get("src_height", clip.height)
        self.width = args.get("src_width", clip.width)
        self.base_height = base_height
        self.base_width = base_width

        self.descaled = self.kernel.descale(clip, **args)
        self.rescaled = descale_rescale(self, self.descaled, width=clip.width, height=clip.height, **self.rescale_args)
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

        return self

    def _errormask(
        self, mask: vs.VideoNode | float = 0.05, maximum_iter: int = 2, inflate_iter: int = 3, expand: int | tuple[int, int | None] = 0
    ) -> vs.VideoNode:
        if isinstance(mask, vs.VideoNode):
            return mask

        err_mask = core.std.Expr([depth(self.funcutil.work_clip, 32), depth(self.rescaled, 32)], f"x y - abs {mask} < 0 1 ?")
        err_mask = depth(err_mask, 16, range_out=ColorRange.FULL, range_in=ColorRange.FULL)
        err_mask = err_mask.rgvs.RemoveGrain(mode=6)
        err_mask = self._process_mask(err_mask, maximum_iter, inflate_iter, expand)
        err_mask = depth(err_mask, get_depth(self.funcutil.work_clip))

        return err_mask

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
        self.errormask_clip = self._errormask(mask, maximum_iter, inflate_iter, expand)
        return self

    def errormask_zoned(
        self,
        ranges: FrameRangesN,
        mask: vs.VideoNode | float = 0.05,
        maximum_iter: int = 2,
        inflate_iter: int = 3,
        expand: int | tuple[int, int | None] = 0,
    ) -> Self:
        """
        A function to apply a basic error mask to the final output.\n
        But with this rfs'd to certain ranges.
        """
        if not ranges:
            return self
        err_mask = self._errormask(mask, maximum_iter, inflate_iter, expand)
        self.errormask_clip = replace_ranges(self.errormask_clip, err_mask, ranges)
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
        self.upscaled = scaler.scale(self.doubled, wclip.width, wclip.height, **self.post_crop)

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


def descale_rescale(builder: RescaleBuilder, clip: vs.VideoNode, **kwargs: KwargsT) -> vs.VideoNode:
    kernel_args = KwargsT(border_handling=builder.kernel.kwargs.get("border_handling", 0))
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
