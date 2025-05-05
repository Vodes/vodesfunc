from vstools import (
    vs,
    core,
    FunctionUtil,
    GenericVSFunction,
    iterate,
    replace_ranges,
    FrameRangesN,
    get_peak_value,
    FieldBasedT,
    FieldBased,
    CustomValueError,
    get_video_format,
)
from vskernels import KernelT, Kernel, ScalerT, Bilinear, Hermite
from vsmasktools import EdgeDetectT, KirschTCanny
from vsrgtools import removegrain
from typing import Self
import inspect

from .scale import Doubler
from .rescale_ext import RescBuildFB, RescBuildNonFB
from .rescale_ext.mixed_rescale import RescBuildMixed

__all__ = ["RescaleBuilder"]


class RescaleBuilder(RescBuildFB, RescBuildNonFB, RescBuildMixed):
    """
    The fancy new rescale wrapper to make life easier.
    Now 99% less buggy and should handle everything.

    Example usage:
    ```py
    builder, rescaled = (
        RescaleBuilder(clip)
        .descale(Bilinear, 1500, 843.75, base_height=846)
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
        self.funcutil = FunctionUtil(clip, self.__class__.__name__, planes=0, color_family=(vs.YUV, vs.GRAY), bitdepth=32)

    def descale(
        self,
        kernel: KernelT,
        width: int | float,
        height: int | float,
        base_height: int | None = None,
        base_width: int | None = None,
        shift: tuple[float, float] = (0, 0),
        field_based: FieldBasedT | None = None,
        mode: str = "hw",
    ) -> Self:
        """
        Performs descale and rescale (with the same kernel).

        :param kernel:              Kernel to descale with
        :param height:              Height to descale to
        :param width:               Width to descale to. Please be absolutely certain of what you're doing if you're using get_w for this.
        :param base_height:         Padded height used in a "fractional" descale
        :param base_width:          Padded width used in a "fractional" descale
                                    Both of these are technically optional but highly recommended to have set for float width/height.

        :param shift:               A custom shift to be applied
        :param mode:                Whether to descale only height, only width, or both.
                                    "h" or "w" respectively for the former two.
        :param field_based:         To descale a cross-converted/interlaced clip.
                                    Will try to take the prop from the clip if `None` was passed.
        """
        clip = self.funcutil.work_clip

        if isinstance(height, float) and len(stack := inspect.stack()) > 1:
            has_getw = [ctx for ctx in stack[1].code_context if "get_w" in ctx.lower()]
            if has_getw:
                print("RescaleBuilder: Please make sure get_w returns the width you really want!")

        self.kernel = Kernel.ensure_obj(kernel)
        self.border_handling = self.kernel.kwargs.pop("border_handling", 0)
        self.field_based = FieldBased.from_param(field_based) or FieldBased.from_video(clip)

        self.height = height if "h" in mode else clip.height
        self.width = width if "w" in mode else clip.width
        self.base_height = base_height
        self.base_width = base_width

        if (isinstance(width, float) or isinstance(height, float)) and self.field_based.is_inter:
            raise CustomValueError("Float is not supported for fieldbased descales!", self.descale)

        if self.field_based.is_inter:
            self._fieldbased_descale(clip, width=self.width, height=self.height, shift=shift, border_handling=self.border_handling)
        else:
            self._non_fieldbased_descale(clip, width, height, base_height, base_width, shift, mode)

        self.descaled = self.descaled.std.CopyFrameProps(clip)
        self.rescaled = self.rescaled.std.CopyFrameProps(clip)

        return self

    def post_descale(self, func: GenericVSFunction | list[GenericVSFunction]) -> Self:
        """
        A function to apply any arbitrary function on the descaled clip.\n
        I can't think of a good usecase/example for this but I was asked to add this before.

        :param func:    This can be any function or list of functions that take a videonode input
                        and returns a videonode. You are responsible for keeping the format the same.
        """
        if not isinstance(func, list):
            func = [func]

        for f in func:
            if not callable(f):
                raise CustomValueError(f"post_descale: Function {f.__name__} is not callable!", self.post_descale)

            self.descaled = f(self.descaled)

        return self

    def linemask(
        self,
        mask: vs.VideoNode | EdgeDetectT | None = None,
        downscaler: ScalerT | None = None,
        maximum_iter: int = 0,
        inflate_iter: int = 0,
        expand: int | tuple[int, int | None] = 0,
        kernel_window: int | None = None,
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
        :param kernel_window:   To override kernel radius used in case of border_handling being used.
        :param **kwargs:        Any other params to pass to the edgemask creation. For example `lthr` or `hthr`.
        """
        if self.upscaled:
            raise SyntaxError("RescaleBuilder: Downscaled clip already created. Create linemasks before calling downscale.")
        if isinstance(mask, vs.VideoNode):
            self.linemask_clip = mask
            return self
        edgemaskFunc = KirschTCanny.ensure_obj(mask)

        # Perform on doubled clip if exists and downscale
        if self.doubled:
            scaler = Bilinear.ensure_obj(downscaler)
            self.linemask_clip = edgemaskFunc.edgemask(self.doubled, **kwargs)
            self.linemask_clip = scaler.scale(self.linemask_clip, self.funcutil.work_clip.width, self.funcutil.work_clip.height, **self.post_crop)
        else:
            self.linemask_clip = edgemaskFunc.edgemask(self.funcutil.work_clip, **kwargs)

        self.linemask_clip = self._process_mask(self.linemask_clip, maximum_iter, inflate_iter, expand)

        if self.border_handling:
            from .misc import get_border_crop

            borders = get_border_crop(self.funcutil.work_clip, self, kernel_window)
            self.linemask_clip = self.linemask_clip.std.Crop(*borders).std.AddBorders(*borders, [get_peak_value(self.linemask_clip)])

        self.linemask_clip = self.linemask_clip.std.Limiter()

        return self

    def _errormask(
        self, mask: vs.VideoNode | float = 0.05, maximum_iter: int = 2, inflate_iter: int = 3, expand: int | tuple[int, int | None] = 0
    ) -> vs.VideoNode:
        if self.upscaled:
            raise SyntaxError("RescaleBuilder: Downscaled clip already created. Create errormasks before calling downscale.")
        if isinstance(mask, vs.VideoNode):
            return mask

        err_mask = core.std.Expr([self.funcutil.work_clip, self.rescaled], f"x y - abs {mask} < 0 1 ?")
        err_mask = removegrain(err_mask, 6)
        err_mask = self._process_mask(err_mask, maximum_iter, inflate_iter, expand)

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
        self.errormask_clip = self._errormask(mask, maximum_iter, inflate_iter, expand).std.Limiter()
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
        if not self.errormask_clip:
            self.errormask_clip = core.std.BlankClip(self.funcutil.work_clip, format=get_video_format(err_mask))

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

    def post_double(self, func: GenericVSFunction | list[GenericVSFunction]) -> Self:
        """
        A function to apply any arbitrary function on the doubled clip.

        :param func:    This can be any function or list of functions that take a videonode input
                        and returns a videonode. You are responsible for keeping the format the same.
        """
        if not self.doubled:
            raise SyntaxError("post_double: Doubled clip has not been generated yet. Please call this after double().")

        if not isinstance(func, list):
            func = [func]

        for f in func:
            if not callable(f):
                raise CustomValueError(f"post_double: Function {f.__name__} is not callable!", self.post_double)

            self.doubled = f(self.doubled)

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
        self._apply_masks()
        self.upscaled = self.upscaled.std.CopyFrameProps(wclip)
        return self

    def _apply_masks(self):
        wclip = self.funcutil.work_clip
        if isinstance(self.errormask_clip, vs.VideoNode) and isinstance(self.linemask_clip, vs.VideoNode):
            self.final_mask = core.std.Expr([self.linemask_clip, self.errormask_clip], "x y - 0 max 1 min")
            self.upscaled = wclip.std.MaskedMerge(self.upscaled, self.final_mask)
        elif isinstance(self.errormask_clip, vs.VideoNode):
            self.upscaled = self.upscaled.std.MaskedMerge(wclip, self.errormask_clip)
        elif isinstance(self.linemask_clip, vs.VideoNode):
            self.upscaled = wclip.std.MaskedMerge(self.upscaled, self.linemask_clip)

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
