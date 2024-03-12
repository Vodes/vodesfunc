from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Callable, Sequence, Union

from vskernels import Catrom, Kernel, KernelT, Scaler, ScalerT
from vsmasktools import EdgeDetect, EdgeDetectT, KirschTCanny
from vstools import (
    ColorRange,
    FieldBased,
    FieldBasedT,
    core,
    depth,
    get_depth,
    get_h,
    get_lowest_value,
    get_peak_value,
    get_w,
    get_y,
    iterate,
    join,
    padder,
    GenericVSFunction,
    vs,
)

from .scale import Doubler, NNEDI_Doubler

__all__ = ["DescaleTarget", "MixedRescale", "DT"]


def get_args(clip: vs.VideoNode, base_height: int, height: float, width: float, base_width: float = None, shift: tuple[float, float] = (0.0, 0.0)):
    base_height = float(base_height)
    if not base_width:
        base_width = clip.width
    cropped_width = base_width - 2 * floor((base_width - width) / 2)
    cropped_height = base_height - 2 * floor((base_height - height) / 2)
    fractional_args = dict(
        height=cropped_height,
        width=cropped_width,
        src_width=width,
        src_height=height,
        src_left=(cropped_width - width) / 2 + shift[1],
        src_top=(cropped_height - height) / 2 + shift[0],
    )
    return fractional_args


class TargetVals:
    input_clip: vs.VideoNode | None = None
    descale: vs.VideoNode | None = None
    rescale: vs.VideoNode | None = None
    doubled: vs.VideoNode | None = None
    upscale: vs.VideoNode | None = None

    index: int = 0


@dataclass
class DescaleTarget(TargetVals):
    """
    Basically an entirely self contained rescaling utility class that can do pretty much everything.

    :param height:          Height to be descaled to.
    :param kernel:          The kernel used for descaling.
    :param upscaler:        Either a vodesfunc doubler or any scaler from vsscale used to upscale/double the descaled clip.
    :param downscaler:      Any kernel or scaler used for downscaling the upscaled/doubled clip back to input res.
    :param base_height:     Needed for fractional descales.
    :param width:           Width to be descaled to. (will be calculated if None)
    :param base_width:      Needed for fractional descales. (will be calculated if None)
    :param shift:           Shifts to apply during the descaling and reupscaling.
    :param do_post_double:  A function that's called on the doubled clip. Can be used to do sensitive processing on a bigger clip. (e. g. Dehaloing)
    :param credit_mask:     Can be used to pass a mask that'll be used or False to disable error masking.
    :param credit_mask_thr: The error threshold of the automatically generated credit mask.
    :param credit_mask_bh:  Generates an error mask based on a descale using the base_height. For some reason had better results with this on some shows.
    :param line_mask:       Can be used to pass a mask that'll be used or False to disable line masking.
                            You can also pass a list containing edgemask function, scaler and thresholds to generate the mask on the doubled clip for potential better results.
                            If None is passed to the first threshold then the mask won't be binarized. It will also run a Maximum and Inflate call on the mask.
                            Example: `line_mask=(KirschTCanny, Bilinear, 50 / 255, 150 / 255)`
                            You can also pass a function to create a mask on the doubled clip.
                            Example: `line_mask=lambda clip: create_linemask(clip)`
    :param field_based:     Per-field descaling. Must be a FieldBased object. For example, `field_based=FieldBased.TFF`.
                            This indicates the order the fields get operated in, and whether it needs special attention.
                            Defaults to checking the input clip for the frameprop.
    :param border_handling: Adjust the way the clip is padded internally during the scaling process. Accepted values are:
                                0: Assume the image was resized with mirror padding.
                                1: Assume the image was resized with zero padding.
                                2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far.
                            Defaults to 0.
    :param border_radius:   Radius for the border mask. Only used when border_handling is set to 1 or 2.
                            Defaults to kernel radius if possible, else 2.
    """

    height: float
    kernel: KernelT = Catrom
    upscaler: Doubler | ScalerT = NNEDI_Doubler()
    downscaler: ScalerT = Catrom
    base_height: int | None = None
    width: float | None = None
    base_width: int | None = None
    shift: tuple[float, float] = (0, 0)
    do_post_double: Callable[[vs.VideoNode], vs.VideoNode] | None = None
    credit_mask: vs.VideoNode | bool | None = None
    credit_mask_thr: float = 0.04
    credit_mask_bh: bool = False
    line_mask: vs.VideoNode | bool | Sequence[Union[EdgeDetectT, ScalerT, float | None]] | GenericVSFunction | None = None
    field_based: FieldBasedT | None = None
    border_handling: int = 0
    border_radius: int | None = None

    def generate_clips(self, clip: vs.VideoNode) -> "DescaleTarget":
        """
        Generates descaled and rescaled clips of the given input clip

        :param clip:    Clip to descale and rescale
        """
        self.kernel = Kernel.ensure_obj(self.kernel)
        self.input_clip = clip.std.SetFrameProp("Target", self.index + 1)
        bits, clip = get_depth(clip), get_y(clip)
        self.height = float(self.height)

        self.field_based = FieldBased.from_param(self.field_based) or FieldBased.from_video(clip)

        self.border_handling = self.kernel.kwargs.pop("border_handling", self.border_handling)

        if not self.width:
            self.width = float((self.height * clip.width / clip.height) if not self.height.is_integer() else get_w(self.height, clip))

        if self.field_based.is_inter:
            if not self.height.is_integer():
                raise ValueError("DescaleTarget: `height` must be an integer when descaling an interlaced clip, not float.")
            if not self.width.is_integer():
                raise ValueError("DescaleTarget: `width` must be an integer when descaling an interlaced clip, not float.")

            self._descale_fields(clip)

            ref_y = self.rescale
            clip = FieldBased.PROGRESSIVE.apply(clip)
            self.line_mask = self.line_mask or False
        elif self.height.is_integer():
            self.descale = self.kernel.descale(clip, self.width, self.height, self.shift, border_handling=self.border_handling)
            self.rescale = self._perform_rescale(self.descale)
            ref_y = self.rescale
        else:
            if self.base_height is None:
                raise ValueError("DescaleTarget: height cannot be fractional if you don't pass a base_height.")
            if not float(self.base_height).is_integer():
                raise ValueError("DescaleTarget: Your base_height has to be an integer.")
            if self.base_height < self.height:
                raise ValueError("DescaleTarget: Your base_height has to be bigger than your height.")
            self.frac_args = get_args(clip, self.base_height, self.height, self.width, self.base_width, self.shift)
            self.descale = (
                self.kernel.descale(clip, **self.frac_args, border_handling=self.border_handling)
                .std.CopyFrameProps(clip)
                .std.SetFrameProp("Descale", self.index + 1)
            )
            self.frac_args.pop("width")
            self.frac_args.pop("height")
            self.rescale = self._perform_rescale(self.descale, **self.frac_args)
            self.rescale = self.rescale.std.CopyFrameProps(clip).std.SetFrameProp("Rescale", self.index + 1)
            if self.credit_mask_bh:
                base_height_desc = self.kernel.descale(
                    clip, self.base_height * (clip.width / clip.height), self.base_height, border_handling=self.border_handling
                )
                ref_y = self.kernel.scale(base_height_desc, clip.width, clip.height)
            else:
                ref_y = self.rescale

        if self.line_mask != False and not isinstance(self.line_mask, Sequence) and not isinstance(self.line_mask, Callable):
            try:
                # Scaling was changed so I abuse the new param to check if its the newer version
                self.line_mask = KirschTCanny().edgemask(clip, lthr=80 / 255, hthr=150 / 255, planes=(0, True))
            except:
                self.line_mask = KirschTCanny().edgemask(clip, lthr=80 << 8, hthr=150 << 8)

            if self.do_post_double is not None:
                self.line_mask = self.line_mask.std.Inflate()

            self.line_mask = depth(self.line_mask, bits)

        if self.credit_mask != False or self.credit_mask_thr <= 0:
            if not isinstance(self.credit_mask, vs.VideoNode):
                self.credit_mask = core.std.Expr([depth(clip, 32), depth(ref_y, 32)], f"x y - abs {self.credit_mask_thr} < 0 1 ?")
                self.credit_mask = depth(self.credit_mask, 16, range_out=ColorRange.FULL, range_in=ColorRange.FULL)
                self.credit_mask = self.credit_mask.rgvs.RemoveGrain(mode=6)
                self.credit_mask = iterate(self.credit_mask, core.std.Maximum, 2)
                self.credit_mask = iterate(self.credit_mask, core.std.Inflate, 2 if self.do_post_double is None else 4)

            self.credit_mask = depth(self.credit_mask, bits)

        return self

    def _perform_rescale(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        match int(self.border_handling):
            case 1:
                clip = clip.std.AddBorders(
                    *((0, 0) if self.width == self.input_clip.width else (10, 10)),
                    *((0, 0) if self.height == self.input_clip.height else (10, 10)),
                    get_lowest_value(clip, False, ColorRange.from_video(clip)),
                )
            case 2:
                clip = padder(
                    clip,
                    *((0, 0) if self.width == self.input_clip.width else (10, 10)),
                    *((0, 0) if self.height == self.input_clip.height else (10, 10)),
                    reflect=False,
                )
            case _:
                pass

        shift_top = kwargs.pop("src_top", False) or self.shift[0]
        shift_left = kwargs.pop("src_left", False) or self.shift[1]

        shift = [
            shift_top + (self.height != self.input_clip.height and self.border_handling) * 10,
            shift_left + (self.width != self.input_clip.width and self.border_handling) * 10,
        ]

        src_width = kwargs.pop("src_width", clip.width)
        src_height = kwargs.pop("src_height", clip.height)

        return self.kernel.scale(
            clip,
            self.input_clip.width,
            self.input_clip.height,
            shift,
            src_width=src_width - ((clip.width - self.width) if float(self.width).is_integer() else 0),
            src_height=src_height - ((clip.height - self.height) if float(self.width).is_integer() else 0),
            **kwargs,
        )

    def get_diff(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Returns a clip used for diff measuring ala getnative

        :param clip:    Clip to compare the rescaled clip to
        :return:        Diff clip
        """
        clip = depth(get_y(clip), 32)
        diff = core.std.Expr([depth(self.rescale, 32), clip], ["x y - abs dup 0.015 > swap 0 ?"])
        return diff.std.Crop(5, 5, 5, 5).std.PlaneStats()

    def get_upscaled(self, clip: vs.VideoNode, chroma: vs.VideoNode | None = None) -> vs.VideoNode:
        """
        Generates and returns the fully upscaled & masked & what not clip
        """
        if self.descale is None or self.rescale is None:
            self.generate_clips(clip)

        bits, y = get_depth(clip), get_y(clip)

        if isinstance(self.upscaler, Doubler):
            self.doubled = self.upscaler.double(self.descale)
        else:
            self.upscaler = Scaler.ensure_obj(self.upscaler)

            self.doubled = self.upscaler.scale(
                self.descale,
                self.descale.width * ((self.width != self.input_clip.width) + 1),
                self.descale.height * ((self.height != self.input_clip.height) + 1),
            )

        if self.do_post_double is not None:
            self.doubled = self.do_post_double(self.doubled)

        self.downscaler = Scaler.ensure_obj(self.downscaler)
        if hasattr(self, "frac_args"):
            # TODO: Figure out how to counteract shift during descaling (if we want to?), maybe additive?
            self.frac_args.update({key: value * 2 for (key, value) in self.frac_args.items()})
            self.upscale = self.downscaler.scale(self.doubled, clip.width, clip.height, **self.frac_args)
            self.upscale = self.upscale.std.CopyFrameProps(self.rescale)
        else:
            shift = (self.shift[0] * 2, self.shift[1] * 2)
            self.upscale = self.downscaler.scale(self.doubled, clip.width, clip.height, shift)

        self.upscale = depth(self.upscale, bits)
        self.rescale = depth(self.rescale, bits)

        if self.line_mask != False:
            if isinstance(self.line_mask, Sequence):
                if len(self.line_mask) < 4:
                    raise ValueError("DescaleTarget: line_mask must contain an Edgemask, Downscaler, lthr and hthr if you passed a list.")
                mask_fun = EdgeDetect.ensure_obj(self.line_mask[0])
                if self.line_mask[2] is None:
                    mask = mask_fun.edgemask(self.doubled, planes=0).std.Maximum().std.Inflate()
                else:
                    mask = mask_fun.edgemask(self.doubled, self.line_mask[2], self.line_mask[3], planes=0)
                self.line_mask = Scaler.ensure_obj(self.line_mask[1]).scale(mask, clip.width, clip.height)
            elif isinstance(self.line_mask, Callable):
                self.line_mask = self.line_mask(self.doubled)

        if self.border_handling:
            self._add_border_mask()

        if isinstance(self.credit_mask, vs.VideoNode) and isinstance(self.line_mask, vs.VideoNode):
            self.final_mask = core.std.Expr([self.line_mask.std.Limiter(), self.credit_mask], "x y -")
            self.upscale = y.std.MaskedMerge(self.upscale, self.final_mask.std.Limiter())
        elif isinstance(self.credit_mask, vs.VideoNode):
            self.upscale = self.upscale.std.MaskedMerge(y, self.credit_mask.std.Limiter())
        elif isinstance(self.line_mask, vs.VideoNode):
            self.upscale = y.std.MaskedMerge(self.upscale, self.line_mask.std.Limiter())

        self.upscale = depth(self.upscale, bits)
        self.upscale = self.upscale if clip.format.color_family == vs.GRAY else join(self.upscale, clip)
        return self.upscale if not chroma else join(depth(self.upscale, get_depth(chroma)), depth(chroma, get_depth(chroma)))

    def _return_creditmask(self) -> vs.VideoNode:
        return core.std.BlankClip(self.input_clip) if self.credit_mask == False else self.credit_mask

    def _return_linemask(self) -> vs.VideoNode:
        return core.std.BlankClip(self.input_clip) if self.line_mask == False else self.line_mask

    def _return_doubled(self) -> vs.VideoNode:
        return core.std.BlankClip(self.input_clip, width=self.input_clip * 2) if not self.doubled else self.doubled

    def _descale_fields(self, clip: vs.VideoNode) -> None:
        wclip = self.field_based.apply(clip)

        self.descale = self.kernel.descale(wclip, self.width, self.height, border_handling=self.border_handling)
        self.rescale = self.kernel.scale(self.descale, clip.width, clip.height)

        self.descale = FieldBased.PROGRESSIVE.apply(self.descale)
        self.rescale = FieldBased.PROGRESSIVE.apply(self.rescale)

    @staticmethod
    def crossconv_shift_calc_irregular(clip: vs.VideoNode, native_height: int) -> float:
        """Calculate the shift for an irregular cross-conversion."""
        return 0.25 / (clip.height / native_height)

    def _add_border_mask(self) -> None:
        """Add the borders to the line mask for border_handling."""
        if self.border_radius == 0:
            return

        if self.line_mask:
            self.line_mask = self._crop_mask_bord(self.line_mask, get_peak_value(self.input_clip))

        if self.credit_mask:
            self.credit_mask = self._crop_mask_bord(self.credit_mask)

    def _crop_mask_bord(self, mask: vs.VideoNode, color: float = 0.0) -> vs.VideoNode:
        if not hasattr(self, "_bord_crop_args"):
            self._bord_crop_args = self._get_border_crop()

        return mask.std.Crop(*self._bord_crop_args).std.AddBorders(*self._bord_crop_args, [color])

    # fmt: off
    def _get_border_crop(self) -> tuple:
        """Get the crops for the border handling masking."""
        if self.height == self.input_clip.height:
            vertical_crop = (0, 0)
        else:
            base_height = self.base_height or get_h(self.base_width, self.descale) if self.base_width else self.height
            src_top = self.frac_args.get("src_top", 0) if hasattr(self, "frac_args") else self.shift[0]

            top = max(ceil(
                (-(self.height - 1) / 2 + self._kernel_window - src_top - 1)
                * self.input_clip.height / self.height + (self.input_clip.height - 1) / 2
            ), 0)

            bottom = max(ceil(
                (-(self.height - 1) / 2 + self._kernel_window - (base_height - self.height - src_top) - 1)
                * self.input_clip.height / self.height + (self.input_clip.height - 1) / 2
            ), 0)

            vertical_crop = (top, bottom)

        if self.width == self.input_clip.width:
            horizontal_crop = (0, 0)
        else:
            base_width = self.base_width or get_w(self.base_height, self.descale) if self.base_height else self.width
            src_left = self.frac_args.get("src_left", 0) if hasattr(self, "frac_args") else self.shift[1]

            left = max(ceil(
                (-(self.width - 1) / 2 + self._kernel_window - src_left - 1)
                * self.input_clip.width / self.width + (self.input_clip.width - 1) / 2
            ), 0)

            right = max(ceil(
                (-(self.width - 1) / 2 + self._kernel_window - (base_width - self.width - src_left) - 1)
                * self.input_clip.width / self.width + (self.input_clip.width - 1) / 2
            ), 0)

            horizontal_crop = (left, right)

        return horizontal_crop + vertical_crop
    # fmt: on

    @property
    def _kernel_window(self) -> int:
        if (bord_rad := self.border_radius) is None:
            try:
                bord_rad = self.kernel.kernel_radius
            except (AttributeError, NotImplementedError):
                bord_rad = 2

        return bord_rad


DT = DescaleTarget


class MixedRescale:
    def __init__(self, src: vs.VideoNode, *targets: DescaleTarget) -> None:
        """
        A naive per-frame diff approach of trying to get the best descale.
        Credits to Setsu for most of this class.


        Example usage:
        ```
        t1 = DT(847.1, Bilinear(), base_height=848)
        t2 = DescaleTarget(843.75, Bilinear(), base_height=846)

        rescaled = MixedRescale(clip, t1, t2)

        out(rescaled.final)
        out(rescaled.line_mask)
        ```
        """
        y = get_y(src)

        for i, d in enumerate(targets):
            d.index = i
            d.generate_clips(y)

        prop_srcs = [d.get_diff(y) for d in targets]
        targets_idx = tuple(range(len(targets)))

        blank = core.std.BlankClip(None, 1, 1, vs.GRAY8, src.num_frames, keep=True)

        map_prop_srcs = [blank.std.CopyFrameProps(prop_src).akarin.Expr("x.PlaneStatsAverage", vs.GRAYS) for prop_src in prop_srcs]

        base_frame, idx_frames = blank.get_frame(0), []

        for i in targets_idx:
            fcurr = base_frame.copy()

            fcurr[0][0, 0] = i

            idx_frames.append((i, fcurr))

        def _select(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            return min(idx_frames, key=lambda i: f[i[0]][0][0, 0])[1]

        _select_clip = blank.std.ModifyFrame(map_prop_srcs, _select)

        def _selector(clips: list[vs.VideoNode | None]) -> vs.VideoNode:
            base = next(filter(None, clips), None)

            if base is None:
                raise ValueError("Requested clip was None")

            base = base.std.BlankClip(keep=True)
            clips = [c or base for c in clips]

            return core.std.FrameEval(base, lambda n, f: clips[f[0][0, 0]], _select_clip)

        self.upscaled = _selector(
            [t.get_upscaled(t.input_clip) if src.format.color_family == vs.GRAY else t.get_upscaled(t.input_clip, src) for t in targets]
        )
        # self.upscaled = depth(self.upscaled, get_depth(src))
        self.final = self.upscaled

        self.rescaled = _selector([t.rescale for t in targets])
        # These two are not working yet because I need to figure out how to make the base clip up there use varres
        # self.descaled = _selector([t.descale for t in targets])
        # self.doubled = _selector([t._return_doubled() for t in targets])
        self.credit_mask = _selector([t._return_creditmask() for t in targets])
        self.line_mask = _selector([t._return_linemask() for t in targets])

    def get_upscaled(self, *_) -> vs.VideoNode:
        return self.upscaled
