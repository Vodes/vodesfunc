from dataclasses import dataclass
from math import floor
from typing import Callable, Sequence, Union

from vskernels import Catrom, Kernel, KernelT, ScalerT
from vsmasktools import EdgeDetectT, KirschTCanny
from vstools import FieldBasedT, core, depth, get_y, GenericVSFunction, vs, CustomValueError, join

from .scale import Doubler, NNEDI_Doubler
from .rescale import RescaleBuilder

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

    builder: RescaleBuilder | None = None

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
    :param width:           Width to be descaled to.
    :param base_width:      Needed for fractional descales. (will be calculated if None)
    :param shift:           Shifts to apply during the descaling and reupscaling.
    :param do_post_double:  A function that's called on the doubled clip. Can be used to do sensitive processing on a bigger clip. (e. g. Dehaloing)
    :param do_post_descale: Same thing as do_post_double but instead on the descaled clip.
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

    height: float | int
    kernel: KernelT = Catrom
    upscaler: Doubler | ScalerT = NNEDI_Doubler()
    downscaler: ScalerT = Catrom
    base_height: int | None = None
    width: float | int | None = None
    base_width: int | None = None
    shift: tuple[float, float] = (0, 0)
    do_post_double: Callable[[vs.VideoNode], vs.VideoNode] | None = None
    do_post_descale: Callable[[vs.VideoNode], vs.VideoNode] | None = None
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
        self.border_handling = self.kernel.kwargs.pop("border_handling", self.border_handling)
        self.kernel.kwargs.update({"border_handling": self.border_handling})

        if not self.width:
            raise CustomValueError("Width is a mandatory parameter now.", self.__class__.__name__)

        self.builder = RescaleBuilder(self.input_clip)
        self.builder.descale(self.kernel, self.width, self.height, self.base_height, self.base_width, self.shift, self.field_based)
        if self.do_post_descale:
            self.builder.post_descale(self.do_post_descale)
        self.descale = self.builder.descaled
        self.rescale = self.builder.rescaled

        if self.line_mask is not False and not isinstance(self.line_mask, Sequence) and not isinstance(self.line_mask, Callable):
            if isinstance(self.line_mask, vs.VideoNode):
                self.builder.linemask(self.line_mask)
            else:
                self.builder.linemask(
                    KirschTCanny, inflate_iter=1 if self.do_post_double else 0, lthr=80 / 255, hthr=150 / 255, kernel_window=self._kernel_window
                )
            self.line_mask = self.builder.linemask_clip

        if self.credit_mask is not False or self.credit_mask_thr <= 0:
            if not isinstance(self.credit_mask, vs.VideoNode):
                self.builder.errormask(self.credit_mask_thr, inflate_iter=4 if self.do_post_double else 3)
            else:
                self.builder.errormask(self.credit_mask)
            self.credit_mask = self.builder.errormask_clip

        return self

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

        self.builder.double(self.upscaler)
        if self.do_post_double:
            self.builder.post_double(self.do_post_double)

        self.doubled = self.builder.doubled

        if self.line_mask is not False:
            if isinstance(self.line_mask, Sequence):
                if len(self.line_mask) < 4:
                    raise CustomValueError(
                        "line_mask must contain an Edgemask, Downscaler, lthr and hthr if you passed a list.", self.__class__.__name__
                    )
                self.builder.linemask(
                    self.line_mask[0], self.line_mask[1], lthr=self.line_mask[2], hthr=self.line_mask[3], kernel_window=self._kernel_window
                )
            elif isinstance(self.line_mask, Callable):
                self.builder.linemask(self.line_mask(self.doubled))

            self.line_mask = self.builder.linemask_clip

        self.builder.downscale(self.downscaler)
        final = self.builder.final()[1]
        if chroma:
            final = join(get_y(final), chroma, vs.YUV)
        return final

    def _return_creditmask(self) -> vs.VideoNode:
        return self.credit_mask if isinstance(self.credit_mask, vs.VideoNode) else core.std.BlankClip(self.input_clip)

    def _return_linemask(self) -> vs.VideoNode:
        return self.line_mask if isinstance(self.line_mask, vs.VideoNode) else core.std.BlankClip(self.input_clip)

    def _return_doubled(self) -> vs.VideoNode:
        return core.std.BlankClip(self.input_clip, width=self.input_clip * 2) if not self.doubled else self.doubled

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
