from vstools import vs, core, get_y, depth, iterate, ColorRange, join, get_depth, FieldBased, FieldBasedT
from vskernels import Scaler, ScalerT, Kernel, KernelT, Catrom
from vsmasktools import EdgeDetectT, EdgeDetect
from typing import Callable, Sequence, Union
from math import floor
from dataclasses import dataclass

from .scale import Doubler, NNEDI_Doubler

__all__ = ['DescaleTarget', 'MixedRescale', 'DT']

def get_args(clip: vs.VideoNode, base_height: int, height: float, base_width: float = None):
    base_height = float(base_height)
    src_width = height * clip.width / clip.height
    if not base_width:
        base_width = clip.width
    cropped_width = base_width - 2 * floor((base_width - src_width) / 2)
    cropped_height = base_height - 2 * floor((base_height - height) / 2)
    fractional_args = dict(height = cropped_height, width = cropped_width,
        src_width = src_width, src_height = height, src_left = (cropped_width - src_width) / 2,
        src_top = (cropped_height - height) / 2)
    return fractional_args

class TargetVals():
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
                                Example: line_mask=(KirschTCanny, Bilinear, 50 / 255, 150 / 255)
        :param field_based:     Per-field descaling. Must be a FieldBased object. For example, `field_based=FieldBased.TFF`.
                                This indicates the order the fields get operated in, and whether it needs special attention.
                                Defaults to checking the input clip for the frameprop.
        :param bbmod_masks:     Specify rows to be bbmod'ed for a clip to generate the masks on. Will probably be useful for the new border param in descale.
    """
    height: float
    kernel: KernelT = Catrom
    upscaler: Doubler | ScalerT = NNEDI_Doubler()
    downscaler: ScalerT = Catrom
    base_height: int | None = None
    width: float | None = None
    base_width: int | None = None
    shift: tuple[int, int] = (0, 0)
    do_post_double: Callable[[vs.VideoNode], vs.VideoNode] | None = None
    credit_mask: vs.VideoNode | bool | None = None
    credit_mask_thr: float = 0.04
    credit_mask_bh: bool = False
    line_mask: vs.VideoNode | bool | Sequence[Union[EdgeDetectT, ScalerT, float | None]] | None = None
    field_based: FieldBasedT | None = None
    bbmod_masks: int | list[int] = 0 # Not actually implemented yet lol

    def generate_clips(self, clip: vs.VideoNode) -> 'DescaleTarget':
        """
            Generates descaled and rescaled clips of the given input clip

            :param clip:    Clip to descale and rescale
        """
        self.kernel = Kernel.ensure_obj(self.kernel)
        self.input_clip = clip.std.SetFrameProp('Target', self.index + 1)
        bits, clip = get_depth(clip), get_y(clip)
        self.height = float(self.height)

        self.field_based = FieldBased.from_param(self.field_based) or FieldBased.from_video(clip)

        if not self.width:
            self.width = float(self.height * clip.width / clip.height)

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
            self.descale = self.kernel.descale(clip, self.width, self.height, self.shift)
            self.rescale = self.kernel.scale(self.descale, clip.width, clip.height, self.shift)
            ref_y = self.rescale
        else:
            if self.base_height is None:
                raise ValueError("DescaleTarget: height cannot be fractional if you don't pass a base_height.")
            if not float(self.base_height).is_integer():
                raise ValueError("DescaleTarget: Your base_height has to be an integer.")
            if self.base_height < self.height:
                raise ValueError("DescaleTarget: Your base_height has to be bigger than your height.")
            self.frac_args = get_args(clip, self.base_height, self.height, self.base_width)
            self.descale = self.kernel.descale(clip, **self.frac_args)  \
                .std.CopyFrameProps(clip).std.SetFrameProp('Descale', self.index + 1)
            self.frac_args.pop('width')
            self.frac_args.pop('height')
            self.rescale = self.kernel.scale(self.descale, clip.width, clip.height, **self.frac_args) \
                .std.CopyFrameProps(clip).std.SetFrameProp('Rescale', self.index + 1)
            if self.credit_mask_bh:
                base_height_desc = self.kernel.descale(clip, self.base_height * (clip.width / clip.height), self.base_height)
                ref_y = self.kernel.scale(base_height_desc, clip.width, clip.height)
            else:
                ref_y = self.rescale

        if self.line_mask != False and not isinstance(self.line_mask, Sequence):
            if not isinstance(self.line_mask, vs.VideoNode):
                try:
                    from vsmasktools.edge import KirschTCanny
                    try:
                        # Scaling was changed so I abuse the new param to check if its the newer version
                        self.line_mask = KirschTCanny().edgemask(clip, lthr=80 / 255, hthr=150 / 255, planes=(0, True))
                    except:
                        self.line_mask = KirschTCanny().edgemask(clip, lthr=80 << 8, hthr=150 << 8)
                except:
                    from vsmask.edge import KirschTCanny
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
        if self.descale == None or self.rescale == None:
            self.generate_clips(clip)

        bits, y = get_depth(clip), get_y(clip)

        if isinstance(self.upscaler, Doubler):
            self.doubled = self.upscaler.double(self.descale)
        else:
            self.upscaler = Scaler.ensure_obj(self.upscaler)
            self.doubled = self.upscaler.scale(self.descale, self.descale.width * 2, self.descale.height * 2)

        if self.do_post_double is not None:
            self.doubled = self.do_post_double(self.doubled)

        self.downscaler = Scaler.ensure_obj(self.downscaler)
        if hasattr(self, 'frac_args'):
            # TODO: Figure out how to counteract shift during descaling (if we want to?), maybe additive?
            self.frac_args.update({key: value * 2 for (key, value) in self.frac_args.items()})
            self.upscale = self.downscaler.scale(self.doubled, clip.width, clip.height, **self.frac_args)
            self.upscale = self.upscale.std.CopyFrameProps(self.rescale)
        else:
            self.upscale = self.downscaler.scale(self.doubled, clip.width, clip.height, tuple(-sh for sh in self.shift))

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

            if get_depth(self.line_mask) == 32:
                self.line_mask = self.line_mask.std.Limiter()
            self.upscale = y.std.MaskedMerge(self.upscale, self.line_mask)

        if self.credit_mask != False or self.credit_mask_thr <= 0:
            if get_depth(self.credit_mask) == 32:
                self.credit_mask = self.credit_mask.std.Limiter()
            self.upscale = self.upscale.std.MaskedMerge(y, self.credit_mask)

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

        self.descale = self.kernel.descale(wclip, self.width, self.height)
        self.rescale = self.kernel.scale(self.descale, clip.width, clip.height)

        self.descale = FieldBased.PROGRESSIVE.apply(self.descale)
        self.rescale = FieldBased.PROGRESSIVE.apply(self.rescale)

    @staticmethod
    def crossconv_shift_calc_irregular(clip: vs.VideoNode, native_height: int) -> float:
        """Calculate the shift for an irregular cross-conversion."""
        return 0.25 / (clip.height / native_height)

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

        map_prop_srcs = [
            blank.std.CopyFrameProps(prop_src).akarin.Expr('x.PlaneStatsAverage', vs.GRAYS)
            for prop_src in prop_srcs
        ]

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
                raise ValueError('Requested clip was None')

            base = base.std.BlankClip(keep=True)
            clips = [c or base for c in clips]

            return core.std.FrameEval(
                base, lambda n, f: clips[f[0][0, 0]], _select_clip
            )

        self.upscaled = _selector([t.get_upscaled(t.input_clip) if src.format.color_family == vs.GRAY else t.get_upscaled(t.input_clip, src) for t in targets])
        #self.upscaled = depth(self.upscaled, get_depth(src))
        self.final = self.upscaled

        self.rescaled = _selector([t.rescale for t in targets])
        # These two are not working yet because I need to figure out how to make the base clip up there use varres
        #self.descaled = _selector([t.descale for t in targets])
        #self.doubled = _selector([t._return_doubled() for t in targets])
        self.credit_mask = _selector([t._return_creditmask() for t in targets])
        self.line_mask = _selector([t._return_linemask() for t in targets])

    def get_upscaled(self, *_) -> vs.VideoNode:
        return self.upscaled