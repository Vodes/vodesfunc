from vstools import vs, core, depth, vs_object
from typing import TYPE_CHECKING, MutableMapping
from enum import IntEnum
from .base import RescaleBase

__all__ = ["DiffMode", "MixedRB"]


class DiffMode(IntEnum):
    """Mode used to calculate the difference between rescale and input clip."""

    MAE = 1
    """Mean Absolute Error"""

    MSE = 2
    """Mean Squared Error"""

    GET_NATIVE = 3
    """Weird headcraft from getnative"""


class RescBuildMixed(RescaleBase):
    diffmode: DiffMode = DiffMode.GET_NATIVE
    crop_diff: bool = True
    index = 0

    def get_diff(self) -> vs.VideoNode:
        clip = depth(self.funcutil.work_clip, 32)
        match self.diffmode:
            case DiffMode.MAE:
                metric = "x y - abs"
            case DiffMode.MSE:
                metric = "x y - 2 pow"
            case _:
                metric = "x y - abs dup 0.015 > swap 0 ?"

        diff = core.std.Expr([depth(self.rescaled, 32), clip], metric)
        if self.crop_diff:
            diff = diff.std.Crop(5, 5, 5, 5)
        return diff.std.PlaneStats()

    def _add_index_to_clips(self):
        self.descaled = self.descaled.std.SetFrameProp("RB_Target", self.index)
        self.rescaled = self.rescaled.std.SetFrameProp("RB_Target", self.index)
        self.upscaled = self.upscaled.std.SetFrameProp("RB_Target", self.index)


class MixedRB(vs_object):
    """
    Implementation of MixedRescale for RescaleBuilder(s)

    This is just a stop-gap solution until we (mostly Setsu) can cook up something better.

    Example Usage:

    ```py
    upscaler = Waifu2x("trt", 1, fp16=True)

    builders = [
        RescaleBuilder(src).descale(Bilinear(border_handling=1), 1280, 720),
        RescaleBuilder(src).descale(BicubicSharp, 1280, 720),
    ]

    # This will be run on all of the above
    builders = [
        b.double(upscaler)
        .linemask(KirschTCanny, Bilinear, lthr=50 / 255, hthr=150 / 255, inflate_iter=2)
        .errormask(expand=2)
        .downscale(Hermite(linear=True))
        for b in builders
    ]

    mixed = MixedRB(*builders)
    rescaled = mixed.get_upscaled()
    ```
    """

    def __init__(self, *targets: RescBuildMixed, diffmode: DiffMode = DiffMode.GET_NATIVE, crop_diff: bool = True) -> None:
        """
        A naive per-frame diff approach of trying to get the best descale.
        Credits to Setsu for most of this class.
        """
        y = targets[0].funcutil.work_clip

        for i, d in enumerate(targets):
            d.index = i + 1
            d.diffmode = diffmode
            d.crop_diff = crop_diff
            d._add_index_to_clips()

        prop_srcs = [d.get_diff() for d in targets]
        targets_idx = tuple(range(len(targets)))

        blank = core.std.BlankClip(None, 1, 1, vs.GRAY8, y.num_frames, keep=True)

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

        self.upscaled = _selector([t.final()[1] for t in targets])
        self.final = self.upscaled

        self.rescaled = _selector([t.rescaled for t in targets])
        self.credit_mask = _selector([t._return_creditmask() for t in targets])
        self.line_mask = _selector([t._return_linemask() for t in targets])

    def get_upscaled(self, *_) -> vs.VideoNode:
        return self.upscaled

    def __vs_del__(self, core_id: int) -> None:
        if not TYPE_CHECKING:
            for v in self.__dict__.values():
                if not isinstance(v, MutableMapping):
                    continue

                for k2, v2 in v.items():
                    if isinstance(v2, vs.VideoNode):
                        v[k2] = None
