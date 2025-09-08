from vstools import vs, KwargsT
from vskernels import BorderHandling
from vsscale import ScalingArgs

from .base import RescaleBase, descale_rescale
from .ignoremask import border_clipping_mask

__all__ = ["RescBuildNonFB"]


class RescBuildNonFB(RescaleBase):
    def _non_fieldbased_descale(
        self,
        clip: vs.VideoNode,
        width: int | float,
        height: int | float,
        base_height: int | None = None,
        base_width: int | None = None,
        shift: tuple[float, float] = (0, 0),
        mode: str = "hw",
    ) -> None:
        sc_args = ScalingArgs.from_args(
            clip, height=height, width=width, base_height=base_height, base_width=base_width, src_top=shift[0], src_left=shift[1], mode=mode
        )

        args = KwargsT(width=sc_args.width, height=sc_args.height, border_handling=self.border_handling) | sc_args.kwargs()

        if isinstance(self.ignore_mask, vs.VideoNode):
            args = args | dict(ignore_mask=self.ignore_mask)

        self.post_crop = sc_args.kwargs(2)
        self.rescale_args = sc_args.kwargs()

        self.descale_func_args = KwargsT() | args

        self.height = args.get("src_height", clip.height)
        self.width = args.get("src_width", clip.width)
        self.base_height = base_height
        self.base_width = base_width

        if self.ignore_mask and not isinstance(self.ignore_mask, vs.VideoNode) and mode.lower() in ("wh", "hw"):
            if not callable(self.ignore_mask):
                self.ignore_mask = border_clipping_mask

            sc_args_w = ScalingArgs.from_args(
                clip, height=height, width=width, base_height=base_height, base_width=base_width, src_top=shift[0], src_left=shift[1], mode="w"
            )
            sc_args_h = ScalingArgs.from_args(
                clip, height=height, width=width, base_height=base_height, base_width=base_width, src_top=shift[0], src_left=shift[1], mode="h"
            )

            ignore_mask_w = self.ignore_mask(clip, sc_args_w, self.kernel, BorderHandling(self.border_handling))
            self.descaled = self.kernel.descale(
                clip, **(sc_args_w.kwargs() | dict(border_handling=self.border_handling, width=sc_args_w.width, ignore_mask=ignore_mask_w))
            )

            ignore_mask_h = self.ignore_mask(self.descaled, sc_args_h, self.kernel, BorderHandling(self.border_handling))
            self.descaled = self.kernel.descale(
                self.descaled, **(sc_args_h.kwargs() | dict(border_handling=self.border_handling, height=sc_args_w.height, ignore_mask=ignore_mask_h))
            )

            self.ignore_masks = tuple(x.resize.Point(clip.width, clip.height) for x in (ignore_mask_w, ignore_mask_h))
        else:
            self.descaled = self.kernel.descale(clip, **args)

        self.rescaled = descale_rescale(
            self.descaled, self.kernel, **(self.rescale_args | dict(width=clip.width, height=clip.height, border_handling=self.border_handling))
        )
