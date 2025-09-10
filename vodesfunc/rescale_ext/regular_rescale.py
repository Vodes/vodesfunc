from vstools import vs, KwargsT, limiter
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

            def _descale(clip: vs.VideoNode, sc_args: ScalingArgs, order: str) -> vs.VideoNode:
                ignore_mask = self.ignore_mask(clip, sc_args, self.kernel, BorderHandling(self.border_handling))  # type:ignore[reportCallableArgument]
                direction = {"width" if order == "w" else "height": out_width if order == "w" else out_height}

                return self.kernel.descale(
                    clip,
                    **(sc_args.kwargs() | dict(border_handling=self.border_handling, ignore_mask=ignore_mask) | direction),
                )

            self.descaled = clip
            out_width, out_height = sc_args.width, sc_args.height

            for order in mode.lower():
                self.descaled = _descale(self.descaled, sc_args, order)
        else:
            self.descaled = self.kernel.descale(clip, **args)

        self.rescaled = limiter(
            descale_rescale(
                self.descaled, self.kernel, **(self.rescale_args | dict(width=clip.width, height=clip.height, border_handling=self.border_handling))
            )
        )
