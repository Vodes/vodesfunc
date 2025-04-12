from vstools import vs, KwargsT

from .base import RescaleBase, descale_rescale
from .scaling_args import ScalingArgs

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
        self.post_crop = sc_args.kwargs(2)
        self.rescale_args = sc_args.kwargs()

        self.descale_func_args = KwargsT() | args

        self.height = args.get("src_height", clip.height)
        self.width = args.get("src_width", clip.width)
        self.base_height = base_height
        self.base_width = base_width

        self.descaled = self.kernel.descale(clip, **args)
        self.rescaled = descale_rescale(self, self.descaled, width=clip.width, height=clip.height, **self.rescale_args)
