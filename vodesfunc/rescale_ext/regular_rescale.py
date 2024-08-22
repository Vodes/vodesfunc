from vstools import vs, KwargsT
from vsscale import fdescale_args

from .base import RescaleBase, descale_rescale

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
        sanitized_shift = (shift[0] if shift[0] else None, shift[1] if shift[1] else None)
        args, self.post_crop = fdescale_args(clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode)
        _, self.rescale_args = fdescale_args(clip, height, base_height, base_width, sanitized_shift[0], sanitized_shift[1], width, mode, up_rate=1)
        args.update({"border_handling": self.border_handling})

        self.descale_func_args = KwargsT()
        self.descale_func_args.update(args)

        self.height = args.get("src_height", clip.height)
        self.width = args.get("src_width", clip.width)
        self.base_height = base_height
        self.base_width = base_width

        self.descaled = self.kernel.descale(clip, **args)
        self.rescaled = descale_rescale(self, self.descaled, width=clip.width, height=clip.height, **self.rescale_args)
