from vstools import vs, KwargsT
from vsscale import ScalingArgs

from .base import RescaleBase, descale_rescale
from .ignoremask import IgnoreMask, DescaleDirection

__all__ = ["RescBuildNonFB"]


class RescBuildNonFB(RescaleBase, IgnoreMask):
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

        if self.ignore_mask and mode in ("wh", "hw"):
            self.descaled = self._descale_with_ignore_mask(clip, args, mode)
        else:
            self.descaled = self.kernel.descale(clip, **args)

        self.rescaled = descale_rescale(self, self.descaled, **(self.rescale_args | dict(width=clip.width, height=clip.height)))

    def _descale_with_ignore_mask(self, clip: vs.VideoNode, args: KwargsT, mode: str) -> vs.VideoNode:
        """Helper function to handle descaling with ignore_mask by splitting into two operations."""

        self.ignore_masks = []

        def _remove_direction_params(clip: vs.VideoNode, args_copy: KwargsT, direction: str) -> KwargsT:
            """Remove parameters for a specific direction from args."""

            if direction == "h":
                direction = DescaleDirection.HORIZONTAL
                args_copy["height"] = clip.height
                args_copy["src_height"] = clip.height
            else:
                direction = DescaleDirection.VERTICAL
                args_copy["width"] = clip.width
                args_copy["src_width"] = clip.width

            args_copy["ignore_mask"] = self._clipping_mask(clip, args_copy["width"], args_copy["height"], direction=direction)

            return args_copy

        args_first = _remove_direction_params(clip, args.copy(), mode[0])
        first_descaled = self.kernel.descale(clip, **args_first)

        args_second = _remove_direction_params(first_descaled, args.copy(), mode[1])
        second_descaled = self.kernel.descale(first_descaled, **args_second)

        return second_descaled
