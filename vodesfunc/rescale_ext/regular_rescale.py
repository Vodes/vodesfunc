from jetpytools import KwargsT
from vstools import vs
from vskernels import BorderHandling
from vsscale import ScalingArgs

from .base import RescaleBase
from .ignoremask import border_clipping_mask

__all__ = ["RescBuildNonFB"]


class RescBuildNonFB(RescaleBase):
    def _scaling_args(self, **kwargs: KwargsT) -> ScalingArgs:
        if self.sample_grid_model:
            if "sample_grid_model" not in ScalingArgs.from_args.__annotations__:
                raise ValueError("'sample_grid_model' is currently not supported. Please update vsjetpack.")
            kwargs = kwargs | dict(sample_grid_model=self.sample_grid_model)
        return ScalingArgs.from_args(**kwargs)

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
        sc_args = self._scaling_args(
            base_clip=clip, height=height, width=width, base_height=base_height, base_width=base_width, src_top=shift[0], src_left=shift[1], mode=mode
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

            def _descale_step(workclip: vs.VideoNode, order: str) -> vs.VideoNode:
                sc_args_direction = self._scaling_args(
                    base_clip=clip,
                    height=height,
                    width=width,
                    base_height=base_height,
                    base_width=base_width,
                    src_top=shift[0],
                    src_left=shift[1],
                    mode=order,
                )
                ignore_mask = self.ignore_mask(workclip, sc_args_direction, self.kernel, BorderHandling(self.border_handling))  # type:ignore[reportCallableArgument]
                direction = {"width" if order == "w" else "height": out_width if order == "w" else out_height}

                return self.kernel.descale(
                    workclip,
                    **(sc_args_direction.kwargs() | dict(border_handling=self.border_handling, ignore_mask=ignore_mask) | direction),
                )

            self.descaled = clip
            out_width, out_height = sc_args.width, sc_args.height

            for order in mode.lower():
                self.descaled = _descale_step(self.descaled, order)
        else:
            self.descaled = self.kernel.descale(clip, **args)

        self.rescaled = self.kernel.rescale(
            self.descaled, **(self.rescale_args | dict(width=clip.width, height=clip.height, border_handling=self.border_handling))
        )
