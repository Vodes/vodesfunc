from __future__ import annotations

from enum import StrEnum
from math import ceil

from vskernels import Kernel, KernelLike, Lanczos
from vstools import CustomValueError, scale_value, vs

from .base import RescaleBase

__all__ = ["DescaleDirection", "IgnoreMask"]


class DescaleDirection(StrEnum):
    """The direction the descale is performed in."""

    HORIZONTAL = "X"
    """Descale is performed horizontally (ex. 1920x1080 => 1280x1080)"""

    VERTICAL = "Y"
    """Descale is performed vertically (ex. 1920x1080 => 1920x720)"""

    @classmethod
    def from_ref(cls, src: vs.VideoNode | tuple[int, int], width: int, height: int) -> DescaleDirection:
        """Get the direction from a reference."""

        if isinstance(src, vs.VideoNode):
            src = (src.width, src.height)

        assert isinstance(src, tuple) and all(int(x) for x in src), "You must pass a tuple of ints or a VideoNode!"

        if src == (width, height):
            raise CustomValueError("Reference dimensions are the same as the output dimensions!", cls.from_ref)

        if src[0] != width and src[1] == height:
            return cls.HORIZONTAL
        elif src[0] == width and src[1] != height:
            return cls.VERTICAL

        raise CustomValueError("Cannot determine descale direction from given dimensions!", cls.from_ref)


class IgnoreMask(RescaleBase):
    ignore_mask: vs.VideoNode | None = None
    """User-passed ignore mask"""

    ignore_masks: list[vs.VideoNode] | None = None
    """Generated ignore masks."""

    def _clipping_mask(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        dark_thr: float = 0,
        bright_thr: float = 235,
        direction: DescaleDirection = DescaleDirection.HORIZONTAL,
        kernel: KernelLike = Lanczos,
    ) -> vs.VideoNode:
        """Create a clipping mask to pass to descale as an ignore_mask."""

        kernel = Kernel.ensure_obj(kernel)

        scale_factor = clip.width / width if direction == DescaleDirection.HORIZONTAL else clip.height / height
        kernel_radius = ceil(kernel.kernel_radius * scale_factor) - 1

        mask = clip.akarin.Expr(
            f"x {scale_value(dark_thr, 8, clip)} < x {scale_value(bright_thr, 8, clip)} > or "
            f"{'height' if direction == DescaleDirection.VERTICAL else 'width'} "
            f"{'Y' if direction == DescaleDirection.VERTICAL else 'X'} - {kernel_radius} < "
            f"{'Y' if direction == DescaleDirection.VERTICAL else 'X'} {kernel_radius - 1} < or "
            "and 255 0 ?",
            format=vs.GRAY8,
        )

        if self.ignore_masks is None:
            self.ignore_masks = []

        self.ignore_masks += [mask]

        return mask
