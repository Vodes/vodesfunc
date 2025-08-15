from __future__ import annotations

from enum import StrEnum

from vskernels import Kernel, KernelLike, Lanczos
from vstools import ColorRange, CustomValueError, get_lowest_value, get_peak_value, scale_value, vs

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

    @property
    def expr_size(self) -> str:
        """Get the size expr param"""

        if self == DescaleDirection.VERTICAL:
            return "height"

        return "width"


class IgnoreMask(RescaleBase):
    def _clipping_mask(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        direction: DescaleDirection = DescaleDirection.HORIZONTAL,
        kernel: KernelLike = Lanczos,
    ) -> vs.VideoNode:
        """Create a clipping mask to pass to descale as an ignore_mask."""

        kernel = Kernel.ensure_obj(kernel)

        scale_factor = clip.width / width if direction == DescaleDirection.HORIZONTAL else clip.height / height
        threshold = scale_value(get_peak_value(clip, range_in=ColorRange.LIMITED) - 8, 8, 32, scale_offsets=True)
        kernel_radius = kernel.kernel_radius * scale_factor

        return clip.akarin.Expr(
            f"x {threshold} >= "
            f"{direction.expr_size} {direction.value} - {kernel_radius + 1} < "
            f"{direction.value} {kernel_radius} < or and "
            f"{get_peak_value(8, range_in=ColorRange.FULL)} "
            f"{get_lowest_value(8, range_in=ColorRange.FULL)} ?",
            format=vs.GRAY8,
        )
