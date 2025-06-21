from vskernels import Lanczos
from vstools import (
    inject_self,
    vs,
)


__all__: list[str] = ["Lanczos_PreSS"]


class Lanczos_PreSS(Lanczos):
    """
    Convenience class to pass to a dehalo function.
    This serves the same purpose as NNEDI to double and reverse using point.
    Except it is a quite a bit faster and (if using opencl) takes a lot of load off the GPU.
    """

    @inject_self.init_kwargs.clean
    def scale(self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs) -> vs.VideoNode:
        if width != clip.width * 2 or height != clip.height * 2:
            raise ValueError("Lanczos_PreSS: You're probably not using this correctly.")
        return Lanczos.scale(clip, width, height, (-0.25, -0.25))


def mod_padding(clip: vs.VideoNode, mod: int = 4, min: int = 4):
    from math import floor

    width = clip.width + min * 2
    height = clip.height + min * 2
    ph = mod - ((width - 1) % mod + 1)
    pv = mod - ((height - 1) % mod + 1)

    left = floor(ph / 2)
    right = ph - left
    top = floor(pv / 2)
    bottom = pv - top
    return (left + min, right + min, top + min, bottom + min)
