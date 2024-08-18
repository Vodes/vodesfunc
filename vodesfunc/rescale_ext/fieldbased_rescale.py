from vstools import vs, KwargsT, FieldBased

from .base import RescaleBase, descale_rescale

__all__ = ["RescBuildFB"]


class RescBuildFB(RescaleBase):
    def _fieldbased_descale(self, clip: vs.VideoNode, **descale_args: KwargsT) -> None:
        clip = self.field_based.apply(clip)

        self.descaled = self.kernel.descale(clip, **descale_args)
        self.rescaled = descale_rescale(self, self.descaled, width=clip.width, height=clip.height)

        self.descaled = FieldBased.PROGRESSIVE.apply(self.descaled)
        self.rescaled = FieldBased.PROGRESSIVE.apply(self.rescaled)
