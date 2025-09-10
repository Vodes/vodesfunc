from vstools import vs, KwargsT, FieldBased, padder

from .base import RescaleBase

__all__ = ["RescBuildFB"]


class RescBuildFB(RescaleBase):
    def _fieldbased_descale(self, clip: vs.VideoNode, shift: tuple[float, float] = (0, 0), **descale_args: KwargsT) -> None:
        clip = self.field_based.apply(clip)
        if shift != (0, 0):
            descale_args.update({"shift": shift})
        self.descaled = self.kernel.descale(clip, **descale_args)
        self.descale_func_args = KwargsT()
        self.descale_func_args.update(descale_args)
        self.post_crop = KwargsT()

        if not self.border_handling:
            self.rescaled = self.kernel.scale(self.descaled, width=clip.width, height=clip.height)
        else:
            # Reimplementation of border_handling because regular scale operations aren't aware of it yet.
            # Can't use descale scale because we need vskernels to handle the field shifting.
            wclip = self.descaled
            left = right = 10 if self.width != clip.width else 0
            top = bottom = 10 if self.height != clip.height else 0

            match int(self.border_handling):
                case 1:
                    wclip = padder.COLOR(wclip, left, right, top, bottom, color=0)
                case 2:
                    wclip = padder.REPEAT(wclip, left, right, top, bottom)
                case _:
                    pass

            shift_top = descale_args.pop("src_top", False) or shift[0]
            shift_left = descale_args.pop("src_left", False) or shift[1]

            shift = [
                shift_top + (self.height != wclip.height and self.border_handling) * 10,
                shift_left + (self.width != wclip.width and self.border_handling) * 10,
            ]

            src_width = descale_args.pop("src_width", wclip.width)
            src_height = descale_args.pop("src_height", wclip.height)
            self.rescaled = self.kernel.scale(
                wclip,
                clip.width,
                clip.height,
                shift=shift,
                src_width=src_width - (wclip.width - self.width),
                src_height=src_height - (wclip.height - self.height),
            )
        self.descaled = FieldBased.PROGRESSIVE.apply(self.descaled)
        self.rescaled = FieldBased.PROGRESSIVE.apply(self.rescaled)
