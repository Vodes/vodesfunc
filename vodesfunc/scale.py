from typing import Any, Callable
from vskernels import Catrom, Kernel, Scaler, ScalerT, Lanczos, Point
from vstools import inject_self, vs, core, depth, get_depth, get_y, Matrix, KwargsT, get_nvidia_version, Transfer
from vsrgtools.sharp import unsharp_masked
from .types import PathLike
from abc import ABC, abstractmethod


__all__: list[str] = [
    "NNEDI_Doubler",
    "Clamped_Doubler",
    "Shader_Doubler",
    "Waifu2x_Doubler",
    "LinearScaler",
    "Lanczos_PreSS",
    "SigmoidScaler",
]


class LinearScaler(Scaler):
    def __init__(self, scaler: ScalerT, **kwargs: KwargsT) -> None:
        """
        Simple scaler class to do your scaling business in linear light.

        :params scaler:     Any vsscale scaler class/object
        """
        self.scaler = Scaler.ensure_obj(scaler)
        self.kwargs = kwargs

    def scale(self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs) -> vs.VideoNode:
        if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
            print(clip.format.subsampling_h, clip.format.subsampling_w)
            raise ValueError("LinearScaler: Using linear scaling on a subsampled clip results in very poor output. Please don't do it.")

        trans_in = Transfer.from_video(clip)
        clip = clip.resize.Point(transfer_in=trans_in, transfer=Transfer.LINEAR)
        args = KwargsT(clip=clip, width=width, height=height, shift=shift)
        args.update(**kwargs)
        args.update(**self.kwargs)
        scaled = self.scaler.scale(**args)
        return scaled.resize.Point(transfer_in=Transfer.LINEAR, transfer=trans_in)


class SigmoidScaler(Scaler):
    def __init__(self, scaler: ScalerT, sig_slope: float = 6.5, sig_center: float = 0.75, **kwargs: KwargsT) -> None:
        """
        Simple scaler class to do your scaling business in sigmoid light.

        :params scaler:     Any vsscale scaler class/object
        :param sig_slope:   Curvature value for the sigmoid curve, in range 1-20. The higher the curvature value, the more non-linear the function.
        :param sig_center:  Inflection point for the sigmoid curve, in range 0-1. The closer to 1, the more the curve looks like a standard power curve.
        """
        self.scaler = Scaler.ensure_obj(scaler)
        self.sig_slope = sig_slope
        self.sig_center = sig_center
        self.kwargs = kwargs

    def scale(self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs) -> vs.VideoNode:
        from math import exp
        from vsexprtools import norm_expr, ExprOp

        sig_slope = self.sig_slope
        sig_center = self.sig_center
        if not 1.0 <= sig_slope <= 20.0:
            raise ValueError("sig_slope only accepts values from 1.0 to 20.0 inclusive.")
        if not 0.0 <= sig_center <= 1.0:
            raise ValueError("sig_center only accepts values from 0.0 to 1.0 inclusive.")
        if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
            raise ValueError("SigmoidScaler: Using sigmoid scaling on a subsampled clip results in very poor output. Please don't do it.")

        trans_in = Transfer.from_video(clip)
        convert_csp = (Matrix.from_transfer(trans_in), clip.format)
        sig_offset = 1.0 / (1 + exp(sig_slope * sig_center))
        sig_scale = 1.0 / (1 + exp(sig_slope * (sig_center - 1))) - sig_offset

        bits, clip = get_depth(clip), depth(clip, 32)
        if clip.format and clip.format.color_family is not vs.RGB:
            clip = Point.resample(clip, vs.RGBS, None, convert_csp[0])
        clip = clip.resize.Point(transfer_in=trans_in, transfer=Transfer.LINEAR)
        expr = f"{sig_center} 1 x {sig_scale} * {sig_offset} + / 1 - log {sig_slope} / -"
        clip = norm_expr(clip, f"{expr} {ExprOp.clamp(0, 1)}")
        args = KwargsT(clip=clip, width=width, height=height, shift=shift)
        args.update(**kwargs)
        args.update(**self.kwargs)
        scaled = self.scaler.scale(**args)
        expr = f"1 1 {sig_slope} {sig_center} x - * exp + / {sig_offset} - {sig_scale} /"
        scaled = norm_expr(scaled, f"{expr} {ExprOp.clamp(0, 1)}")
        scaled = Point.resample(scaled, convert_csp[1], convert_csp[0], transfer_in=Transfer.LINEAR, transfer=trans_in)
        return depth(scaled, bits)


class Lanczos_PreSS(Scaler):
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


class Doubler(ABC):
    kwargs: dict[str, Any]
    """Arguments passed to the internal scale function"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Returns doubled clip
        """
        pass


class NNEDI_Doubler(Doubler):
    ediargs: dict[str, Any]
    opencl: bool

    def __init__(self, opencl: bool = True, nns: int = 4, nsize: int = 4, qual: int = 2, pscrn: int = 1, **kwargs) -> None:
        """
        Simple utility class for doubling a clip using znedi or nnedi3cl (also fixes the shift)

        :param opencl:          Will use nnedi3cl if True and znedi3 if False
        """
        self.ediargs = {"qual": qual, "nsize": nsize, "nns": nns, "pscrn": pscrn}
        self.ediargs.update(**kwargs)
        self.opencl = opencl

    def double(self, clip: vs.VideoNode, correct_shift: bool = True) -> vs.VideoNode:
        y = get_y(clip)
        # nnedi3cl needs padding, to avoid issues on edges (https://slow.pics/c/QcJef38u)
        if self.opencl:
            (left, right, top, bottom) = mod_padding(y, 2, 2)
            width = clip.width + left + right
            height = clip.height + top + bottom
            pad = y.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height).std.Transpose()
            doubled_y = pad.nnedi3cl.NNEDI3CL(dh=True, dw=True, field=0, **self.ediargs).std.Transpose()
            doubled_y = doubled_y.std.Crop(left * 2, right * 2, top * 2, bottom * 2)
        else:
            doubled_y = (
                depth(y, 16)
                .znedi3.nnedi3(dh=True, field=0, **self.ediargs)
                .std.Transpose()
                .znedi3.nnedi3(dh=True, field=0, **self.ediargs)
                .std.Transpose()
            )
            doubled_y = depth(doubled_y, get_depth(clip))

        if correct_shift:
            doubled_y = doubled_y.resize.Bicubic(src_top=0.5, src_left=0.5)

        return doubled_y.std.CopyFrameProps(y)


class Shader_Doubler(Doubler):
    shaderfile: str

    def __init__(self, shaderfile: PathLike = r"C:\FSRCNNX_x2_56-16-4-1.glsl") -> None:
        """
        Simple utility class for doubling a clip using a glsl shader

        :param shaderfile:      The glsl shader used to double the resolution
        """
        self.shaderfile = shaderfile if isinstance(shaderfile, str) else str(shaderfile.resolve())

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        y = depth(get_y(clip), 16)
        filler_chroma = core.std.BlankClip(y, format=vs.YUV444P16)
        doubled = core.std.ShufflePlanes([y, filler_chroma], [0, 1, 2], vs.YUV).placebo.Shader(
            self.shaderfile, filter="box", width=y.width * 2, height=y.height * 2
        )
        doubled_y = get_y(doubled)
        return depth(doubled_y, get_depth(clip))


class Waifu2x_Doubler(Doubler):
    backend: any
    kwargs: KwargsT
    w2xargs: KwargsT = {}

    def __init__(
        self,
        cuda: bool | str | None = None,
        fp16: bool = True,
        num_streams: int = 1,
        tiles: int | tuple[int, int] | None = None,
        model: int = 6,
        **kwargs,
    ) -> None:
        """
        Simple utility class for doubling a clip using Waifu2x

        :param cuda:            ORT-Cuda if True, NCNN-VK or CPU (depending on what you have installed) if False, TRT if some string
                                Automatically chosen and tuned when None
        :param fp16:            Uses 16 bit floating point internally if True.
        :param num_streams:     Amount of streams to use for Waifu2x; Sacrifices a lot of vram for a speedup.
        :param tiles:           Splits up the upscaling process into multiple tiles.
                                You will likely have to use atleast `2` if you have less than 16 GB of VRAM.
        :param model:           Model to use from vsmlrt.
        :param kwargs:          Args that get passed to both the Backend and actual scaling function.
        """
        from vsmlrt import Backend

        self.kwargs = {"num_streams": num_streams, "fp16": fp16}

        # Partially stolen from setsu but removed useless stuff that is default in mlrt already and added version checks
        if cuda is None:
            nv = get_nvidia_version()
            cuda = nv is not None
            try:
                if nv is not None and not hasattr(core, "trt") and hasattr(core, "ort"):
                    self.kwargs.update({"use_cuda_graph": True})
                else:
                    props: KwargsT = core.trt.DeviceProperties(kwargs.get("device_id", 0))
                    version_props: KwargsT = core.trt.Version()

                    vram = props.get("total_global_memory", 0)
                    trt_version = float(version_props.get("tensorrt_version", 0))

                    cuda = "trt"

                    presumedArgs = KwargsT(
                        workspace=vram / (1 << 22) if vram else None,
                        use_cuda_graph=True,
                        use_cublas=True,
                        use_cudnn=trt_version < 8400,
                        heuristic=trt_version >= 8500,
                        output_format=int(fp16),
                    )

                    # Swinunet doesn't like forced 16. Further testing for the other models needed.
                    if model <= 6:
                        presumedArgs.update({"tf32": not fp16, "force_fp16": fp16})

                    self.kwargs.update(presumedArgs)
            except:
                cuda = nv is not None

        self.w2xargs = KwargsT(
            model=model,
            tiles=tiles,
            preprocess=kwargs.pop("preprocess", True),
            scale=kwargs.pop("scale", 2),
            tilesize=kwargs.pop("tilesize", None),
            overlap=kwargs.pop("overlap", None),
        )

        self.kwargs.update(kwargs)

        if cuda is False:
            if hasattr(core, "ncnn"):
                self.backend = Backend.NCNN_VK(**self.kwargs)
            else:
                self.kwargs.pop("device_id")
                self.backend = Backend.ORT_CPU(**self.kwargs) if hasattr(core, "ort") else Backend.OV_CPU(**self.kwargs)
        elif cuda is True:
            self.backend = Backend.ORT_CUDA(**self.kwargs) if hasattr(core, "ort") else Backend.OV_GPU(**self.kwargs)
        else:
            self.backend = Backend.TRT(**self.kwargs)

        self.kwargs = kwargs
        self.model = model

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        from vsmlrt import Waifu2x

        pre = depth(clip, 32).std.Limiter()

        (left, right, top, bottom) = mod_padding(pre)
        width = pre.width + left + right
        height = pre.height + top + bottom
        pad = pre.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

        # Model 0 wants a gray input
        needs_gray = self.w2xargs.get("model", 6) == 0
        was_444 = pre.format.color_family == vs.YUV and pre.format.subsampling_w == 0 and pre.format.subsampling_h == 0 and not needs_gray

        if was_444:
            pad = Catrom().resample(pad, format=vs.RGBS, matrix=Matrix.RGB, matrix_in=Matrix.from_video(pre))
        elif needs_gray is True:
            pad = get_y(pad)
        else:
            pad = get_y(pad).std.ShufflePlanes(0, vs.RGB)

        up = Waifu2x(pad, noise=-1, backend=self.backend, **self.w2xargs)

        if was_444:
            up = Catrom().resample(up, format=vs.YUV444PS, matrix=Matrix.from_video(pre), matrix_in=Matrix.RGB)
        elif needs_gray is False:
            up = up.std.ShufflePlanes(0, vs.GRAY)

        up = up.std.Crop(left * 2, right * 2, top * 2, bottom * 2)

        # Only Model 6 has the tint
        if self.w2xargs.get("model", 6) == 6:
            up = up.std.Expr("x 0.5 255 / +")

        return depth(up, get_depth(clip)).std.CopyFrameProps(pre)


class Clamped_Doubler(Doubler):
    sharp_doubler: Doubler
    sharpen_smooth: bool | vs.VideoNode | Callable[[vs.VideoNode], vs.VideoNode] = None

    def __init__(
        self,
        sharpen_smooth: bool | vs.VideoNode | Callable[[vs.VideoNode], vs.VideoNode] = False,
        sharp_doubler: Doubler | str = Shader_Doubler(),
        ratio: int = 100,
        **kwargs,
    ) -> None:
        """
        Simple utility class for doubling a clip using fsrcnnx / any shader clamped to nnedi.
        Using sharpen will be basically the same as the zastin profile in varde's fsrcnnx upscale.
        Not sharpening will on the other hand be the same as the slow profile.

        :param sharpen_smooth:  Sharpened "smooth upscale" clip or a sharpener function. Will use z4usm if True.
                                Uses the other mode if False or None.
        :param sharp_doubler:   The doubler used for the sharp upscale. Defaults to Shader_Doubler (which defaults to fsrcnnx)
        :param ratio:           Does a weighted average of the clamped and nnedi clips.
                                The higher, the more of the clamped clip will be used.
        :param kwargs:          You can pass all kinds of stuff here, ranging from the default sharpener params to nnedi args.
                                z4usm params: radius (default 2), strength (default 35)
                                nnedi params: see `NNEDI_Doubler`
                                overshoot, undershoot for non-sharpen mode (defaults to ratio / 100)
        """
        self.sharp_doubler = Shader_Doubler(sharp_doubler) if isinstance(sharp_doubler, str) else sharp_doubler
        self.sharpen_smooth = sharpen_smooth

        if ratio > 100 or ratio < 1:
            raise "Clamped_Doubler: ratio should be a value between 1 and 100"
        self.ratio = ratio
        self.kwargs = kwargs

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        y = depth(get_y(clip), 16)

        overshoot = self.kwargs.pop("overshoot", self.ratio / 100)
        undershoot = self.kwargs.pop("undershoot", overshoot)
        radius = self.kwargs.pop("radius", 2)
        strength = self.kwargs.pop("strength", 35)

        smooth = NNEDI_Doubler(**self.kwargs).double(y)
        shader = self.sharp_doubler.double(y)

        if self.sharpen_smooth != None and self.sharpen_smooth != False:
            if isinstance(self.sharpen_smooth, vs.VideoNode):
                sharpened_smooth = self.sharpen_smooth
            elif isinstance(self.sharpen_smooth, Callable):
                sharpened_smooth = self.sharpen_smooth(smooth)
            elif self.sharpen_smooth == True:
                sharpened_smooth = unsharp_masked(smooth, radius, strength)

            clamped = core.std.Expr([smooth, shader, sharpened_smooth], "x y z min max y z max min")
            if self.ratio != 100:
                clamped = core.std.Expr([clamped, smooth], f"{self.ratio / 100} x * {1 - (self.ratio / 100)} y * +")
        else:
            upscaled = core.std.Expr([shader, smooth], "x {ratio} * y 1 {ratio} - * +".format(ratio=self.ratio / 100))
            dark_limit = core.std.Minimum(smooth)
            bright_limit = core.std.Maximum(smooth)
            overshoot *= 2**8
            undershoot *= 2**8
            clamped = core.std.Expr(
                [upscaled, bright_limit, dark_limit],
                f"x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?",
            )
        return depth(clamped, get_depth(clip))


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
