from typing import Any
from vskernels import Catrom, Lanczos
from vstools import (
    inject_self,
    vs,
    core,
    depth,
    get_depth,
    get_y,
    Matrix,
    KwargsT,
    get_nvidia_version,
)
from abc import ABC, abstractmethod


__all__: list[str] = ["NNEDI_Doubler", "Waifu2x_Doubler", "Lanczos_PreSS"]


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


class Waifu2x_Doubler(Doubler):
    backend: Any
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
        from vsmlrt import Backend, backendT

        self.kwargs = {"num_streams": num_streams, "fp16": fp16}

        if "backend" in kwargs.keys():
            cuda = False

        # Partially stolen from setsu but removed useless stuff that is default in mlrt already and added version checks
        if cuda is None:
            nv = get_nvidia_version()
            cuda = nv is not None
            try:
                if nv is not None and not hasattr(core, "trt") and hasattr(core, "ort"):
                    self.kwargs.update({"use_cuda_graph": True})
                else:
                    props: KwargsT = core.trt.DeviceProperties(kwargs.get("device_id", 0))  # type: ignore
                    version_props: KwargsT = core.trt.Version()  # type: ignore

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
            backend = kwargs.pop("backend", None)
            if backend and isinstance(backend, backendT):
                self.backend = backend
            else:
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
        was_444 = pre.format.color_family == vs.YUV and pre.format.subsampling_w == 0 and pre.format.subsampling_h == 0 and not needs_gray  # type: ignore

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
