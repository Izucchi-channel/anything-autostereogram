import subprocess
import logging
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None
    prange = None


APP_NAME = "autostereogram_gradio"

logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


def log_line(buf, message: str) -> None:
    print(message)
    if buf is not None:
        buf.append(message)
    logger.info(message)


def extract_path(value) -> Optional[str]:
    """Robustly extract actual file path from Gradio input."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            if key in value and value[key]:
                return value[key]
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            p = extract_path(item)
            if p:
                return p
    return None


def safe_open_rgb(path: str) -> Image.Image:
    if not path:
        raise ValueError("Input file path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with Image.open(path) as im:
        return im.convert("RGB")


def pil_to_uint8_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def rgb_to_gray01(arr: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 gray [0,1]."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB array required. shape={arr.shape}")
    arrf = arr.astype(np.float32)
    gray = (0.299 * arrf[..., 0] + 0.587 * arrf[..., 1] + 0.114 * arrf[..., 2]) / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def resize_gray(gray01: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize grayscale float32 [0,1] to target size."""
    if gray01.ndim != 2:
        raise ValueError("gray01 must be a 2D array.")

    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        try:
            t = torch.from_numpy(gray01).to(device="cuda", dtype=torch.float32)[None, None, ...]
            t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
            return t[0, 0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            logger.exception("GPU resize failed, continuing on CPU.")

    pil = Image.fromarray((np.clip(gray01, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return (np.asarray(pil, dtype=np.float32) / 255.0).astype(np.float32)


def resize_rgb(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB array required. shape={arr.shape}")
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr.astype(np.uint8, copy=False)
    interp = cv2.INTER_AREA if out_w < arr.shape[1] or out_h < arr.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(arr, (out_w, out_h), interpolation=interp).astype(np.uint8)


def apply_binary_and_noise_style(
    rgb: np.ndarray,
    noise_color_mode: str,
    allow_noise_color_mode: bool,
) -> np.ndarray:
    """Apply color settings to pattern side only."""
    rgb = np.asarray(rgb, dtype=np.uint8)

    if allow_noise_color_mode and noise_color_mode == "grayscale":
        gray = rgb_to_gray01(rgb)
        g8 = (gray * 255.0).astype(np.uint8)
        return np.repeat(g8[..., None], 3, axis=2)

    return rgb


def fit_rgb_to_size(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    pil = ImageOps.fit(pil, (out_w, out_h), method=Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def prepare_depth_from_rgb(
    rgb: np.ndarray,
    out_w: int,
    out_h: int,
    invert: bool = False,
    force_binary: bool = False,
) -> np.ndarray:
    gray = rgb_to_gray01(rgb)
    gray = resize_gray(gray, out_w, out_h)
    if invert:
        gray = 1.0 - gray
    if force_binary:
        gray = (gray >= 0.5).astype(np.float32)
    return gray.astype(np.float32)


def generate_sample_pattern(h: int, w: int, seed: int = 1234) -> np.ndarray:
    """Generate a visually appealing sample pattern."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (40, 60, 90))
    draw = ImageDraw.Draw(img)

    stripe_h = max(8, h // 18)
    colors = [(60, 90, 160), (200, 170, 120), (90, 150, 100), (170, 110, 130)]
    for y in range(0, h, stripe_h):
        color = colors[(y // stripe_h) % len(colors)]
        draw.rectangle([0, y, w, min(h, y + stripe_h)], fill=color)

    for _ in range(max(18, w // 24)):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(max(6, min(w, h) // 30), max(12, min(w, h) // 10)))
        col = tuple(int(x) for x in rng.integers(70, 240, size=3))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=max(1, r // 6))
        if rng.random() > 0.55:
            draw.line((cx - r, cy, cx + r, cy), fill=col, width=max(1, r // 10))
            draw.line((cx, cy - r, cx, cy + r), fill=col, width=max(1, r // 10))

    arr = np.asarray(img, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    mix = (0.65 * xx + 0.35 * yy)[..., None]
    arr = arr * (0.75 + 0.25 * mix)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def generate_noise_texture(
    out_h: int,
    out_w: int,
    noise_color_mode: str,
    block_size: int,
    seed: int = 1234,
) -> np.ndarray:
    """Generate noise texture with block size."""
    rng = np.random.default_rng(int(seed))
    block_size = max(1, int(block_size))
    grid_h = max(1, math.ceil(out_h / block_size))
    grid_w = max(1, math.ceil(out_w / block_size))

    if noise_color_mode == "grayscale":
        vals = rng.integers(0, 256, size=(grid_h, grid_w, 1), dtype=np.uint8)
        arr = np.repeat(vals, 3, axis=2)
    else:
        arr = rng.integers(0, 256, size=(grid_h, grid_w, 3), dtype=np.uint8)

    arr = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.uint8)


def prepare_texture_from_source_rgb(
    rgb: np.ndarray,
    out_w: int,
    out_h: int,
    noise_color_mode: str,
    allow_noise_color_mode: bool,
) -> np.ndarray:
    rgb = resize_rgb(rgb, out_w, out_h)
    rgb = apply_binary_and_noise_style(
        rgb,
        noise_color_mode=noise_color_mode,
        allow_noise_color_mode=allow_noise_color_mode,
    )
    return rgb


def prepare_depth_from_image(path: str, out_w: int, out_h: int, invert: bool = False, force_binary: bool = False) -> np.ndarray:
    logger.info(f"Loading depth image: {path}")
    img = safe_open_rgb(path)
    return prepare_depth_from_rgb(pil_to_uint8_rgb(img), out_w, out_h, invert=invert, force_binary=force_binary)


def open_video_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def read_video_frame_loop(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    return None


def prepare_texture_frame_from_bgr(
    frame_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    noise_color_mode: str,
    allow_noise_color_mode: bool,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return prepare_texture_from_source_rgb(
        rgb,
        out_w=out_w,
        out_h=out_h,
        noise_color_mode=noise_color_mode,
        allow_noise_color_mode=allow_noise_color_mode,
    )


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def synthesize_autostereogram_numba(depth_map, pattern, pad, scale):
        h, w = depth_map.shape
        out = np.empty((h, w, 3), dtype=np.uint8)

        for r in prange(h):
            for c in range(pad):
                out[r, c, 0] = pattern[r, c, 0]
                out[r, c, 1] = pattern[r, c, 1]
                out[r, c, 2] = pattern[r, c, 2]

            for c in range(pad, w):
                shift = int(depth_map[r, c] * scale + 0.5)
                src = c - pad + shift

                if src < 0:
                    src = 0
                elif src >= c:
                    src = c - 1

                out[r, c, 0] = out[r, src, 0]
                out[r, c, 1] = out[r, src, 1]
                out[r, c, 2] = out[r, src, 2]

        return out
else:
    def synthesize_autostereogram_numba(depth_map, pattern, pad, scale):
        h, w = depth_map.shape
        out = np.empty((h, w, 3), dtype=np.uint8)

        for r in range(h):
            out[r, :pad, :] = pattern[r, :pad, :]

            for c in range(pad, w):
                shift = int(depth_map[r, c] * scale + 0.5)
                src = c - pad + shift

                if src < 0:
                    src = 0
                elif src >= c:
                    src = c - 1

                out[r, c, :] = out[r, src, :]

        return out


def draw_alignment_dots(img: np.ndarray, pad: int) -> np.ndarray:
    """Draw red dots at the same position as JS version."""
    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size

    y = max(12, int(h * 0.015))
    x1 = w // 2 - pad // 2
    x2 = w // 2 + pad // 2
    r = max(4, w // 200)
    color = (255, 0, 0)

    for x in (x1, x2):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return np.asarray(pil, dtype=np.uint8)


def save_png(arr: np.ndarray, prefix: str = "my-autostereogram") -> str:
    out_dir = tempfile.mkdtemp(prefix="stereogram_")
    path = os.path.join(out_dir, f"{prefix}.png")
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def save_video(frames_bgr, fps: float, prefix: str = "my-autostereogram") -> str:
    out_dir = tempfile.mkdtemp(prefix="stereogram_")
    path = os.path.join(out_dir, f"{prefix}.mp4")

    if not frames_bgr:
        raise ValueError("No frames to save.")

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        path = os.path.join(out_dir, f"{prefix}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to initialize VideoWriter.")

    try:
        for f in frames_bgr:
            if f.shape[:2] != (h, w):
                raise ValueError("Frame size mismatch.")
            writer.write(f)
    finally:
        writer.release()

    return path


def _make_output_size(src_w: int, src_h: int, out_w: int) -> Tuple[int, int]:
    out_w = int(max(64, out_w))
    out_h = max(64, int(round(out_w * src_h / max(1, src_w))))
    return out_w, out_h


def get_single_source_frame(
    source_image_path: Optional[str],
    source_video_path: Optional[str],
) -> Optional[np.ndarray]:
    if source_video_path:
        cap = open_video_capture(source_video_path)
        try:
            frame = read_video_frame_loop(cap)
        finally:
            cap.release()
        return frame
    if source_image_path:
        return cv2.cvtColor(pil_to_uint8_rgb(safe_open_rgb(source_image_path)), cv2.COLOR_RGB2BGR)
    return None


def has_nvenc():
    """Check if ffmpeg supports NVENC."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def compress_video(input_path, output_path, crf=20, cpu_preset="medium", gpu_preset="p7", use_gpu=False):
    """
    Compress video with specified settings.

    Args:
        input_path: Source video path
        output_path: Output video path
        crf: Quality (0-51, lower is better)
        cpu_preset: CPU encoding preset (ultrafast to veryslow)
        gpu_preset: GPU encoding preset (p1 to p7)
        use_gpu: Use GPU encoding if available
    """
    actual_use_gpu = use_gpu and has_nvenc()

    if actual_use_gpu:
        # Map GPU preset names to NVENC preset values
        gpu_preset_map = {
            "p1": "p1", "p2": "p2", "p3": "p3",
            "p4": "p4", "p5": "p5", "p6": "p6", "p7": "p7"
        }
        preset = gpu_preset_map.get(gpu_preset, "p7")
        print(f"[compress] Using GPU (NVENC) with preset={preset}, crf={crf}")
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-i",
            input_path,
            "-vcodec",
            "h264_nvenc",
            "-preset",
            preset,
            "-cq",
            str(crf),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
    else:
        if use_gpu and not actual_use_gpu:
            print("[compress] GPU requested but NVENC not available, falling back to CPU")
        print(f"[compress] Using CPU (libx264) with preset={cpu_preset}, crf={crf}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            cpu_preset,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[compress] FFmpeg error: {e.stderr}")
        raise


def process_image(
    depth_image_path: str,
    pattern_image_path: Optional[str],
    pattern_video_path: Optional[str],
    use_noise: bool,
    noise_color_mode: str,
    regenerate_noise_per_frame: bool,
    out_width: int,
    max_shift_ratio: float,
    invert_depth: bool,
    add_alignment_dots: bool,
    force_binary: bool,
    noise_block_size: int,
    seed: int,
):
    logs = []
    t0 = cv2.getTickCount()

    try:
        depth_image_path = extract_path(depth_image_path)
        pattern_image_path = extract_path(pattern_image_path)
        pattern_video_path = extract_path(pattern_video_path)

        if not depth_image_path:
            raise ValueError("Depth image is not specified.")

        src = safe_open_rgb(depth_image_path)
        src_w, src_h = src.size
        out_w, out_h = _make_output_size(src_w, src_h, out_width)
        pad = max(8, out_w // 10)
        scale = pad * max_shift_ratio

        log_line(logs, f"[image] Input depth image: {depth_image_path}")
        log_line(logs, f"[image] Input size: {src_w}x{src_h}")
        log_line(logs, f"[image] Output size: {out_w}x{out_h}")
        log_line(logs, f"[image] pad={pad}, max_shift_ratio={max_shift_ratio:.3f}, scale={scale:.3f}")
        log_line(logs, f"[image] Maximum shift: {int(scale)} pixels")
        log_line(logs, f"[image] Noise mode: {'ON' if use_noise else 'OFF'}")
        log_line(logs, f"[image] Noise color: {noise_color_mode}")
        log_line(logs, f"[image] Regenerate noise per frame: {'ON' if regenerate_noise_per_frame else 'OFF'}")
        log_line(logs, f"[image] Force binary depth: {'ON' if force_binary else 'OFF'}")
        log_line(logs, f"[image] Block size: {int(noise_block_size)}px")

        depth = prepare_depth_from_image(depth_image_path, out_w, out_h, invert=invert_depth, force_binary=force_binary)

        if use_noise:
            if pattern_video_path:
                frame_bgr = get_single_source_frame(None, pattern_video_path)
                if frame_bgr is None:
                    raise RuntimeError("Failed to load first frame from noise video.")
                pattern = prepare_texture_frame_from_bgr(
                    frame_bgr,
                    pad,
                    out_h,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=True,
                )
                log_line(logs, f"[image] Noise source: video -> {pattern_video_path}")
            else:
                pattern = generate_noise_texture(
                    out_h=out_h,
                    out_w=pad,
                    noise_color_mode=noise_color_mode,
                    block_size=noise_block_size,
                    seed=int(seed),
                )
                log_line(logs, "[image] Noise source: generated noise")
        else:
            if pattern_video_path:
                frame_bgr = get_single_source_frame(None, pattern_video_path)
                if frame_bgr is None:
                    raise RuntimeError("Failed to load first frame from pattern video.")
                pattern = prepare_texture_frame_from_bgr(
                    frame_bgr,
                    pad,
                    out_h,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )
                log_line(logs, f"[image] Pattern source: video -> {pattern_video_path}")
            elif pattern_image_path:
                img = safe_open_rgb(pattern_image_path)
                pattern = fit_rgb_to_size(pil_to_uint8_rgb(img), pad, out_h)
                pattern = apply_binary_and_noise_style(
                    pattern,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )
                log_line(logs, f"[image] Pattern source: image -> {pattern_image_path}")
            else:
                pattern = generate_sample_pattern(out_h, pad, seed=int(seed))
                pattern = apply_binary_and_noise_style(
                    pattern,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )
                log_line(logs, "[image] Pattern source: sample pattern")

        out = synthesize_autostereogram_numba(depth, pattern, pad, scale)
        if add_alignment_dots:
            out = draw_alignment_dots(out, pad)

        out_path = save_png(out)
        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        log_line(logs, f"[image] Generation completed: {out_path}")
        log_line(logs, f"[image] Processing time: {elapsed:.2f} seconds")
        return (
            gr.update(value=out_path, visible=True),
            gr.update(value=None, visible=False),
            "\n".join(logs),
        )
    except Exception as e:
        tb = traceback.format_exc()
        log_line(logs, f"[image][ERROR] {e}")
        log_line(logs, tb)
        raise gr.Error(f"Image processing failed: {e}")


def process_video(
    depth_video_path: str,
    pattern_image_path: Optional[str],
    pattern_video_path: Optional[str],
    use_noise: bool,
    noise_color_mode: str,
    regenerate_noise_per_frame: bool,
    out_width: int,
    max_shift_ratio: float,
    invert_depth: bool,
    add_alignment_dots: bool,
    force_binary: bool,
    noise_block_size: int,
    seed: int,
    merge_audio: bool,
    compression_crf: int,
    compression_cpu_preset: str,
    compression_gpu_preset: str,
    use_gpu_encoding: bool,
):
    logs = []
    t0 = cv2.getTickCount()

    cap = None
    source_cap = None
    try:
        depth_video_path = extract_path(depth_video_path)
        pattern_image_path = extract_path(pattern_image_path)
        pattern_video_path = extract_path(pattern_video_path)

        if not depth_video_path:
            raise ValueError("Depth video is not specified.")

        cap = open_video_capture(depth_video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3 or np.isnan(fps):
            fps = 30.0

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to load first frame from video.")

        src_h, src_w = frame.shape[:2]
        out_w, out_h = _make_output_size(src_w, src_h, out_width)
        pad = max(8, out_w // 10)
        scale = pad * max_shift_ratio

        log_line(logs, f"[video] Input video: {depth_video_path}")
        log_line(logs, f"[video] Output size: {out_w}x{out_h}")
        log_line(logs, f"[video] pad={pad}, max_shift_ratio={max_shift_ratio:.3f}, scale={scale:.3f}")
        log_line(logs, f"[video] Noise mode: {'ON' if use_noise else 'OFF'}")
        log_line(logs, f"[video] Noise color: {noise_color_mode}")
        log_line(logs, f"[video] Regenerate noise per frame: {'ON' if regenerate_noise_per_frame else 'OFF'}")
        log_line(logs, f"[video] Force binary depth: {'ON' if force_binary else 'OFF'}")
        log_line(logs, f"[video] Block size: {int(noise_block_size)}px")
        log_line(logs, f"[video] Compression settings: CRF={compression_crf}, CPU Preset={compression_cpu_preset}, GPU Preset={compression_gpu_preset}, GPU={'ON' if use_gpu_encoding else 'OFF'}")
        log_line(logs, "[video] Memory saving mode: sequential frame writing")

        if pattern_video_path:
            source_cap = open_video_capture(pattern_video_path)

        static_pattern = None
        if use_noise and source_cap is None and not regenerate_noise_per_frame:
            static_pattern = generate_noise_texture(
                out_h=out_h,
                out_w=pad,
                noise_color_mode=noise_color_mode,
                block_size=noise_block_size,
                seed=int(seed),
            )
        elif not use_noise and source_cap is None:
            if pattern_image_path:
                img = safe_open_rgb(pattern_image_path)
                static_pattern = fit_rgb_to_size(pil_to_uint8_rgb(img), pad, out_h)
                static_pattern = apply_binary_and_noise_style(
                    static_pattern,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )
            else:
                static_pattern = generate_sample_pattern(out_h, pad, seed=int(seed))
                static_pattern = apply_binary_and_noise_style(
                    static_pattern,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )

        out_dir = tempfile.mkdtemp(prefix="stereogram_")
        out_path = os.path.join(out_dir, "my-autostereogram.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("Failed to initialize VideoWriter")

        processed = 0

        def make_pattern_for_frame(frame_index: int) -> np.ndarray:
            nonlocal static_pattern, source_cap
            if use_noise:
                if source_cap is not None:
                    src_frame = read_video_frame_loop(source_cap)
                    if src_frame is None:
                        raise RuntimeError("Failed to load frame from noise video.")
                    return prepare_texture_frame_from_bgr(
                        src_frame,
                        pad,
                        out_h,
                        noise_color_mode=noise_color_mode,
                        allow_noise_color_mode=True,
                    )
                if regenerate_noise_per_frame or static_pattern is None:
                    return generate_noise_texture(
                        out_h=out_h,
                        out_w=pad,
                        noise_color_mode=noise_color_mode,
                        block_size=noise_block_size,
                        seed=int(seed) + frame_index,
                    )
                return static_pattern

            if source_cap is not None:
                src_frame = read_video_frame_loop(source_cap)
                if src_frame is None:
                    raise RuntimeError("Failed to load frame from pattern video.")
                return prepare_texture_frame_from_bgr(
                    src_frame,
                    pad,
                    out_h,
                    noise_color_mode=noise_color_mode,
                    allow_noise_color_mode=False,
                )

            if static_pattern is not None:
                return static_pattern

            return generate_sample_pattern(out_h, pad, seed=int(seed) + frame_index)

        def process_frame(bgr_frame: np.ndarray, frame_index: int) -> np.ndarray:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            depth = prepare_depth_from_rgb(rgb, out_w, out_h, invert=invert_depth, force_binary=force_binary)
            pattern = make_pattern_for_frame(frame_index)
            out_rgb = synthesize_autostereogram_numba(depth, pattern, pad, scale)
            if add_alignment_dots:
                out_rgb = draw_alignment_dots(out_rgb, pad)
            return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        writer.write(process_frame(frame, processed))
        processed += 1

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            writer.write(process_frame(frame, processed))
            processed += 1

            if processed % 10 == 0:
                log_line(logs, f"[video] Progress: {processed} frames")

        writer.release()

        compressed_path = out_path.replace(".mp4", "_compressed.mp4")
        log_line(logs, "[video] Starting compression (ffmpeg)...")
        compress_video(
            out_path, compressed_path,
            crf=compression_crf,
            cpu_preset=compression_cpu_preset,
            gpu_preset=compression_gpu_preset,
            use_gpu=use_gpu_encoding
        )
        log_line(logs, f"[video] Compression completed: {compressed_path}")

        final_path = compressed_path
        if merge_audio:
            log_line(logs, "[video] Starting audio merging (ffmpeg)...")
            merged_path = compressed_path.replace(".mp4", "_with_audio.mp4")

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                compressed_path,
                "-i",
                depth_video_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                merged_path,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                final_path = merged_path
                log_line(logs, f"[video] Audio merging completed: {merged_path}")
            except Exception as e:
                log_line(logs, f"[video] Audio merging failed: {e}")
                log_line(logs, "[video] Returning compressed video without audio.")

        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        log_line(logs, f"[video] Total frames: {processed}")
        log_line(logs, f"[video] Processing time: {elapsed:.2f}s")

        return (
            gr.update(value=None, visible=False),
            gr.update(value=final_path, visible=True),
            "\n".join(logs),
        )
    except Exception as e:
        tb = traceback.format_exc()
        log_line(logs, f"[video][ERROR] {e}")
        log_line(logs, tb)
        raise gr.Error(f"Video processing failed: {e}")
    finally:
        if cap is not None:
            cap.release()
        if source_cap is not None:
            source_cap.release()


def run(
    depth_image,
    depth_video,
    pattern_image,
    pattern_video,
    use_noise,
    noise_color_mode,
    regenerate_noise_per_frame,
    out_width,
    max_shift_ratio,
    invert_depth,
    add_alignment_dots,
    force_binary,
    noise_block_size,
    seed,
    merge_audio,
    compression_crf,
    compression_cpu_preset,
    compression_gpu_preset,
    use_gpu_encoding,
):
    logger.info("=== Generation Started ===")
    logger.info(f"Numba available: {NUMBA_AVAILABLE}")
    logger.info(f"Torch available: {TORCH_AVAILABLE}")

    depth_image_path = extract_path(depth_image)
    depth_video_path = extract_path(depth_video)
    pattern_image_path = extract_path(pattern_image)
    pattern_video_path = extract_path(pattern_video)

    if depth_video_path:
        return process_video(
            depth_video_path=depth_video_path,
            pattern_image_path=pattern_image_path,
            pattern_video_path=pattern_video_path,
            use_noise=use_noise,
            noise_color_mode=noise_color_mode,
            regenerate_noise_per_frame=regenerate_noise_per_frame,
            out_width=out_width,
            max_shift_ratio=max_shift_ratio,
            invert_depth=invert_depth,
            add_alignment_dots=add_alignment_dots,
            force_binary=force_binary,
            noise_block_size=noise_block_size,
            seed=seed,
            merge_audio=merge_audio,
            compression_crf=compression_crf,
            compression_cpu_preset=compression_cpu_preset,
            compression_gpu_preset=compression_gpu_preset,
            use_gpu_encoding=use_gpu_encoding,
        )
    if depth_image_path:
        return process_image(
            depth_image_path=depth_image_path,
            pattern_image_path=pattern_image_path,
            pattern_video_path=pattern_video_path,
            use_noise=use_noise,
            noise_color_mode=noise_color_mode,
            regenerate_noise_per_frame=regenerate_noise_per_frame,
            out_width=out_width,
            max_shift_ratio=max_shift_ratio,
            invert_depth=invert_depth,
            add_alignment_dots=add_alignment_dots,
            force_binary=force_binary,
            noise_block_size=noise_block_size,
            seed=seed,
        )
    raise gr.Error("Please specify either a depth image or a depth video.")


def build_demo():
    with gr.Blocks(title="Autostereogram Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Autostereogram (Magic Eye) Generator

            Generate autostereograms from depth images (grayscale: white = foreground, black = background) that appear 3D when viewed with parallel viewing method.

            **How to use:**
            1. Upload a depth image (or video)
            2. Optionally specify a pattern image or pattern video
            3. Adjust noise settings if using noise
            4. Click "Generate" button
            5. Use **parallel viewing** (look at a distant point, not cross-eyed) to merge the two red dots at the top of the output image to see the 3D effect

            **Tip:** For beginners, set `max_shift_ratio` to 0.15-0.2 for easier viewing.
            """
        )

        with gr.Tabs():
            with gr.TabItem("English", id="en"):
                with gr.Row():
                    with gr.Column(scale=1):
                        depth_image = gr.Image(
                            label="Depth Image (white=foreground, black=background)",
                            type="filepath",
                            sources=["upload"],
                        )
                        depth_video = gr.Video(
                            label="Depth Video (optional)",
                            sources=["upload"],
                            format="mp4",
                        )
                        pattern_image = gr.Image(
                            label="Pattern Image (optional)",
                            type="filepath",
                            sources=["upload"],
                        )
                        pattern_video = gr.Video(
                            label="Pattern Video / Noise Video (optional)",
                            sources=["upload"],
                            format="mp4",
                        )

                        with gr.Row():
                            use_noise = gr.Checkbox(
                                value=False,
                                label="Use Noise",
                            )
                            invert_depth = gr.Checkbox(
                                value=False,
                                label="Invert Depth",
                            )
                            add_alignment_dots = gr.Checkbox(
                                value=True,
                                label="Add Alignment Dots",
                            )

                        with gr.Row():
                            noise_color_mode = gr.Radio(
                                choices=[("Color", "color"), ("Grayscale", "grayscale")],
                                value="color",
                                label="Noise Color",
                            )
                            regenerate_noise_per_frame = gr.Checkbox(
                                value=False,
                                label="Regenerate Noise Per Frame",
                            )

                        force_binary = gr.Checkbox(
                            value=False,
                            label="Force Binary Depth (Black/White)",
                        )

                        merge_audio = gr.Checkbox(
                            value=False,
                            label="Merge Audio from Original Depth Video",
                        )

                        out_width = gr.Slider(
                            minimum=256,
                            maximum=20000,
                            value=1000,
                            step=1,
                            label="Output Width (pixels)",
                        )
                        max_shift_ratio = gr.Slider(
                            minimum=0.05,
                            maximum=0.40,
                            value=0.20,
                            step=0.01,
                            label="Maximum Disparity (ratio of pad) - Smaller is easier to view",
                        )
                        noise_block_size = gr.Slider(
                            minimum=1,
                            maximum=64,
                            value=1,
                            step=1,
                            label="Noise Block Size (px)",
                        )
                        seed = gr.Number(
                            value=1234,
                            precision=0,
                            label="Random Seed",
                        )

                        with gr.Group():
                            gr.Markdown("### Compression Settings")
                            compression_crf = gr.Slider(
                                minimum=0,
                                maximum=51,
                                value=20,
                                step=1,
                                label="CRF (Quality) - Lower = Better Quality, Higher = Smaller File",
                            )
                            use_gpu_encoding = gr.Checkbox(
                                value=False,
                                label="Use GPU Encoding (NVENC) if available",
                            )
                            compression_cpu_preset = gr.Dropdown(
                                choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                                value="medium",
                                label="CPU Preset (for libx264) - Faster = Lower Quality, Slower = Better Quality",
                            )
                            compression_gpu_preset = gr.Dropdown(
                                choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                                value="p7",
                                label="GPU Preset (for NVENC) - p1=Fastest/Lowest Quality, p7=Slowest/Highest Quality",
                            )

                        run_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Output Image", type="filepath", visible=False)
                        output_video = gr.Video(label="Output Video", visible=False)
                        log_box = gr.Textbox(label="Processing Log", lines=20, interactive=False)

            with gr.TabItem("日本語", id="ja"):
                with gr.Row():
                    with gr.Column(scale=1):
                        depth_image_ja = gr.Image(
                            label="深度画像（白=手前、黒=奥）",
                            type="filepath",
                            sources=["upload"],
                        )
                        depth_video_ja = gr.Video(
                            label="深度動画（任意）",
                            sources=["upload"],
                            format="mp4",
                        )
                        pattern_image_ja = gr.Image(
                            label="パターン画像（任意）",
                            type="filepath",
                            sources=["upload"],
                        )
                        pattern_video_ja = gr.Video(
                            label="パターン動画 / ノイズ動画（任意）",
                            sources=["upload"],
                            format="mp4",
                        )

                        with gr.Row():
                            use_noise_ja = gr.Checkbox(
                                value=False,
                                label="ノイズを使う",
                            )
                            invert_depth_ja = gr.Checkbox(
                                value=False,
                                label="深度を反転",
                            )
                            add_alignment_dots_ja = gr.Checkbox(
                                value=True,
                                label="視点合わせ用の点を追加",
                            )

                        with gr.Row():
                            noise_color_mode_ja = gr.Radio(
                                choices=[("カラー", "color"), ("白黒", "grayscale")],
                                value="color",
                                label="ノイズの色",
                            )
                            regenerate_noise_per_frame_ja = gr.Checkbox(
                                value=False,
                                label="ノイズをフレームごとに再生成",
                            )

                        force_binary_ja = gr.Checkbox(
                            value=False,
                            label="深度画像・深度動画を白黒2値に強制変換",
                        )

                        merge_audio_ja = gr.Checkbox(
                            value=False,
                            label="元の深度動画の音声を結合する",
                        )

                        out_width_ja = gr.Slider(
                            minimum=256,
                            maximum=20000,
                            value=1000,
                            step=1,
                            label="出力の横幅（ピクセル）",
                        )
                        max_shift_ratio_ja = gr.Slider(
                            minimum=0.05,
                            maximum=0.40,
                            value=0.20,
                            step=0.01,
                            label="最大視差（padに対する比率）- 小さいほど見やすい",
                        )
                        noise_block_size_ja = gr.Slider(
                            minimum=1,
                            maximum=64,
                            value=1,
                            step=1,
                            label="ノイズのブロックサイズ（px）",
                        )
                        seed_ja = gr.Number(
                            value=1234,
                            precision=0,
                            label="乱数シード",
                        )

                        with gr.Group():
                            gr.Markdown("### 圧縮設定")
                            compression_crf_ja = gr.Slider(
                                minimum=0,
                                maximum=51,
                                value=20,
                                step=1,
                                label="CRF（品質）- 低いほど高品質、高いほどファイルサイズ小",
                            )
                            use_gpu_encoding_ja = gr.Checkbox(
                                value=False,
                                label="GPUエンコーディング（NVENC）を使用（利用可能な場合）",
                            )
                            compression_cpu_preset_ja = gr.Dropdown(
                                choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                                value="medium",
                                label="CPUプリセット（libx264用）- 高速=低品質、低速=高品質",
                            )
                            compression_gpu_preset_ja = gr.Dropdown(
                                choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                                value="p7",
                                label="GPUプリセット（NVENC用）- p1=最速/最低品質、p7=最遅/最高品質",
                            )

                        run_btn_ja = gr.Button("生成", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_image_ja = gr.Image(label="出力画像", type="filepath", visible=False)
                        output_video_ja = gr.Video(label="出力動画", visible=False)
                        log_box_ja = gr.Textbox(label="処理ログ", lines=20, interactive=False)

        # Link English tab components
        run_btn.click(
            fn=run,
            inputs=[
                depth_image,
                depth_video,
                pattern_image,
                pattern_video,
                use_noise,
                noise_color_mode,
                regenerate_noise_per_frame,
                out_width,
                max_shift_ratio,
                invert_depth,
                add_alignment_dots,
                force_binary,
                noise_block_size,
                seed,
                merge_audio,
                compression_crf,
                compression_cpu_preset,
                compression_gpu_preset,
                use_gpu_encoding,
            ],
            outputs=[output_image, output_video, log_box],
        )

        # Link Japanese tab components
        def run_ja(
            depth_image_ja, depth_video_ja, pattern_image_ja, pattern_video_ja,
            use_noise_ja, noise_color_mode_ja, regenerate_noise_per_frame_ja,
            out_width_ja, max_shift_ratio_ja, invert_depth_ja, add_alignment_dots_ja,
            force_binary_ja, noise_block_size_ja, seed_ja, merge_audio_ja,
            compression_crf_ja, compression_cpu_preset_ja, compression_gpu_preset_ja, use_gpu_encoding_ja,
        ):
            return run(
                depth_image_ja, depth_video_ja, pattern_image_ja, pattern_video_ja,
                use_noise_ja, noise_color_mode_ja, regenerate_noise_per_frame_ja,
                out_width_ja, max_shift_ratio_ja, invert_depth_ja, add_alignment_dots_ja,
                force_binary_ja, noise_block_size_ja, seed_ja, merge_audio_ja,
                compression_crf_ja, compression_cpu_preset_ja, compression_gpu_preset_ja, use_gpu_encoding_ja,
            )

        run_btn_ja.click(
            fn=run_ja,
            inputs=[
                depth_image_ja,
                depth_video_ja,
                pattern_image_ja,
                pattern_video_ja,
                use_noise_ja,
                noise_color_mode_ja,
                regenerate_noise_per_frame_ja,
                out_width_ja,
                max_shift_ratio_ja,
                invert_depth_ja,
                add_alignment_dots_ja,
                force_binary_ja,
                noise_block_size_ja,
                seed_ja,
                merge_audio_ja,
                compression_crf_ja,
                compression_cpu_preset_ja,
                compression_gpu_preset_ja,
                use_gpu_encoding_ja,
            ],
            outputs=[output_image_ja, output_video_ja, log_box_ja],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=8).launch(share=True, debug=True)
