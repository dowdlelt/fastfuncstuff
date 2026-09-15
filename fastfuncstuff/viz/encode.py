"""Write ``(T, H, W, 3)`` uint8 frames as a small movie file.

mp4 goes through the system ffmpeg (h264, no pip dependency, a fraction of the size
of a GIF); GIF goes through imageio and is the fallback when ffmpeg is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

MOVIE_FORMATS = ("mp4", "gif")


def find_ffmpeg() -> str | None:
    """System ffmpeg binary path, if available."""
    return shutil.which("ffmpeg")


def default_movie_format() -> str:
    """mp4 when ffmpeg is on PATH, else gif."""
    return "mp4" if find_ffmpeg() else "gif"


def _write_mp4_ffmpeg(frames: np.ndarray, path: str, fps: int, ffmpeg: str) -> None:
    # h264 + yuv420p needs even dimensions; pad the sheet if odd.
    _, h, w, _ = frames.shape
    ph, pw = h + (h % 2), w + (w % 2)
    if (ph, pw) != (h, w):
        # Replicate the border so the pad row is invisible on any background.
        frames = np.pad(frames, ((0, 0), (0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        h, w = ph, pw
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        path,
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])


def write_movie(frames: np.ndarray, path: str, fps: int, fmt: str) -> str:
    """Write ``(T, H, W, 3)`` uint8 frames as a movie; return the path actually written.

    An mp4 request without ffmpeg on PATH writes a GIF next to ``path`` instead.
    """
    if fmt not in MOVIE_FORMATS:
        raise ValueError(f"movie format must be one of {MOVIE_FORMATS}, got {fmt!r}")
    if fmt == "mp4":
        ffmpeg = find_ffmpeg()
        if ffmpeg is not None:
            _write_mp4_ffmpeg(frames, path, fps, ffmpeg)
            return path
        gif_path = str(Path(path).with_suffix(".gif"))
        print(f"  ⚠️  no ffmpeg on PATH; writing GIF instead: {gif_path}")
        path = gif_path

    import imageio.v2 as imageio

    imageio.mimwrite(path, list(frames), duration=1000.0 / max(fps, 1), loop=0)
    return path


def movie_path(prefix: str, fmt: str) -> str:
    """``prefix`` with the container extension, unless it already names one."""
    p = Path(prefix)
    if p.suffix.lower().lstrip(".") in MOVIE_FORMATS:
        return str(p)
    return f"{prefix}.{fmt}"
