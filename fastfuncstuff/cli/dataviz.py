"""ffs_dataviz — watch a 4-D run play out as data-driven pixel art (and sound).

Thin by design: parse, load and compute features, then either open the window
or render a movie headless.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg

EPILOG = """\
every masked voxel becomes a point (or several). nothing is fitted -- no
events, no model -- the run is replayed and the eye finds what is there. the
same drives feed every scene; a scene is only a way of drawing them:

  mandala   ICA families as concentric rings in N-fold mirrored symmetry.
            a ring spins and sprouts petals while its family is active, the
            figure breathes with the global signal, DVARS spikes scatter it.
  milkdrop  a feedback tunnel: the last frame is zoomed, swirled and
            recoloured, then this frame's voxels are drawn on top. the zoom
            pumps with the global signal; each family twists its own radial
            band; DVARS spikes jolt the warp.
  flow      ink in water: voxels laid out in correlation space, stirred by
            one vortex per family that spins with the family's time course.
  sand      falling sand: each family pours grains from its own lane while
            active; pile heights are cumulative activity, strata are time.
            the global signal is wind, DVARS spikes are quakes.
  brain     the anatomical 3-D swarm: dust on the anatomy, sparks lit by the
            live signal that bloom, flock with their family and shimmer.

most of the brain carries no task signal, so sparks are pooled before they are
shown: across space (each voxel blended with its neighbourhood: pool, reach)
and across correlation (toward the low-rank reconstruction: clarity). they are
weighted by salience -- how much of the voxel's variance is shared with its
partners and its neighbours -- so white matter stays quiet however high its
tSNR. whole-brain events reach every particle: the global signal shifts the
colour temperature and a DVARS spike flashes and jolts the swarm.

sound follows the same drives: each ICA family is a voice on a pentatonic
scale (pitch from its height in the head, pan from its side), a drone opens up
with the global signal, and DVARS spikes are a soft swell.

keys: 1-5 scene, space play/pause, tab or m controls, h timeline, f fullscreen,
s sound, c colour source, [ ] speed, arrows step, wheel zoom, q quit;
in brain: drag orbits, l animates anatomy <-> correlation layout.

examples
  ffs_dataviz -input run1.nii.zst
  ffs_dataviz -input run1.nii.zst run2.nii.zst -set scene=sand speed=4
  ffs_dataviz -input run1.nii.zst -movie tunnel.mp4 -duration 60 -set scene=milkdrop
"""


def build_parser() -> FfsArgumentParser:
    from fastfuncstuff.viz.dataviz.engine import PARAMS

    params = ", ".join(f"{p.name}" for p in PARAMS)
    parser = FfsArgumentParser(
        prog="ffs_dataviz",
        description="Data-driven pixel-art visualiser (and sonifier) for 4-D fMRI.",
        epilog=EPILOG,
    )
    parser.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="DSET",
        help="4-D run(s) on one grid; several are concatenated with per-run drift removal",
    )
    parser.add_argument(
        "-mask", metavar="DSET", help="brain mask on the run's grid (default: automask)"
    )
    parser.add_argument("-TR", type=float, metavar="S", help="override the header TR")
    parser.add_argument(
        "-polort",
        type=int,
        metavar="N",
        help="Legendre drift degree per run (default: AFNI's 1 + duration/150 s)",
    )
    parser.add_argument("-ncomp", type=int, default=16, metavar="K", help="PCA components kept")
    parser.add_argument(
        "-nfam", type=int, default=10, metavar="F", help="ICA families (voices, flocks)"
    )
    parser.add_argument(
        "-period",
        type=float,
        metavar="S",
        help="colour 'wave' by Fourier phase at this period (phase-encoded designs) "
        "instead of complex-PCA phase",
    )
    parser.add_argument(
        "-max_voxels", type=int, metavar="N", help="randomly subsample the mask to N voxels"
    )
    parser.add_argument(
        "-size", default="640x360", metavar="WxH", help="simulation pixels (default 640x360)"
    )
    parser.add_argument(
        "-set",
        nargs="+",
        default=[],
        metavar="NAME=VALUE",
        help=f"starting parameter values; names: {params}. choice parameters "
        "(scene, color, inject) also take their names",
    )
    parser.add_argument("-no_sound", action="store_true", help="start without audio")
    parser.add_argument(
        "-movie", metavar="PATH", help="render an mp4 headless instead of opening a window"
    )
    parser.add_argument(
        "-duration",
        type=float,
        metavar="S",
        help="movie length in seconds (default: one pass through the run)",
    )
    parser.add_argument(
        "-start",
        type=float,
        default=0.0,
        metavar="TR",
        help="data time to start playing from (TRs)",
    )
    parser.add_argument("-fps", type=int, default=30, help="movie frame rate (default 30)")
    parser.add_argument("-seed", type=int, default=0, help="random seed (ICA, particle jitter)")
    add_device_arg(parser)
    return parser


def parse_sets(items: list[str]) -> dict[str, float]:
    from fastfuncstuff.viz.dataviz.engine import PARAM_BY_NAME

    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"ffs_dataviz: -set expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        prm = PARAM_BY_NAME.get(name)
        if prm is None:
            raise SystemExit(f"ffs_dataviz: unknown parameter {name!r}")
        if prm.choices and value in prm.choices:
            out[name] = float(prm.choices.index(value))
        else:
            try:
                out[name] = float(value)
            except ValueError:
                raise SystemExit(
                    f"ffs_dataviz: {name} wants a number{' or one of ' + ', '.join(prm.choices) if prm.choices else ''}"
                ) from None
    return out


def render_movie(
    feats,
    path: str,
    *,
    size: tuple[int, int],
    params: dict,
    duration: float | None,
    fps: int,
    sound: bool,
    start: float = 0.0,
) -> None:
    import numpy as np
    from tqdm.auto import tqdm

    from fastfuncstuff.viz.dataviz.engine import Visualizer
    from fastfuncstuff.viz.dataviz.sound import render_track
    from fastfuncstuff.viz.encode import find_ffmpeg

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise SystemExit("ffs_dataviz: -movie needs ffmpeg on PATH")
    sw = Visualizer(feats, *size)
    for k, v in params.items():
        sw.set_param(k, v)
    sw.seek(start)
    speed = sw.p["speed"]
    if duration is None:
        duration = (feats.n_time - 1) / speed
    n_frames = int(round(duration * fps))
    w, h = size
    with tempfile.TemporaryDirectory() as tmp:
        video = str(Path(tmp) / "video.mp4")
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
            "-vf",
            "scale=iw*2:ih*2:flags=neighbor",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            video,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for _ in tqdm(range(n_frames), desc="frames", leave=True, disable=n_frames < 30):
            sw.step(1.0 / fps)
            proc.stdin.write(np.ascontiguousarray(sw.render()).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffs_dataviz: ffmpeg failed writing video")
        if not sound:
            Path(video).replace(path)
            return
        import scipy.io.wavfile

        audio = render_track(feats, n_frames, fps, speed, start=start)
        wav = str(Path(tmp) / "audio.wav")
        scipy.io.wavfile.write(wav, 44100, (audio * 32767).astype(np.int16))
        mux = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            video,
            "-i",
            wav,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            path,
        ]
        if subprocess.run(mux).returncode != 0:
            raise SystemExit("ffs_dataviz: ffmpeg failed muxing audio")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"ffs_dataviz: -size wants WxH, got {args.size!r}") from None
    params = parse_sets(args.set)

    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.viz.dataviz.features import load_features

    device = setup_device(args.device)
    feats, info = load_features(
        args.input,
        device,
        mask_path=args.mask,
        tr=args.TR,
        max_voxels=args.max_voxels,
        seed=args.seed,
        n_components=args.ncomp,
        n_families=args.nfam,
        period_s=args.period,
        polort=args.polort,
    )
    print(
        f"ffs_dataviz: {feats.n_voxels} voxels x {feats.n_time} TRs (TR {feats.tr:g} s) on {device}"
    )

    if args.movie:
        render_movie(
            feats,
            args.movie,
            size=(w, h),
            params=params,
            duration=args.duration,
            start=args.start,
            fps=args.fps,
            sound=not args.no_sound,
        )
        print(f"ffs_dataviz: wrote {args.movie}")
        return 0

    try:
        from fastfuncstuff.viz.dataviz.app import launch
    except ImportError as exc:
        print(
            f"ffs_dataviz needs the GUI extra: pip install 'fastfuncstuff[viewer]'\n  ({exc})",
            file=sys.stderr,
        )
        return 1
    return launch(
        feats,
        " + ".join(info["names"]),
        size=(w, h),
        params=params,
        sound=not args.no_sound,
        start=args.start,
    )


if __name__ == "__main__":
    raise SystemExit(main())
