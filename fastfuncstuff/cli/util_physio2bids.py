"""CLI: CMRR physio logs to BIDS ``_physio.tsv.gz``, matched to runs by acquisition clock.

Command: ffs_util_physio2bids (registered as entry point in pyproject.toml)

Usage:
    ffs_util_physio2bids -logs derivatives/sub-01/ -bids rawdata/sub-01/ -dry_run
    ffs_util_physio2bids -logs derivatives/sub-01/ -bids rawdata/sub-01/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser

_EPILOG = """\
Every *_Info.log under -logs is paired with the BOLD run under -bids whose
AcquisitionTime equals the log's first acquisition tick (CMRR ticks count
2.5 ms from midnight), within -tol seconds and with the same volume count.
Folder and run names are never used, so a renumbered or repeated run cannot
be mislabelled. Unmatched runs are listed, not guessed.

Writes next to each matched run, one file per channel (PULS and RESP are
sampled at different rates):
  <stem>_recording-cardiac_physio.tsv.gz / .json
  <stem>_recording-respiratory_physio.tsv.gz / .json
with SamplingFrequency, StartTime (relative to the first volume) and Columns.
Existing files are left alone unless -overwrite.
"""


def parse_args(argv: list[str] | None = None):
    p = FfsArgumentParser(
        prog="ffs_util_physio2bids",
        description="Convert CMRR physio logs (PULS/RESP/ECG) to BIDS _physio.tsv.gz.",
        epilog=_EPILOG,
    )
    p.add_argument("-logs", type=Path, required=True, help="Directory searched for *_Info.log.")
    p.add_argument(
        "-bids", type=Path, required=True, help="BIDS subject/session/dataset directory."
    )
    p.add_argument("-tol", type=float, default=0.5, help="Clock tolerance, seconds.")
    p.add_argument("-dry_run", action="store_true", help="Print the pairing, write nothing.")
    p.add_argument("-overwrite", action="store_true", help="Replace existing physio files.")
    return p.parse_args(argv)


def _bolds(root: Path) -> list[tuple[Path, dict, int]]:
    """One representative image per run: magnitude, first echo."""
    import nibabel as nib

    out = []
    for nii in sorted(root.rglob("*_bold.nii*")):
        name = nii.name
        if "part-phase" in name or ("echo-" in name and "echo-1_" not in name):
            continue
        side = Path(str(nii).split(".nii")[0] + ".json")
        if not side.exists():
            continue
        meta = json.loads(side.read_text())
        shape = nib.Nifti1Image.from_filename(str(nii)).shape
        out.append((nii, meta, shape[3] if len(shape) > 3 else 1))
    return out


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.io.physio_bids import find_cmrr_recordings, match_runs, write_bids_physio

    args = parse_args(argv)
    recs = find_cmrr_recordings(args.logs)
    bolds = _bolds(args.bids)
    print(
        f"{len(recs)} CMRR recordings under {args.logs}, {len(bolds)} BOLD runs under {args.bids}"
    )
    pairs = match_runs(recs, bolds, tol_s=args.tol)
    n_ok = n_written = 0
    for bold, rec, why in pairs:
        rel = bold.relative_to(args.bids)
        if rec is None:
            print(f"  --  {rel}: {why}")
            continue
        n_ok += 1
        chans = "+".join(rec.channels)
        print(f"  ok  {rel} <- {rec.info.parent.name}/{rec.info.name[:30]}… [{chans}] {why}")
        if not args.dry_run:
            try:
                n_written += len(write_bids_physio(rec, bold, overwrite=args.overwrite)) // 2
            except FileExistsError as e:
                print(f"      exists, skipped: {Path(str(e)).name}")
    used = {r.info for _, r, _ in pairs if r is not None}
    orphans = [r for r in recs if r.info not in used]
    print(f"{n_ok}/{len(pairs)} runs matched; {len(orphans)} recordings matched no run")
    if not args.dry_run:
        print(f"{n_written} channel files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
