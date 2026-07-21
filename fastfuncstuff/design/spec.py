"""
Design specification (``design.spec``) — a TOML file describing an fMRI GLM
design plus contrasts. Front-end for :mod:`fastfuncstuff.design.builder`,
consumed by :mod:`fastfuncstuff.cli.design_spec` and (eventually) ``ffs_reml``.

The spec is the single source of truth for: which runs, which events, which
HRF model per event, how to round onsets/durations, which nuisance regressors,
which contrasts (with wildcards, ALLOTHERS, balancing, F-tests).

Format reference: see ``../fmri_wiki/notes/design contrast rebuild.md``.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    bold: str
    events: str | None = None  # may be omitted when using -onsets vector form


@dataclass
class EventsColumns:
    onset: str = "onset"
    duration: str = "duration"
    trial_type: str = "trial_type"


@dataclass
class MetaSpec:
    runs: list[RunSpec]
    tr: float
    n_timepoints_per_run: list[int]
    polort: int | list[int] | Literal["auto"] = "auto"
    events_columns: EventsColumns = field(default_factory=EventsColumns)
    drop_trial_types: list[str] = field(
        default_factory=lambda: ["rest", "Rest", "REST", "baseline"]
    )
    censor: str | None = None  # path to outcount.1D-style keep mask (1=keep,0=cut)


@dataclass
class EventSpec:
    trial_type: str
    duration: float | Literal["from_events"] = "from_events"
    hrf: str = "SPMG1(0)"
    mode: Literal["condition", "im"] = "condition"
    round_onset: float | Literal["TR"] | None = None
    round_duration: float | None = None


@dataclass
class NuisanceSpec:
    """A single nuisance regressor block.

    Three modes, mirroring ``ffs_reml``'s ``-ortvec`` / ``-ortvec_run`` /
    ``-ortvec_glob``:

    - ``scope = "full"`` — ``file`` has one row per *concatenated* timepoint
      (sum of n_timepoints_per_run). Used as-is, no padding. This is the right
      mode for AFNI's ``mot_demean.r0N.1D`` files, which are already block-
      diagonal full-length.
    - ``scope = "run:N"`` — ``file`` covers exactly run N (1-indexed); other
      runs are zero-padded at compile time.
    - ``scope = "glob"`` — ``pattern`` matches one file per run (run index
      inferred from the filename; BIDS ``_run-NN_`` preferred). Each match is
      one-run length and gets zero-padded into its slot. ``file`` is ignored
      in this mode.
    """

    file: str | None = None  # required unless scope == "glob"
    label: str = ""
    scope: str = "full"  # "full" | "run:N" | "glob"
    pattern: str | None = None  # required when scope == "glob"
    rescale: str = "as-is"  # "as-is" | "demean"


@dataclass
class ContrastSpec:
    label: str
    sym: str | list[str]  # str = t-test, list = F-test rows
    balance: Literal["none", "sum1", "zero"] = "none"


@dataclass
class Spec:
    meta: MetaSpec
    events: list[EventSpec]
    nuisance: list[NuisanceSpec] = field(default_factory=list)
    contrasts: list[ContrastSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_spec(path: str | Path) -> Spec:
    """Parse a ``design.spec`` TOML file into a Spec object. Raises on
    missing required fields or unknown enum values."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    if "meta" not in raw:
        raise ValueError(f"{path}: missing [meta] section")
    meta_raw = raw["meta"]

    if "runs" not in meta_raw:
        raise ValueError(f"{path}: [meta].runs is required")
    runs = [RunSpec(**r) for r in meta_raw["runs"]]

    ec_raw = meta_raw.get("events_columns", {})
    events_columns = EventsColumns(**ec_raw) if ec_raw else EventsColumns()

    meta = MetaSpec(
        runs=runs,
        tr=float(meta_raw["tr"]),
        n_timepoints_per_run=[int(x) for x in meta_raw["n_timepoints_per_run"]],
        polort=meta_raw.get("polort", "auto"),
        events_columns=events_columns,
        drop_trial_types=list(
            meta_raw.get("drop_trial_types", ["rest", "Rest", "REST", "baseline"])
        ),
        censor=meta_raw.get("censor"),
    )

    events = [EventSpec(**e) for e in raw.get("events", [])]

    nuisance: list[NuisanceSpec] = []
    for n_raw in raw.get("nuisance", []):
        n = NuisanceSpec(**n_raw)
        if not n.label:
            raise ValueError(f"{path}: [[nuisance]] block missing 'label'")
        if n.scope == "glob":
            if not n.pattern:
                raise ValueError(f"{path}: nuisance '{n.label}' has scope='glob' but no pattern")
        else:
            if not n.file:
                raise ValueError(f"{path}: nuisance '{n.label}' (scope={n.scope!r}) needs 'file'")
            if n.scope != "full" and not n.scope.startswith("run:"):
                raise ValueError(
                    f"{path}: nuisance '{n.label}': scope must be "
                    f"'full', 'run:N', or 'glob' (got {n.scope!r})"
                )
        if n.rescale not in ("as-is", "demean"):
            raise ValueError(
                f"{path}: nuisance '{n.label}': rescale must be "
                f"'as-is' or 'demean' (got {n.rescale!r})"
            )
        nuisance.append(n)

    contrasts = [ContrastSpec(**c) for c in raw.get("contrasts", [])]

    # Validate enums
    for ev in events:
        if ev.mode not in ("condition", "im"):
            raise ValueError(f"event '{ev.trial_type}': mode must be 'condition' or 'im'")
    for c in contrasts:
        if c.balance not in ("none", "sum1", "zero"):
            raise ValueError(f"contrast '{c.label}': balance must be none|sum1|zero")

    return Spec(meta=meta, events=events, nuisance=nuisance, contrasts=contrasts)


# ---------------------------------------------------------------------------
# Write — hand-formatted so the file stays comment-friendly and diffable
# ---------------------------------------------------------------------------


def write_spec(
    spec: Spec,
    path: str | Path,
    *,
    header_comment: str = "",
    event_notes: dict[str, str] | None = None,
    include_contrast_examples: bool = False,
) -> None:
    """Write a Spec out as TOML with section comments to guide editing.

    Parameters
    ----------
    event_notes : dict, optional
        ``trial_type -> note`` strings rendered as comments above each
        ``[[events]]`` block. Intended for stub-time informational summaries
        (e.g. observed duration stats) that have no effect on compile.
    include_contrast_examples : bool
        If True, emit a few commented-out example contrasts after the
        ``[[contrasts]]`` header so the user can copy/edit instead of typing
        from scratch.
    """
    event_notes = event_notes or {}
    lines: list[str] = []
    if header_comment:
        for line in header_comment.splitlines():
            lines.append(f"# {line}")
        lines.append("")

    lines.append("# ===========================================================================")
    lines.append("# [meta] — dataset geometry and global options. Most values are populated")
    lines.append("# automatically at stub time from your -input/-events files; only edit if")
    lines.append("# you know the auto-detected value is wrong.")
    lines.append("# ===========================================================================")
    lines.append("[meta]")
    lines.append("")
    lines.append("# Repetition time in seconds. Read from the first -input NIfTI header at")
    lines.append("# stub time. Override with -TR if the header is wrong.")
    lines.append(f"tr = {spec.meta.tr}")
    lines.append("")
    lines.append("# Number of TRs in each run (4th dim of each -input file). Must match the")
    lines.append("# data you eventually feed to ffs_reml / ffs_deconvolve. Edit only if the")
    lines.append("# stub captured the wrong files.")
    lines.append(f"n_timepoints_per_run = {list(spec.meta.n_timepoints_per_run)}")
    lines.append("")
    lines.append("# Polynomial drift order per run. Auto-rule: 1 + floor(run_seconds / 150),")
    lines.append("# matching AFNI's 3dDeconvolve default. Accepts:")
    lines.append("#   int          — same order for every run")
    lines.append("#   list of int  — explicit per-run order (e.g. [2, 2, 3])")
    lines.append('#   "auto"       — recompute at compile time from tr * n_timepoints_per_run')
    lines.append(f"polort = {_toml_value(spec.meta.polort)}")
    lines.append("")
    lines.append("# trial_type values to silently drop from the events TSVs before modelling.")
    lines.append("# Edit to add anything you treat as implicit baseline (e.g. 'iti', 'null').")
    lines.append("drop_trial_types = " + _toml_value(spec.meta.drop_trial_types))
    if spec.meta.censor:
        lines.append("")
        lines.append(
            "# AFNI-style outcount.1D keep mask (one value per *full* TR, nonzero = keep)."
        )
        lines.append(
            "# Drives GoodList / NRowFull in the output xmat. Remove this line for no censoring."
        )
        lines.append(f'censor = "{spec.meta.censor}"')
    ec = spec.meta.events_columns
    lines.append("")
    lines.append("# Column names inside the events.tsv files. Override for non-BIDS data")
    lines.append("# (e.g. older AFNI exports that use 'onset_s', 'dur', 'condition').")
    lines.append(
        f'events_columns = {{ onset = "{ec.onset}", '
        f'duration = "{ec.duration}", trial_type = "{ec.trial_type}" }}'
    )
    lines.append("")
    lines.append("# One entry per imaging run. 'bold' is the 4D NIfTI; 'events' is the BIDS")
    lines.append("# events.tsv that goes with it. Order here MUST match the order in which")
    lines.append("# you concatenate runs downstream (run_starts are derived from this list).")
    lines.append("runs = [")
    for r in spec.meta.runs:
        if r.events is None:
            lines.append(f'  {{ bold = "{r.bold}" }},')
        else:
            lines.append(f'  {{ bold = "{r.bold}", events = "{r.events}" }},')
    lines.append("]")
    lines.append("")

    lines.append("# ===========================================================================")
    lines.append("# [[events]] — one entry per modelled trial_type. Order does not matter; the")
    lines.append("# resulting xmat columns are in the order events appear here.")
    lines.append("# ===========================================================================")
    lines.append("#")
    lines.append("# Fields:")
    lines.append("#")
    lines.append("# trial_type     Must match the value in the events.tsv trial_type column.")
    lines.append("#")
    lines.append("# duration       Event duration in seconds. Two forms:")
    lines.append("#                  <number>      — every event of this type uses this duration.")
    lines.append('#                  "from_events" — read each event\'s duration from the TSV.')
    lines.append("#                If from_events produces multiple distinct (rounded) durations")
    lines.append("#                and mode is 'condition', the regressor is split into one")
    lines.append("#                column per duration, labelled '{trial_type}_dur{d}'.")
    lines.append("#")
    lines.append("# hrf            HRF model to convolve the onsets with. Preferred form is the")
    lines.append("#                BARE model name — the duration above is injected automatically:")
    lines.append("#                  SPMG1, SPMG2, SPMG3   SPM canonical (1/2/3 basis functions)")
    lines.append("#                  BLOCK                 AFNI boxcar; emitted as BLOCK(d,1)")
    lines.append("#                  TENT(a,b,n)           N-tent FIR basis from lag a..b seconds")
    lines.append("#                  hrfopt:<lib.tsv>      Per-voxel HRF picked from an hrfopt")
    lines.append("#                                        library (ffs_reml -spec only;")
    lines.append("#                                        ffs_design_spec compile will refuse)")
    lines.append("#                You may also write an explicit form (SPMG1(5), BLOCK(20,1),")
    lines.append("#                TENT(0,16,9)) — that is treated as a user override and the")
    lines.append("#                duration field is ignored for this event.")
    lines.append("#")
    lines.append('# mode           "condition" — one regressor column for the whole condition (the')
    lines.append("#                              normal first-level GLM column).")
    lines.append('#                "im"        — one column per event (AFNI -stim_times_IM), for')
    lines.append("#                              single-trial / amplitude-modulation analyses.")
    lines.append("#")
    lines.append("# round_onset    Pre-convolution onset rounding (applied before grouping).")
    lines.append("#                  <number>  — round to this many decimal places (0 = integers)")
    lines.append('#                  "TR"      — snap to the nearest TR boundary')
    lines.append("#                  omitted   — no rounding")
    lines.append("#")
    lines.append("# round_duration Same form as round_onset, applied to event durations. Useful")
    lines.append("#                with from_events to collapse near-equal durations (e.g.")
    lines.append("#                10.03 and 10.06 → 10) so they share a regressor.")
    for ev in spec.events:
        lines.append("")
        note = event_notes.get(ev.trial_type)
        if note:
            for note_line in note.splitlines():
                lines.append(f"# {note_line}")
        lines.append("[[events]]")
        lines.append(f'trial_type = "{ev.trial_type}"')
        lines.append(f"duration = {_toml_value(ev.duration)}")
        lines.append(f'hrf = "{ev.hrf}"')
        lines.append(f'mode = "{ev.mode}"')
        if ev.round_onset is not None:
            lines.append(f"round_onset = {_toml_value(ev.round_onset)}")
        if ev.round_duration is not None:
            lines.append(f"round_duration = {_toml_value(ev.round_duration)}")

    lines.append("")
    lines.append("# ===========================================================================")
    lines.append("# [[nuisance]] — regressors of no interest (motion, physio, scrub spikes …).")
    lines.append("# Three input modes, harmonised with ffs_reml's -ortvec / -ortvec_run /")
    lines.append("# -ortvec_glob flags. All three end up in the same block-diagonal nuisance")
    lines.append("# slot in the design matrix.")
    lines.append("# ===========================================================================")
    lines.append("#")
    lines.append("# Fields:")
    lines.append("#")
    lines.append("# file     Path to a 1D file (one column per regressor, one row per TR).")
    lines.append('#          Ignored when scope = "glob"; required otherwise.')
    lines.append("#")
    lines.append("# label    Name written into the xmat column header. Multi-column files get")
    lines.append("#          '#0', '#1', … suffixes automatically.")
    lines.append("#")
    lines.append("# scope    How the file's rows map onto the concatenated run grid:")
    lines.append("#")
    lines.append('#          "full"   The file already spans all runs (sum(n_timepoints_per_run)')
    lines.append("#                    rows). Used as-is — NO padding is applied. This is the")
    lines.append("#                    right mode for AFNI's mot_demean.r0N.1D files, which are")
    lines.append("#                    already full-length and block-diagonal (zeros outside")
    lines.append("#                    their own run). Pass one [[nuisance]] block per file.")
    lines.append("#")
    lines.append('#          "run:N"  The file is exactly run N long (n_timepoints_per_run[N-1]')
    lines.append("#                    rows). The other runs are zero-padded at compile time, so")
    lines.append("#                    the regressor only acts inside run N. RUN is 1-indexed.")
    lines.append("#                    Use one [[nuisance]] block per run when you have separate")
    lines.append(
        "#                    *un-padded* per-run files. ffs_reml's -ortvec_run does this."
    )
    lines.append("#")
    lines.append('#          "glob"   The pattern field is a shell glob matching one file per')
    lines.append("#                    run. Each file must be one run long; the run index is")
    lines.append("#                    inferred from the filename (BIDS '_run-NN_' preferred,")
    lines.append("#                    falling back to trailing digits). Missing runs are")
    lines.append("#                    zero-padded. The convenience over many run:N blocks.")
    lines.append("#")
    lines.append('# pattern  Shell glob — only set this when scope = "glob".')
    lines.append("#")
    lines.append("# rescale  Per-column preprocessing applied before the regressor enters the")
    lines.append("#          design matrix.")
    lines.append('#            "as-is"  Use the file values verbatim. Default.')
    lines.append('#            "demean" Subtract the per-column mean. Useful for raw motion or')
    lines.append("#                     physio files that have a non-zero mean. WARNING: do not")
    lines.append("#                     use on already-block-diagonal full-length files (e.g.")
    lines.append("#                     AFNI mot_demean.r0N.1D) — subtracting the column mean")
    lines.append("#                     would propagate nonzero values into the zero rows of")
    lines.append("#                     the other runs and break the block-diagonal structure.")
    if not spec.nuisance:
        lines.append("#")
        lines.append("# (No nuisance regressors were passed to the stub. Add [[nuisance]] blocks")
        lines.append("#  below, or re-run stub with -ortvec / -ortvec_run / -ortvec_glob.)")
        lines.append("#")
        lines.append("# Examples:")
        lines.append("#")
        lines.append("# [[nuisance]]                        # full-length motion (sum of all runs)")
        lines.append('# file = "motion_demean.1D"')
        lines.append('# label = "motion"')
        lines.append('# scope = "full"')
        lines.append("#")
        lines.append("# [[nuisance]]                        # AFNI-style block-diagonal demean")
        lines.append('# file = "mot_demean.r01.1D"')
        lines.append('# label = "motion_r01"')
        lines.append('# scope = "full"')
        lines.append("#")
        lines.append("# [[nuisance]]                        # un-padded per-run file, zero-padded")
        lines.append('# file = "motion_run01.1D"')
        lines.append('# label = "motion"')
        lines.append('# scope = "run:1"')
        lines.append("#")
        lines.append("# [[nuisance]]                        # glob over per-run files")
        lines.append('# pattern = "motion_run-*.1D"')
        lines.append('# label = "motion"')
        lines.append('# scope = "glob"')
    for n in spec.nuisance:
        lines.append("")
        lines.append("[[nuisance]]")
        if n.scope == "glob":
            lines.append(f'pattern = "{n.pattern}"')
            lines.append(f'label = "{n.label}"')
            lines.append('scope = "glob"')
        else:
            lines.append(f'file = "{n.file}"')
            lines.append(f'label = "{n.label}"')
            lines.append(f'scope = "{n.scope}"')
        if n.rescale != "as-is":
            lines.append(f'rescale = "{n.rescale}"')

    lines.append("")
    # Echo the trial_types back as a copy-paste reference — these are exactly
    # the names that go inside the `sym` strings below.
    if spec.events:
        lines.append(
            "# ---------------------------------------------------------------------------"
        )
        lines.append("# Stim labels available for the `sym` strings below (one per [[events]]):")
        for ev in spec.events:
            lines.append(f"#   {ev.trial_type}")
        lines.append("#")
        lines.append("# Multi-column bases (SPMG2/3, TENT, FIR) and condition-mode 'from_events'")
        lines.append("# duration splits expand these into <label>[k] / <label>_durN at compile")
        lines.append("# time; address a single basis column with label[a..b].")
        lines.append(
            "# ---------------------------------------------------------------------------"
        )
    lines.append("# ===========================================================================")
    lines.append("# [[contrasts]] — symbolic linear contrasts evaluated against the resolved")
    lines.append("# stim labels (each [[events]] entry contributes one or more — multi-column")
    lines.append("# bases like SPMG2/3, TENT, and condition-mode 'from_events' splits all expand).")
    lines.append("# ===========================================================================")
    lines.append("#")
    lines.append("# Fields:")
    lines.append("#")
    lines.append("# label    Short name written into the output bucket sub-brick(s).")
    lines.append("#")
    lines.append("# sym      Either a SYM: string (t-test, one row) or an array of SYM: strings")
    lines.append("#          (F-test, one row per array element). Token grammar:")
    lines.append("#            +w*label         coefficient w on `label`")
    lines.append("#            -w*label         coefficient -w on `label`")
    lines.append("#            +w*label[a..b]   coefficient on basis cols a..b of a multi-column")
    lines.append("#                             stim (TENT/FIR/SPMG2/3). Inclusive, 0-indexed.")
    lines.append("#            +w**pattern      glob matching trial_types (fnmatch syntax,")
    lines.append("#                             case-sensitive). w is split as w/N across the N")
    lines.append("#                             matches, so '+1**_instruct' is the *average*.")
    lines.append("#            +w*ALLOTHERS     every label NOT named elsewhere in this row gets")
    lines.append("#                             share w/(N - k_named). Sign matters; balance below")
    lines.append("#                             can normalise the result.")
    lines.append("#          Bare 'ALLOTHERS' (no '*') is shorthand for '+1*ALLOTHERS'.")
    lines.append("#")
    lines.append("# balance  Post-resolution rescaling of the row weights:")
    lines.append('#            "none"  — leave weights as written.')
    lines.append('#            "sum1"  — divide every weight so they sum to 1 (an "average").')
    lines.append(
        '#            "zero"  — shift every weight by mean so they sum to 0 (a "contrast")'
    )
    lines.append("#                      — useful for clean one-vs-rest tests where you do not")
    lines.append("#                      want to hand-balance the signs.")
    if include_contrast_examples:
        lines.append("#")
        lines.append("# Examples — uncomment and edit (replace stimA/stimB with your trial_types):")
        lines.append("#")
        lines.append("# [[contrasts]]                       # t-test, classic A minus B")
        lines.append('# label = "stimA_vs_stimB"')
        lines.append('# sym = "SYM: +1*stimA -1*stimB"')
        lines.append('# balance = "none"')
        lines.append("#")
        lines.append("# [[contrasts]]                       # t-test, one-vs-rest (auto-balanced)")
        lines.append('# label = "stimA_vs_rest"')
        lines.append('# sym = "SYM: +1*stimA -ALLOTHERS"')
        lines.append('# balance = "zero"')
        lines.append("#")
        lines.append("# [[contrasts]]                       # F-test, any-of-N (multi-row sym)")
        lines.append('# label = "any_stim_F"')
        lines.append('# sym = ["SYM: +1*stimA", "SYM: +1*stimB"]')
        lines.append('# balance = "none"')
    for c in spec.contrasts:
        lines.append("")
        lines.append("[[contrasts]]")
        lines.append(f'label = "{c.label}"')
        if isinstance(c.sym, list):
            lines.append("sym = [")
            for row in c.sym:
                lines.append(f'  "{row}",')
            lines.append("]")
        else:
            lines.append(f'sym = "{c.sym}"')
        lines.append(f'balance = "{c.balance}"')

    Path(path).write_text("\n".join(lines) + "\n")


def _toml_value(v: object) -> str:
    """Minimal scalar/list/literal renderer for write_spec."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return repr(v)


# ---------------------------------------------------------------------------
# Contrast resolver — wildcards, ALLOTHERS, balance
# ---------------------------------------------------------------------------


# Token regex (mirrors design.builder.parse_glt_string but also matches '*'
# in label position for globs and the special token ALLOTHERS).
import re  # noqa: E402

_TOKEN_RE = re.compile(
    r"([+-]?\s*\d+\.?\d*)\s*\*\s*([A-Za-z_*][\w\-*]*)"
    r"(?:\[\s*(\d+)\s*\.\.\s*(\d+)\s*\])?"
)
_ALLOTHERS_RE = re.compile(r"([+-]?\s*\d*\.?\d*)\s*\*?\s*ALLOTHERS")


def resolve_contrast_row(
    sym: str,
    stim_labels: list[str],
    balance: str = "none",
) -> dict[str, tuple[float, tuple[int, int] | None]]:
    """
    Expand one symbolic contrast row into ``{label: (weight, range_or_None)}``
    suitable for :func:`fastfuncstuff.design.builder.glt_weights_to_vector`.

    Supports:
    - Plain ``+w*label`` / ``-w*label`` (label must exist in ``stim_labels``).
    - Sub-ranges ``label[a..b]`` (passed through unchanged).
    - Globs ``+w*pattern`` (e.g. ``+1**_instruct``) — distributes ``w/N`` over
      all ``N`` matching labels.
    - ``+w*ALLOTHERS`` / ``-w*ALLOTHERS`` / bare ``ALLOTHERS`` (default w=1) —
      expands to every label not named explicitly in this row, weight
      ``w/(N - k_explicit)``.
    - ``balance`` post-processing: ``"sum1"`` divides every weight by the sum;
      ``"zero"`` adds a constant so weights sum to 0.
    """
    s = sym.strip()
    if s.upper().startswith("SYM:"):
        s = s[4:].strip()

    # 1) Pull ALLOTHERS out first so it doesn't compete with the label regex.
    others_weight: float | None = None
    m = _ALLOTHERS_RE.search(s)
    if m:
        w_str = m.group(1).replace(" ", "")
        if w_str in ("", "+", "-"):
            others_weight = 1.0 if w_str != "-" else -1.0
        else:
            others_weight = float(w_str)
        s = (s[: m.start()] + s[m.end() :]).strip()

    # 2) Walk explicit tokens (including globs).
    explicit_labels: set[str] = set()
    resolved: dict[str, tuple[float, tuple[int, int] | None]] = {}

    for w_str, label, lo, hi in _TOKEN_RE.findall(s):
        weight = float(w_str.replace(" ", ""))
        rng = (int(lo), int(hi)) if lo and hi else None

        if "*" in label:
            matches = [lbl for lbl in stim_labels if fnmatch.fnmatchcase(lbl, label)]
            if not matches:
                raise ValueError(f"Glob '{label}' matched no stim labels. Available: {stim_labels}")
            if rng is not None:
                raise ValueError(f"Sub-range on a glob pattern '{label}{rng}' is not supported.")
            share = weight / len(matches)
            for lbl in matches:
                resolved[lbl] = (resolved.get(lbl, (0.0, None))[0] + share, None)
                explicit_labels.add(lbl)
        else:
            if label not in stim_labels:
                raise ValueError(f"Contrast label '{label}' not in stim labels: {stim_labels}")
            # Accumulate so '+1*A +1*A' sums (rare but well-defined).
            prev = resolved.get(label, (0.0, None))
            resolved[label] = (prev[0] + weight, rng)
            explicit_labels.add(label)

    # 3) Expand ALLOTHERS.
    if others_weight is not None:
        others = [lbl for lbl in stim_labels if lbl not in explicit_labels]
        if not others:
            raise ValueError("ALLOTHERS used but every stim label was already named explicitly.")
        share = others_weight / len(others)
        for lbl in others:
            resolved[lbl] = (share, None)

    if not resolved:
        raise ValueError(f"Contrast row '{sym}' resolved to no labels.")

    # 4) Balance.
    if balance == "sum1":
        total = sum(w for w, _ in resolved.values())
        if abs(total) < 1e-12:
            raise ValueError(
                f"balance='sum1' on row '{sym}' but weights sum to 0; cannot rescale to sum=1."
            )
        resolved = {k: (w / total, r) for k, (w, r) in resolved.items()}
    elif balance == "zero":
        total = sum(w for w, _ in resolved.values())
        shift = total / len(resolved)
        resolved = {k: (w - shift, r) for k, (w, r) in resolved.items()}
    elif balance != "none":
        raise ValueError(f"Unknown balance mode: {balance}")

    return resolved


def resolve_contrast(
    spec: ContrastSpec,
    stim_labels: list[str],
) -> list[dict[str, tuple[float, tuple[int, int] | None]]]:
    """Resolve a ContrastSpec into a list of resolved rows (1 = t-test, >1 = F)."""
    sym = spec.sym
    if isinstance(sym, list):
        return [resolve_contrast_row(row, stim_labels, spec.balance) for row in sym]
    return [resolve_contrast_row(sym, stim_labels, spec.balance)]
