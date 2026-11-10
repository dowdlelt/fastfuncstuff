"""Quality-control figures and metrics, written by each tool under ``-qc DIR``.

Every tool that takes ``-qc`` writes two kinds of output into one folder:

- **PNGs that tell the story.** Each is self-explanatory: a header naming the
  tool, the run and the check, the headline numbers, and a line saying what a
  good result looks like. Names sort check-first, run-second, so scrolling the
  folder in any image viewer flips between runs on the same check.
- **A metrics record** (``records/*.json``): flat, named numbers with units and
  warning limits. ``ffs_qc_report`` turns one folder into a subject table and an
  HTML page that simply stitches the PNGs together; ``ffs_qc_table`` gathers many
  folders into one cross-subject table and flags outliers.

The modules split by what they import, so a report never pays for torch:
``record`` / ``report`` / ``table`` are numpy-only; ``render`` adds PIL;
``plots`` adds matplotlib (Agg, no pyplot).
"""
