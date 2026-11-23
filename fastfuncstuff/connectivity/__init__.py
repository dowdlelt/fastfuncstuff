"""Functional connectivity: the series a connectivity measure sees, and the measures.

``prep`` removes nuisance per run on the uncensored TRs (one cleaner shared by ROI
series, voxel series and external seeds, so all of them live in the same residual
space). ``matrix`` turns cleaned series into correlation / partial / shrunk
matrices, ``maps`` correlates seeds with every voxel, ``edge`` builds edge time
series (co-fluctuation) and their root-sum-square amplitude, and ``netcc`` reads and
writes AFNI's ``.netcc`` text layout so results drop into the FATCAT tools.

See ``../fmri_wiki/concepts/Functional connectivity.md``.
"""
