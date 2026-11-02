"""ffs_dataviz — a data-driven particle visualiser for 4-D fMRI.

Every voxel becomes one or more particles. Where a particle lives, how it moves,
what colour it is and what it sounds like are all read off the data: anatomy and
correlation structure set the layout, the low-rank (PCA/ICA) structure sets who
flocks with whom, the live time series sets brightness and motion, and the
whole-brain signals (global mean, DVARS) tint and shake everything at once.

The core (``features``, ``swarm``, ``sound``) is headless torch/numpy, so a movie
renders without Qt; ``app`` is the interactive window.
"""
