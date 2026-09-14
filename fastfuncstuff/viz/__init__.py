"""Rendering helpers for diagnostic movies and slice figures.

- :mod:`.encode` -- frames to mp4/gif.
- :mod:`.slices` -- orientation-correct display planes through a 3-D grid.
- :mod:`.compose` -- greyscale panels, edge overlays and caption strips.
- :mod:`.warp_movie` -- record an optimizer's warp as it develops, render it later.

Submodules are imported explicitly; this package imports nothing on its own so the
CLI startup path stays torch-free.
"""
