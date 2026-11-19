"""Shared pytest environment guards for parallel worker stability."""

from __future__ import annotations

import os

# Cap BLAS/OpenMP/PyTorch thread pools per pytest-xdist worker to avoid thread
# oversubscription and excessive virtual memory commit on multi-core runs.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
