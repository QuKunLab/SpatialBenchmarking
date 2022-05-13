import numba
import numpy as np
import multiprocessing as mp
from functools import partial

# Cite from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in numba.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges

# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = np.searchsorted(bin_edges, x)-1
    if bin < 0 or bin >= n:
        return None
    else:
        return bin

# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def numba_histogram(a, bin_edges):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.intp)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1
    return hist, bin_edges


# Modified from https://rdrr.io/bioc/scRecover/src/R/countsSampling.R
# Downsample cell reads to a fraction
@numba.jit(nopython=True, parallel=True)
def downsample_cell(cell_counts,fraction):
    n = np.floor(np.sum(cell_counts) * fraction)
    readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(n), replace=False))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

def downsample_matrix_by_cell(matrix,fraction,n_cpus=None):
    if n_cpus is not None:
        with mp.Pool(16) as p:
            matrix_ds = p.map(partial(downsample_cell,fraction=fraction), matrix)
    else:
        matrix_ds = [downsample_cell(i,fraction) for i in matrix]
    return np.array(matrix_ds)

# ps. slow speed.
def downsample_matrix_total(matrix,fraction):
    matrix_flat = matrix.reshape(-1)
    matrix_flat_ds = downsample_cell(matrix_flat,fraction)
    matrix_ds = matrix_flat_ds.reshape(matrix.shape)
    return matrix_ds

