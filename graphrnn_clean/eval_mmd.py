import concurrent.futures
from functools import partial

import numpy as np
from scipy.linalg import toeplitz


def _lazy_import_pyemd():
    try:
        import pyemd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pyemd import failed. Use --no-emd or install a lightweight pyemd build."
        ) from exc
    return pyemd


def emd(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    pyemd = _lazy_import_pyemd()
    return pyemd.emd(x, y, distance_mat)


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    pyemd = _lazy_import_pyemd()
    emd_value = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def _kernel_parallel_unpacked(x, samples2, kernel):
    return sum(kernel(x, s2) for s2 in samples2)


def _kernel_parallel_worker(t):
    return _kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
    d = 0.0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            tasks = [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]
            for dist in executor.map(_kernel_parallel_worker, tasks):
                d += dist
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, is_parallel=False, *args, **kwargs):
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    return (
        disc(samples1, samples1, kernel, is_parallel=is_parallel, *args, **kwargs)
        + disc(samples2, samples2, kernel, is_parallel=is_parallel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, is_parallel=is_parallel, *args, **kwargs)
    )
