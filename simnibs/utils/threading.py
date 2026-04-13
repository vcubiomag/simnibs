from concurrent.futures import ThreadPoolExecutor
import functools
import multiprocessing as mp
import sys
from typing import Callable


def _default_pool_start_method() -> str:
    # On Linux we explicitly request "fork" because the pool call sites in
    # simnibs.simulation.fem rely on copy-on-write inheritance of module-global
    # solver state and of unittest.mock patches, and pass user-supplied
    # post_pro callables through initargs that may not pickle. Python 3.14
    # changed the Linux default from "fork" to "forkserver"; this restores the
    # pre-3.14 behavior. macOS (spawn since 3.8) and Windows (spawn-only) fall
    # through to the platform default.
    if sys.platform.startswith("linux"):
        return "fork"
    return mp.get_start_method()


def run_in_new_thread(fn):
    """Decorator that runs `fn` in a new thread."""

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(fn, *args, **kwargs).result()

    return wrapped_fn


@run_in_new_thread
def run_in_multiprocessing_pool(
    n_workers: int,
    fn: Callable,
    iterable,
    pool_kwargs: dict | None = None,
    start_method: str | None = None,
):
    """Submit an iterable to a pool of workers.  `fn` is executed as

            fn(iterable[0]), fn(iterable[1]), ...

    and the result returned as a list.

    Parameters
    ----------
    n_workers : int
        Number of workers.
    fn : Callable
        The function to call.
    iterable : _type_
        Iterable of arguments to `fn`.

    Returns
    -------
    result
        The concatenated result from running `fn`.
    """
    start_method = _default_pool_start_method() if start_method is None else start_method
    pool_kwargs = pool_kwargs or {}
    with mp.get_context(start_method).Pool(processes=n_workers, **pool_kwargs) as pool:
        result = pool.starmap_async(fn, iterable)
        pool.close()
        pool.join()
    return result.get()
