import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import b2luigi

CUDA_FORK_ERROR_MSG = "Cannot re-initialize CUDA in forked subprocess"


class AIDOTask(b2luigi.Task):
    """ Shallow wrapper around b2luigi.Task
    """

    @property
    def htcondor_settings(self):
        return {
            "request_cpus": "1",
            "getenv": "true",
        }


def torch_safe_wrapper(
    func: Callable,
    *args,
    **kwargs,
):
    """
    b2luigi safe wrapper for calls to a torch function. Otherwise torch will raise
    'RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    To use CUDA with multiprocessing, you must use the 'spawn' start method'

    We avoid this by first calling the function 'func' and excepting any errors. If
    that Error is raised by CUDA, we catch it and call that function again but inside
    a subprocess. If any further errors are raised afterwards, they are return. In
    case of no errors, we return the result of the function.
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if CUDA_FORK_ERROR_MSG not in str(e):
            raise

        with ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result()
