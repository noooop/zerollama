
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple, Callable, TypeVar
from concurrent.futures import Future
from functools import lru_cache, partial, wraps
from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from gevent.threadpool import ThreadPoolExecutor

T = TypeVar("T")


class ExecutorGeventBase(ExecutorBase):

    @abstractmethod
    def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()


def make_async(func: Callable[..., T]) -> Callable[..., Future[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args, **kwargs) -> Future:
        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(func, *args, **kwargs)
            return f.result()

    return _async_wrapper