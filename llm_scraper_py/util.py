import inspect
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    AsyncGenerator,
    Coroutine,
    Callable,
    Awaitable,
)

T = TypeVar("T")


def is_async_context():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def run_sync(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """
    Run an async coroutine from sync code with improved error reporting.
    Preserves original stack trace and adds caller context.
    """

    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    # Grab caller info for context
    caller = inspect.stack()[1]
    caller_info = f"{caller.filename}:{caller.lineno}"

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(coroutine)
        except Exception as e:
            raise RuntimeError(
                f"[run_sync] Coroutine {getattr(coroutine, '__name__', type(coroutine))} "
                f"failed in caller {caller_info}"
            ) from e

    try:
        if threading.current_thread() is threading.main_thread():
            if not loop.is_running():
                return loop.run_until_complete(coroutine)
            else:
                with ThreadPoolExecutor() as pool:
                    future = pool.submit(run_in_new_loop)
                    return future.result(timeout=timeout)
        else:
            return asyncio.run_coroutine_threadsafe(coroutine, loop).result(
                timeout=timeout
            )
    except Exception as e:
        # Add context but keep full traceback from original error
        raise RuntimeError(
            f"[run_sync] Coroutine {getattr(coroutine, '__name__', type(coroutine))} "
            f"failed in caller {caller_info}"
        ) from e


async def run_maybe_async(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Async wrapper that runs either a sync or async function.
    - If the function is sync, runs it directly.
    - If the function is async, awaits it.
    """
    result = func(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result
    return result
