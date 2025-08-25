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

from pydantic import BaseModel

T = TypeVar("T")


def dict_to_model_type(data: dict, model_class: Type[T]) -> T:
    """
    Convert a dict into a Pydantic model instance.

    Args:
        data (dict): The dictionary to convert
        model_class (Type[T]): A Pydantic model class (e.g. Unit, Project)

    Returns:
        T: An instance of the model_class
    """
    if not data or not model_class:
        return None
    if not issubclass(model_class, BaseModel):
        return data
    return model_class(**data)
