# index.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, AsyncGenerator
from pydantic import BaseModel
from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage

from .preprocess import preprocess, apreprocess, PreProcessOptions, PreProcessResult
from .models import (
    LanguageModel,
    OpenAIModel,  # default adapter; you can swap this for others
    ScraperLLMOptions,
    ScraperGenerateOptions,
    generate_llm_object,
    generate_llm_code,
    agenerate_llm_code,
    agenerate_llm_object,
)

T = TypeVar("T")

SchemaType = Union[
    Type[BaseModel], Dict[str, Any]
]  # Pydantic model class or JSON Schema dict


class LLMScraper(Generic[T]):
    """
    Python port of your TypeScript LLMScraper.
    """

    def __init__(self, client: LanguageModel):
        self.client = client

    async def arun(
        self,
        page: AsyncPage,
        schema: SchemaType,
        options: Optional[
            ScraperLLMOptions | PreProcessOptions | Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and get a structured JSON object back.
        Schema can be a Pydantic model class or a JSON Schema dict.
        """
        opts = {} if options is None else dict(options)  # shallow copy
        pre: PreProcessResult = await apreprocess(page, opts or {"format": "html"})
        obj = await agenerate_llm_object(self.client, pre, schema, opts)
        return {"data": obj, "url": pre.url}

    def run(
        self,
        page: SyncPage,
        schema: SchemaType,
        options: Optional[
            ScraperLLMOptions | PreProcessOptions | Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and get a structured JSON object back.
        Schema can be a Pydantic model class or a JSON Schema dict.
        """
        opts = {} if options is None else dict(options)  # shallow copy
        pre: PreProcessResult = preprocess(page, opts or {"format": "html"})
        obj = generate_llm_object(self.client, pre, schema, opts)
        return {"data": obj, "url": pre.url}

    async def astream(
        self,
        page: AsyncPage,
        schema: SchemaType,
        options: Optional[
            ScraperLLMOptions | PreProcessOptions | Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and stream partial JSON (as dict deltas).
        Returns a dict with:
          - "stream": an async generator yielding partial dicts
          - "url": page url
        """
        raise NotImplementedError("stream is not implemented yet")

    def stream(
        self,
        page: SyncPage,
        schema: SchemaType,
        options: Optional[
            ScraperLLMOptions | PreProcessOptions | Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and stream partial JSON (as dict deltas).
        Returns a dict with:
          - "stream": an async generator yielding partial dicts
          - "url": page url
        """
        raise NotImplementedError("stream is not implemented yet")

    async def agenerate(
        self,
        page: AsyncPage,
        schema: SchemaType,
        options: Optional[ScraperGenerateOptions | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and ask the LLM to output runnable JavaScript (IIFE) that
        extracts the data per schema from the CURRENT DOM.
        """
        opts = {} if options is None else dict(options)
        pre: PreProcessResult = await apreprocess(page, opts or {"format": "html"})
        code = await agenerate_llm_code(self.client, pre, schema, opts)
        return {"code": code, "url": pre.url}

    def generate(
        self,
        page: SyncPage,
        schema: SchemaType,
        options: Optional[ScraperGenerateOptions | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process the page and ask the LLM to output runnable JavaScript (IIFE) that
        extracts the data per schema from the CURRENT DOM.
        """
        opts = {} if options is None else dict(options)
        pre: PreProcessResult = preprocess(page, opts or {"format": "html"})
        code = generate_llm_code(self.client, pre, schema, opts)
        return {"code": code, "url": pre.url}
