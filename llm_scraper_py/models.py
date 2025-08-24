# models.py
from __future__ import annotations

import os
import json
from json_repair import repair_json
from typing import Any, AsyncGenerator, Dict, Optional, Protocol, Type, Union

from pydantic import BaseModel
from pydantic import BaseModel, ValidationError as PydanticValidationError
from jsonschema import validate as jsonschema_validate, ValidationError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from .preprocess import PreProcessResult
from .playwright_js import DEFAULT_PROMPT, DEFAULT_CODE_PROMPT

# ----------------------------
# Options (kept close to your TS)
# ----------------------------


class ScraperLLMOptions(Dict[str, Any]):
    """
    Recognized keys:
      prompt: str
      temperature: float
      maxTokens: int
      topP: float
      mode: 'auto' | 'json' | 'tool'   (hinting only; adapter-dependent)
      output: 'array'                  (hinting only; adapter-dependent)
    """


class ScraperGenerateOptions(Dict[str, Any]):
    """
    Recognized keys:
      prompt: str
      temperature: float
      maxTokens: int
      topP: float
      format: 'html' | 'raw_html'
    """


# ----------------------------
# Schema helpers
# ----------------------------


def schema_dumps(schema: Union[Type[BaseModel], Dict[str, Any]]) -> Any:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.model_json_schema()
    else:
        schema_dict = schema
    return json.dumps(schema_dict)


def validate_against_schema(
    obj: Any, schema: Union[Type[BaseModel], Dict[str, Any]]
) -> Any:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(obj).model_dump()
    # JSON schema validation
    jsonschema_validate(obj, schema)  # raises ValidationError if mismatch
    return obj


def strip_markdown_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove opening fence with optional language
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


# ----------------------------
# Model provider protocol + OpenAI adapter
# ----------------------------


class LanguageModel(Protocol):
    async def agenerate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    async def astream_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]: ...

    async def agenerate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str: ...

    def generate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def stream_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]: ...

    def generate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str: ...


class OpenAIModel:
    """
    Minimal OpenAI adapter.
    - Uses Chat Completions with response_format='json_object' for JSON.
    - Streams JSON by accumulating text deltas and re-parsing to best-effort dict deltas.
    You can swap this out for Bedrock/Anthropic/etc by implementing LanguageModel.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        from openai import AsyncOpenAI, OpenAI

        self._aclient = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", None)
        )
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", None))
        self._model = model

    # ----------------------------
    # Sync LLM request functions
    # ----------------------------

    def generate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        client_settings = {
            "model": self._model,
            "input": messages,
            "temperature": temperature,
            "top_p": top_p,
            "text_format": schema,
        }

        try:
            return self._client.responses.parse(**client_settings).output_parsed
        except Exception:
            # Fallback: ask for JSON in the prompt and parse
            client_settings["input"] = messages + [
                {
                    "role": "system",
                    "content": "Return only valid minified JSON that strictly matches the provided schema.",
                }
            ]
            return self._client.responses.parse(**client_settings).output_parsed

    def stream_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Emits best-effort partial dicts as content streams in.
        """
        raise NotImplementedError("Streaming not implemented for OpenAI model")

    def generate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        client_settings = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        resp = self._client.chat.completions.create(**client_settings)
        return resp.choices[0].message.content or ""

    # ----------------------------
    # Async LLM request functions
    # ----------------------------

    async def agenerate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        client_settings = {
            "model": self._model,
            "input": messages,
            "temperature": temperature,
            "top_p": top_p,
            "text_format": schema,
        }

        try:
            return await self._aclient.responses.parse(**client_settings).output_parsed
        except Exception:
            # Fallback: ask for JSON in the prompt and parse
            client_settings["input"] = messages + [
                {
                    "role": "system",
                    "content": "Return only valid minified JSON that strictly matches the provided schema.",
                }
            ]
            return await self._aclient.responses.parse(**client_settings).output_parsed

    async def astream_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Emits best-effort partial dicts as content streams in.
        """
        raise NotImplementedError("Streaming not implemented for OpenAI model")

    async def agenerate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        client_settings = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        resp = await self._aclient.chat.completions.create(**client_settings)
        return resp.choices[0].message.content or ""


# ----------------------------
# Shared helpers for the three entry points
# ----------------------------


def _prepare_object_messages(
    pre: PreProcessResult, prompt: Optional[str]
) -> list[dict]:
    sys = {"role": "system", "content": prompt or DEFAULT_PROMPT}
    if pre.format == "image":
        # Send as inline data URL
        user = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"URL: {pre.url}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pre.content}"},
                },
            ],
        }
    else:
        user = {
            "role": "user",
            "content": f"Website: {pre.url}\n\nContent:\n{pre.content}",
        }
    return [sys, user]


def _prepare_code_messages(
    pre: PreProcessResult,
    prompt: Optional[str],
    schema: Union[type[BaseModel], Dict[str, Any]],
) -> list[dict]:
    sys = {"role": "system", "content": prompt or DEFAULT_CODE_PROMPT}
    user = {
        "role": "user",
        "content": f"""
            Website: {pre.url}
            Schema: {schema_dumps(schema)}
            Content:
            {pre.content}
        """,
    }
    return [sys, user]


# ----------------------------
# Sync functions
# ----------------------------


def generate_llm_object(
    model: LanguageModel,
    pre: PreProcessResult,
    schema: Union[type[BaseModel], Dict[str, Any]],
    options: Optional[ScraperLLMOptions] = None,
) -> Dict[str, Any]:
    opts = options or {}
    messages = _prepare_object_messages(pre, opts.get("prompt"))

    raw = model.generate_json(
        messages=messages,
        schema=schema,
        temperature=opts.get("temperature"),
        max_tokens=opts.get("maxTokens"),
        top_p=opts.get("topP"),
        mode=opts.get("mode"),
    )
    try:
        return validate_against_schema(raw, schema)
    except (PydanticValidationError, JSONSchemaValidationError) as e:
        raise ValueError(f"LLM output failed schema validation: {e}") from e


def generate_llm_code(
    model: LanguageModel,
    pre: PreProcessResult,
    schema: Union[type[BaseModel], Dict[str, Any]],
    options: Optional[ScraperGenerateOptions] = None,
) -> str:
    opts = options or {}
    messages = _prepare_code_messages(pre, opts.get("prompt"), schema)
    txt = model.generate_text(
        messages=messages,
        temperature=opts.get("temperature"),
        top_p=opts.get("topP"),
    )
    return strip_markdown_backticks(txt)


# ----------------------------
# Sync functions
# ----------------------------


async def agenerate_llm_object(
    model: LanguageModel,
    pre: PreProcessResult,
    schema: Union[type[BaseModel], Dict[str, Any]],
    options: Optional[ScraperLLMOptions] = None,
) -> Dict[str, Any]:
    opts = options or {}
    messages = _prepare_messages(pre, opts.get("prompt"))

    raw = await model.agenerate_json(
        messages=messages,
        schema=schema,
        temperature=opts.get("temperature"),
        max_tokens=opts.get("maxTokens"),
        top_p=opts.get("topP"),
        mode=opts.get("mode"),
    )
    try:
        return validate_against_schema(raw, schema)
    except (PydanticValidationError, JSONSchemaValidationError) as e:
        raise ValueError(f"LLM output failed schema validation: {e}") from e


async def agenerate_llm_code(
    model: LanguageModel,
    pre: PreProcessResult,
    schema: Union[type[BaseModel], Dict[str, Any]],
    options: Optional[ScraperGenerateOptions] = None,
) -> str:
    opts = options or {}
    messages = _prepare_code_messages(pre, opts.get("prompt"), schema)
    txt = await model.agenerate_text(
        messages=messages,
        temperature=opts.get("temperature"),
        top_p=opts.get("topP"),
    )
    return strip_markdown_backticks(txt)
