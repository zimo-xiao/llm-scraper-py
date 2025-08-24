# preprocess.py
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, TypedDict, Union, Awaitable

from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage
from .playwright_js import CLEANUP_JS, TO_MARKDOWN_JS, TO_READABILITY_TEXT_JS


Format = Literal["html", "text", "markdown", "raw_html", "image", "custom"]


class PreProcessOptions(TypedDict, total=False):
    format: Format
    # for custom:
    formatFunction: (
        Callable[[AsyncPage], Awaitable[str]]
        | Callable[[AsyncPage], str]
        | Callable[[SyncPage], str]
    )
    # for image:
    fullPage: bool


@dataclass
class PreProcessResult:
    url: str
    content: str
    format: Format


# ----------------------------
# Sync main functions
# ----------------------------


def _to_markdown(page: SyncPage) -> str:
    """
    Best-effort HTML -> Markdown in the browser (keeps dependency-light).
    If you prefer Python-side conversion, install `markdownify` and fetch innerHTML here.
    """
    body_html = page.inner_html("body")
    try:
        return page.evaluate(TO_MARKDOWN_JS, body_html)
    except Exception:
        return page.inner_text("body")


def _readability_text(page: SyncPage) -> str:
    """
    Use @mozilla/readability directly in the page context (mirrors your TS code).
    """
    readable = page.evaluate(TO_READABILITY_TEXT_JS)
    title = readable.get("title") or ""
    text = readable.get("text") or ""
    return f"Page Title: {title}\n{text}"


def preprocess(
    page: SyncPage, options: Optional[PreProcessOptions] = None
) -> PreProcessResult:
    opts: PreProcessOptions = {"format": "html"}
    if options:
        opts.update(options)

    fmt: Format = opts.get("format", "html")  # type: ignore
    url = page.url

    if fmt == "raw_html":
        content = page.content()

    elif fmt == "markdown":
        content = _to_markdown(page)

    elif fmt == "text":
        content = _readability_text(page)

    elif fmt == "html":
        page.evaluate(CLEANUP_JS)
        content = page.content()

    elif fmt == "image":
        full = bool(opts.get("fullPage", False))
        png_bytes = page.screenshot(full_page=full)
        content = base64.b64encode(png_bytes).decode("ascii")

    elif fmt == "custom":
        fn = opts.get("formatFunction")
        if not callable(fn):
            raise ValueError(
                "format='custom' requires a callable formatFunction(page) -> str"
            )
        content = fn(page)

    else:
        raise ValueError(f"Unknown format: {fmt}")

    return PreProcessResult(url=url, content=content, format=fmt)


# ----------------------------
# Async main functions
# ----------------------------


async def _ato_markdown(page: AsyncPage) -> str:
    """
    Best-effort HTML -> Markdown in the browser (keeps dependency-light).
    If you prefer Python-side conversion, install `markdownify` and fetch innerHTML here.
    """
    body_html = await page.inner_html("body")
    try:
        return await page.evaluate(TO_MARKDOWN_JS, body_html)
    except Exception:
        return await page.inner_text("body")


async def _areadability_text(page: AsyncPage) -> str:
    """
    Use @mozilla/readability directly in the page context (mirrors your TS code).
    """
    readable = await page.evaluate(TO_READABILITY_TEXT_JS)
    title = readable.get("title") or ""
    text = readable.get("text") or ""
    return f"Page Title: {title}\n{text}"


async def apreprocess(
    page: AsyncPage, options: Optional[PreProcessOptions] = None
) -> PreProcessResult:
    opts: PreProcessOptions = {"format": "html"}
    if options:
        opts.update(options)

    fmt: Format = opts.get("format", "html")  # type: ignore
    url = page.url

    if fmt == "raw_html":
        content = await page.content()

    elif fmt == "markdown":
        content = await _ato_markdown(page)

    elif fmt == "text":
        content = await _areadability_text(page)

    elif fmt == "html":
        await page.evaluate(CLEANUP_JS)
        content = await page.content()

    elif fmt == "image":
        full = bool(opts.get("fullPage", False))
        png_bytes = await page.screenshot(full_page=full)
        content = base64.b64encode(png_bytes).decode("ascii")

    elif fmt == "custom":
        fn = opts.get("formatFunction")
        if not callable(fn):
            raise ValueError(
                "format='custom' requires a callable formatFunction(page) -> str"
            )
        res = fn(page)
        content = (
            await res if hasattr(res, "__await__") else res
        )  # supports async or sync

    else:
        raise ValueError(f"Unknown format: {fmt}")

    return PreProcessResult(url=url, content=content, format=fmt)
