import os
import asyncio
from typing import Annotated, List
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

from llm_scraper_py import LLMScraper, OpenAIModel


# ---- 1) Define the schema (Pydantic v2) ----
class Story(BaseModel):
    title: str
    points: int
    by: str
    commentsURL: str


class HNTop5(BaseModel):
    # exactly 5 items
    top: Annotated[List[Story], Field(min_length=5, max_length=5)]


# ---- 2) Main flow (generate code, run it in the page, validate) ----
async def main():
    # LLM client (reads OPENAI_API_KEY from env)
    llm = OpenAIModel(model="gpt-4o-mini")
    scraper = LLMScraper(llm)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://news.ycombinator.com", wait_until="domcontentloaded")

        # Ask the LLM to generate a JS IIFE that extracts data per schema
        gen = await scraper.agenerate(page, HNTop5, {"format": "raw_html"})
        code = gen["code"]
        print("Generated code:\n", code, "\n")

        # Execute the generated code in the page context
        result = await page.evaluate(code)

        # Validate/parse the result against the schema (v2)
        data = HNTop5.model_validate(result)

        print("Parsed result:")
        for i, s in enumerate(data.top, 1):
            print(
                f"{i:>2}. {s.title}  —  {s.points} points  —  {s.by}  —  {s.commentsURL}"
            )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
