import asyncio
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from typing import List
from llm_scraper_py import LLMScraper, OpenAIModel


# Define the data structure using Pydantic
class Story(BaseModel):
    title: str
    points: int
    by: str
    comments_url: str = Field(alias="commentsURL")


class HackerNewsData(BaseModel):
    top: List[Story] = Field(max_length=5, description="Top 5 stories on Hacker News")


async def main():
    # Launch browser
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # Initialize LLM
        llm = OpenAIModel(model="gpt-4o")

        # Create scraper
        scraper = LLMScraper(llm)

        # Open page
        page = await browser.new_page()
        await page.goto("https://news.ycombinator.com")

        # Run the scraper
        result = await scraper.arun(page, HackerNewsData, {"format": "html"})

        # Show results
        print("Top Stories:")
        for story in result["data"]["top"]:
            print(f"- {story['title']} ({story['points']} points by {story['by']})")

        await browser.close()


# Run the example
asyncio.run(main())
