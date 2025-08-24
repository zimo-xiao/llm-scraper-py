from playwright.sync_api import sync_playwright
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


def main():
    # Launch browser
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # Initialize LLM
        llm = OpenAIModel(model="gpt-4o")

        # Create scraper
        scraper = LLMScraper(llm)

        # Open page
        page = browser.new_page()
        page.goto("https://news.ycombinator.com")

        # Run the scraper (sync wrapper)
        result = scraper.run(page, HackerNewsData, {"format": "html"})

        # Show results
        print("Top Stories:")
        for story in result["data"]["top"]:
            print(f"- {story['title']} ({story['points']} points by {story['by']})")

        browser.close()


if __name__ == "__main__":
    main()
