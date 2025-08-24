# LLM Scraper (Python)

LLM Scraper Python is a Python library that allows you to extract structured data from **any** webpage using LLMs. This is a Python port of the popular [TypeScript LLM Scraper](https://github.com/mishushakov/llm-scraper).

> [!IMPORTANT]
> This is a Python implementation of the original TypeScript LLM Scraper library, providing the same powerful functionality with Python-native APIs and both sync/async support.

> [!TIP]
> Under the hood, it uses structured output generation to convert pages to structured data. You can find more about this approach [here](https://til.simonwillison.net/gpt3/openai-python-functions-data-extraction).

### Features

- **Dual API Support**: Both synchronous and asynchronous operations
- **OpenAI Integration**: Built-in support for OpenAI GPT models with structured outputs
- **Extensible**: Protocol-based design allows custom LLM providers
- **Schema Flexibility**: Supports both Pydantic models and JSON Schema
- **Type Safety**: Full type-safety with Python type hints
- **Playwright Integration**: Built on the robust Playwright framework
- **Multiple Formats**: 6 content processing modes including image support
- **Code Generation**: Generate reusable JavaScript extraction code
- **Error Handling**: Comprehensive validation and error reporting

**Supported Content Formats:**

- `html` - Pre-processed HTML (cleaned, scripts/styles removed)
- `raw_html` - Raw HTML (no processing)
- `markdown` - HTML converted to markdown
- `text` - Extracted readable text (using Readability.js)
- `image` - Page screenshot for multi-modal models
- `custom` - User-defined extraction function

**Make sure to give the original project a star! ⭐️**

## Getting Started

### Installation

1. Install the package and dependencies:

```bash
pip install llm_scraper_py
```

2. Install Playwright browsers:

```bash
playwright install
```

### Quick Setup

```python
import os
from llm_scraper_py import LLMScraper, OpenAIModel

# Initialize OpenAI model
llm = OpenAIModel(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")  # or pass directly
)

# Create scraper instance
scraper = LLMScraper(llm)
```

## Examples

### Async Example (Recommended)

Extract top stories from Hacker News using async/await:

```python
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
    top: List[Story] = Field(
        max_length=5,
        description="Top 5 stories on Hacker News"
    )

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Initialize LLM and scraper
        llm = OpenAIModel(model="gpt-4o")
        scraper = LLMScraper(llm)

        # Navigate and scrape
        await page.goto("https://news.ycombinator.com")
        result = await scraper.arun(page, HackerNewsData, {"format": "html"})

        # Display results
        print("Top Stories:")
        for story in result["data"]["top"]:
            print(f"- {story['title']} ({story['points']} points by {story['by']})")

        await browser.close()

asyncio.run(main())
```

### Sync Example

For simpler use cases, use the synchronous API:

```python
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from llm_scraper_py import LLMScraper, OpenAIModel

class ArticleData(BaseModel):
    title: str
    content: str
    author: str = None

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Initialize LLM and scraper
        llm = OpenAIModel(model="gpt-4o-mini")
        scraper = LLMScraper(llm)

        # Navigate and scrape
        page.goto("https://example-blog.com/article")
        result = scraper.run(page, ArticleData, {"format": "text"})

        print(f"Title: {result['data']['title']}")
        print(f"Author: {result['data']['author']}")

        browser.close()

main()
```

## Schema Options

### Using JSON Schema

You can use JSON Schema instead of Pydantic models:

```python
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price": {"type": "number"},
        "availability": {"type": "boolean"},
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5
        }
    },
    "required": ["title", "price"]
}

# Works with both sync and async
result = await scraper.arun(page, schema, {"format": "html"})
# or
result = scraper.run(page, schema, {"format": "html"})
```

### Pydantic Models (Recommended)

Pydantic provides better type safety and validation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Product(BaseModel):
    title: str
    price: float = Field(gt=0, description="Price must be positive")
    availability: bool
    features: List[str] = Field(max_length=5)
    description: Optional[str] = None

result = await scraper.arun(page, Product)
```

## Content Format Options

The scraper supports different content processing formats:

```python
# HTML (default) - cleaned HTML with scripts/styles removed
result = await scraper.arun(page, schema, {"format": "html"})

# Raw HTML - unprocessed HTML content
result = await scraper.arun(page, schema, {"format": "raw_html"})

# Markdown - HTML converted to markdown format
result = await scraper.arun(page, schema, {"format": "markdown"})

# Text - extracted readable text using Readability.js
result = await scraper.arun(page, schema, {"format": "text"})

# Image - page screenshot for multi-modal models
result = await scraper.arun(page, schema, {"format": "image"})

# Custom - user-defined extraction function
def extract_custom_data(page):
    return page.locator(".main-content").inner_text()

result = await scraper.arun(page, schema, {
    "format": "custom",
    "formatFunction": extract_custom_data
})
```

## Code Generation

Generate reusable JavaScript code for data extraction:

```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float
    rating: float

# Generate extraction code (async)
result = await scraper.agenerate(page, ProductInfo)
generated_code = result["code"]

# Or synchronous
result = scraper.generate(page, ProductInfo)
generated_code = result["code"]

# Execute the generated code on any similar page
extracted_data = await page.evaluate(generated_code)

# Validate and use the data
product = ProductInfo.model_validate(extracted_data)
print(f"Product: {product.name}, Price: ${product.price}")
```

The generated code is a self-contained JavaScript function that can be reused across similar pages without additional LLM calls.

## Advanced Configuration

### LLM Options

Customize the LLM behavior with detailed options:

```python
from llm_scraper_py import ScraperLLMOptions

options = ScraperLLMOptions(
    format="html",
    prompt="Extract the data carefully and accurately",
    temperature=0.1,        # Lower = more deterministic
    maxTokens=2000,         # Response length limit
    topP=0.9,              # Nucleus sampling
    mode="json"            # Response format hint
)

result = await scraper.arun(page, schema, options)
```

### Generation Options

For code generation, use specialized options:

```python
from llm_scraper_py import ScraperGenerateOptions

gen_options = ScraperGenerateOptions(
    format="html",
    prompt="Generate efficient extraction code",
    temperature=0.2
)

result = await scraper.agenerate(page, schema, gen_options)
```

## Error Handling

The library provides comprehensive error handling:

```python
from llm_scraper_py import LLMScraper, OpenAIModel
from pydantic import ValidationError
from playwright.async_api import TimeoutError

try:
    llm = OpenAIModel(model="gpt-4o")
    scraper = LLMScraper(llm)

    result = await scraper.arun(page, schema, {"format": "html"})

except ValidationError as e:
    print(f"Schema validation failed: {e}")
except TimeoutError:
    print("Page load timeout")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Custom LLM Providers

Implement custom LLM providers using the `LanguageModel` protocol:

```python
from llm_scraper_py import LanguageModel, LLMScraper
from typing import Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel

class CustomLLMProvider:
    """Example custom LLM provider implementation"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    # Sync methods
    def generate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Implement your JSON generation logic
        # Return structured data matching the schema
        pass

    def generate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        # Implement your text generation logic
        pass

    # Async methods
    async def agenerate_json(
        self,
        messages: list[dict],
        schema: BaseModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Async version of generate_json
        pass

    async def agenerate_text(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        # Async version of generate_text
        pass

    # Streaming methods (optional)
    def stream_json(self, *args, **kwargs):
        raise NotImplementedError("Streaming not supported")

    async def astream_json(self, *args, **kwargs):
        raise NotImplementedError("Streaming not supported")

# Use your custom provider
custom_llm = CustomLLMProvider(api_key="your-key", base_url="https://api.example.com")
scraper = LLMScraper(custom_llm)
```

## API Reference

### LLMScraper Methods

**Async Methods (Recommended):**

- `arun(page, schema, options=None)` - Extract structured data asynchronously
- `agenerate(page, schema, options=None)` - Generate extraction code asynchronously
- `astream(page, schema, options=None)` - Stream partial results (not implemented)

**Sync Methods:**

- `run(page, schema, options=None)` - Extract structured data synchronously
- `generate(page, schema, options=None)` - Generate extraction code synchronously
- `stream(page, schema, options=None)` - Stream partial results (not implemented)

### Response Format

All extraction methods return a dictionary with:

```python
{
    "data": {...},      # Extracted data matching your schema
    "url": "https://..." # Source page URL
}
```

Generation methods return:

```python
{
    "code": "...",      # Generated JavaScript code
    "url": "https://..." # Source page URL
}
```

## Installation & Dependencies

```bash
pip install llm_scraper_py
```

**Core Dependencies:**

- `playwright` - Web automation and browser control
- `pydantic` - Data validation and serialization
- `openai` - OpenAI API client (for built-in OpenAI support)
- `jsonschema` - JSON Schema validation

## Sync vs Async Usage

### When to Use Async (Recommended)

Use async methods for:

- Better performance with multiple concurrent scraping tasks
- Integration with async web frameworks (FastAPI, aiohttp)
- Non-blocking operations in async applications

```python
import asyncio
from playwright.async_api import async_playwright

async def scrape_multiple_pages(urls):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        tasks = []

        for url in urls:
            page = await browser.new_page()
            await page.goto(url)
            task = scraper.arun(page, schema)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        await browser.close()
        return results
```

### When to Use Sync

Use sync methods for:

- Simple scripts and one-off tasks
- Integration with sync codebases
- Learning and prototyping

```python
from playwright.sync_api import sync_playwright

def scrape_single_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        result = scraper.run(page, schema)
        browser.close()
        return result
```

## Performance Tips

1. **Reuse browser instances** when scraping multiple pages
2. **Use async methods** for concurrent operations
3. **Choose appropriate content formats** - `text` is fastest, `image` is slowest
4. **Set reasonable token limits** to control costs and response times
5. **Use code generation** for repeated scraping of similar pages

## Contributing

We welcome contributions! This project is a Python port of the original [TypeScript LLM Scraper](https://github.com/mishushakov/llm-scraper) by [mishushakov](https://github.com/mishushakov).

**Ways to contribute:**

- Report bugs and request features via GitHub issues
- Submit pull requests for improvements
- Add support for new LLM providers
- Improve documentation and examples
- Write tests for edge cases

## License

This project follows the same license as the original LLM Scraper project.
