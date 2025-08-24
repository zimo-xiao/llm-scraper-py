"""
Pytest configuration and fixtures for llm-scraper-py tests
"""

import pytest
from unittest.mock import Mock, MagicMock
from playwright.sync_api import sync_playwright, Page
from llm_scraper_py import LLMScraper, OpenAIModel, LanguageModel
from llm_scraper_py.preprocess import PreProcessResult


@pytest.fixture
def mock_language_model():
    """Mock language model for testing"""
    mock = Mock(spec=LanguageModel)
    mock.generate_json.return_value = {"title": "Test Title", "content": "Test Content"}
    mock.generate_text.return_value = (
        "(() => { return { title: 'Test', content: 'Mock' }; })();"
    )
    return mock


@pytest.fixture
def mock_page():
    """Mock Playwright page for testing"""
    page = Mock(spec=Page)
    page.url = "https://example.com"
    page.content.return_value = "<html><body><h1>Test</h1></body></html>"
    page.inner_html.return_value = "<h1>Test</h1>"
    page.inner_text.return_value = "Test"
    page.evaluate.return_value = "Test content"
    page.screenshot.return_value = b"fake_png_data"
    return page


@pytest.fixture
def scraper(mock_language_model):
    """LLMScraper instance with mock model"""
    return LLMScraper(mock_language_model)


@pytest.fixture
def sample_preprocess_result():
    """Sample PreProcessResult for testing"""
    return PreProcessResult(
        url="https://example.com",
        content="<html><body><h1>Test</h1></body></html>",
        format="html",
    )


@pytest.fixture(scope="session")
def browser():
    """Real browser instance for integration tests"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    """Real page instance for integration tests"""
    page = browser.new_page()
    yield page
    page.close()
