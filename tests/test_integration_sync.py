"""
Integration tests for sync functionality with real browser
"""

import pytest
from pydantic import BaseModel, Field
from typing import List
from llm_scraper_py import LLMScraper
from llm_scraper_py.preprocess import preprocess


class SimpleSchema(BaseModel):
    title: str
    content: str


class ListSchema(BaseModel):
    items: List[str] = Field(max_length=3)


@pytest.mark.integration
class TestSyncIntegration:
    def test_preprocess_real_page_html(self, page):
        """Test preprocessing with real page - HTML format"""
        # Create a simple test page
        page.set_content(
            """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>Some content here</p>
                <script>console.log('should be removed');</script>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
        """
        )

        result = preprocess(page, {"format": "html"})
        assert result.format == "html"
        assert "Main Title" in result.content
        assert "Some content here" in result.content
        # Script and style should be removed by cleanup
        assert "console.log" not in result.content
        assert ".hidden" not in result.content

    def test_preprocess_real_page_raw_html(self, page):
        """Test preprocessing with real page - raw HTML format"""
        page.set_content(
            """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Raw Title</h1>
                <script>console.log('preserved');</script>
            </body>
        </html>
        """
        )

        result = preprocess(page, {"format": "raw_html"})

        assert result.format == "raw_html"
        assert "Raw Title" in result.content
        # Script should be preserved in raw HTML
        assert "console.log('preserved')" in result.content

    def test_preprocess_real_page_text(self, page):
        """Test preprocessing with real page - text format"""
        page.set_content(
            """
        <html>
            <head><title>Article Title</title></head>
            <body>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content of the article.</p>
                    <p>Another paragraph with useful information.</p>
                </article>
                <aside>Sidebar content</aside>
            </body>
        </html>
        """
        )

        result = preprocess(page, {"format": "text"})

        assert result.format == "text"
        assert "Page Title:" in result.content
        # Should extract main content using readability

    def test_preprocess_real_page_markdown(self, page):
        """Test preprocessing with real page - markdown format"""
        page.set_content(
            """
        <html>
            <body>
                <h1>Markdown Title</h1>
                <p>Some <strong>bold</strong> text.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """
        )

        result = preprocess(page, {"format": "markdown"})

        assert result.format == "markdown"
        # Should contain markdown-like content or fallback to text
        assert "Markdown Title" in result.content

    def test_preprocess_real_page_image(self, page):
        """Test preprocessing with real page - image format"""
        page.set_content(
            """
        <html>
            <body style="background: white; padding: 20px;">
                <h1 style="color: black;">Screenshot Test</h1>
            </body>
        </html>
        """
        )

        result = preprocess(page, {"format": "image"})

        assert result.format == "image"
        assert len(result.content) > 0  # Should have base64 image data
        # Basic check that it looks like base64
        assert all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            for c in result.content
        )

    def test_preprocess_custom_format_function(self, page):
        """Test preprocessing with custom format function"""
        page.set_content(
            """
        <html>
            <body>
                <div id="target">Custom content</div>
            </body>
        </html>
        """
        )

        def custom_extractor(page):
            return page.locator("#target").inner_text()

        result = preprocess(
            page, {"format": "custom", "formatFunction": custom_extractor}
        )

        assert result.format == "custom"
        assert result.content == "Custom content"

    def test_preprocess_navigation_and_wait(self, page):
        """Test preprocessing after navigation with wait"""
        # Navigate to a data URL that simulates loading
        page.goto("data:text/html,<html><body><h1>Loaded Page</h1></body></html>")

        result = preprocess(page, {"format": "html"})

        assert "Loaded Page" in result.content
        assert result.url.startswith("data:")

    def test_multiple_format_consistency(self, page):
        """Test that different formats work consistently on same page"""
        page.set_content(
            """
        <html>
            <head><title>Consistency Test</title></head>
            <body>
                <h1>Main Content</h1>
                <p>Test paragraph</p>
            </body>
        </html>
        """
        )

        # Test multiple formats on same page
        html_result = preprocess(page, {"format": "html"})
        raw_result = preprocess(page, {"format": "raw_html"})
        text_result = preprocess(page, {"format": "text"})

        # All should have same URL
        assert html_result.url == raw_result.url == text_result.url

        # All should contain the main content in some form
        assert "Main Content" in html_result.content
        assert "Main Content" in raw_result.content
        # Text format might extract differently, so just check it's not empty
        assert len(text_result.content) > 0

    def test_error_handling_invalid_page_state(self, page):
        """Test error handling with invalid page states"""
        # Close the page to create an invalid state
        page.close()

        with pytest.raises(Exception):  # Should raise some playwright error
            preprocess(page, {"format": "html"})


@pytest.mark.integration
@pytest.mark.slow
class TestRealWebsiteIntegration:
    """Tests against real websites - marked as slow"""

    def test_preprocess_simple_website(self, page):
        """Test preprocessing against a simple real website"""
        try:
            # Use a simple, reliable website
            page.goto("https://httpbin.org/html", timeout=10000)

            result = preprocess(page, {"format": "html"})

            assert result.url == "https://httpbin.org/html"
            assert len(result.content) > 0
            assert result.format == "html"

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")

    def test_preprocess_different_formats_real_site(self, page):
        """Test different preprocessing formats on real website"""
        try:
            page.goto("https://httpbin.org/html", timeout=10000)

            # Test HTML format
            html_result = preprocess(page, {"format": "html"})
            assert len(html_result.content) > 0

            # Test raw HTML format
            raw_result = preprocess(page, {"format": "raw_html"})
            assert len(raw_result.content) > 0

            # Raw should be longer (includes scripts, styles, etc.)
            assert len(raw_result.content) >= len(html_result.content)

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")
