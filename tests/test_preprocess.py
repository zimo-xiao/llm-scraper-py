"""
Tests for preprocess.py - sync functionality only
"""

import pytest
import base64
from unittest.mock import Mock, patch
from llm_scraper_py.preprocess import (
    preprocess,
    PreProcessOptions,
    PreProcessResult,
    _to_markdown,
    _readability_text,
)


class TestPreprocessSync:
    def test_preprocess_html_format(self, mock_page):
        """Test preprocessing with HTML format"""
        mock_page.evaluate.return_value = None  # CLEANUP_JS returns None
        mock_page.content.return_value = "<html><body><h1>Clean HTML</h1></body></html>"

        result = preprocess(mock_page, {"format": "html"})

        assert isinstance(result, PreProcessResult)
        assert result.url == "https://example.com"
        assert result.content == "<html><body><h1>Clean HTML</h1></body></html>"
        assert result.format == "html"
        mock_page.evaluate.assert_called_once()  # CLEANUP_JS called

    def test_preprocess_raw_html_format(self, mock_page):
        """Test preprocessing with raw HTML format"""
        result = preprocess(mock_page, {"format": "raw_html"})

        assert result.format == "raw_html"
        assert result.content == "<html><body><h1>Test</h1></body></html>"
        mock_page.evaluate.assert_not_called()  # No cleanup for raw HTML

    def test_preprocess_markdown_format(self, mock_page):
        """Test preprocessing with markdown format"""
        mock_page.inner_html.return_value = "<h1>Test</h1>"
        mock_page.evaluate.return_value = "# Test"

        result = preprocess(mock_page, {"format": "markdown"})

        assert result.format == "markdown"
        assert result.content == "# Test"

    def test_preprocess_markdown_fallback(self, mock_page):
        """Test markdown preprocessing with fallback to inner_text"""
        mock_page.inner_html.return_value = "<h1>Test</h1>"
        mock_page.evaluate.side_effect = Exception("TurndownService not available")
        mock_page.inner_text.return_value = "Test"

        result = preprocess(mock_page, {"format": "markdown"})

        assert result.content == "Test"

    def test_preprocess_text_format(self, mock_page):
        """Test preprocessing with text format using readability"""
        mock_page.evaluate.return_value = {
            "title": "Test Title",
            "text": "Test content",
        }

        result = preprocess(mock_page, {"format": "text"})

        assert result.format == "text"
        assert result.content == "Page Title: Test Title\nTest content"

    def test_preprocess_image_format(self, mock_page):
        """Test preprocessing with image format"""
        fake_png = b"fake_png_data"
        mock_page.screenshot.return_value = fake_png

        result = preprocess(mock_page, {"format": "image"})

        assert result.format == "image"
        expected_b64 = base64.b64encode(fake_png).decode("ascii")
        assert result.content == expected_b64
        mock_page.screenshot.assert_called_once_with(full_page=False)

    def test_preprocess_image_format_full_page(self, mock_page):
        """Test preprocessing with image format and full page option"""
        fake_png = b"fake_png_data"
        mock_page.screenshot.return_value = fake_png

        result = preprocess(mock_page, {"format": "image", "fullPage": True})

        assert result.format == "image"
        mock_page.screenshot.assert_called_once_with(full_page=True)

    def test_preprocess_custom_format_sync_function(self, mock_page):
        """Test preprocessing with custom sync format function"""

        def custom_formatter(page):
            return "Custom formatted content"

        options = PreProcessOptions(format="custom", formatFunction=custom_formatter)
        result = preprocess(mock_page, options)

        assert result.format == "custom"
        assert result.content == "Custom formatted content"

    def test_preprocess_custom_format_missing_function(self, mock_page):
        """Test preprocessing with custom format but missing function"""
        options = PreProcessOptions(format="custom")

        with pytest.raises(
            ValueError, match="format='custom' requires a callable formatFunction"
        ):
            preprocess(mock_page, options)

    def test_preprocess_unknown_format(self, mock_page):
        """Test preprocessing with unknown format"""
        with pytest.raises(ValueError, match="Unknown format: unknown"):
            preprocess(mock_page, {"format": "unknown"})

    def test_preprocess_no_options(self, mock_page):
        """Test preprocessing with no options (defaults to HTML)"""
        mock_page.evaluate.return_value = None
        mock_page.content.return_value = "<html><body>Default</body></html>"

        result = preprocess(mock_page)

        assert result.format == "html"
        assert result.content == "<html><body>Default</body></html>"

    def test_preprocess_empty_options(self, mock_page):
        """Test preprocessing with empty options dict"""
        mock_page.evaluate.return_value = None
        mock_page.content.return_value = "<html><body>Empty opts</body></html>"

        result = preprocess(mock_page, {})

        assert result.format == "html"
        assert result.content == "<html><body>Empty opts</body></html>"


class TestMarkdownHelper:
    def test_to_markdown_with_turndown(self, mock_page):
        """Test markdown conversion with TurndownService available"""
        mock_page.inner_html.return_value = "<h1>Test</h1>"
        mock_page.evaluate.return_value = "# Test"

        result = _to_markdown(mock_page)

        assert result == "# Test"
        mock_page.inner_html.assert_called_once_with("body")

    def test_to_markdown_fallback(self, mock_page):
        """Test markdown conversion fallback to inner_text"""
        mock_page.inner_html.return_value = "<h1>Test</h1>"
        mock_page.evaluate.side_effect = Exception("No TurndownService")
        mock_page.inner_text.return_value = "Test"

        result = _to_markdown(mock_page)

        assert result == "Test"
        mock_page.inner_text.assert_called_once_with("body")


class TestReadabilityHelper:
    def test_readability_text_with_title_and_content(self, mock_page):
        """Test readability text extraction with title and content"""
        mock_page.evaluate.return_value = {
            "title": "Article Title",
            "text": "Article content here",
        }

        result = _readability_text(mock_page)

        assert result == "Page Title: Article Title\nArticle content here"

    def test_readability_text_empty_response(self, mock_page):
        """Test readability text extraction with empty response"""
        mock_page.evaluate.return_value = {}

        result = _readability_text(mock_page)

        assert result == "Page Title: \n"

    def test_readability_text_partial_response(self, mock_page):
        """Test readability text extraction with partial response"""
        mock_page.evaluate.return_value = {"title": "Only Title"}

        result = _readability_text(mock_page)

        assert result == "Page Title: Only Title\n"


class TestPreProcessOptions:
    def test_preprocess_options_type_hints(self):
        """Test that PreProcessOptions accepts expected keys"""
        # This is more of a documentation test for the TypedDict
        options = PreProcessOptions(
            format="html", fullPage=True, formatFunction=lambda x: "test"
        )

        assert options["format"] == "html"
        assert options["fullPage"] is True
        assert callable(options["formatFunction"])

    def test_preprocess_result_dataclass(self):
        """Test PreProcessResult dataclass"""
        result = PreProcessResult(
            url="https://test.com", content="test content", format="html"
        )

        assert result.url == "https://test.com"
        assert result.content == "test content"
        assert result.format == "html"
