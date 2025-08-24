"""
Tests for LLMScraper class - sync functionality only
"""

import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel
from llm_scraper_py import LLMScraper, ScraperLLMOptions, ScraperGenerateOptions
from llm_scraper_py.preprocess import PreProcessResult


class TestDataSchema(BaseModel):
    title: str
    content: str


class TestLLMScraperSync:
    def test_init(self, mock_language_model):
        """Test LLMScraper initialization"""
        scraper = LLMScraper(mock_language_model)
        assert scraper.client == mock_language_model

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_with_pydantic_schema(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync run method with Pydantic schema"""
        # Setup mocks
        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate.return_value = {"title": "Test Title", "content": "Test Content"}

        scraper = LLMScraper(mock_language_model)
        result = scraper.run(mock_page, TestDataSchema)

        # Verify result structure
        assert "data" in result
        assert "url" in result
        assert result["data"] == {"title": "Test Title", "content": "Test Content"}
        assert result["url"] == "https://example.com"

        # Verify function calls
        mock_preprocess.assert_called_once_with(mock_page, {"format": "html"})
        mock_generate.assert_called_once_with(
            mock_language_model, mock_preprocess.return_value, TestDataSchema, {}
        )

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_with_dict_schema(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync run method with dict schema"""
        schema_dict = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
        }

        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate.return_value = {"title": "Test", "content": "Content"}

        scraper = LLMScraper(mock_language_model)
        result = scraper.run(mock_page, schema_dict)

        assert result["data"] == {"title": "Test", "content": "Content"}
        mock_generate.assert_called_once_with(
            mock_language_model, mock_preprocess.return_value, schema_dict, {}
        )

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_with_options(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync run method with options"""
        options = ScraperLLMOptions(
            format="markdown", temperature=0.5, prompt="Custom prompt"
        )

        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com", content="# Test", format="markdown"
        )
        mock_generate.return_value = {"title": "Test", "content": "Content"}

        scraper = LLMScraper(mock_language_model)
        result = scraper.run(mock_page, TestDataSchema, options)

        # Verify preprocess called with merged options
        expected_preprocess_options = {
            "format": "markdown",
            "temperature": 0.5,
            "prompt": "Custom prompt",
        }
        mock_preprocess.assert_called_once_with(mock_page, expected_preprocess_options)

        # Verify generate called with options
        mock_generate.assert_called_once_with(
            mock_language_model,
            mock_preprocess.return_value,
            TestDataSchema,
            expected_preprocess_options,
        )

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_no_options(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync run method with no options"""
        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate.return_value = {"title": "Test", "content": "Content"}

        scraper = LLMScraper(mock_language_model)
        result = scraper.run(mock_page, TestDataSchema, None)

        mock_preprocess.assert_called_once_with(mock_page, {"format": "html"})
        mock_generate.assert_called_once_with(
            mock_language_model, mock_preprocess.return_value, TestDataSchema, {}
        )

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_code")
    def test_generate_with_pydantic_schema(
        self, mock_generate_code, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync generate method with Pydantic schema"""
        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate_code.return_value = (
            "(() => { return {title: 'Test', content: 'Content'}; })();"
        )

        scraper = LLMScraper(mock_language_model)
        result = scraper.generate(mock_page, TestDataSchema)

        assert "code" in result
        assert "url" in result
        assert (
            result["code"]
            == "(() => { return {title: 'Test', content: 'Content'}; })();"
        )
        assert result["url"] == "https://example.com"

        mock_preprocess.assert_called_once_with(mock_page, {"format": "html"})
        mock_generate_code.assert_called_once_with(
            mock_language_model, mock_preprocess.return_value, TestDataSchema, {}
        )

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_code")
    def test_generate_with_options(
        self, mock_generate_code, mock_preprocess, mock_language_model, mock_page
    ):
        """Test sync generate method with options"""
        options = ScraperGenerateOptions(
            format="raw_html", temperature=0.3, prompt="Generate extraction code"
        )

        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Raw</body></html>",
            format="raw_html",
        )
        mock_generate_code.return_value = "(() => { return {}; })();"

        scraper = LLMScraper(mock_language_model)
        result = scraper.generate(mock_page, TestDataSchema, options)

        expected_options = {
            "format": "raw_html",
            "temperature": 0.3,
            "prompt": "Generate extraction code",
        }
        mock_preprocess.assert_called_once_with(mock_page, expected_options)
        mock_generate_code.assert_called_once_with(
            mock_language_model,
            mock_preprocess.return_value,
            TestDataSchema,
            expected_options,
        )

    def test_stream_not_implemented(self, mock_language_model, mock_page):
        """Test that sync stream method raises NotImplementedError"""
        scraper = LLMScraper(mock_language_model)

        with pytest.raises(NotImplementedError, match="stream is not implemented yet"):
            scraper.stream(mock_page, TestDataSchema)

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_options_shallow_copy(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test that options are shallow copied and don't mutate original"""
        original_options = {"format": "html", "temperature": 0.5}

        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate.return_value = {"title": "Test", "content": "Content"}

        scraper = LLMScraper(mock_language_model)
        scraper.run(mock_page, TestDataSchema, original_options)

        # Original options should be unchanged
        assert original_options == {"format": "html", "temperature": 0.5}

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_object")
    def test_run_error_propagation(
        self, mock_generate, mock_preprocess, mock_language_model, mock_page
    ):
        """Test that errors from underlying functions are propagated"""
        mock_preprocess.side_effect = ValueError("Preprocessing failed")

        scraper = LLMScraper(mock_language_model)

        with pytest.raises(ValueError, match="Preprocessing failed"):
            scraper.run(mock_page, TestDataSchema)

    @patch("llm_scraper_py.index.preprocess")
    @patch("llm_scraper_py.index.generate_llm_code")
    def test_generate_error_propagation(
        self, mock_generate_code, mock_preprocess, mock_language_model, mock_page
    ):
        """Test that errors from code generation are propagated"""
        mock_preprocess.return_value = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        mock_generate_code.side_effect = ValueError("Code generation failed")

        scraper = LLMScraper(mock_language_model)

        with pytest.raises(ValueError, match="Code generation failed"):
            scraper.generate(mock_page, TestDataSchema)
