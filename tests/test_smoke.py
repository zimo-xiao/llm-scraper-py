"""
Smoke tests to verify basic functionality and imports work
"""

import pytest
from unittest.mock import Mock, patch, ANY


class TestImports:
    """Test that all main components can be imported"""

    def test_import_main_classes(self):
        """Test importing main classes"""
        from llm_scraper_py import LLMScraper, OpenAIModel

        assert LLMScraper is not None
        assert OpenAIModel is not None

    def test_import_options(self):
        """Test importing option classes"""
        from llm_scraper_py import ScraperLLMOptions, ScraperGenerateOptions

        assert ScraperLLMOptions is not None
        assert ScraperGenerateOptions is not None

    def test_import_language_model_protocol(self):
        """Test importing LanguageModel protocol"""
        from llm_scraper_py import LanguageModel

        assert LanguageModel is not None

    def test_import_preprocess_functions(self):
        """Test importing preprocess functions"""
        from llm_scraper_py.preprocess import (
            preprocess,
            PreProcessResult,
            PreProcessOptions,
        )

        assert preprocess is not None
        assert PreProcessResult is not None
        assert PreProcessOptions is not None

    def test_import_model_functions(self):
        """Test importing model utility functions"""
        from llm_scraper_py.models import (
            generate_llm_object,
            generate_llm_code,
            schema_dumps,
            validate_against_schema,
        )

        assert generate_llm_object is not None
        assert generate_llm_code is not None
        assert schema_dumps is not None
        assert validate_against_schema is not None

    def test_import_javascript_constants(self):
        """Test importing JavaScript constants"""
        from llm_scraper_py.playwright_js import (
            CLEANUP_JS,
            TO_MARKDOWN_JS,
            TO_READABILITY_TEXT_JS,
            DEFAULT_PROMPT,
            DEFAULT_CODE_PROMPT,
        )

        assert isinstance(CLEANUP_JS, str)
        assert isinstance(TO_MARKDOWN_JS, str)
        assert isinstance(TO_READABILITY_TEXT_JS, str)
        assert isinstance(DEFAULT_PROMPT, str)
        assert isinstance(DEFAULT_CODE_PROMPT, str)


class TestBasicFunctionality:
    """Test basic functionality without external dependencies"""

    def test_llm_scraper_creation(self):
        """Test LLMScraper can be created with mock client"""
        from unittest.mock import Mock
        from llm_scraper_py import LLMScraper, LanguageModel

        mock_client = Mock(spec=LanguageModel)
        scraper = LLMScraper(mock_client)

        assert scraper.client == mock_client

    def test_openai_model_creation(self):
        """Test OpenAIModel can be created (without actual API calls)"""
        from unittest.mock import patch
        from llm_scraper_py import OpenAIModel

        model = OpenAIModel(model="gpt-4", api_key="test-key")
        assert model._model == "gpt-4"

    def test_schema_dumps_basic(self):
        """Test schema_dumps with simple dict"""
        from llm_scraper_py.models import schema_dumps

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = schema_dumps(schema)

        assert isinstance(result, str)
        assert "name" in result

    def test_preprocess_result_creation(self):
        """Test PreProcessResult can be created"""
        from llm_scraper_py.preprocess import PreProcessResult

        result = PreProcessResult(
            url="https://example.com", content="<html>test</html>", format="html"
        )

        assert result.url == "https://example.com"
        assert result.content == "<html>test</html>"
        assert result.format == "html"

    def test_options_creation(self):
        """Test that option classes can be created"""
        from llm_scraper_py import ScraperLLMOptions, ScraperGenerateOptions

        llm_opts = ScraperLLMOptions(temperature=0.5, prompt="test")
        gen_opts = ScraperGenerateOptions(format="html")

        assert llm_opts["temperature"] == 0.5
        assert gen_opts["format"] == "html"


class TestPydanticIntegration:
    """Test Pydantic integration works correctly"""

    def test_pydantic_model_creation(self):
        """Test creating and validating Pydantic models"""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        # Valid data
        model = TestModel(name="test", age=25)
        assert model.name == "test"
        assert model.age == 25

        # Invalid data should raise error
        with pytest.raises(Exception):  # ValidationError or similar
            TestModel(name="test")  # missing age

    def test_schema_dumps_with_pydantic(self):
        """Test schema_dumps works with Pydantic models"""
        from pydantic import BaseModel
        from llm_scraper_py.models import schema_dumps

        class TestModel(BaseModel):
            title: str
            count: int

        result = schema_dumps(TestModel)
        assert isinstance(result, str)
        assert "title" in result
        assert "count" in result

    def test_validate_against_schema_pydantic(self):
        """Test validate_against_schema with Pydantic model"""
        from pydantic import BaseModel
        from llm_scraper_py.models import validate_against_schema

        class TestModel(BaseModel):
            name: str

        data = {"name": "test"}
        result = validate_against_schema(data, TestModel)
        assert result == data


class TestErrorHandling:
    """Test basic error handling"""

    def test_unknown_preprocess_format(self):
        """Test that unknown preprocess format raises error"""
        from unittest.mock import Mock
        from llm_scraper_py.preprocess import preprocess

        mock_page = Mock()
        mock_page.url = "https://example.com"

        with pytest.raises(ValueError, match="Unknown format"):
            preprocess(mock_page, {"format": "unknown_format"})

    def test_custom_format_without_function(self):
        """Test custom format without function raises error"""
        from unittest.mock import Mock
        from llm_scraper_py.preprocess import preprocess

        mock_page = Mock()
        mock_page.url = "https://example.com"

        with pytest.raises(ValueError, match="formatFunction"):
            preprocess(mock_page, {"format": "custom"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
