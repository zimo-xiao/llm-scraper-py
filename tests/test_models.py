"""
Tests for models.py - sync functionality only
"""

import pytest
from unittest.mock import Mock, patch, ANY
from pydantic import BaseModel
from pydantic_core import ValidationError as PydanticValidationError
from jsonschema import ValidationError as JSONValidationError
from llm_scraper_py.models import (
    OpenAIModel,
    ScraperLLMOptions,
    ScraperGenerateOptions,
    schema_dumps,
    validate_against_schema,
    strip_markdown_backticks,
    generate_llm_object,
    generate_llm_code,
    _prepare_object_messages,
    _prepare_code_messages,
)
from llm_scraper_py.preprocess import PreProcessResult


class TestSchema(BaseModel):
    title: str
    content: str


class TestSchemaHelpers:
    def test_schema_dumps_pydantic_model(self):
        result = schema_dumps(TestSchema)
        assert isinstance(result, str)
        assert "title" in result
        assert "content" in result

    def test_schema_dumps_dict_schema(self):
        schema_dict = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
        }
        result = schema_dumps(schema_dict)
        assert isinstance(result, str)
        assert "title" in result

    def test_validate_against_schema_pydantic_valid(self):
        data = {"title": "Test", "content": "Content"}
        result = validate_against_schema(data, TestSchema)
        assert result == data

    def test_validate_against_schema_pydantic_invalid(self):
        data = {"title": "Test"}  # missing content
        with pytest.raises(PydanticValidationError):
            validate_against_schema(data, TestSchema)

    def test_validate_against_schema_dict_valid(self):
        schema_dict = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }
        data = {"title": "Test"}
        result = validate_against_schema(data, schema_dict)
        assert result == data

    def test_strip_markdown_backticks(self):
        text = "```javascript\nconst x = 1;\n```"
        assert strip_markdown_backticks(text) == "const x = 1;"

        text = "```\nconst x = 1;\n```"
        assert strip_markdown_backticks(text) == "const x = 1;"

        text = "const x = 1;"
        assert strip_markdown_backticks(text) == "const x = 1;"


class TestOpenAIModel:
    @patch("openai.OpenAI")
    @patch("openai.AsyncOpenAI")
    def test_init_with_api_key(self, mock_async_openai, mock_openai):
        model = OpenAIModel(model="gpt-4", api_key="test-key")
        assert model._model == "gpt-4"
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    @patch("openai.OpenAI")
    @patch("openai.AsyncOpenAI")
    def test_init_with_env_key(self, mock_async_openai, mock_openai):
        model = OpenAIModel()
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()

    def test_generate_text_sync(self):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel()
        model._client = mock_client

        messages = [{"role": "user", "content": "Test"}]
        result = model.generate_text(messages, temperature=0.7)

        assert result == "Generated text"
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_text_sync_empty_response(self):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel()
        model._client = mock_client

        messages = [{"role": "user", "content": "Test"}]
        result = model.generate_text(messages)

        assert result == ""


class TestMessagePreparation:
    def test_prepare_object_messages_html(self):
        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        messages = _prepare_object_messages(pre, "Custom prompt")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom prompt"
        assert messages[1]["role"] == "user"
        assert "https://example.com" in messages[1]["content"]
        assert "<html><body>Test</body></html>" in messages[1]["content"]

    def test_prepare_object_messages_image(self):
        pre = PreProcessResult(
            url="https://example.com", content="base64imagedata", format="image"
        )
        messages = _prepare_object_messages(pre, None)

        assert len(messages) == 2
        assert messages[1]["content"][0]["text"] == "URL: https://example.com"
        assert messages[1]["content"][1]["type"] == "image_url"
        assert (
            "data:image/png;base64,base64imagedata"
            in messages[1]["content"][1]["image_url"]["url"]
        )

    def test_prepare_code_messages(self):
        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )
        messages = _prepare_code_messages(pre, "Code prompt", TestSchema)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Code prompt"
        assert messages[1]["role"] == "user"
        assert "https://example.com" in messages[1]["content"]
        assert "Schema:" in messages[1]["content"]


class TestSyncGenerationFunctions:
    def test_generate_llm_object_success(self):
        mock_model = Mock()
        mock_model.generate_json.return_value = {"title": "Test", "content": "Content"}

        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )

        result = generate_llm_object(mock_model, pre, TestSchema)
        assert result == {"title": "Test", "content": "Content"}
        mock_model.generate_json.assert_called_once()

    def test_generate_llm_object_validation_error(self):
        mock_model = Mock()
        mock_model.generate_json.return_value = {"title": "Test"}  # missing content

        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )

        with pytest.raises(ValueError, match="LLM output failed schema validation"):
            generate_llm_object(mock_model, pre, TestSchema)

    def test_generate_llm_object_with_options(self):
        mock_model = Mock()
        mock_model.generate_json.return_value = {"title": "Test", "content": "Content"}

        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )

        options = ScraperLLMOptions(
            temperature=0.5, maxTokens=1000, prompt="Custom prompt"
        )

        result = generate_llm_object(mock_model, pre, TestSchema, options)

        assert result == {"title": "Test", "content": "Content"}
        mock_model.generate_json.assert_called_once_with(
            messages=ANY,
            schema=TestSchema,
            temperature=0.5,
            max_tokens=1000,
            top_p=None,
            mode=None,
        )

    def test_generate_llm_code_success(self):
        mock_model = Mock()
        mock_model.generate_text.return_value = (
            "```javascript\n(() => { return {}; })();\n```"
        )

        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )

        result = generate_llm_code(mock_model, pre, TestSchema)

        assert result == "(() => { return {}; })();"
        mock_model.generate_text.assert_called_once()

    def test_generate_llm_code_with_options(self):
        mock_model = Mock()
        mock_model.generate_text.return_value = "(() => { return {}; })();"

        pre = PreProcessResult(
            url="https://example.com",
            content="<html><body>Test</body></html>",
            format="html",
        )

        options = ScraperGenerateOptions(temperature=0.3, prompt="Generate code")

        result = generate_llm_code(mock_model, pre, TestSchema, options)

        assert result == "(() => { return {}; })();"
        mock_model.generate_text.assert_called_once_with(
            messages=ANY, temperature=0.3, top_p=None
        )
