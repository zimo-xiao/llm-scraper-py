# LLM Scraper Python - Test Suite

This directory contains comprehensive test cases for the **sync functionality only** of llm-scraper-py.

## Test Structure

### Unit Tests (No Browser Required)

- `test_models.py` - Tests for model classes, schema validation, and LLM integration
- `test_preprocess.py` - Tests for preprocessing functionality with mocked pages
- `test_llm_scraper.py` - Tests for the main LLMScraper class
- `test_playwright_js.py` - Tests for JavaScript constants and validation
- `test_examples_sync.py` - Tests that validate example code patterns

### Integration Tests (Browser Required)

- `test_integration_sync.py` - Tests with real browser instances
- `test_playwright_js.py` (integration tests) - Tests JavaScript execution in browser

## Running Tests

### Quick Start

```bash
# Run standard test suite (excludes slow tests)
python run_tests.py

# Run only unit tests (no browser needed)
python run_tests.py unit

# Run integration tests (browser required)
python run_tests.py integration

# Run all tests including slow ones
python run_tests.py all
```

### Specific Test Patterns

```bash
# Run specific test file
python run_tests.py test_models

# Run specific test method
python run_tests.py "test_preprocess_html_format"

# Run tests with complex patterns
python run_tests.py "sync and not slow"
```

## Test Markers

- `integration` - Tests requiring a real browser instance
- `slow` - Tests that may take longer (e.g., real website requests)

## Test Coverage

The test suite covers:

### Core Functionality

- ✅ LLMScraper.run() - sync data extraction
- ✅ LLMScraper.generate() - sync code generation
- ✅ Preprocessing with all formats (html, raw_html, text, markdown, image, custom)
- ✅ Schema validation (Pydantic models and JSON schemas)
- ✅ Error handling and edge cases

### Model Integration

- ✅ OpenAI model adapter (mocked)
- ✅ Language model protocol compliance
- ✅ Message preparation for different content types
- ✅ JSON and text generation functions

### Browser Integration

- ✅ Playwright page preprocessing
- ✅ JavaScript execution (cleanup, markdown, readability)
- ✅ Screenshot capture and base64 encoding
- ✅ Custom format functions

### Example Validation

- ✅ Schema definitions from examples
- ✅ Code generation workflow
- ✅ Data extraction workflow
- ✅ Playwright usage patterns

## Dependencies

### Required for Unit Tests

- pytest
- pydantic
- unittest.mock (built-in)

### Required for Integration Tests

- playwright
- All unit test dependencies

### Installation

```bash
pip install -r requirements.txt
playwright install chromium  # For integration tests
```

## Notes

- **Sync Only**: These tests focus exclusively on synchronous functionality
- **No Async**: Async methods are not tested in this suite
- **Mocked LLM**: Most tests use mocked language models to avoid API costs
- **Real Browser**: Integration tests use real Playwright browser instances
- **Fast by Default**: Standard test run excludes slow tests for quick feedback
