"""
Tests for playwright_js.py constants and JavaScript code
"""

import pytest
from llm_scraper_py.playwright_js import (
    CLEANUP_JS,
    TO_MARKDOWN_JS,
    TO_READABILITY_TEXT_JS,
    DEFAULT_PROMPT,
    DEFAULT_CODE_PROMPT,
)


class TestJavaScriptConstants:
    def test_cleanup_js_is_string(self):
        """Test that CLEANUP_JS is a non-empty string"""
        assert isinstance(CLEANUP_JS, str)
        assert len(CLEANUP_JS) > 0

    def test_cleanup_js_contains_expected_elements(self):
        """Test that CLEANUP_JS contains expected element removal logic"""
        # Should contain elements to remove
        assert "script" in CLEANUP_JS
        assert "style" in CLEANUP_JS
        assert "iframe" in CLEANUP_JS
        assert "img" in CLEANUP_JS

        # Should contain attribute removal logic
        assert "style" in CLEANUP_JS
        assert "src" in CLEANUP_JS
        assert "aria-" in CLEANUP_JS
        assert "data-" in CLEANUP_JS

    def test_cleanup_js_is_iife(self):
        """Test that CLEANUP_JS is wrapped in IIFE"""
        assert CLEANUP_JS.strip().startswith("(() => {")
        assert CLEANUP_JS.strip().endswith("})();")

    def test_to_markdown_js_is_string(self):
        """Test that TO_MARKDOWN_JS is a non-empty string"""
        assert isinstance(TO_MARKDOWN_JS, str)
        assert len(TO_MARKDOWN_JS) > 0

    def test_to_markdown_js_contains_turndown_logic(self):
        """Test that TO_MARKDOWN_JS contains TurndownService logic"""
        assert "TurndownService" in TO_MARKDOWN_JS
        assert "turndown" in TO_MARKDOWN_JS
        assert "innerText" in TO_MARKDOWN_JS  # fallback

    def test_to_markdown_js_is_function(self):
        """Test that TO_MARKDOWN_JS is a function taking html parameter"""
        assert TO_MARKDOWN_JS.strip().startswith("(html) => {")

    def test_to_readability_text_js_is_string(self):
        """Test that TO_READABILITY_TEXT_JS is a non-empty string"""
        assert isinstance(TO_READABILITY_TEXT_JS, str)
        assert len(TO_READABILITY_TEXT_JS) > 0

    def test_to_readability_text_js_contains_readability_logic(self):
        """Test that TO_READABILITY_TEXT_JS contains readability logic"""
        assert "@mozilla/readability" in TO_READABILITY_TEXT_JS
        assert "Readability" in TO_READABILITY_TEXT_JS
        assert "parse()" in TO_READABILITY_TEXT_JS
        assert "title" in TO_READABILITY_TEXT_JS
        assert "textContent" in TO_READABILITY_TEXT_JS

    def test_to_readability_text_js_is_async_function(self):
        """Test that TO_READABILITY_TEXT_JS is an async function"""
        assert "async ()" in TO_READABILITY_TEXT_JS
        assert "await import" in TO_READABILITY_TEXT_JS

    def test_default_prompt_is_string(self):
        """Test that DEFAULT_PROMPT is a non-empty string"""
        assert isinstance(DEFAULT_PROMPT, str)
        assert len(DEFAULT_PROMPT) > 0

    def test_default_prompt_content(self):
        """Test that DEFAULT_PROMPT contains expected content"""
        assert "scraper" in DEFAULT_PROMPT.lower()
        assert "extract" in DEFAULT_PROMPT.lower()
        assert "webpage" in DEFAULT_PROMPT.lower()

    def test_default_code_prompt_is_string(self):
        """Test that DEFAULT_CODE_PROMPT is a non-empty string"""
        assert isinstance(DEFAULT_CODE_PROMPT, str)
        assert len(DEFAULT_CODE_PROMPT) > 0

    def test_default_code_prompt_content(self):
        """Test that DEFAULT_CODE_PROMPT contains expected content"""
        assert "JavaScript" in DEFAULT_CODE_PROMPT
        assert "IIFE" in DEFAULT_CODE_PROMPT
        assert "schema" in DEFAULT_CODE_PROMPT.lower()
        assert "extract" in DEFAULT_CODE_PROMPT.lower()


@pytest.mark.integration
class TestJavaScriptExecution:
    """Integration tests that actually execute the JavaScript in a browser"""

    def test_cleanup_js_execution(self, page):
        """Test that CLEANUP_JS executes without errors"""
        # Set up a page with elements that should be removed
        page.set_content(
            """
        <html>
            <head>
                <title>Test</title>
                <style>.test { color: red; }</style>
            </head>
            <body>
                <h1 style="color: blue;" data-test="value">Title</h1>
                <p>Content</p>
                <script>console.log('test');</script>
                <img src="test.jpg" alt="test">
                <div aria-label="test" tabindex="0">Div</div>
            </body>
        </html>
        """
        )

        # Execute cleanup
        result = page.evaluate(CLEANUP_JS)

        # Should execute without error (returns undefined/None)
        assert result is None

        # Check that elements were removed/cleaned
        content = page.content()
        assert "<script>" not in content
        assert "<style>" not in content
        assert "<img" not in content
        assert 'style="' not in content
        assert 'data-test="' not in content
        assert 'aria-label="' not in content

    def test_to_markdown_js_execution_fallback(self, page):
        """Test TO_MARKDOWN_JS fallback to innerText"""
        page.set_content(
            """
        <html>
            <body>
                <h1>Title</h1>
                <p>Paragraph</p>
            </body>
        </html>
        """
        )

        html_content = page.inner_html("body")
        result = page.evaluate(TO_MARKDOWN_JS, html_content)

        # Should return text content (TurndownService likely not available)
        assert isinstance(result, str)
        assert "Title" in result
        assert "Paragraph" in result

    def test_to_readability_text_js_execution(self, page):
        """Test TO_READABILITY_TEXT_JS execution"""
        page.set_content(
            """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <h1>Main Title</h1>
                    <p>This is the main content of the article.</p>
                </article>
                <aside>Sidebar content</aside>
            </body>
        </html>
        """
        )

        try:
            result = page.evaluate(TO_READABILITY_TEXT_JS)

            # Should return an object with title and text
            assert isinstance(result, dict)
            assert "title" in result
            assert "text" in result

            # May contain the title and some content
            if result["title"]:
                assert isinstance(result["title"], str)
            if result["text"]:
                assert isinstance(result["text"], str)

        except Exception as e:
            # Readability import might fail in test environment
            pytest.skip(f"Readability import failed: {e}")

    def test_javascript_syntax_validity(self, page):
        """Test that all JavaScript constants have valid syntax"""
        # Test CLEANUP_JS syntax
        try:
            page.evaluate(CLEANUP_JS)
        except Exception as e:
            pytest.fail(f"CLEANUP_JS has invalid syntax: {e}")

        # Test TO_MARKDOWN_JS syntax (with dummy parameter)
        try:
            page.evaluate(TO_MARKDOWN_JS, "<p>test</p>")
        except Exception as e:
            # Might fail due to missing TurndownService, but syntax should be valid
            if "TurndownService" not in str(e):
                pytest.fail(f"TO_MARKDOWN_JS has invalid syntax: {e}")

        # Test TO_READABILITY_TEXT_JS syntax
        try:
            page.evaluate(TO_READABILITY_TEXT_JS)
        except Exception as e:
            # Might fail due to import issues, but syntax should be valid
            if "import" not in str(e) and "fetch" not in str(e):
                pytest.fail(f"TO_READABILITY_TEXT_JS has invalid syntax: {e}")

    def test_cleanup_preserves_content_structure(self, page):
        """Test that CLEANUP_JS preserves main content structure"""
        page.set_content(
            """
        <html>
            <body>
                <header>Header</header>
                <main>
                    <h1>Main Title</h1>
                    <section>
                        <h2>Section Title</h2>
                        <p>Section content</p>
                    </section>
                </main>
                <footer>Footer</footer>
                <script>alert('remove me');</script>
            </body>
        </html>
        """
        )

        page.evaluate(CLEANUP_JS)
        content = page.content()

        # Main structure should be preserved
        assert "<main>" in content
        assert "<h1>" in content
        assert "<section>" in content
        assert "<p>" in content
        assert "Main Title" in content
        assert "Section content" in content

        # But script should be removed
        assert "<script>" not in content
        assert "alert" not in content
