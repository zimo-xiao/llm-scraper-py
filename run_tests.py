#!/usr/bin/env python3
"""
Test runner script for llm-scraper-py sync functionality
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the full test suite (sync only)"""
    test_dir = Path(__file__).parent / "tests"

    if not test_dir.exists():
        print("❌ Tests directory not found!")
        return 1

    print("🧪 Running llm-scraper-py sync test suite...")
    print(f"📁 Test directory: {test_dir}")

    try:
        # Run pytest with verbose output, excluding slow tests by default
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir),
                "-v",
                "--tb=short",
                "--color=yes",
                "-m",
                "not slow",  # Skip slow tests by default
            ],
            check=False,
        )

        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   pip install pytest pytest-asyncio")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_unit_tests():
    """Run only unit tests (fast, no browser)"""
    test_dir = Path(__file__).parent / "tests"

    print("🧪 Running unit tests (no browser required)...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir),
                "-v",
                "--tb=short",
                "--color=yes",
                "-m",
                "not integration and not slow",
            ],
            check=False,
        )

        return result.returncode
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return 1


def run_integration_tests():
    """Run integration tests (requires browser)"""
    test_dir = Path(__file__).parent / "tests"

    print("🧪 Running integration tests (browser required)...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir),
                "-v",
                "--tb=short",
                "--color=yes",
                "-m",
                "integration",
            ],
            check=False,
        )

        return result.returncode
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return 1


def run_all_tests():
    """Run all tests including slow ones"""
    test_dir = Path(__file__).parent / "tests"

    print("🧪 Running ALL tests (including slow ones)...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir),
                "-v",
                "--tb=short",
                "--color=yes",
            ],
            check=False,
        )

        return result.returncode
    except Exception as e:
        print(f"❌ Error running all tests: {e}")
        return 1


def run_specific_test(test_pattern):
    """Run specific tests matching a pattern"""
    test_dir = Path(__file__).parent / "tests"

    print(f"🧪 Running tests matching: {test_pattern}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir),
                "-v",
                "-k",
                test_pattern,
                "--tb=short",
                "--color=yes",
            ],
            check=False,
        )

        return result.returncode

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def show_help():
    """Show help message"""
    print(
        """
llm-scraper-py Test Runner

Usage:
    python run_tests.py [command]

Commands:
    (no args)    Run standard test suite (excludes slow tests)
    unit         Run unit tests only (no browser required)
    integration  Run integration tests (browser required)
    all          Run all tests including slow ones
    help         Show this help message
    <pattern>    Run tests matching the pattern

Examples:
    python run_tests.py                    # Standard test run
    python run_tests.py unit              # Unit tests only
    python run_tests.py integration       # Integration tests only
    python run_tests.py all               # All tests including slow
    python run_tests.py test_models       # Tests matching 'test_models'
    python run_tests.py "sync and not slow"  # Complex pattern

Test Markers:
    integration  - Tests that require a real browser
    slow         - Tests that may take longer (e.g., real websites)
    """
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: run standard test suite
        exit_code = run_tests()
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()

        if command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "all":
            exit_code = run_all_tests()
        elif command in ["help", "-h", "--help"]:
            show_help()
            exit_code = 0
        else:
            # Treat as test pattern
            exit_code = run_specific_test(sys.argv[1])
    else:
        print("❌ Too many arguments. Use 'python run_tests.py help' for usage.")
        exit_code = 1

    sys.exit(exit_code)
