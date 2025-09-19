#!/usr/bin/env python3
"""
Comprehensive test runner for Golf Video Anonymizer.
Provides different test execution modes and reporting options.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    print(f"Exit code: {result.returncode}")
    
    return result.returncode == 0


def run_unit_tests(verbose=False, coverage=True):
    """Run unit tests only."""
    cmd = ["python", "-m", "pytest", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    
    cmd.append("tests/")
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests only."""
    cmd = ["python", "-m", "pytest", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append("tests/")
    
    return run_command(cmd, "Integration Tests")


def run_performance_tests(verbose=False):
    """Run performance tests only."""
    cmd = ["python", "-m", "pytest", "-m", "performance"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append("tests/")
    
    return run_command(cmd, "Performance Tests")


def run_all_tests(verbose=False, coverage=True, html_report=False):
    """Run all tests with comprehensive reporting."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
        
        if html_report:
            cmd.append("--cov-report=html")
    
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "tests/"
    ])
    
    return run_command(cmd, "All Tests")


def run_fast_tests(verbose=False):
    """Run fast tests only (exclude slow tests)."""
    cmd = ["python", "-m", "pytest", "-m", "not slow"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append("tests/")
    
    return run_command(cmd, "Fast Tests (excluding slow tests)")


def run_specific_test_file(test_file, verbose=False):
    """Run tests from a specific file."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append(f"tests/{test_file}")
    
    return run_command(cmd, f"Tests from {test_file}")


def run_tests_with_keyword(keyword, verbose=False):
    """Run tests matching a keyword."""
    cmd = ["python", "-m", "pytest", "-k", keyword]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append("tests/")
    
    return run_command(cmd, f"Tests matching keyword: {keyword}")


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\nGenerating detailed coverage report...")
    
    # Generate HTML coverage report
    html_success = run_command(
        ["python", "-m", "pytest", "--cov=.", "--cov-report=html", "--cov-report=term", "tests/"],
        "Coverage Report Generation"
    )
    
    if html_success:
        print(f"\nHTML coverage report generated in: {Path.cwd() / 'htmlcov' / 'index.html'}")
        print("Open this file in a web browser to view detailed coverage information.")
    
    return html_success


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("✗ pytest not installed")
        return False
    
    # Check if coverage is available
    try:
        import coverage
        print(f"✓ coverage version: {coverage.__version__}")
    except ImportError:
        print("✗ coverage not installed")
    
    # Check if OpenCV is available
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    # Check if Redis is available (optional)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis connection successful")
    except (ImportError, Exception):
        print("⚠ Redis not available (some tests may be skipped)")
    
    # Check test data directory
    test_data_dir = Path("tests/test_data")
    if test_data_dir.exists():
        print(f"✓ Test data directory exists: {test_data_dir}")
    else:
        print(f"⚠ Test data directory will be created: {test_data_dir}")
    
    print("Environment check completed.\n")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Golf Video Anonymizer Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit                   # Run unit tests only
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --performance            # Run performance tests only
  python run_tests.py --fast                   # Run fast tests only
  python run_tests.py --file test_models.py    # Run specific test file
  python run_tests.py --keyword face           # Run tests matching keyword
  python run_tests.py --coverage               # Generate coverage report
  python run_tests.py --check                  # Check test environment
        """
    )
    
    # Test execution modes
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow)")
    
    # Specific test selection
    parser.add_argument("--file", help="Run tests from specific file (e.g., test_models.py)")
    parser.add_argument("--keyword", help="Run tests matching keyword")
    
    # Reporting options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Environment check
    parser.add_argument("--check", action="store_true", help="Check test environment")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Check environment if requested
    if args.check:
        if not check_test_environment():
            sys.exit(1)
        return
    
    # Ensure test environment is set up
    if not check_test_environment():
        print("Test environment check failed. Please install required dependencies.")
        sys.exit(1)
    
    success = True
    
    # Run tests based on arguments
    if args.all:
        success = run_all_tests(args.verbose, coverage=True, html_report=args.html)
    elif args.unit:
        success = run_unit_tests(args.verbose, coverage=True)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.performance:
        success = run_performance_tests(args.verbose)
    elif args.fast:
        success = run_fast_tests(args.verbose)
    elif args.file:
        success = run_specific_test_file(args.file, args.verbose)
    elif args.keyword:
        success = run_tests_with_keyword(args.keyword, args.verbose)
    elif args.coverage:
        success = generate_coverage_report()
    else:
        print("No test mode specified. Use --help for options.")
        parser.print_help()
        return
    
    # Exit with appropriate code
    if success:
        print("\n✓ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()