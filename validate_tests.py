#!/usr/bin/env python3
"""
Test validation script to verify the comprehensive test suite is working correctly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"✗ FAILED: {description}")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {description} - {e}")
        return False


def validate_test_environment():
    """Validate that the test environment is properly set up."""
    print("Validating test environment...")
    
    # Check if pytest is available
    success = run_command(["python", "-c", "import pytest; print(f'pytest {pytest.__version__}')"], 
                         "Check pytest installation")
    if not success:
        return False
    
    # Check if coverage is available
    success = run_command(["python", "-c", "import coverage; print(f'coverage {coverage.__version__}')"], 
                         "Check coverage installation")
    if not success:
        print("Warning: coverage not available, some features may not work")
    
    # Check if OpenCV is available
    success = run_command(["python", "-c", "import cv2; print(f'OpenCV {cv2.__version__}')"], 
                         "Check OpenCV installation")
    if not success:
        print("Warning: OpenCV not available, some tests may be skipped")
    
    return True


def validate_test_structure():
    """Validate that test files and structure are correct."""
    print("\nValidating test structure...")
    
    required_files = [
        "pytest.ini",
        "tests/conftest.py",
        "tests/README.md",
        "tests/test_comprehensive_unit.py",
        "tests/test_comprehensive_integration.py", 
        "tests/test_comprehensive_performance.py",
        "tests/setup_test_data.py",
        "run_tests.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    
    print("✓ All required test files present")
    return True


def run_sample_tests():
    """Run a sample of tests to verify they work."""
    print("\nRunning sample tests...")
    
    # Test pytest discovery
    success = run_command(["python", "-m", "pytest", "--collect-only", "-q"], 
                         "Test discovery")
    if not success:
        return False
    
    # Run a simple unit test
    success = run_command(["python", "-m", "pytest", "tests/test_models.py::TestProcessingJob::test_create_new_job", "-v"], 
                         "Sample unit test")
    if not success:
        return False
    
    # Test markers
    success = run_command(["python", "-m", "pytest", "-m", "unit", "--collect-only", "-q"], 
                         "Unit test marker")
    if not success:
        return False
    
    return True


def validate_coverage_setup():
    """Validate coverage configuration."""
    print("\nValidating coverage setup...")
    
    # Test coverage configuration
    success = run_command(["python", "-m", "pytest", "--cov=models", "--cov-report=term", 
                          "tests/test_models.py", "-v"], 
                         "Coverage test")
    
    return success


def main():
    """Main validation function."""
    print("Golf Video Anonymizer - Test Suite Validation")
    print("=" * 60)
    
    validation_steps = [
        ("Environment Setup", validate_test_environment),
        ("Test Structure", validate_test_structure),
        ("Sample Tests", run_sample_tests),
        ("Coverage Setup", validate_coverage_setup),
    ]
    
    results = {}
    
    for step_name, step_function in validation_steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            results[step_name] = step_function()
        except Exception as e:
            print(f"✗ ERROR in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for step_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{step_name:.<40} {status}")
        if not success:
            all_passed = False
    
    print(f"{'='*60}")
    
    if all_passed:
        print("🎉 All validations passed! Test suite is ready to use.")
        print("\nNext steps:")
        print("1. Run full test suite: python run_tests.py --all")
        print("2. Generate test data: python tests/setup_test_data.py --all")
        print("3. View coverage report: python run_tests.py --coverage --html")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())