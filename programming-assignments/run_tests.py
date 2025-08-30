#!/usr/bin/env python3
"""
General test runner for programming assignments.

This script can find and run test files in a specified assignment directory,
supporting multiple test file patterns and generating YAML output.

Usage:
    python run_tests.py PA1                    # Run tests in PA1 directory
    python run_tests.py PA1 --yaml-only       # Generate YAML output only
    python run_tests.py PA1 --pattern "test_*.py"  # Use custom pattern
"""

import argparse
import os
import sys
import importlib.util
import traceback
import yaml
from pathlib import Path
import inspect
import subprocess


class TestRunner:
    def __init__(self, assignment_dir, test_patterns=None, verbose=True):
        self.assignment_dir = Path(assignment_dir)
        self.test_patterns = test_patterns or [
            "tests.py",
            "test_*.py", 
            "unittests_*.py",
            "*_test.py",
            "*_tests.py"
        ]
        self.verbose = verbose
        self.results = []
        
    def find_test_files(self):
        """Find all test files matching the specified patterns."""
        test_files = []
        
        for pattern in self.test_patterns:
            matches = list(self.assignment_dir.glob(pattern))
            test_files.extend(matches)
        
        # Remove duplicates and sort
        test_files = sorted(list(set(test_files)))
        
        if self.verbose:
            print(f"Found test files: {[f.name for f in test_files]}")
            
        return test_files
    
    def try_pytest_runner(self, test_file):
        """Try to run tests using pytest if available."""
        try:
            import pytest
            
            # Capture pytest output
            result = subprocess.run([
                sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short'
            ], cwd=self.assignment_dir, capture_output=True, text=True)
            
            return {
                'runner': 'pytest',
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except ImportError:
            return None
        except Exception as e:
            return {
                'runner': 'pytest',
                'success': False,
                'error': str(e)
            }
    
    def run_test_functions_directly(self, test_file):
        """Run test functions directly by importing and executing them."""
        results = []
        
        # Add the assignment directory to Python path
        sys.path.insert(0, str(self.assignment_dir))
        
        try:
            # Import the test module
            spec = importlib.util.spec_from_file_location("test_module", test_file)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Find all test functions
            test_functions = [
                (name, obj) for name, obj in inspect.getmembers(test_module)
                if (inspect.isfunction(obj) and 
                    (name.startswith('test_') or name.endswith('_test')))
            ]
            
            if self.verbose:
                print(f"\nRunning tests from {test_file.name}:")
                print("=" * 50)
            
            for test_name, test_func in test_functions:
                try:
                    test_func()
                    status = "PASSED"
                    error_message = None
                    if self.verbose:
                        print(f"✓ {test_name}")
                except Exception as e:
                    status = "FAILED"
                    error_message = str(e)
                    if self.verbose:
                        print(f"✗ {test_name}: {error_message}")
                
                results.append({
                    'test_name': test_name,
                    'test_file': test_file.name,
                    'status': status,
                    'error_message': error_message
                })
        
        except Exception as e:
            results.append({
                'test_name': 'MODULE_IMPORT',
                'test_file': test_file.name,
                'status': 'FAILED',
                'error_message': f"Failed to import test module: {str(e)}"
            })
        
        finally:
            # Clean up Python path
            if str(self.assignment_dir) in sys.path:
                sys.path.remove(str(self.assignment_dir))
        
        return results
    
    def run_tests_in_file(self, test_file):
        """Run all tests in a specific file."""
        if self.verbose:
            print(f"\nProcessing: {test_file}")
        
        # First try pytest if available
        pytest_result = self.try_pytest_runner(test_file)
        if pytest_result and pytest_result.get('success'):
            # Parse pytest output for individual test results
            return self.parse_pytest_output(pytest_result, test_file)
        
        # Fall back to direct function execution
        return self.run_test_functions_directly(test_file)
    
    def parse_pytest_output(self, pytest_result, test_file):
        """Parse pytest output to extract individual test results."""
        results = []
        
        stdout = pytest_result.get('stdout', '')
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0].split('::')[-1]
                    status = parts[-1]
                    
                    results.append({
                        'test_name': test_name,
                        'test_file': test_file.name,
                        'status': status,
                        'error_message': None if status == 'PASSED' else 'See pytest output'
                    })
        
        return results
    
    def run_all_tests(self):
        """Run all tests in the assignment directory."""
        test_files = self.find_test_files()
        
        if not test_files:
            print(f"No test files found in {self.assignment_dir}")
            return
        
        all_results = []
        
        for test_file in test_files:
            file_results = self.run_tests_in_file(test_file)
            all_results.extend(file_results)
        
        self.results = all_results
        return self.generate_summary()
    
    def generate_summary(self):
        """Generate a summary of test results."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        
        summary = {
            'assignment_directory': str(self.assignment_dir),
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': round((passed_tests / total_tests * 100) if total_tests > 0 else 0, 1)
            },
            'test_results': self.results
        }
        
        if self.verbose:
            print("\n" + "=" * 50)
            print(f"Test Summary for {self.assignment_dir.name}:")
            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Success rate: {summary['test_summary']['success_rate']}%")
        
        return summary
    
    def save_yaml_results(self, filename=None):
        """Save test results to YAML file."""
        if not filename:
            filename = f"{self.assignment_dir.name}_test_results.yaml"
        
        summary = self.generate_summary()
        
        with open(filename, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        if self.verbose:
            print(f"\nResults saved to: {filename}")
        
        return filename


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for programming assignments and generate YAML output"
    )
    parser.add_argument(
        "assignment", 
        help="Assignment directory (e.g., PA1, PA2, etc.)"
    )
    parser.add_argument(
        "--pattern", 
        action="append",
        help="Test file pattern to search for (can be specified multiple times)"
    )
    parser.add_argument(
        "--yaml-only", 
        action="store_true",
        help="Only generate YAML output, suppress console output"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output YAML filename (default: {assignment}_test_results.yaml)"
    )
    
    args = parser.parse_args()
    
    # Resolve assignment directory path
    # Try relative to current directory first, then relative to script directory
    assignment_dir = Path(args.assignment)
    
    if not assignment_dir.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        assignment_dir = script_dir / args.assignment
    
    if not assignment_dir.exists():
        print(f"Error: Assignment directory '{args.assignment}' not found")
        print(f"Searched in:")
        print(f"  - {Path(args.assignment).resolve()}")
        print(f"  - {(Path(__file__).parent / args.assignment).resolve()}")
        sys.exit(1)
    
    # Set up test patterns
    patterns = args.pattern if args.pattern else None
    verbose = not args.yaml_only
    
    # Create and run test runner
    runner = TestRunner(assignment_dir, patterns, verbose)
    summary = runner.run_all_tests()
    
    # Save YAML results
    output_file = args.output or f"{args.assignment}_test_results.yaml"
    runner.save_yaml_results(output_file)
    
    # Print YAML if requested
    if args.yaml_only:
        print(yaml.dump(summary, default_flow_style=False, indent=2))
    
    # Exit with appropriate code
    success_rate = summary['test_summary']['success_rate']
    sys.exit(0 if success_rate == 100 else 1)


if __name__ == "__main__":
    main()