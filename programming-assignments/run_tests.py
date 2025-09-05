#!/usr/bin/env python3
"""
General test runner for programming assignments.

This script can find and run test files in a specified assignment directory,
supporting multiple test file patterns and generating YAML output.

Usage:
    python run_tests.py PA1                      # Output YAML to stdout (default)
    python run_tests.py PA1 --verbose           # Show detailed output + YAML
    python run_tests.py PA1 --output results.yaml  # Save YAML to file
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
import logging
import signal

# Set up project logger
log = logging.getLogger('test_runner')


class TestRunner:
    def __init__(self, assignment_dir, test_patterns=None, verbose=True, scoring_file=None):
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
        self.scoring_config = self.load_scoring_config(scoring_file)
    
    def load_scoring_config(self, scoring_file=None):
        """Load scoring configuration from YAML file."""
        if scoring_file:
            config_path = Path(scoring_file)
        else:
            # Default to scoring.yaml in assignment directory
            config_path = self.assignment_dir / "scoring.yaml"
        
        if not config_path.exists():
            if scoring_file:
                log.warning(f"Scoring file '{config_path}' not found. Using default scoring.")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            log.info(f"Loaded scoring configuration from: {config_path}")
            return config
        except Exception as e:
            log.warning(f"Failed to load scoring config: {e}. Using default scoring.")
            return None
    
    def calculate_test_score(self, test_name, status):
        """Calculate score for a single test based on scoring configuration."""
        if not self.scoring_config:
            # Default scoring: 1 point for pass, 0 for fail
            return 1.0 if status == "PASSED" else 0.0
        
        # Check for specific test scoring
        test_scores = self.scoring_config.get('test_scores', {})
        if test_name in test_scores:
            score_config = test_scores[test_name]
            if isinstance(score_config, (int, float)):
                return float(score_config) if status == "PASSED" else 0.0
            elif isinstance(score_config, dict):
                return float(score_config.get('points', 1.0)) if status == "PASSED" else 0.0
        
        # Check for group scoring
        groups = self.scoring_config.get('groups', {})
        for group_name, group_config in groups.items():
            patterns = group_config.get('patterns', [])
            for pattern in patterns:
                if pattern in test_name or test_name.startswith(pattern.replace('*', '')):
                    points = group_config.get('points', 1.0)
                    return float(points) if status == "PASSED" else 0.0
        
        # Default scoring
        default_points = self.scoring_config.get('default_points', 1.0)
        return float(default_points) if status == "PASSED" else 0.0
        
    def find_test_files(self):
        """Find all test files matching the specified patterns."""
        test_files = []
        
        for pattern in self.test_patterns:
            matches = list(self.assignment_dir.glob(pattern))
            test_files.extend(matches)
        
        # Remove duplicates and sort
        test_files = sorted(list(set(test_files)))
        
        log.info(f"Found test files: {[f.name for f in test_files]}")
            
        return test_files
    
    def try_pytest_runner(self, test_file, timeout=60):
        """Try to run tests using pytest if available with timeout protection."""
        try:
            import pytest
        except ImportError:
            return None
        
        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Test execution timed out")
        
        # Collect results using a custom pytest plugin
        class TestCollector:
            def __init__(self):
                self.results = []
            
            def pytest_runtest_logreport(self, report):
                if report.when == "call":
                    self.results.append({
                        'test_name': report.nodeid.split("::")[-1],
                        'status': 'PASSED' if report.passed else 'FAILED',
                        'error_message': str(report.longrepr) if report.failed else None
                    })
        
        old_cwd = None
        old_handler = None
        try:
            old_cwd = os.getcwd()
            os.chdir(self.assignment_dir)
            
            collector = TestCollector()
            
            # Set up timeout protection (only on Unix-like systems)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            # Run pytest programmatically with our custom plugin
            pytest.main([
                str(test_file), 
                '-v', 
                '--tb=short'
            ], plugins=[collector])
            
            # Cancel timeout if everything completed normally
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            return {
                'runner': 'pytest',
                'success': True,  # Always return success if we can run
                'test_results': collector.results
            }
            
        except TimeoutError:
            log.error(f"Test execution timed out after {timeout} seconds")
            return {
                'runner': 'pytest', 
                'success': False,
                'error': f'Test execution timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'runner': 'pytest', 
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up timeout and restore signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            
            # Restore working directory
            if old_cwd is not None:
                try:
                    os.chdir(old_cwd)
                except OSError:
                    pass
    
    def run_test_functions_directly(self, test_file, timeout=30):
        """Run test functions directly by importing and executing them with timeout protection."""
        results = []
        
        # Timeout handler for individual tests
        def timeout_handler(signum, frame):
            raise TimeoutError("Individual test timed out")
        
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
            
            log.info(f"Running tests from {test_file.name}")
            
            for test_name, test_func in test_functions:
                old_handler = None
                try:
                    # Set timeout for individual test (only on Unix-like systems)
                    if hasattr(signal, 'SIGALRM'):
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout)
                    
                    test_func()
                    status = "PASSED"
                    error_message = None
                    log.info(f"✓ {test_name}")
                    
                except TimeoutError:
                    status = "FAILED" 
                    error_message = f"Test timed out after {timeout} seconds"
                    log.error(f"✗ {test_name}: {error_message}")
                    
                except Exception as e:
                    status = "FAILED"
                    error_message = str(e)
                    log.error(f"✗ {test_name}: {error_message}")
                
                finally:
                    # Clean up timeout
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        if old_handler is not None:
                            signal.signal(signal.SIGALRM, old_handler)
                
                score = self.calculate_test_score(test_name, status)
                results.append({
                    'test_name': test_name,
                    'test_file': test_file.name,
                    'status': status,
                    'error_message': error_message,
                    'points_earned': score,
                    'points_possible': self.calculate_test_score(test_name, "PASSED")
                })
        
        except Exception as e:
            score = self.calculate_test_score('MODULE_IMPORT', 'FAILED')
            results.append({
                'test_name': 'MODULE_IMPORT',
                'test_file': test_file.name,
                'status': 'FAILED',
                'error_message': f"Failed to import test module: {str(e)}",
                'points_earned': score,
                'points_possible': self.calculate_test_score('MODULE_IMPORT', "PASSED")
            })
        
        finally:
            # Clean up Python path
            if str(self.assignment_dir) in sys.path:
                sys.path.remove(str(self.assignment_dir))
        
        return results
    
    def run_tests_in_file(self, test_file):
        """Run all tests in a specific file."""
        log.info(f"Processing: {test_file}")
        
        # First try pytest if available
        pytest_result = self.try_pytest_runner(test_file)
        if pytest_result and pytest_result.get('success'):
            # Use the structured test results from pytest
            return self.process_pytest_results(pytest_result, test_file)
        
        # Fall back to direct function execution
        return self.run_test_functions_directly(test_file)
    
    def process_pytest_results(self, pytest_result, test_file):
        """Process structured pytest results."""
        results = []
        
        test_results = pytest_result.get('test_results', [])
        
        for test_result in test_results:
            test_name = test_result['test_name']
            status = test_result['status']
            error_message = test_result['error_message']
            
            score = self.calculate_test_score(test_name, status)
            results.append({
                'test_name': test_name,
                'test_file': test_file.name,
                'status': status,
                'error_message': error_message,
                'points_earned': score,
                'points_possible': self.calculate_test_score(test_name, "PASSED")
            })
        
        return results
    
    def run_all_tests(self):
        """Run all tests in the assignment directory."""
        test_files = self.find_test_files()
        
        if not test_files:
            log.error(f"No test files found in {self.assignment_dir}")
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
        
        # Calculate scoring totals
        total_points_earned = sum(r.get('points_earned', 0) for r in self.results)
        total_points_possible = sum(r.get('points_possible', 0) for r in self.results)
        if total_points_possible > 0:
            score_percentage = round((total_points_earned / total_points_possible * 100), 1)
        else:
            score_percentage = 0
        
        summary = {
            'assignment_directory': str(self.assignment_dir),
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': round((passed_tests / total_tests * 100) if total_tests > 0 else 0, 1),
                'total_points_earned': total_points_earned,
                'total_points_possible': total_points_possible,
                'score_percentage': score_percentage
            },
            'test_results': self.results
        }
        
        log.info(f"Test Summary for {self.assignment_dir.name}:")
        log.info(f"Total tests: {total_tests}")
        log.info(f"Passed: {passed_tests}")
        log.info(f"Failed: {failed_tests}")
        log.info(f"Success rate: {summary['test_summary']['success_rate']}%")
        log.info(f"Score: {total_points_earned}/{total_points_possible} ({score_percentage}%)")
        
        return summary
    
    def save_yaml_results(self, filename=None):
        """Save test results to YAML file."""
        if not filename:
            filename = f"{self.assignment_dir.name}_test_results.yaml"
        
        summary = self.generate_summary()
        
        with open(filename, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        log.info(f"Results saved to: {filename}")
        
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
        "--verbose", "-v",
        action="store_true",
        help="Show detailed console output in addition to YAML"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output YAML to file instead of stdout"
    )
    parser.add_argument(
        "--scoring-file", "-s",
        help="Path to scoring configuration YAML file (default: {assignment}/scoring.yaml)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            stream=sys.stderr  # Send logs to stderr so stdout is clean for YAML
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
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
    
    # Set up test patterns and verbosity
    patterns = args.pattern if args.pattern else None
    verbose = args.verbose
    
    # Create and run test runner
    runner = TestRunner(assignment_dir, patterns, verbose, args.scoring_file)
    summary = runner.run_all_tests()
    
    # Output YAML results
    if args.output:
        # Save to file
        with open(args.output, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        log.info(f"Results saved to: {args.output}")
    else:
        # Output to stdout by default
        print(yaml.dump(summary, default_flow_style=False, indent=2))
    
    # Exit with appropriate code
    success_rate = summary['test_summary']['success_rate']
    sys.exit(0 if success_rate == 100 else 1)


if __name__ == "__main__":
    main()
