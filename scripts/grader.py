#!/usr/bin/env python3
"""
Programming Assignment Grader

Migrated from run_tests.py to provide a unified grading interface.
This script can find and run test files in a specified assignment directory,
supporting multiple test file patterns and generating YAML output.

Usage:
    python scripts/grader.py --PA PA1                                    # Grade PA1, output to /tmp/feedback.yaml
    python scripts/grader.py --PA PA1 --output-path results.yaml        # Save results to custom file
    python scripts/grader.py --PA PA1 --verbose                         # Show detailed output
    python scripts/grader.py --PA PA1 --pattern "test_*.py"             # Use custom test pattern
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
import dataclasses
from typing import List, Dict, Any, Optional


# Set up project logger
log = logging.getLogger('grader')


@dataclasses.dataclass
class GradingResult:
    grade: float
    comments: str
    logs: str


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
            return None
        
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


def parse_flags():
    parser = argparse.ArgumentParser(
        description="Grade programming assignments by running tests and generating feedback"
    )
    
    parser.add_argument(
        "--PA", "--pa",
        required=True,
        help="Name of the PA (e.g. \"PA1\") to grade. If it matches Canvas or your LMS it can be easier."
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        default="/tmp/feedback.yaml",
        help="Override for where to output the feedback.yaml file."
    )
    parser.add_argument(
        "--pattern", 
        action="append",
        help="Test file pattern to search for (can be specified multiple times)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed console output"
    )
    parser.add_argument(
        "--scoring-file", "-s",
        help="Path to scoring configuration YAML file (default: {PA}/scoring.yaml)"
    )
    
    return parser.parse_args()


def grade(**kwargs) -> GradingResult:
    """
    Grade an assignment by running tests and calculating scores.
    
    Args:
        PA: Programming assignment name (e.g. "PA1")
        output_path: Path for output file
        pattern: Test file patterns (optional)
        verbose: Enable verbose logging
        scoring_file: Custom scoring configuration file
    
    Returns:
        GradingResult with grade, comments, and logs
    """
    pa_name = kwargs.get('PA') or kwargs.get('pa')
    verbose = kwargs.get('verbose', False)
    patterns = kwargs.get('pattern')
    scoring_file = kwargs.get('scoring_file')
    
    # Set up logging
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    # Find assignment directory
    script_dir = Path(__file__).parent
    assignment_dir = script_dir.parent / "programming-assignments" / pa_name
    
    if not assignment_dir.exists():
        error_msg = f"Assignment directory '{pa_name}' not found at {assignment_dir}"
        log.error(error_msg)
        return GradingResult(
            grade=0.0,
            comments=f"Error: {error_msg}",
            logs=error_msg
        )
    
    # Create and run test runner
    try:
        runner = TestRunner(assignment_dir, patterns, verbose, scoring_file)
        summary = runner.run_all_tests()
        
        if summary is None:
            return GradingResult(
                grade=0.0,
                comments="No test files found or tests could not be executed",
                logs="No tests found"
            )
        
        # Extract results for GradingResult format
        test_summary = summary['test_summary']
        score_percentage = test_summary['score_percentage']
        total_points_earned = test_summary['total_points_earned']
        total_points_possible = test_summary['total_points_possible']
        
        # Generate comments
        comments_lines = [
            f"Assignment: {pa_name}",
            f"Score: {total_points_earned}/{total_points_possible} ({score_percentage}%)",
            f"Tests: {test_summary['passed_tests']}/{test_summary['total_tests']} passed",
            ""
        ]
        
        # Add failed test details
        failed_tests = [r for r in summary['test_results'] if r['status'] == 'FAILED']
        if failed_tests:
            comments_lines.append("Failed Tests:")
            for test in failed_tests:
                comments_lines.append(f"- {test['test_name']}: {test['error_message']}")
        else:
            comments_lines.append("All tests passed!")
        
        # Generate logs (detailed test information)
        logs_lines = [f"Grading log for {pa_name}:", ""]
        for result in summary['test_results']:
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            logs_lines.append(
                f"{status_symbol} {result['test_name']} "
                f"({result['points_earned']}/{result['points_possible']} pts)"
            )
            if result['error_message']:
                logs_lines.append(f"  Error: {result['error_message']}")
        
        return GradingResult(
            grade=score_percentage,
            comments="\n".join(comments_lines),
            logs="\n".join(logs_lines)
        )
        
    except Exception as e:
        error_msg = f"Error during grading: {str(e)}"
        log.error(error_msg)
        return GradingResult(
            grade=0.0,
            comments=f"Grading failed: {error_msg}",
            logs=traceback.format_exc()
        )

def main():
    flags = parse_flags()
    
    result = grade(**vars(flags))
    
    with open(flags.output_path, 'w') as yaml_fid:
        yaml.safe_dump(
            dataclasses.asdict(result),
            yaml_fid,
            sort_keys=False
        )
    
    # Also print summary to stdout if verbose
    if flags.verbose:
        print(f"\nGrading completed. Results saved to: {flags.output_path}")
        print(f"Final grade: {result.grade}%")


if __name__ == "__main__":
    main()