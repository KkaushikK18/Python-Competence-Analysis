"""
Evaluation Benchmarks for Python Competence Analysis
Test cases designed to validate the effectiveness of the competence analysis system.
"""

import json
from typing import Dict, List, Any

# Define benchmark test cases with expected competence assessments
BENCHMARK_TESTS = [
    {
        "test_id": "basic_function_1",
        "code": """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
""",
        "expected_competence": "beginner",
        "expected_concepts": ["functions", "variables", "basic_arithmetic"],
        "expected_issues": [],
        "description": "Simple function definition and call"
    },
    
    {
        "test_id": "range_len_misconception",
        "code": """
def print_items(items):
    for i in range(len(items)):
        print(items[i])

my_list = ['a', 'b', 'c']
print_items(my_list)
""",
        "expected_competence": "beginner",
        "expected_concepts": ["functions", "loops", "lists"],
        "expected_issues": ["Using range(len()) instead of direct iteration"],
        "description": "Common beginner misconception with iteration"
    },
    
    {
        "test_id": "list_comprehension_intermediate",
        "code": """
def filter_and_transform(numbers):
    return [x**2 for x in numbers if x > 0]

def process_data(data):
    positive_squares = filter_and_transform(data)
    return sum(positive_squares)

test_data = [-2, 3, -1, 4, 5]
result = process_data(test_data)
print(f"Result: {result}")
""",
        "expected_competence": "intermediate",
        "expected_concepts": ["list_comprehensions", "functions", "conditionals"],
        "expected_issues": [],
        "description": "Good use of list comprehensions and function composition"
    },
    
    {
        "test_id": "exception_handling_intermediate",
        "code": """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    except TypeError:
        print("Invalid input types")
        return None

def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(calculate_average([1, 2, 3, 4, 5]))
""",
        "expected_competence": "intermediate",
        "expected_concepts": ["exception_handling", "functions", "conditionals"],
        "expected_issues": [],
        "description": "Proper exception handling with multiple exception types"
    },
    
    {
        "test_id": "class_inheritance_advanced",
        "code": """
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print(f"{shape.name}: Area = {shape.area():.2f}, Perimeter = {shape.perimeter():.2f}")
""",
        "expected_competence": "advanced",
        "expected_concepts": ["classes", "inheritance", "abstract_methods", "imports"],
        "expected_issues": [],
        "description": "Abstract base classes with inheritance and polymorphism"
    },
    
    {
        "test_id": "missing_error_handling",
        "code": """
def read_file_content(filename):
    file = open(filename, 'r')
    content = file.read()
    file.close()
    return content

def parse_number(text):
    return int(text)

filename = "data.txt"
content = read_file_content(filename)
number = parse_number(content)
print(f"Number: {number}")
""",
        "expected_competence": "beginner",
        "expected_concepts": ["functions", "file_operations"],
        "expected_issues": ["No error handling for file operations", "Not using context managers"],
        "description": "Missing error handling and resource management"
    },
    
    {
        "test_id": "generator_advanced",
        "code": """
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

def process_sequence(generator, transform_func):
    return [transform_func(x) for x in generator]

# Usage
fib_gen = fibonacci_generator(10)
squared_fibs = process_sequence(fib_gen, lambda x: x**2)
print(f"First 10 squared Fibonacci numbers: {squared_fibs}")

# Memory-efficient processing
def large_number_processor():
    for i in range(1000000):
        if i % 2 == 0:
            yield i ** 2

# Process in chunks to avoid memory issues
def process_in_chunks(generator, chunk_size=1000):
    chunk = []
    for item in generator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

processor = large_number_processor()
for chunk in process_in_chunks(processor, 100):
    print(f"Processed chunk of {len(chunk)} items, max value: {max(chunk)}")
    break  # Just show first chunk for demo
""",
        "expected_competence": "advanced",
        "expected_concepts": ["generators", "lambda_functions", "memory_optimization"],
        "expected_issues": [],
        "description": "Advanced generator usage with memory-efficient processing"
    },
    
    {
        "test_id": "decorator_pattern",
        "code": """
import functools
import time

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def memoize(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper

@timing_decorator
@memoize
def expensive_calculation(n):
    if n <= 1:
        return n
    return expensive_calculation(n-1) + expensive_calculation(n-2)

# Test the decorated function
result1 = expensive_calculation(30)
result2 = expensive_calculation(30)  # Should be cached
print(f"Result: {result1}")
""",
        "expected_competence": "advanced",
        "expected_concepts": ["decorators", "functools", "caching", "recursion"],
        "expected_issues": [],
        "description": "Multiple decorators with memoization and timing"
    }
]

# Expected prompt categories for different competence levels
EXPECTED_PROMPT_PATTERNS = {
    "beginner": [
        "syntax_guidance",
        "concept_explanation", 
        "simple_debugging",
        "basic_improvement"
    ],
    "intermediate": [
        "best_practices",
        "performance_awareness",
        "error_handling",
        "code_organization"
    ],
    "advanced": [
        "architectural_thinking",
        "design_patterns",
        "optimization_strategies",
        "maintainability"
    ]
}

# Evaluation metrics for prompt quality
PROMPT_QUALITY_METRICS = {
    "educational_value": {
        "description": "Does the prompt encourage learning rather than just giving answers?",
        "weight": 0.3
    },
    "appropriate_difficulty": {
        "description": "Is the prompt difficulty appropriate for the student's level?",
        "weight": 0.25
    },
    "clarity": {
        "description": "Is the prompt clear and understandable?",
        "weight": 0.2
    },
    "actionability": {
        "description": "Does the prompt provide clear guidance on what to do next?",
        "weight": 0.15
    },
    "engagement": {
        "description": "Is the prompt engaging and likely to motivate the student?",
        "weight": 0.1
    }
}

def load_benchmark_tests() -> List[Dict[str, Any]]:
    """Load benchmark test cases"""
    return BENCHMARK_TESTS

def evaluate_competence_prediction(predicted: str, expected: str) -> float:
    """Evaluate competence level prediction accuracy"""
    levels = ["beginner", "intermediate", "advanced"]
    
    if predicted == expected:
        return 1.0
    elif abs(levels.index(predicted) - levels.index(expected)) == 1:
        return 0.5  # Adjacent levels get partial credit
    else:
        return 0.0

def evaluate_concept_identification(predicted: List[str], expected: List[str]) -> float:
    """Evaluate concept identification accuracy"""
    if not expected:
        return 1.0 if not predicted else 0.5
    
    correct = len(set(predicted) & set(expected))
    total = len(expected)
    
    return correct / total

def evaluate_issue_detection(predicted: List[str], expected: List[str]) -> float:
    """Evaluate issue detection accuracy"""
    if not expected:
        return 1.0 if not predicted else 0.8  # False positives less severe
    
    # Check if the essence of expected issues is captured
    detected_issues = 0
    for expected_issue in expected:
        for predicted_issue in predicted:
            # Simple keyword matching for now
            if any(keyword in predicted_issue.lower() 
                   for keyword in expected_issue.lower().split()):
                detected_issues += 1
                break
    
    return detected_issues / len(expected) if expected else 1.0

def create_evaluation_report(results: List[Dict]) -> Dict[str, Any]:
    """Create comprehensive evaluation report"""
    total_tests = len(results)
    
    competence_accuracy = sum(r['competence_accuracy'] for r in results) / total_tests
    concept_accuracy = sum(r['concept_accuracy'] for r in results) / total_tests
    issue_accuracy = sum(r['issue_accuracy'] for r in results) / total_tests
    
    report = {
        "overall_score": (competence_accuracy + concept_accuracy + issue_accuracy) / 3,
        "competence_prediction_accuracy": competence_accuracy,
        "concept_identification_accuracy": concept_accuracy,
        "issue_detection_accuracy": issue_accuracy,
        "total_tests": total_tests,
        "detailed_results": results,
        "recommendations": []
    }
    
    # Generate recommendations based on results
    if competence_accuracy < 0.7:
        report["recommendations"].append(
            "Improve competence level classification algorithm"
        )
    
    if concept_accuracy < 0.8:
        report["recommendations"].append(
            "Enhance concept identification patterns"
        )
    
    if issue_accuracy < 0.6:
        report["recommendations"].append(
            "Expand issue detection rules and patterns"
        )
    
    return report

# Sample benchmark runner for testing
def run_benchmark_evaluation(analyzer_func):
    """Run benchmark evaluation on a given analyzer function"""
    benchmark_tests = load_benchmark_tests()
    results = []
    
    for test in benchmark_tests:
        # Run the analyzer on the test code
        analysis_result = analyzer_func(test["code"])
        
        # Evaluate the results
        competence_accuracy = evaluate_competence_prediction(
            analysis_result.get("competence_level", "unknown"),
            test["expected_competence"]
        )
        
        concept_accuracy = evaluate_concept_identification(
            analysis_result.get("identified_concepts", []),
            test["expected_concepts"]
        )
        
        issue_accuracy = evaluate_issue_detection(
            analysis_result.get("potential_issues", []),
            test["expected_issues"]
        )
        
        results.append({
            "test_id": test["test_id"],
            "description": test["description"],
            "competence_accuracy": competence_accuracy,
            "concept_accuracy": concept_accuracy,
            "issue_accuracy": issue_accuracy,
            "overall_score": (competence_accuracy + concept_accuracy + issue_accuracy) / 3
        })
    
    return create_evaluation_report(results)

if __name__ == "__main__":
    # Example of how to use the benchmark
    def mock_analyzer(code):
        """Mock analyzer for demonstration"""
        # This would be replaced with your actual analyzer
        return {
            "competence_level": "intermediate",
            "identified_concepts": ["functions", "loops"],
            "potential_issues": ["Some issue detected"]
        }
    
    # Run benchmark evaluation
    report = run_benchmark_evaluation(mock_analyzer)
    
    print("=== BENCHMARK EVALUATION REPORT ===")
    print(f"Overall Score: {report['overall_score']:.3f}")
    print(f"Competence Accuracy: {report['competence_prediction_accuracy']:.3f}")
    print(f"Concept Accuracy: {report['concept_identification_accuracy']:.3f}")
    print(f"Issue Detection: {report['issue_detection_accuracy']:.3f}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    print(f"\nDetailed results available for {report['total_tests']} test cases.")