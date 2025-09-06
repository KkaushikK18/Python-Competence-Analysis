#!/usr/bin/env python3
"""
Main runner for Python Student Competence Analysis System
Provides command-line interface and demonstration capabilities.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from competence_analyzer import PythonCompetenceAnalyzer, CompetenceLevel
from model_evaluation import ModelEvaluator
from prompt_generator import EducationalPromptGenerator

def analyze_file(filepath: str, expected_level: Optional[str] = None, output: Optional[str] = None):
    """Analyze a Python file and generate educational prompts"""
    
    # Read the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    # Initialize analyzer
    print("Loading model... (this may take a few minutes on first run)")
    try:
        analyzer = PythonCompetenceAnalyzer()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure you have the required dependencies and sufficient memory/GPU resources.")
        return
    
    # Perform analysis
    print(f"Analyzing file: {filepath}")
    start_time = time.time()
    
    try:
        # Convert string level to enum if provided
        expected_competence = None
        if expected_level:
            expected_competence = CompetenceLevel(expected_level.lower())
        
        report = analyzer.get_competence_report(code, expected_competence)
        analysis_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*60)
        print("PYTHON COMPETENCE ANALYSIS REPORT")
        print("="*60)
        
        analysis = report['code_analysis']
        print(f"File: {filepath}")
        print(f"Analysis time: {analysis_time:.2f} seconds")
        print(f"Detected competence level: {analysis['competence_level'].upper()}")
        print(f"Complexity score: {analysis['complexity_score']:.2f}/1.0")
        print(f"Best practices score: {analysis['best_practices_score']:.2f}/1.0")
        print(f"Syntax correct: {'Yes' if analysis['syntax_correct'] else 'No'}")
        
        if analysis['identified_concepts']:
            print(f"Identified concepts: {', '.join(analysis['identified_concepts'])}")
        
        if analysis['misconceptions']:
            print(f"Common misconceptions detected:")
            for misconception in analysis['misconceptions']:
                print(f"  • {misconception}")
        
        if analysis['potential_issues']:
            print(f"Potential issues:")
            for issue in analysis['potential_issues']:
                print(f"  • {issue}")
        
        # Educational prompts
        print("\n" + "="*60)
        print("EDUCATIONAL PROMPTS")
        print("="*60)
        
        for i, prompt in enumerate(report['educational_prompts'], 1):
            print(f"\nPrompt {i} ({prompt['type'].upper()}):")
            print("-" * 40)
            print(prompt['prompt'])
            if prompt.get('learning_objectives'):
                print(f"Learning objectives: {', '.join(prompt['learning_objectives'])}")
        
        # Recommendations
        if report['recommendations']:
            print("\n" + "="*60)
            print("LEARNING RECOMMENDATIONS")
            print("="*60)
            for rec in report['recommendations']:
                print(f"• {rec}")
        
        # Next steps
        if report['next_steps']:
            print("\n" + "="*60)
            print("SUGGESTED NEXT STEPS")
            print("="*60)
            for step in report['next_steps']:
                print(f"• {step}")
        
        # Save to file if requested
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\nDetailed report saved to: {output}")
            except Exception as e:
                print(f"Error saving report: {str(e)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def run_evaluation():
    """Run model evaluation on benchmark tests"""
    print("Starting model evaluation...")
    print("This will test the model against benchmark cases to validate performance.")
    
    try:
        evaluator = ModelEvaluator()
        
        # Test with a subset of models (can expand based on resources)
        models_to_test = ["codellama-7b"]  # Start with smaller model
        
        print(f"Testing models: {models_to_test}")
        results = evaluator.run_comprehensive_evaluation(models_to_test)
        
        # Display results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\nModel: {result.model_name}")
            print(f"Code Understanding: {result.code_understanding_score:.2f}/1.0")
            print(f"Prompt Quality: {result.prompt_quality_score:.2f}/1.0") 
            print(f"Educational Alignment: {result.educational_alignment_score:.2f}/1.0")
            print(f"Memory Usage: {result.memory_usage:.2f} GB")
            
            overall = (result.code_understanding_score + 
                      result.prompt_quality_score + 
                      result.educational_alignment_score) / 3
            print(f"Overall Score: {overall:.2f}/1.0")
        
        # Save results
        evaluator.save_results(results, "evaluation_results.json")
        print(f"\nDetailed results saved to: evaluation_results.json")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def run_demo():
    """Run demonstration with sample student code"""
    print("Running demonstration with sample student code...")
    
    # Sample code with common beginner issues
    demo_code = '''
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):  # Common misconception
        total = total + numbers[i]
    average = total / len(numbers)  # No error handling
    return average

def find_maximum(items):
    max_item = items[0]  # No empty list check
    for item in items:
        if item > max_item:
            max_item = item
    return max_item

# Test the functions
my_numbers = [1, 2, 3, 4, 5]
avg = calculate_average(my_numbers)
print("Average:", avg)

my_items = [10, 5, 8, 3, 9]
max_val = find_maximum(my_items)
print("Maximum:", max_val)
'''
    
    print("Sample code to analyze:")
    print("-" * 40)
    print(demo_code)
    print("-" * 40)
    
    # Create temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(demo_code)
        temp_file = f.name
    
    try:
        analyze_file(temp_file, expected_level="beginner")
    finally:
        # Clean up temporary file
        Path(temp_file).unlink()

def run_benchmark():
    """Run benchmark tests to validate system performance"""
    print("Running benchmark tests...")
    
    try:
        from data.evaluation_benchmarks.benchmark_tests import run_benchmark_evaluation
        from competence_analyzer import PythonCompetenceAnalyzer
        
        # Create analyzer wrapper for benchmark
        analyzer = PythonCompetenceAnalyzer()
        
        def analyze_for_benchmark(code):
            analysis = analyzer.analyze_code(code)
            return {
                'competence_level': analysis.competence_level.value,
                'identified_concepts': analysis.identified_concepts,
                'potential_issues': analysis.misconceptions
            }
        
        # Run benchmark evaluation
        results = run_benchmark_evaluation(analyze_for_benchmark)
        
        # Display results
        print("\n" + "="*60)
        print("BENCHMARK TEST RESULTS")
        print("="*60)
        
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Competence Prediction Accuracy: {results['competence_prediction_accuracy']:.3f}")
        print(f"Concept Identification Accuracy: {results['concept_identification_accuracy']:.3f}")
        print(f"Issue Detection Accuracy: {results['issue_detection_accuracy']:.3f}")
        print(f"Total Tests: {results['total_tests']}")
        
        if results['recommendations']:
            print("\nRecommendations for improvement:")
            for rec in results['recommendations']:
                print(f"• {rec}")
        
        # Save detailed results
        with open("benchmark_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed benchmark results saved to: benchmark_results.json")
        
    except ImportError:
        print("Benchmark module not found. Please ensure all dependencies are installed.")
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()

def check_system():
    """Check system requirements and dependencies"""
    print("Checking system requirements...")
    
    import torch
    import psutil
    
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("No CUDA GPUs detected. System will use CPU (slower performance).")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # CPU info
    print(f"CPU cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
    
    # Check critical imports
    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    dependencies = [
        ('transformers', 'Hugging Face Transformers'),
        ('accelerate', 'Accelerate for model loading'),
        ('bitsandbytes', 'Quantization support'),
        ('torch', 'PyTorch'),
    ]
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {description}: OK")
        except ImportError:
            print(f"❌ {description}: MISSING - install with 'pip install {module}'")
    
    # Test model loading
    print("\n" + "="*60)
    print("MODEL LOADING TEST")
    print("="*60)
    
    try:
        print("Attempting to load CodeLlama-13B-Instruct...")
        start_time = time.time()
        analyzer = PythonCompetenceAnalyzer()
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
        
        # Quick functionality test
        test_code = "def hello_world():\n    print('Hello, World!')"
        analysis = analyzer.analyze_code(test_code)
        print(f"✅ Analysis test passed - detected level: {analysis.competence_level.value}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print("This could be due to insufficient memory, missing model files, or network issues.")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Python Student Competence Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze student_code.py
  python main.py analyze student_code.py -l intermediate -o report.json
  python main.py demo
  python main.py evaluate
  python main.py benchmark
  python main.py check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a Python file')
    analyze_parser.add_argument('file', help='Path to Python file to analyze')
    analyze_parser.add_argument('-l', '--level', 
                               choices=['beginner', 'intermediate', 'advanced'],
                               help='Expected student competence level')
    analyze_parser.add_argument('-o', '--output', 
                               help='Save detailed report to JSON file')
    
    # Other commands
    subparsers.add_parser('demo', help='Run demonstration with sample code')
    subparsers.add_parser('evaluate', help='Run model evaluation')
    subparsers.add_parser('benchmark', help='Run benchmark tests')
    subparsers.add_parser('check', help='Check system requirements and setup')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_file(args.file, args.level, args.output)
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'evaluate':
        run_evaluation()
    elif args.command == 'benchmark':
        run_benchmark()
    elif args.command == 'check':
        check_system()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()