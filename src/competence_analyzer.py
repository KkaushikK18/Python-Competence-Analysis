"""
Python Competence Analyzer
Main class for analyzing student Python code and assessing competence levels.
"""

import ast
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CompetenceLevel(Enum):
    """Student competence levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class CodeAnalysis:
    """Results of code analysis"""
    syntax_correct: bool
    complexity_score: float
    best_practices_score: float
    identified_concepts: List[str]
    potential_issues: List[str]
    competence_level: CompetenceLevel
    misconceptions: List[str]

@dataclass 
class EducationalPrompt:
    """Generated educational prompt"""
    prompt_text: str
    competence_level: CompetenceLevel
    learning_objectives: List[str]
    prompt_type: str  # "question", "hint", "challenge", "explanation_request"

class PythonCompetenceAnalyzer:
    """Main class for analyzing Python student competence"""
    
    def __init__(self, model_name: str = "meta-llama/CodeLlama-13b-Instruct-hf"):
        """Initialize the competence analyzer with specified model"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Define Python concepts by competence level
        self.concept_hierarchy = {
            CompetenceLevel.BEGINNER: [
                "variables", "basic_data_types", "print_statements", "input", 
                "basic_arithmetic", "string_operations", "if_statements", 
                "basic_loops", "lists", "basic_functions"
            ],
            CompetenceLevel.INTERMEDIATE: [
                "list_comprehensions", "dictionaries", "file_handling", 
                "exception_handling", "classes", "inheritance", "modules",
                "lambda_functions", "decorators", "generators"
            ],
            CompetenceLevel.ADVANCED: [
                "metaclasses", "context_managers", "async_programming",
                "design_patterns", "algorithms", "data_structures",
                "performance_optimization", "testing", "debugging"
            ]
        }
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def analyze_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Analyze code syntax and identify issues"""
        issues = []
        
        try:
            ast.parse(code)
            syntax_correct = True
        except SyntaxError as e:
            syntax_correct = False
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            syntax_correct = False
            issues.append(f"Parse error: {str(e)}")
        
        return syntax_correct, issues
    
    def calculate_complexity_score(self, code: str) -> float:
        """Calculate code complexity score (0-1 scale)"""
        try:
            tree = ast.parse(code)
            
            complexity_factors = {
                'nested_loops': 0,
                'conditional_depth': 0,
                'function_count': 0,
                'class_count': 0,
                'line_count': len(code.split('\n')),
                'imports': 0
            }
            
            # Traverse AST to count complexity factors
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    complexity_factors['nested_loops'] += 1
                elif isinstance(node, ast.If):
                    complexity_factors['conditional_depth'] += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity_factors['function_count'] += 1
                elif isinstance(node, ast.ClassDef):
                    complexity_factors['class_count'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    complexity_factors['imports'] += 1
            
            # Calculate normalized complexity score
            base_score = min(1.0, (
                complexity_factors['nested_loops'] * 0.2 +
                complexity_factors['conditional_depth'] * 0.15 +
                complexity_factors['function_count'] * 0.3 +
                complexity_factors['class_count'] * 0.4 +
                complexity_factors['line_count'] * 0.01 +
                complexity_factors['imports'] * 0.1
            ) / 10)
            
            return base_score
            
        except:
            return 0.0
    
    def assess_best_practices(self, code: str) -> float:
        """Assess adherence to Python best practices"""
        score = 1.0
        
        # Check various best practices
        practices = {
            'proper_naming': self._check_naming_conventions(code),
            'documentation': self._check_documentation(code),
            'error_handling': self._check_error_handling(code),
            'code_structure': self._check_code_structure(code),
            'pythonic_patterns': self._check_pythonic_patterns(code)
        }
        
        # Calculate weighted score
        weights = {
            'proper_naming': 0.2,
            'documentation': 0.15,
            'error_handling': 0.25,
            'code_structure': 0.2,
            'pythonic_patterns': 0.2
        }
        
        weighted_score = sum(practices[key] * weights[key] for key in practices)
        return max(0.0, min(1.0, weighted_score))
    
    def _check_naming_conventions(self, code: str) -> float:
        """Check if naming conventions are followed"""
        try:
            tree = ast.parse(code)
            violations = 0
            total_names = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_names += 1
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        violations += 1
                elif isinstance(node, ast.ClassDef):
                    total_names += 1
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        violations += 1
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    total_names += 1
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                        violations += 1
            
            if total_names == 0:
                return 1.0
            
            return max(0.0, 1.0 - (violations / total_names))
            
        except:
            return 0.5
    
    def _check_documentation(self, code: str) -> float:
        """Check for documentation and comments"""
        lines = code.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 0.0
        
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        docstring_lines = len([line for line in lines if '"""' in line or "'''" in line])
        
        doc_ratio = (comment_lines + docstring_lines * 2) / total_lines
        return min(1.0, doc_ratio * 2)
    
    def _check_error_handling(self, code: str) -> float:
        """Check for proper error handling"""
        try:
            tree = ast.parse(code)
            
            try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            
            if functions == 0:
                return 1.0 if try_blocks > 0 else 0.7
            
            return min(1.0, try_blocks / functions + 0.3)
            
        except:
            return 0.5
    
    def _check_code_structure(self, code: str) -> float:
        """Check code structure and organization"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) == 0:
            return 0.0
        
        # Check for proper function/class organization
        structure_score = 0.7  # Base score
        
        # Bonus for using functions
        if 'def ' in code:
            structure_score += 0.2
        
        # Bonus for using classes appropriately
        if 'class ' in code and len(non_empty_lines) > 20:
            structure_score += 0.1
        
        return min(1.0, structure_score)
    
    def _check_pythonic_patterns(self, code: str) -> float:
        """Check for Pythonic patterns and idioms"""
        pythonic_patterns = [
            r'for .+ in .+:',  # for loops instead of range(len())
            r'with open\(',    # context managers
            r'\[.+ for .+ in .+\]',  # list comprehensions
            r'\.join\(',       # string joining
            r'enumerate\(',    # enumerate usage
            r'zip\(',          # zip usage
        ]
        
        score = 0.5  # Base score
        
        for pattern in pythonic_patterns:
            if re.search(pattern, code):
                score += 0.1
        
        return min(1.0, score)
    
    def identify_concepts(self, code: str) -> List[str]:
        """Identify Python concepts present in the code"""
        concepts = []
        
        try:
            tree = ast.parse(code)
            
            # Check for various Python concepts
            concept_checks = {
                'variables': lambda: any(isinstance(node, ast.Name) for node in ast.walk(tree)),
                'functions': lambda: any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)),
                'classes': lambda: any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
                'loops': lambda: any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)),
                'conditionals': lambda: any(isinstance(node, ast.If) for node in ast.walk(tree)),
                'lists': lambda: any(isinstance(node, ast.List) for node in ast.walk(tree)),
                'dictionaries': lambda: any(isinstance(node, ast.Dict) for node in ast.walk(tree)),
                'list_comprehensions': lambda: any(isinstance(node, ast.ListComp) for node in ast.walk(tree)),
                'exception_handling': lambda: any(isinstance(node, ast.Try) for node in ast.walk(tree)),
                'imports': lambda: any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
                'lambda_functions': lambda: any(isinstance(node, ast.Lambda) for node in ast.walk(tree)),
                'generators': lambda: any(isinstance(node, ast.GeneratorExp) for node in ast.walk(tree)),
                'decorators': lambda: any(len(node.decorator_list) > 0 for node in ast.walk(tree) 
                                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
            }
            
            for concept, check_func in concept_checks.items():
                if check_func():
                    concepts.append(concept)
            
        except:
            pass
        
        return concepts
    
    def detect_misconceptions(self, code: str) -> List[str]:
        """Detect common Python misconceptions in student code"""
        misconceptions = []
        
        # Common misconception patterns
        misconception_patterns = [
            (r'for i in range\(len\(.+\)\):', 'Using range(len()) instead of direct iteration'),
            (r'==\s*True|==\s*False', 'Explicitly comparing to True/False'),
            (r'if len\(.+\) > 0:', 'Using len() to check if container is empty'),
            (r'\.append\(.+\)\s*\n\s*return', 'Trying to return result of append()'),
            (r'global \w+\s*\n.*\w+\s*=', 'Overusing global variables'),
            (r'except:', 'Using bare except clause'),
            (r'import \*', 'Using wildcard imports')
        ]
        
        for pattern, description in misconception_patterns:
            if re.search(pattern, code, re.MULTILINE):
                misconceptions.append(description)
        
        return misconceptions
    
    def determine_competence_level(self, analysis: CodeAnalysis) -> CompetenceLevel:
        """Determine overall competence level based on analysis"""
        
        # Count concepts by level
        concept_counts = {level: 0 for level in CompetenceLevel}
        
        for concept in analysis.identified_concepts:
            for level, level_concepts in self.concept_hierarchy.items():
                if concept in level_concepts:
                    concept_counts[level] += 1
        
        # Calculate scores
        complexity_factor = analysis.complexity_score
        practices_factor = analysis.best_practices_score
        
        # Determine level based on multiple factors
        if (concept_counts[CompetenceLevel.ADVANCED] >= 2 and 
            complexity_factor > 0.6 and 
            practices_factor > 0.7):
            return CompetenceLevel.ADVANCED
        elif (concept_counts[CompetenceLevel.INTERMEDIATE] >= 3 or
              concept_counts[CompetenceLevel.ADVANCED] >= 1) and complexity_factor > 0.3:
            return CompetenceLevel.INTERMEDIATE
        else:
            return CompetenceLevel.BEGINNER
    
    def analyze_code(self, code: str, student_level: Optional[CompetenceLevel] = None) -> CodeAnalysis:
        """Complete code analysis pipeline"""
        
        # Basic syntax analysis
        syntax_correct, syntax_issues = self.analyze_syntax(code)
        
        # Calculate metrics
        complexity_score = self.calculate_complexity_score(code)
        best_practices_score = self.assess_best_practices(code)
        
        # Identify concepts and issues
        identified_concepts = self.identify_concepts(code)
        misconceptions = self.detect_misconceptions(code)
        
        # Create preliminary analysis
        preliminary_analysis = CodeAnalysis(
            syntax_correct=syntax_correct,
            complexity_score=complexity_score,
            best_practices_score=best_practices_score,
            identified_concepts=identified_concepts,
            potential_issues=syntax_issues + misconceptions,
            competence_level=CompetenceLevel.BEGINNER,  # Will be updated
            misconceptions=misconceptions
        )
        
        # Determine competence level
        competence_level = student_level or self.determine_competence_level(preliminary_analysis)
        preliminary_analysis.competence_level = competence_level
        
        return preliminary_analysis
    
    def generate_llm_response(self, prompt: str) -> str:
        """Generate response using the loaded LLM"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Format prompt for instruction-tuned models
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        # Tokenize and generate
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_educational_prompts(self, analysis: CodeAnalysis, code: str) -> List[EducationalPrompt]:
        """Generate educational prompts based on code analysis"""
        prompts = []
        
        # Generate different types of prompts based on analysis
        if not analysis.syntax_correct:
            prompts.extend(self._generate_syntax_prompts(analysis, code))
        
        if analysis.misconceptions:
            prompts.extend(self._generate_misconception_prompts(analysis, code))
        
        if analysis.best_practices_score < 0.7:
            prompts.extend(self._generate_best_practices_prompts(analysis, code))
        
        # Generate level-appropriate challenge prompts
        prompts.extend(self._generate_challenge_prompts(analysis, code))
        
        return prompts
    
    def _generate_syntax_prompts(self, analysis: CodeAnalysis, code: str) -> List[EducationalPrompt]:
        """Generate prompts for syntax issues"""
        prompts = []
        
        for issue in analysis.potential_issues:
            if "Syntax error" in issue:
                prompt_text = f"""
Looking at your code, there seems to be a syntax issue. 
Can you review the code and identify what might be causing the problem?
Think about:
- Are all parentheses, brackets, and quotes properly closed?
- Are the indentation levels consistent?
- Are there any typos in keywords?

Don't worry about fixing it immediately - first try to spot what looks unusual.
"""
                
                prompts.append(EducationalPrompt(
                    prompt_text=prompt_text.strip(),
                    competence_level=CompetenceLevel.BEGINNER,
                    learning_objectives=["syntax_awareness", "debugging_skills"],
                    prompt_type="question"
                ))
        
        return prompts
    
    def _generate_misconception_prompts(self, analysis: CodeAnalysis, code: str) -> List[EducationalPrompt]:
        """Generate prompts to address misconceptions"""
        prompts = []
        
        misconception_prompts = {
            "Using range(len()) instead of direct iteration": """
I notice you're using range(len()) to iterate through a list. 
This works, but Python offers a more elegant approach. 
Can you think of a way to iterate directly over the list elements 
without needing to access indices? What would be the advantages?
""",
            "Explicitly comparing to True/False": """
I see you're comparing a value to True or False explicitly. 
In Python, we can often simplify these comparisons. 
How could you rewrite this condition to be more Pythonic?
""",
            "Using len() to check if container is empty": """
You're using len() to check if a container is empty. 
While this works, Python has a more idiomatic way. 
Can you think of a simpler way to check if a list, string, or dictionary is empty?
"""
        }
        
        for misconception in analysis.misconceptions:
            if misconception in misconception_prompts:
                prompts.append(EducationalPrompt(
                    prompt_text=misconception_prompts[misconception].strip(),
                    competence_level=analysis.competence_level,
                    learning_objectives=["pythonic_patterns", "best_practices"],
                    prompt_type="hint"
                ))
        
        return prompts
    
    def _generate_best_practices_prompts(self, analysis: CodeAnalysis, code: str) -> List[EducationalPrompt]:
        """Generate prompts for improving best practices"""
        prompts = []
        
        # Check specific best practices issues
        if self._check_documentation(code) < 0.3:
            prompts.append(EducationalPrompt(
                prompt_text="""
Good code tells a story not just to the computer, but to other humans too.
Looking at your code, how could you make it clearer what each part does?
Consider adding comments or docstrings to explain your reasoning.
What would a future version of yourself need to know to understand this code?
""",
                competence_level=analysis.competence_level,
                learning_objectives=["documentation", "code_readability"],
                prompt_type="challenge"
            ))
        
        if self._check_error_handling(code) < 0.5 and analysis.competence_level != CompetenceLevel.BEGINNER:
            prompts.append(EducationalPrompt(
                prompt_text="""
What could go wrong when this code runs? Think about edge cases:
- What if the input is unexpected?
- What if a file doesn't exist?
- What if a network connection fails?
How could you make your code more robust to handle these situations gracefully?
""",
                competence_level=analysis.competence_level,
                learning_objectives=["error_handling", "robust_programming"],
                prompt_type="question"
            ))
        
        return prompts
    
    def _generate_challenge_prompts(self, analysis: CodeAnalysis, code: str) -> List[EducationalPrompt]:
        """Generate level-appropriate challenge prompts"""
        prompts = []
        
        level_challenges = {
            CompetenceLevel.BEGINNER: [
                "Can you add input validation to make your code more robust?",
                "How would you modify this to work with different types of data?",
                "What would happen if the input was empty? How could you handle that?"
            ],
            CompetenceLevel.INTERMEDIATE: [
                "How could you optimize this code for better performance?",
                "Can you refactor this using object-oriented principles?",
                "What design patterns might be applicable here?"
            ],
            CompetenceLevel.ADVANCED: [
                "How would you make this code thread-safe?",
                "Can you implement this using functional programming concepts?",
                "What would a comprehensive test suite for this code look like?"
            ]
        }
        
        challenges = level_challenges.get(analysis.competence_level, level_challenges[CompetenceLevel.BEGINNER])
        
        # Select one random challenge
        import random
        selected_challenge = random.choice(challenges)
        
        prompts.append(EducationalPrompt(
            prompt_text=selected_challenge,
            competence_level=analysis.competence_level,
            learning_objectives=["problem_solving", "advanced_concepts"],
            prompt_type="challenge"
        ))
        
        return prompts
    
    def get_competence_report(self, code: str, student_level: Optional[CompetenceLevel] = None) -> Dict[str, Any]:
        """Generate comprehensive competence analysis report"""
        
        # Analyze code
        analysis = self.analyze_code(code, student_level)
        
        # Generate educational prompts
        prompts = self.generate_educational_prompts(analysis, code)
        
        # Create comprehensive report
        report = {
            "code_analysis": {
                "syntax_correct": analysis.syntax_correct,
                "complexity_score": round(analysis.complexity_score, 2),
                "best_practices_score": round(analysis.best_practices_score, 2),
                "identified_concepts": analysis.identified_concepts,
                "potential_issues": analysis.potential_issues,
                "competence_level": analysis.competence_level.value,
                "misconceptions": analysis.misconceptions
            },
            "educational_prompts": [
                {
                    "prompt": prompt.prompt_text,
                    "type": prompt.prompt_type,
                    "learning_objectives": prompt.learning_objectives,
                    "competence_level": prompt.competence_level.value
                }
                for prompt in prompts
            ],
            "recommendations": self._generate_recommendations(analysis),
            "next_steps": self._suggest_next_steps(analysis)
        }
        
        return report
    
    def _generate_recommendations(self, analysis: CodeAnalysis) -> List[str]:
        """Generate learning recommendations based on analysis"""
        recommendations = []
        
        if analysis.best_practices_score < 0.5:
            recommendations.append("Focus on Python coding style and best practices")
        
        if "exception_handling" not in analysis.identified_concepts and analysis.competence_level != CompetenceLevel.BEGINNER:
            recommendations.append("Learn about exception handling and error management")
        
        if analysis.complexity_score < 0.3 and analysis.competence_level != CompetenceLevel.BEGINNER:
            recommendations.append("Practice with more complex algorithmic problems")
        
        if len(analysis.misconceptions) > 2:
            recommendations.append("Review fundamental Python concepts and idioms")
        
        return recommendations
    
    def _suggest_next_steps(self, analysis: CodeAnalysis) -> List[str]:
        """Suggest next learning steps"""
        next_steps = []
        
        level_progressions = {
            CompetenceLevel.BEGINNER: [
                "Practice with more functions and parameter passing",
                "Learn about data structures (lists, dictionaries, sets)",
                "Explore file handling and basic I/O operations"
            ],
            CompetenceLevel.INTERMEDIATE: [
                "Study object-oriented programming concepts",
                "Learn about modules and package organization",
                "Practice with decorators and context managers"
            ],
            CompetenceLevel.ADVANCED: [
                "Explore advanced topics like metaclasses and descriptors",
                "Learn about concurrency and async programming",
                "Study software design patterns and architecture"
            ]
        }
        
        return level_progressions.get(analysis.competence_level, level_progressions[CompetenceLevel.BEGINNER])

# Example usage and testing
if __name__ == "__main__":
    # Example student code for testing
    sample_code = """
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    average = total / len(numbers)
    return average

# Test the function
my_numbers = [1, 2, 3, 4, 5]
result = calculate_average(my_numbers)
print("The average is:", result)
"""
    
    # Initialize analyzer
    try:
        analyzer = PythonCompetenceAnalyzer()
        
        # Generate competence report
        report = analyzer.get_competence_report(sample_code)
        
        # Print results
        print("=== PYTHON COMPETENCE ANALYSIS REPORT ===")
        print(f"Competence Level: {report['code_analysis']['competence_level']}")
        print(f"Complexity Score: {report['code_analysis']['complexity_score']}")
        print(f"Best Practices Score: {report['code_analysis']['best_practices_score']}")
        print(f"Identified Concepts: {', '.join(report['code_analysis']['identified_concepts'])}")
        
        if report['code_analysis']['misconceptions']:
            print(f"Misconceptions Found: {', '.join(report['code_analysis']['misconceptions'])}")
        
        print("\n=== EDUCATIONAL PROMPTS ===")
        for i, prompt in enumerate(report['educational_prompts'], 1):
            print(f"\nPrompt {i} ({prompt['type']}):")
            print(prompt['prompt'])
        
        print("\n=== RECOMMENDATIONS ===")
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        print("\n=== NEXT STEPS ===")
        for step in report['next_steps']:
            print(f"• {step}")
            
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        print("Please ensure you have the required model and dependencies installed.")