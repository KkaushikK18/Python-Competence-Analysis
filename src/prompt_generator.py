"""
Educational Prompt Generator
Specialized module for generating educational prompts and questions
that assess Python competence without giving away solutions.
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class PromptCategory(Enum):
    """Categories of educational prompts"""
    CONCEPTUAL = "conceptual"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    BEST_PRACTICES = "best_practices"
    EXTENSION = "extension"
    REFLECTION = "reflection"

@dataclass
class PromptTemplate:
    """Template for generating educational prompts"""
    category: PromptCategory
    competence_level: str
    template: str
    learning_objectives: List[str]
    follow_up_questions: List[str]
    sample_responses: List[str]

class EducationalPromptGenerator:
    """Advanced prompt generator for Python competence assessment"""
    
    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()
        self.concept_questions = self._load_concept_questions()
        self.debugging_scenarios = self._load_debugging_scenarios()
    
    def _load_prompt_templates(self) -> Dict[str, List[PromptTemplate]]:
        """Load comprehensive prompt templates"""
        templates = {
            "beginner": [
                PromptTemplate(
                    category=PromptCategory.CONCEPTUAL,
                    competence_level="beginner",
                    template="Looking at this {concept}, can you explain in your own words what it does and why you chose to use it here?",
                    learning_objectives=["concept_understanding", "verbalization"],
                    follow_up_questions=[
                        "What would happen if you removed this part?",
                        "Can you think of an alternative approach?"
                    ],
                    sample_responses=["Good explanations show understanding of purpose and context"]
                ),
                PromptTemplate(
                    category=PromptCategory.DEBUGGING,
                    competence_level="beginner",
                    template="I notice something interesting in your code. What do you think would happen if {scenario}? Try to trace through it step by step.",
                    learning_objectives=["debugging_skills", "logical_reasoning"],
                    follow_up_questions=[
                        "How would you test this?",
                        "What output would you expect?"
                    ],
                    sample_responses=["Shows ability to trace code execution"]
                ),
                PromptTemplate(
                    category=PromptCategory.EXTENSION,
                    competence_level="beginner",
                    template="Your code works well for this case. How would you modify it to handle {extension_scenario}?",
                    learning_objectives=["problem_extension", "adaptability"],
                    follow_up_questions=[
                        "What new challenges does this create?",
                        "What additional information would you need?"
                    ],
                    sample_responses=["Demonstrates ability to extend existing solutions"]
                )
            ],
            "intermediate": [
                PromptTemplate(
                    category=PromptCategory.OPTIMIZATION,
                    competence_level="intermediate",
                    template="Your solution works correctly. Now, considering performance, what aspects of this code might become problematic with larger inputs?",
                    learning_objectives=["performance_analysis", "scalability"],
                    follow_up_questions=[
                        "How would you measure the improvement?",
                        "What trade-offs are involved?"
                    ],
                    sample_responses=["Shows awareness of algorithmic complexity"]
                ),
                PromptTemplate(
                    category=PromptCategory.BEST_PRACTICES,
                    competence_level="intermediate",
                    template="I see you've implemented {functionality}. What Python idioms or built-in features could make this more elegant?",
                    learning_objectives=["pythonic_code", "language_features"],
                    follow_up_questions=[
                        "What are the benefits of the Pythonic approach?",
                        "Are there any drawbacks to consider?"
                    ],
                    sample_responses=["Demonstrates knowledge of Python idioms"]
                ),
                PromptTemplate(
                    category=PromptCategory.REFLECTION,
                    competence_level="intermediate",
                    template="Reflect on your problem-solving approach here. What made you choose this particular strategy over alternatives?",
                    learning_objectives=["metacognition", "design_reasoning"],
                    follow_up_questions=[
                        "What other approaches did you consider?",
                        "How did you evaluate the trade-offs?"
                    ],
                    sample_responses=["Shows thoughtful design decision process"]
                )
            ],
            "advanced": [
                PromptTemplate(
                    category=PromptCategory.CONCEPTUAL,
                    competence_level="advanced",
                    template="Your implementation demonstrates {concept}. How does this pattern scale in terms of maintainability and extensibility?",
                    learning_objectives=["architectural_thinking", "design_patterns"],
                    follow_up_questions=[
                        "What design patterns are evident here?",
                        "How would you test this architecture?"
                    ],
                    sample_responses=["Shows understanding of software design principles"]
                ),
                PromptTemplate(
                    category=PromptCategory.OPTIMIZATION,
                    competence_level="advanced",
                    template="Analyze the space and time complexity of your solution. Where are the potential bottlenecks, and how would you profile and optimize them?",
                    learning_objectives=["complexity_analysis", "optimization_strategies"],
                    follow_up_questions=[
                        "What profiling tools would you use?",
                        "How would you validate performance improvements?"
                    ],
                    sample_responses=["Demonstrates deep understanding of performance analysis"]
                )
            ]
        }
        return templates
    
    def _load_concept_questions(self) -> Dict[str, List[str]]:
        """Load concept-specific questions for different Python features"""
        return {
            "list_comprehensions": [
                "What advantages do list comprehensions offer over traditional loops?",
                "When might a traditional loop be preferable to a list comprehension?",
                "How does this list comprehension improve code readability?"
            ],
            "exception_handling": [
                "What specific exceptions are you anticipating here?",
                "How does your error handling improve the user experience?",
                "What information should be preserved when an exception occurs?"
            ],
            "functions": [
                "Why did you choose to break this into separate functions?",
                "How does this function design support code reusability?",
                "What makes this function easy to test?"
            ],
            "classes": [
                "What real-world entity does this class represent?",
                "How does this class design follow object-oriented principles?",
                "What would happen if you needed to add new functionality?"
            ],
            "data_structures": [
                "Why is this data structure the best choice for your use case?",
                "What operations does this structure optimize for?",
                "How would performance change with different data structures?"
            ]
        }
    
    def _load_debugging_scenarios(self) -> Dict[str, List[Dict]]:
        """Load debugging scenarios by competence level"""
        return {
            "beginner": [
                {
                    "scenario": "the input list is empty",
                    "focus": "edge case handling",
                    "hint": "Consider what happens when you try to access elements that don't exist"
                },
                {
                    "scenario": "the user inputs a string instead of a number",
                    "focus": "input validation",
                    "hint": "Think about type checking and conversion"
                },
                {
                    "scenario": "you run this with very large numbers",
                    "focus": "boundary conditions",
                    "hint": "Consider memory and processing limits"
                }
            ],
            "intermediate": [
                {
                    "scenario": "multiple threads try to access this data simultaneously",
                    "focus": "concurrency issues",
                    "hint": "Think about race conditions and data integrity"
                },
                {
                    "scenario": "the file you're trying to read doesn't exist",
                    "focus": "error handling",
                    "hint": "Consider graceful degradation and user feedback"
                },
                {
                    "scenario": "the network connection is slow or unreliable",
                    "focus": "resilience and timeouts",
                    "hint": "Think about retry logic and fallback mechanisms"
                }
            ],
            "advanced": [
                {
                    "scenario": "this needs to process millions of records efficiently",
                    "focus": "scalability and performance",
                    "hint": "Consider memory usage, algorithmic complexity, and optimization strategies"
                },
                {
                    "scenario": "this code needs to be maintained by a team over several years",
                    "focus": "maintainability and documentation",
                    "hint": "Think about code clarity, documentation, and future extensibility"
                }
            ]
        }
    
    def generate_conceptual_prompt(self, concept: str, competence_level: str, code_context: str = "") -> Dict:
        """Generate a prompt that assesses conceptual understanding"""
        
        concept_questions = self.concept_questions.get(concept, [
            f"Can you explain how {concept} works in your implementation?",
            f"What benefits does {concept} provide in this context?",
            f"How might you modify this {concept} for different requirements?"
        ])
        
        selected_question = random.choice(concept_questions)
        
        # Add context-specific elements
        if code_context:
            context_prompt = f"Looking at your use of {concept} in this code, {selected_question.lower()}"
        else:
            context_prompt = selected_question
        
        return {
            "category": "conceptual",
            "competence_level": competence_level,
            "prompt": context_prompt,
            "learning_objectives": ["concept_understanding", "verbalization"],
            "assessment_criteria": [
                "Demonstrates clear understanding of the concept",
                "Can explain benefits and use cases",
                "Shows awareness of alternatives"
            ]
        }
    
    def generate_debugging_prompt(self, competence_level: str, code_snippet: str = "") -> Dict:
        """Generate a debugging-focused prompt"""
        
        scenarios = self.debugging_scenarios.get(competence_level, self.debugging_scenarios["beginner"])
        selected_scenario = random.choice(scenarios)
        
        base_prompt = f"Let's think about potential issues. What do you think would happen if {selected_scenario['scenario']}?"
        
        if code_snippet:
            full_prompt = f"{base_prompt}\n\nLook at your code and trace through what would happen step by step."
        else:
            full_prompt = base_prompt
        
        return {
            "category": "debugging",
            "competence_level": competence_level,
            "prompt": full_prompt,
            "hint": selected_scenario["hint"],
            "focus_area": selected_scenario["focus"],
            "learning_objectives": ["debugging_skills", "edge_case_awareness", "logical_reasoning"],
            "assessment_criteria": [
                "Can identify potential problems",
                "Shows systematic thinking about edge cases",
                "Proposes reasonable solutions"
            ]
        }
    
    def generate_extension_prompt(self, competence_level: str, current_functionality: str) -> Dict:
        """Generate a prompt that asks students to extend their solution"""
        
        extension_scenarios = {
            "beginner": [
                "handle different types of input data",
                "work with empty or invalid inputs",
                "provide more detailed output",
                "add simple error messages"
            ],
            "intermediate": [
                "support multiple data formats",
                "handle concurrent operations",
                "add configuration options",
                "implement caching for performance"
            ],
            "advanced": [
                "scale to handle large datasets",
                "support plugin architecture",
                "implement comprehensive logging",
                "add monitoring and metrics"
            ]
        }
        
        scenarios = extension_scenarios.get(competence_level, extension_scenarios["beginner"])
        selected_extension = random.choice(scenarios)
        
        prompt = f"Your current solution handles {current_functionality} well. " \
                f"How would you extend it to {selected_extension}? " \
                f"Walk me through your thinking process."
        
        return {
            "category": "extension",
            "competence_level": competence_level,
            "prompt": prompt,
            "extension_target": selected_extension,
            "learning_objectives": ["problem_extension", "design_thinking", "adaptability"],
            "assessment_criteria": [
                "Identifies key challenges in the extension",
                "Proposes reasonable architectural changes",
                "Considers backward compatibility"
            ]
        }
    
    def generate_optimization_prompt(self, competence_level: str, code_analysis: Dict) -> Dict:
        """Generate prompts focused on code optimization"""
        
        optimization_focuses = {
            "beginner": [
                "code readability and clarity",
                "removing unnecessary repetition",
                "using appropriate Python built-ins"
            ],
            "intermediate": [
                "algorithmic efficiency",
                "memory usage optimization",
                "leveraging Python idioms and patterns"
            ],
            "advanced": [
                "performance profiling and bottleneck identification",
                "algorithmic complexity analysis",
                "system-level optimization strategies"
            ]
        }
        
        focuses = optimization_focuses.get(competence_level, optimization_focuses["beginner"])
        selected_focus = random.choice(focuses)
        
        if competence_level == "beginner":
            prompt = f"Looking at your code, how could you improve {selected_focus}? " \
                    f"What changes would make it easier to read and understand?"
        elif competence_level == "intermediate":
            prompt = f"Consider {selected_focus} in your solution. " \
                    f"What aspects could be optimized, and how would you measure the improvement?"
        else:  # advanced
            prompt = f"Analyze your solution from the perspective of {selected_focus}. " \
                    f"What tools and techniques would you use to identify and address performance issues?"
        
        return {
            "category": "optimization",
            "competence_level": competence_level,
            "prompt": prompt,
            "focus_area": selected_focus,
            "learning_objectives": ["optimization_thinking", "performance_awareness", "tool_usage"],
            "assessment_criteria": [
                "Identifies relevant optimization opportunities",
                "Understands trade-offs involved",
                "Suggests appropriate measurement techniques"
            ]
        }
    
    def generate_reflection_prompt(self, competence_level: str) -> Dict:
        """Generate prompts that encourage metacognitive reflection"""
        
        reflection_prompts = {
            "beginner": [
                "What was the most challenging part of solving this problem?",
                "If you had to explain your solution to a friend, how would you describe your approach?",
                "What would you do differently if you solved this problem again?"
            ],
            "intermediate": [
                "What alternative approaches did you consider, and why did you choose this one?",
                "How confident are you in your solution, and what would increase that confidence?",
                "What aspects of this problem reminded you of other programming challenges?"
            ],
            "advanced": [
                "How does this solution fit into broader software design principles?",
                "What assumptions did you make, and how did they influence your design decisions?",
                "How would you evaluate the maintainability and extensibility of this approach?"
            ]
        }
        
        prompts = reflection_prompts.get(competence_level, reflection_prompts["beginner"])
        selected_prompt = random.choice(prompts)
        
        return {
            "category": "reflection",
            "competence_level": competence_level,
            "prompt": selected_prompt,