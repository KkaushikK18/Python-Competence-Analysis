"""
Model Evaluation Framework for Python Student Competence Analysis
Compares different open source models for educational code analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Tuple, Optional
import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model evaluation"""
    name: str
    model_id: str
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True

@dataclass
class EvaluationResult:
    """Results from model evaluation"""
    model_name: str
    code_understanding_score: float
    prompt_quality_score: float
    educational_alignment_score: float
    response_time: float
    memory_usage: float

class ModelEvaluator:
    """Evaluates different models for Python competence analysis"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.results = []
        
        # Define model configurations
        # NOTE: Can easily add more models to this list as they become available
        self.model_configs = {
            "codellama-13b": ModelConfig(
                name="CodeLlama-13B-Instruct",
                model_id="meta-llama/CodeLlama-13b-Instruct-hf"
            ),
            "starcoder2-15b": ModelConfig(
                name="StarCoder2-15B",
                model_id="bigcode/starcoder2-15b"
            ),
            "codellama-7b": ModelConfig(
                name="CodeLlama-7B-Instruct", 
                model_id="meta-llama/CodeLlama-7b-Instruct-hf"
            )
        }
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup 4-bit quantization for efficient model loading"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self, model_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with quantization"""
        config = self.model_configs[model_key]
        
        logger.info(f"Loading {config.name}...")
        # This can take a while on first run, but subsequent loads are cached
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Setup quantization
            quantization_config = self.setup_quantization_config() if config.load_in_4bit else None
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Store loaded model and tokenizer
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            logger.info(f"Successfully loaded {config.name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {str(e)}")
            return None, None
    
    def generate_response(self, model_key: str, prompt: str) -> Tuple[str, float]:
        """Generate response from model and measure time"""
        model = self.models.get(model_key)
        tokenizer = self.tokenizers.get(model_key)
        
        if not model or not tokenizer:
            logger.error(f"Model {model_key} not loaded")
            return "", 0.0
        
        # Format prompt for instruction-tuned models
        if "instruct" in model_key.lower():
            formatted_prompt = f"[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip(), response_time
    
    def evaluate_code_understanding(self, model_key: str) -> float:
        """Evaluate model's code understanding capabilities"""
        test_cases = [
            {
                "code": """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Student's implementation has exponential time complexity
result = fibonacci(10)
print(result)
""",
                "prompt": "Analyze this Python code for efficiency and suggest improvements without giving the complete solution. Focus on time complexity issues."
            },
            {
                "code": """
def find_max(numbers):
    max_num = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > max_num:
            max_num = numbers[i]
    return max_num

# Student didn't handle empty list case
""",
                "prompt": "What potential issues exist in this code? Generate a question that helps the student discover the problem themselves."
            },
            {
                "code": """
class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        self.balance -= amount
        return self.balance

# Missing validation and error handling
""",
                "prompt": "Evaluate this class implementation and create educational prompts that guide the student to identify missing validations."
            }
        ]
        
        total_score = 0.0
        for test_case in test_cases:
            full_prompt = f"Code to analyze:\n{test_case['code']}\n\nTask: {test_case['prompt']}"
            response, _ = self.generate_response(model_key, full_prompt)
            
            # Score based on response quality (simplified scoring)
            score = self.score_code_analysis(response, test_case)
            total_score += score
            
            logger.info(f"Code understanding test score: {score:.2f}")
        
        return total_score / len(test_cases)
    
    def evaluate_prompt_generation(self, model_key: str) -> float:
        """Evaluate model's ability to generate educational prompts"""
        scenarios = [
            "Generate a question that tests understanding of Python list comprehensions without giving away the syntax.",
            "Create a prompt that helps a student discover why their recursive function causes stack overflow.",
            "Design a question that assesses understanding of object-oriented inheritance concepts in Python."
        ]
        
        total_score = 0.0
        for scenario in scenarios:
            response, _ = self.generate_response(model_key, scenario)
            score = self.score_prompt_quality(response)
            total_score += score
            
            logger.info(f"Prompt generation score: {score:.2f}")
        
        return total_score / len(scenarios)
    
    def evaluate_educational_alignment(self, model_key: str) -> float:
        """Evaluate how well model aligns with educational objectives"""
        educational_tasks = [
            "A student asks for the complete solution to their homework. How should you respond to maintain educational value?",
            "Generate hints for debugging a Python function without revealing the exact fix.",
            "How would you explain the concept of variable scope to a beginner without using technical jargon?"
        ]
        
        total_score = 0.0
        for task in educational_tasks:
            response, _ = self.generate_response(model_key, task)
            score = self.score_educational_behavior(response)
            total_score += score
            
            logger.info(f"Educational alignment score: {score:.2f}")
        
        return total_score / len(educational_tasks)
    
    def score_code_analysis(self, response: str, test_case: Dict) -> float:
        """Score code analysis quality (simplified heuristic)"""
        score = 0.0
        
        # Check for key indicators of good analysis
        if "efficiency" in response.lower() or "complexity" in response.lower():
            score += 0.3
        if "question" in response.lower() and "?" in response:
            score += 0.3
        if len(response) > 100 and not "def " in response:  # Substantial response without giving code
            score += 0.4
        
        return min(score, 1.0)
    
    def score_prompt_quality(self, response: str) -> float:
        """Score educational prompt quality"""
        score = 0.0
        
        # Check for educational prompt characteristics
        if "?" in response:
            score += 0.2
        if any(word in response.lower() for word in ["why", "how", "what", "explain"]):
            score += 0.3
        if len(response.split()) > 10:  # Substantial prompt
            score += 0.3
        if not any(keyword in response for keyword in ["def ", "import ", "class "]):  # No direct code
            score += 0.2
        
        return min(score, 1.0)
    
    def score_educational_behavior(self, response: str) -> float:
        """Score educational alignment"""
        score = 0.0
        
        # Check for good educational practices
        if any(word in response.lower() for word in ["guide", "help", "learn", "understand"]):
            score += 0.3
        if "solution" not in response.lower() or "complete answer" not in response.lower():
            score += 0.3  # Doesn't give direct solutions
        if any(word in response.lower() for word in ["hint", "suggestion", "consider"]):
            score += 0.4
        
        return min(score, 1.0)
    
    def run_comprehensive_evaluation(self, model_keys: List[str]) -> List[EvaluationResult]:
        """Run complete evaluation for specified models"""
        results = []
        
        for model_key in model_keys:
            logger.info(f"Starting evaluation for {model_key}")
            
            # Load model
            model, tokenizer = self.load_model(model_key)
            if not model:
                continue
            
            # Run evaluations
            code_score = self.evaluate_code_understanding(model_key)
            prompt_score = self.evaluate_prompt_generation(model_key)
            educational_score = self.evaluate_educational_alignment(model_key)
            
            # Calculate memory usage (approximate)
            memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            
            result = EvaluationResult(
                model_name=self.model_configs[model_key].name,
                code_understanding_score=code_score,
                prompt_quality_score=prompt_score,
                educational_alignment_score=educational_score,
                response_time=0.0,  # Average will be calculated separately
                memory_usage=memory_usage
            )
            
            results.append(result)
            logger.info(f"Completed evaluation for {model_key}")
        
        return results
    
    def save_results(self, results: List[EvaluationResult], filepath: str):
        """Save evaluation results to JSON file"""
        results_dict = [
            {
                "model_name": r.model_name,
                "code_understanding_score": r.code_understanding_score,
                "prompt_quality_score": r.prompt_quality_score,
                "educational_alignment_score": r.educational_alignment_score,
                "response_time": r.response_time,
                "memory_usage": r.memory_usage,
                "overall_score": (r.code_understanding_score + r.prompt_quality_score + r.educational_alignment_score) / 3
            }
            for r in results
        ]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")

def main():
    """Main evaluation pipeline"""
    evaluator = ModelEvaluator()
    
    # Models to evaluate (start with smaller models if resources are limited)
    models_to_test = ["codellama-7b"]  # Add others: "codellama-13b", "starcoder2-15b"
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(models_to_test)
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    evaluator.save_results(results, "results/evaluation_results.json")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"\nModel: {result.model_name}")
        print(f"Code Understanding: {result.code_understanding_score:.2f}")
        print(f"Prompt Quality: {result.prompt_quality_score:.2f}")
        print(f"Educational Alignment: {result.educational_alignment_score:.2f}")
        print(f"Memory Usage: {result.memory_usage:.2f} GB")
        overall = (result.code_understanding_score + result.prompt_quality_score + result.educational_alignment_score) / 3
        print(f"Overall Score: {overall:.2f}")

if __name__ == "__main__":
    main()