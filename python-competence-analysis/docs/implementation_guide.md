# Implementation Guide: Python Student Competence Analysis System

## Overview

This guide provides step-by-step instructions for implementing and deploying the Python Student Competence Analysis system using CodeLlama-13B-Instruct. The system is designed for educational institutions wanting to provide automated, intelligent feedback on student Python programming assignments.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface / API                      │
├─────────────────────────────────────────────────────────────┤
│                  Competence Analyzer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │   Code Parser   │  │  LLM Interface  │  │ Prompt Gen.  ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Model Infrastructure                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │ CodeLlama-13B   │  │  Quantization   │  │   Caching    ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
├─────────────────────────────────────────────────────────────┤
│                  Data & Configuration                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │  Student Data   │  │   Benchmarks    │  │    Logs      ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Installation and Setup

### Prerequisites

#### Hardware Requirements

**Minimum Configuration:**
- GPU: 24GB VRAM (RTX 4090, A100-40GB, or equivalent)
- CPU: 16 cores, 3.0GHz+
- RAM: 64GB system memory
- Storage: 100GB free space (SSD recommended)
- Network: Stable internet connection for initial model download

**Recommended Configuration:**
- GPU: 48GB+ VRAM (A100-80GB, H100)
- CPU: 32 cores, 3.5GHz+
- RAM: 128GB system memory
- Storage: 500GB NVMe SSD
- Network: Gigabit ethernet

**Alternative CPU-Only Configuration:**
- CPU: 32+ cores with high memory bandwidth
- RAM: 128GB+ system memory
- Note: ~10x slower inference, suitable for non-real-time applications

#### Software Requirements

```bash
# Operating System
Ubuntu 20.04 LTS or later (recommended)
# Also tested on: CentOS 8, Windows 11 with WSL2

# Python Environment
Python 3.9+ (3.10 recommended)
CUDA 11.8+ (for GPU acceleration)
```

### Step 1: Environment Setup

#### Create Python Environment
```bash
# Using conda (recommended)
conda create -n competence-analysis python=3.10
conda activate competence-analysis

# Or using virtualenv
python -m venv competence-analysis
source competence-analysis/bin/activate  # Linux/Mac
# competence-analysis\Scripts\activate  # Windows
```

#### Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core ML dependencies
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0

# Install application dependencies
pip install -r requirements.txt
```

#### Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### Step 2: Model Setup

#### Download and Setup CodeLlama-13B
```python
# Create setup script: setup_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_and_cache_model():
    model_name = "meta-llama/CodeLlama-13b-Instruct-hf"
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Downloading model (this may take 15-30 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    print("Model successfully downloaded and cached!")
    return model, tokenizer

if __name__ == "__main__":
    download_and_cache_model()
```

```bash
python setup_model.py
```

#### Test Model Loading
```python
# test_model.py
from src.competence_analyzer import PythonCompetenceAnalyzer

def test_model():
    try:
        analyzer = PythonCompetenceAnalyzer()
        print("Model loaded successfully!")
        
        # Simple test
        test_code = "def hello(): print('Hello, World!')"
        result = analyzer.analyze_code(test_code)
        print(f"Analysis completed! Competence level: {result.competence_level.value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_model()
```

### Step 3: Configuration

#### Create Configuration File
```yaml
# config/settings.yaml
model:
  name: "meta-llama/CodeLlama-13b-Instruct-hf"
  max_length: 4096
  temperature: 0.7
  top_p: 0.9
  load_in_4bit: true
  device_map: "auto"

analysis:
  enable_syntax_checking: true
  enable_best_practices: true
  enable_misconception_detection: true
  complexity_weight: 0.3
  practices_weight: 0.4
  concepts_weight: 0.3

prompts:
  max_prompts_per_analysis: 5
  include_socratic_questions: true
  difficulty_adaptation: true
  personalization: false  # Enable after collecting user data

logging:
  level: "INFO"
  file: "logs/competence_analysis.log"
  max_size_mb: 100
  backup_count: 5

deployment:
  batch_size: 1
  max_concurrent_requests: 3
  response_timeout_seconds: 30
  cache_enabled: true
  cache_ttl_seconds: 3600
```

#### Environment Variables
```bash
# .env file
MODEL_CACHE_DIR=/path/to/model/cache
DATA_DIR=/path/to/data
LOG_LEVEL=INFO
MAX_WORKERS=3
CUDA_VISIBLE_DEVICES=0  # Specify GPU if multiple available
```

### Step 4: Basic Usage Examples

#### Command Line Interface
```python
# cli.py
import argparse
from src.competence_analyzer import PythonCompetenceAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze Python code competence')
    parser.add_argument('--file', '-f', required=True, help='Python file to analyze')
    parser.add_argument('--level', '-l', choices=['beginner', 'intermediate', 'advanced'], 
                       help='Expected student level')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    # Read code file
    with open(args.file, 'r') as f:
        code = f.read()
    
    # Initialize analyzer
    analyzer = PythonCompetenceAnalyzer()
    
    # Analyze code
    report = analyzer.get_competence_report(code, args.level)
    
    # Display results
    print("=== COMPETENCE ANALYSIS REPORT ===")
    print(f"File: {args.file}")
    print(f"Detected Level: {report['code_analysis']['competence_level']}")
    print(f"Complexity Score: {report['code_analysis']['complexity_score']}")
    print(f"Best Practices Score: {report['code_analysis']['best_practices_score']}")
    
    if report['code_analysis']['misconceptions']:
        print(f"Misconceptions: {', '.join(report['code_analysis']['misconceptions'])}")
    
    print("\n=== EDUCATIONAL PROMPTS ===")
    for i, prompt in enumerate(report['educational_prompts'], 1):
        print(f"\n{i}. {prompt['prompt']}")
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
```

#### Usage Example
```bash
python cli.py -f student_code.py -l beginner -o analysis_report.json
```

### Step 5: Web Interface Setup (Optional)

#### Simple Flask Web Interface
```python
# web_app.py
from flask import Flask, request, render_template, jsonify
from src.competence_analyzer import PythonCompetenceAnalyzer
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # 1MB max file size

# Initialize analyzer (global instance for efficiency)
analyzer = PythonCompetenceAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_code():
    try:
        data = request.json
        code = data.get('code', '')
        level = data.get('level')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Analyze the code
        report = analyzer.get_competence_report(code, level)
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': analyzer.model is not None})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

#### HTML Template
```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Python Competence Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .code-input { width: 100%; height: 300px; font-family: monospace; }
        .results { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 5px; }
        .prompt { margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }
        .button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        .button:hover { background: #005c8a; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Python Student Competence Analyzer</h1>
        
        <div>
            <label>Student Level:</label>
            <select id="level">
                <option value="">Auto-detect</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
            </select>
        </div>
        
        <div style="margin: 20px 0;">
            <label>Python Code:</label><br>
            <textarea id="code-input" class="code-input" placeholder="Enter Python code here...">
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total / len(numbers)

# Test
result = calculate_average([1, 2, 3, 4, 5])
print("Average:", result)
            </textarea>
        </div>
        
        <button class="button" onclick="analyzeCode()">Analyze Code</button>
        
        <div id="results" class="results" style="display: none;">
            <h2>Analysis Results</h2>
            <div id="analysis-summary"></div>
            <h3>Educational Prompts</h3>
            <div id="prompts-container"></div>
        </div>
    </div>

    <script>
        async function analyzeCode() {
            const code = document.getElementById('code-input').value;
            const level = document.getElementById('level').value;
            
            if (!code.trim()) {
                alert('Please enter some Python code to analyze.');
                return;
            }
            
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({code: code, level: level || null})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.report);
                } else {
                    alert('Analysis failed: ' + result.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function displayResults(report) {
            const analysis = report.code_analysis;
            const prompts = report.educational_prompts;
            
            document.getElementById('analysis-summary').innerHTML = `
                <p><strong>Competence Level:</strong> ${analysis.competence_level}</p>
                <p><strong>Complexity Score:</strong> ${analysis.complexity_score}/1.0</p>
                <p><strong>Best Practices Score:</strong> ${analysis.best_practices_score}/1.0</p>
                <p><strong>Identified Concepts:</strong> ${analysis.identified_concepts.join(', ')}</p>
                ${analysis.misconceptions.length > 0 ? 
                  `<p><strong>Areas for Improvement:</strong> ${analysis.misconceptions.join(', ')}</p>` : ''}
            `;
            
            const promptsHtml = prompts.map((prompt, index) => 
                `<div class="prompt">
                    <strong>Prompt ${index + 1} (${prompt.type}):</strong><br>
                    ${prompt.prompt}
                </div>`
            ).join('');
            
            document.getElementById('prompts-container').innerHTML = promptsHtml;
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
```

### Step 6: Performance Optimization

#### Memory Management
```python
# optimization/memory_manager.py
import torch
import gc
from contextlib import contextmanager

class MemoryManager:
    @staticmethod
    @contextmanager
    def manage_gpu_memory():
        """Context manager for GPU memory cleanup"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'gpu_available': False}
```

#### Batch Processing
```python
# optimization/batch_processor.py
import asyncio
from typing import List, Dict
from src.competence_analyzer import PythonCompetenceAnalyzer

class BatchProcessor:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.analyzer = PythonCompetenceAnalyzer()
    
    async def process_submissions(self, submissions: List[Dict]) -> List[Dict]:
        """Process multiple submissions in batches"""
        results = []
        
        for i in range(0, len(submissions), self.batch_size):
            batch = submissions[i:i + self.batch_size]
            batch_results = []
            
            for submission in batch:
                try:
                    report = self.analyzer.get_competence_report(
                        submission['code'], 
                        submission.get('expected_level')
                    )
                    batch_results.append({
                        'submission_id': submission['id'],
                        'success': True,
                        'report': report
                    })
                except Exception as e:
                    batch_results.append({
                        'submission_id': submission['id'],
                        'success': False,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
            
            # Brief pause between batches to prevent overheating
            await asyncio.sleep(0.1)
        
        return results
```

#### Caching System
```python
# optimization/cache_manager.py
import hashlib
import json
import time
from typing import Optional, Dict, Any

class CacheManager:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds
    
    def _generate_key(self, code: str, level: Optional[str] = None) -> str:
        """Generate cache key from code and level"""
        content = f"{code}:{level or 'auto'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, code: str, level: Optional[str] = None) -> Optional[Dict]:
        """Get cached analysis result"""
        key = self._generate_key(code, level)
        
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item['timestamp'] < self.ttl:
                return cached_item['result']
            else:
                del self.cache[key]  # Expired
        
        return None
    
    def set(self, code: str, result: Dict, level: Optional[str] = None):
        """Cache analysis result"""
        key = self._generate_key(code, level)
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'size_mb': len(json.dumps(self.cache)) / (1024 * 1024)
        }
```

### Step 7: Testing and Validation

#### Unit Testing Framework
```python
# tests/test_competence_analyzer.py
import unittest
from src.competence_analyzer import PythonCompetenceAnalyzer, CompetenceLevel

class TestCompetenceAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize analyzer once for all tests"""
        cls.analyzer = PythonCompetenceAnalyzer()
    
    def test_basic_function_analysis(self):
        """Test analysis of simple function"""
        code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
"""
        analysis = self.analyzer.analyze_code(code)
        
        self.assertTrue(analysis.syntax_correct)
        self.assertIn('functions', analysis.identified_concepts)
        self.assertEqual(analysis.competence_level, CompetenceLevel.BEGINNER)
    
    def test_misconception_detection(self):
        """Test detection of common misconceptions"""
        code = """
def process_list(items):
    for i in range(len(items)):
        print(items[i])
"""
        analysis = self.analyzer.analyze_code(code)
        
        self.assertIn("Using range(len()) instead of direct iteration", 
                     analysis.misconceptions)
    
    def test_complexity_scoring(self):
        """Test complexity score calculation"""
        simple_code = "x = 5"
        complex_code = """
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, items):
        for item in items:
            try:
                if item > 0:
                    self.data.append(item ** 2)
            except TypeError:
                continue
"""
        
        simple_analysis = self.analyzer.analyze_code(simple_code)
        complex_analysis = self.analyzer.analyze_code(complex_code)
        
        self.assertLess(simple_analysis.complexity_score, 
                       complex_analysis.complexity_score)

if __name__ == '__main__':
    unittest.main()
```

#### Integration Testing
```python
# tests/test_integration.py
import unittest
import json
from src.competence_analyzer import PythonCompetenceAnalyzer

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = PythonCompetenceAnalyzer()
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
        
        # Get complete report
        report = self.analyzer.get_competence_report(code)
        
        # Validate report structure
        self.assertIn('code_analysis', report)
        self.assertIn('educational_prompts', report)
        self.assertIn('recommendations', report)
        
        # Validate content
        self.assertTrue(len(report['educational_prompts']) > 0)
        self.assertTrue(len(report['recommendations']) > 0)
        
        # Ensure JSON serializable
        json_str = json.dumps(report)
        self.assertIsInstance(json_str, str)

if __name__ == '__main__':
    unittest.main()
```

#### Benchmark Testing
```python
# tests/run_benchmarks.py
from data.evaluation_benchmarks.benchmark_tests import run_benchmark_evaluation
from src.competence_analyzer import PythonCompetenceAnalyzer
import json

def create_analyzer_function():
    """Create analyzer function for benchmark testing"""
    analyzer = PythonCompetenceAnalyzer()
    
    def analyze_wrapper(code):
        analysis = analyzer.analyze_code(code)
        return {
            'competence_level': analysis.competence_level.value,
            'identified_concepts': analysis.identified_concepts,
            'potential_issues': analysis.misconceptions
        }
    
    return analyze_wrapper

def main():
    print("Running benchmark evaluation...")
    
    analyzer_func = create_analyzer_function()
    results = run_benchmark_evaluation(analyzer_func)
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Competence Accuracy: {results['competence_prediction_accuracy']:.3f}")
    print(f"Concept Accuracy: {results['concept_identification_accuracy']:.3f}")
    print(f"Issue Detection: {results['issue_detection_accuracy']:.3f}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    main()
```

### Step 8: Deployment and Production

#### Docker Deployment
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start command
CMD ["python3", "web_app.py"]
```

#### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  competence-analyzer:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
      - model_cache:/app/model_cache
    environment:
      - MODEL_CACHE_DIR=/app/model_cache
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - competence-analyzer
    restart: unless-stopped

volumes:
  model_cache:
```

#### Production Configuration
```python
# production_config.py
import os
import logging
from logging.handlers import RotatingFileHandler

class ProductionConfig:
    # Model settings
    MODEL_NAME = "meta-llama/CodeLlama-13b-Instruct-hf"
    MAX_CONCURRENT_REQUESTS = int(os.environ.get('MAX_WORKERS', 3))
    REQUEST_TIMEOUT = 30
    
    # Caching
    CACHE_ENABLED = True
    CACHE_TTL = 3600
    
    # Security
    MAX_CODE_LENGTH = 10000  # characters
    ALLOWED_EXTENSIONS = ['.py']
    RATE_LIMIT = "100/hour"  # requests per hour per IP
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/production.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Monitoring
    METRICS_ENABLED = True
    HEALTH_CHECK_INTERVAL = 60
    
    @staticmethod
    def setup_logging():
        """Configure production logging"""
        os.makedirs('logs', exist_ok=True)
        
        handler = RotatingFileHandler(
            ProductionConfig.LOG_FILE,
            maxBytes=ProductionConfig.LOG_MAX_SIZE,
            backupCount=ProductionConfig.LOG_BACKUP_COUNT
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logging.getLogger().setLevel(ProductionConfig.LOG_LEVEL)
        logging.getLogger().addHandler(handler)
```

### Step 9: Monitoring and Maintenance

#### System Monitoring
```python
# monitoring/system_monitor.py
import psutil
import torch
import time
import json
from typing import Dict

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'requests_processed': self.request_count,
            'errors_encountered': self.error_count,
            
            # CPU and Memory
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            
            # Disk
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
        }
        
        # GPU statistics if available
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_utilization_percent': self._get_gpu_utilization()
            })
        
        return stats
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            return 0.0
    
    def log_request(self, success: bool):
        """Log request completion"""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        stats = self.get_system_stats()
        
        health_checks = {
            'memory_ok': stats['memory_percent'] < 85,
            'cpu_ok': stats['cpu_percent'] < 80,
            'disk_ok': stats['disk_usage_percent'] < 90,
            'error_rate_ok': (self.error_count / max(self.request_count, 1)) < 0.05
        }
        
        if torch.cuda.is_available():
            health_checks['gpu_memory_ok'] = stats.get('gpu_memory_reserved_gb', 0) < 20
        
        return {
            'healthy': all(health_checks.values()),
            'checks': health_checks,
            'stats': stats
        }
```

#### Performance Metrics
```python
# monitoring/performance_tracker.py
import time
import statistics
from collections import deque
from typing import Dict, List

class PerformanceTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.request_sizes = deque(maxlen=window_size)
        self.competence_predictions = deque(maxlen=window_size)
    
    def record_request(self, response_time: float, code_length: int, competence_level: str):
        """Record request metrics"""
        self.response_times.append(response_time)
        self.request_sizes.append(code_length)
        self.competence_predictions.append(competence_level)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.response_times:
            return {'status': 'no_data'}
        
        response_times_list = list(self.response_times)
        
        return {
            'requests_analyzed': len(response_times_list),
            'avg_response_time': statistics.mean(response_times_list),
            'median_response_time': statistics.median(response_times_list),
            'p95_response_time': self._percentile(response_times_list, 95),
            'avg_code_length': statistics.mean(self.request_sizes),
            'competence_distribution': self._get_competence_distribution(),
            'throughput_per_minute': len(response_times_list)  # Approximate
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_competence_distribution(self) -> Dict:
        """Get distribution of competence level predictions"""
        from collections import Counter
        counter = Counter(self.competence_predictions)
        total = len(self.competence_predictions)
        
        return {
            level: count / total 
            for level, count in counter.items()
        }
```

### Step 10: Troubleshooting

#### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| CUDA Out of Memory | RuntimeError: CUDA out of memory | Reduce batch_size, enable quantization, or use CPU fallback |
| Model Loading Timeout | Process hangs during initialization | Check internet connection, increase timeout, use cached model |
| Poor Analysis Quality | Inconsistent or irrelevant feedback | Verify model version, check input preprocessing, review prompt templates |
| Slow Response Times | >30 second response times | Enable caching, optimize batch processing, check GPU utilization |
| Memory Leaks | Gradually increasing memory usage | Implement memory cleanup, restart workers periodically |

#### Debugging Tools
```python
# debug/diagnostics.py
import torch
import psutil
import traceback
from src.competence_analyzer import PythonCompetenceAnalyzer

class DiagnosticTool:
    @staticmethod
    def run_system_check():
        """Run comprehensive system diagnostics"""
        print("=== SYSTEM DIAGNOSTICS ===")
        
        # Python environment
        print(f"Python executable: {sys.executable}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        
        # System resources
        memory = psutil.virtual_memory()
        print(f"System memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
        print(f"CPU cores: {psutil.cpu_count()}")
        
        # Model loading test
        try:
            print("\nTesting model loading...")
            analyzer = PythonCompetenceAnalyzer()
            print("Model loaded successfully")
            
            # Quick analysis test
            test_code = "def hello(): print('Hello, World!')"
            result = analyzer.analyze_code(test_code)
            print(f"Analysis test passed: {result.competence_level.value}")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            traceback.print_exc()
    
    @staticmethod
    def benchmark_performance(num_tests: int = 10):
        """Benchmark system performance"""
        print(f"\n=== PERFORMANCE BENCHMARK ({num_tests} tests) ===")
        
        analyzer = PythonCompetenceAnalyzer()
        test_codes = [
            "def add(a, b): return a + b",
            "for i in range(10): print(i)",
            """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self.result
""",
        ] * (num_tests // 3 + 1)
        
        times = []
        for i, code in enumerate(test_codes[:num_tests]):
            start_time = time.time()
            try:
                result = analyzer.analyze_code(code)
                end_time = time.time()
                times.append(end_time - start_time)
                print(f"Test {i+1}: {end_time - start_time:.2f}s - {result.competence_level.value}")
            except Exception as e:
                print(f"Test {i+1}: FAILED - {str(e)}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\nAverage response time: {avg_time:.2f}s")
            print(f"Total throughput: {len(times) / sum(times):.1f} analyses/second")

if __name__ == "__main__":
    import sys
    import time
    
    DiagnosticTool.run_system_check()
    DiagnosticTool.benchmark_performance()
```

## Conclusion

This implementation guide provides a complete framework for deploying the Python Student Competence Analysis system using CodeLlama-13B-Instruct. The system is designed to be:

- **Scalable**: Handle multiple concurrent requests with proper resource management
- **Robust**: Comprehensive error handling and monitoring capabilities  
- **Educational**: Focus on pedagogical effectiveness over pure technical performance
- **Maintainable**: Clear code structure and comprehensive testing framework
- **Production-Ready**: Docker deployment with monitoring and health checks

For production deployment, ensure you have:
1. Adequate hardware resources (24GB+ GPU recommended)
2. Proper monitoring and alerting setup
3. Regular backup procedures for logs and configurations
4. Educational expert validation of system outputs
5. Student feedback collection mechanisms

The system provides a solid foundation for AI-assisted programming education while maintaining the human-centered approach essential for effective learning.