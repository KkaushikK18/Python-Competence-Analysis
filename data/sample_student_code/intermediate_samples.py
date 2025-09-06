# ===== INTERMEDIATE LEVEL SAMPLES =====

import json
from collections import defaultdict

# Sample 1: List comprehensions with some sophistication
def process_student_grades(students):
    # Good use of list comprehensions
    passing_students = [student for student in students if student['grade'] >= 60]
    high_performers = [s['name'] for s in students if s['grade'] >= 90]
    
    # Some complexity in data processing
    grade_categories = {
        'excellent': [s for s in students if s['grade'] >= 90],
        'good': [s for s in students if 80 <= s['grade'] < 90],
        'satisfactory': [s for s in students if 60 <= s['grade'] < 80],
        'needs_improvement': [s for s in students if s['grade'] < 60]
    }
    
    return {
        'passing': passing_students,
        'high_performers': high_performers,
        'categories': grade_categories
    }

# Sample data
students = [
    {'name': 'Alice', 'grade': 92},
    {'name': 'Bob', 'grade': 78},
    {'name': 'Charlie', 'grade': 85},
    {'name': 'Diana', 'grade': 95}
]

results = process_student_grades(students)

# Sample 2: Basic exception handling and file operations
def load_and_process_data(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found. Using empty dataset.")
        data = []
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filename}. Using empty dataset.")
        data = []
    
    # Process the data
    processed = []
    for item in data:
        try:
            processed_item = {
                'id': item['id'],
                'name': item['name'].strip().title(),
                'value': float(item['value'])
            }
            processed.append(processed_item)
        except (KeyError, ValueError) as e:
            print(f"Skipping invalid item: {item}. Error: {e}")
    
    return processed

# Sample 3: Class with inheritance and some advanced features
class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer = 0
    
    def start_engine(self):
        return f"The {self.year} {self.make} {self.model} engine is now running."
    
    def drive(self, miles):
        if miles > 0:
            self.odometer += miles
            return f"Drove {miles} miles. Total odometer: {self.odometer}"
        else:
            return "Invalid distance."

class ElectricVehicle(Vehicle):
    def __init__(self, make, model, year, battery_capacity):
        super().__init__(make, model, year)
        self.battery_capacity = battery_capacity
        self.charge_level = 100
    
    def start_engine(self):
        return f"The {self.year} {self.make} {self.model} is ready to drive silently."
    
    def charge_battery(self, percentage):
        if 0 <= percentage <= 100:
            self.charge_level = min(100, self.charge_level + percentage)
            return f"Battery charged to {self.charge_level}%"
        return "Invalid charge percentage."

# Usage
tesla = ElectricVehicle("Tesla", "Model 3", 2023, 75)
print(tesla.start_engine())
print(tesla.drive(50))
print(tesla.charge_battery(20))

# Sample 4: Dictionary manipulation and defaultdict usage
def analyze_text_frequency(text):
    # Using defaultdict for cleaner code
    word_freq = defaultdict(int)
    char_freq = defaultdict(int)
    
    # Clean and process text
    words = text.lower().split()
    
    for word in words:
        # Remove punctuation (basic approach)
        cleaned_word = ''.join(char for char in word if char.isalnum())
        if cleaned_word:
            word_freq[cleaned_word] += 1
            
            for char in cleaned_word:
                char_freq[char] += 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'word_frequencies': dict(sorted_words),
        'char_frequencies': dict(sorted_chars),
        'total_words': len(words),
        'unique_words': len(word_freq)
    }

sample_text = "The quick brown fox jumps over the lazy dog. The dog was sleeping."
analysis = analyze_text_frequency(sample_text)
print(f"Most common word: {list(analysis['word_frequencies'].keys())[0]}")

# Sample 5: Generator functions and lambda expressions
def fibonacci_generator(n):
    """Generate fibonacci numbers up to n terms"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

def process_numbers(numbers, operations):
    """Apply a series of operations to numbers"""
    result = numbers[:]  # Copy the list
    
    # Using lambda functions
    operation_map = {
        'square': lambda x: x ** 2,
        'double': lambda x: x * 2,
        'increment': lambda x: x + 1,
        'abs': lambda x: abs(x)
    }
    
    for operation in operations:
        if operation in operation_map:
            result = list(map(operation_map[operation], result))
    
    return result

# Usage
fib_numbers = list(fibonacci_generator(10))
print(f"First 10 Fibonacci numbers: {fib_numbers}")

numbers = [-2, 3, -1, 4, -5]
processed = process_numbers(numbers, ['abs', 'square', 'increment'])
print(f"Processed numbers: {processed}")

# Sample 6: Decorator usage (basic)
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    import time
    time.sleep(0.1)  # Simulate slow operation
    return "Done!"

result = slow_function()

# Sample 7: Context manager awareness
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        if exc_type:
            print(f"An exception occurred: {exc_val}")
        return False

# Usage would be:
# with FileManager('test.txt', 'w') as f:
#     f.write("Hello, World!")

# Sample 8: More sophisticated data structure usage
def group_and_analyze_data(data, group_by_field, analyze_field):
    """Group data by a field and analyze another field"""
    grouped = defaultdict(list)
    
    # Group the data
    for item in data:
        if group_by_field in item and analyze_field in item:
            grouped[item[group_by_field]].append(item[analyze_field])
    
    # Analyze each group
    analysis = {}
    for group, values in grouped.items():
        if values:  # Ensure we have data
            analysis[group] = {
                'count': len(values),
                'sum': sum(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    return analysis

# Sample data
sales_data = [
    {'region': 'North', 'salesperson': 'Alice', 'amount': 1000},
    {'region': 'North', 'salesperson': 'Bob', 'amount': 1500},
    {'region': 'South', 'salesperson': 'Charlie', 'amount': 800},
    {'region': 'South', 'salesperson': 'Diana', 'amount': 1200},
    {'region': 'North', 'salesperson': 'Eve', 'amount': 900}
]

regional_analysis = group_and_analyze_data(sales_data, 'region', 'amount')
print("Regional sales analysis:", regional_analysis)