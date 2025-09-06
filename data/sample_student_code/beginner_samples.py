# Sample student code for testing competence analysis
# These represent different levels of Python competence

# ===== BEGINNER LEVEL SAMPLES =====

# Sample 1: Basic function with common misconceptions
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):  # Misconception: using range(len())
        total = total + numbers[i]
    average = total / len(numbers)  # No error handling for empty list
    return average

# Student test
my_numbers = [1, 2, 3, 4, 5]
result = calculate_average(my_numbers)
print("The average is:", result)

# Sample 2: Basic conditional with issues
def check_grade(score):
    if score >= 90:
        print("A grade")
    if score >= 80:  # Should use elif
        print("B grade") 
    if score >= 70:
        print("C grade")
    if score >= 60:
        print("D grade")
    if score < 60:
        print("F grade")

check_grade(85)  # Will print multiple grades

# Sample 3: Simple loop with boundary issue
def print_numbers():
    i = 1
    while i < 10:  # Off-by-one error, should be <= 10
        print(i)
        i = i + 1

print_numbers()

# Sample 4: Function without return statement
def find_largest(numbers):
    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    print("The largest number is:", largest)  # Should return, not print

numbers = [3, 7, 2, 9, 1]
find_largest(numbers)

# Sample 5: String manipulation with inefficiency
def reverse_string(text):
    reversed_text = ""
    for i in range(len(text)):
        reversed_text = text[i] + reversed_text  # Inefficient string concatenation
    return reversed_text

print(reverse_string("hello"))

# Sample 6: List operations with common mistakes
def remove_duplicates(items):
    unique_items = []
    for item in items:
        if item not in unique_items:  # Inefficient membership testing
            unique_items.append(item)
    return unique_items

test_list = [1, 2, 2, 3, 3, 4, 5, 5]
print(remove_duplicates(test_list))

# Sample 7: Input validation missing
def divide_numbers(a, b):
    result = a / b  # No check for division by zero
    return result

print(divide_numbers(10, 2))
# print(divide_numbers(10, 0))  # Would cause error

# Sample 8: Basic file handling without proper error handling
def read_file(filename):
    file = open(filename, 'r')  # No error handling, not using context manager
    content = file.read()
    file.close()
    return content

# Commented out to avoid actual file error
# print(read_file("test.txt"))

# Sample 9: Simple class with basic structure
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def display_info(self):
        print("Name:", self.name)
        print("Age:", self.age)

student1 = Student("Alice", 20)
student1.display_info()

# Sample 10: Basic error-prone calculator
def calculator(num1, num2, operation):
    if operation == "add":
        return num1 + num2
    if operation == "subtract":
        return num1 - num2
    if operation == "multiply":
        return num1 * num2
    if operation == "divide":
        return num1 / num2  # No division by zero check
    # No handling for invalid operations

print(calculator(10, 5, "add"))
print(calculator(10, 5, "divide"))