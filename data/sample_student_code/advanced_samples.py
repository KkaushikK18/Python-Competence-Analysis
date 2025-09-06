# ===== ADVANCED LEVEL SAMPLES =====

import asyncio
import threading
import functools
import weakref
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Generic, TypeVar
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from enum import Enum

# Sample 1: Advanced OOP with design patterns
T = TypeVar('T')

class Observer(ABC):
    """Observer pattern implementation"""
    @abstractmethod
    def update(self, subject: 'Subject', event: str, data: any) -> None:
        pass

class Subject:
    """Subject in observer pattern"""
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self, event: str, data: any = None) -> None:
        for observer in self._observers:
            observer.update(self, event, data)

class DataProcessor(Subject):
    """Data processor with observer notifications"""
    def __init__(self):
        super().__init__()
        self._data: List[Dict] = []
        self._processing_stats = {'processed': 0, 'errors': 0}
    
    def process_batch(self, batch: List[Dict]) -> None:
        self.notify('batch_started', {'size': len(batch)})
        
        for item in batch:
            try:
                processed_item = self._process_item(item)
                self._data.append(processed_item)
                self._processing_stats['processed'] += 1
            except Exception as e:
                self._processing_stats['errors'] += 1
                self.notify('processing_error', {'item': item, 'error': str(e)})
        
        self.notify('batch_completed', self._processing_stats.copy())
    
    def _process_item(self, item: Dict) -> Dict:
        """Process individual item with validation"""
        if 'id' not in item or 'value' not in item:
            raise ValueError("Item must have 'id' and 'value' fields")
        
        return {
            'id': item['id'],
            'value': item['value'] * 2,  # Example processing
            'processed_at': __import__('datetime').datetime.now().isoformat()
        }

class ProcessingLogger(Observer):
    """Observer that logs processing events"""
    def __init__(self):
        self.logger = logging.getLogger('ProcessingLogger')
    
    def update(self, subject: Subject, event: str, data: any) -> None:
        if event == 'batch_started':
            self.logger.info(f"Started processing batch of {data['size']} items")
        elif event == 'processing_error':
            self.logger.error(f"Error processing item {data['item']}: {data['error']}")
        elif event == 'batch_completed':
            self.logger.info(f"Completed batch: {data['processed']} processed, {data['errors']} errors")

# Sample 2: Advanced async programming with concurrency control
class AsyncTaskManager:
    """Manages async tasks with concurrency limits and error handling"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
    
    async def process_urls(self, urls: List[str]) -> Dict[str, any]:
        """Process multiple URLs concurrently with rate limiting"""
        tasks = [self._process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.errors.append({'url': urls[i], 'error': str(result)})
            else:
                self.results.append(result)
        
        return {
            'successful': len(self.results),
            'failed': len(self.errors),
            'results': self.results,
            'errors': self.errors
        }
    
    async def _process_single_url(self, url: str) -> Dict:
        """Process single URL with semaphore-controlled concurrency"""
        async with self.semaphore:
            # Simulate async HTTP request
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate occasional failures
            if 'error' in url:
                raise Exception(f"Failed to process {url}")
            
            return {
                'url': url,
                'status': 'success',
                'content_length': len(url) * 100,  # Simulated
                'processed_at': __import__('time').time()
            }

# Sample 3: Metaclass usage for advanced class behavior
class SingletonMeta(type):
    """Metaclass that implements the Singleton pattern"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class ConfigManager(metaclass=SingletonMeta):
    """Singleton configuration manager with validation"""
    
    def __init__(self):
        self._config: Dict[str, any] = {}
        self._validators: Dict[str, callable] = {}
        self._observers: List[callable] = []
    
    def register_validator(self, key: str, validator: callable) -> None:
        """Register a validator function for a config key"""
        self._validators[key] = validator
    
    def add_observer(self, observer: callable) -> None:
        """Add observer for configuration changes"""
        self._observers.append(observer)
    
    def set(self, key: str, value: any) -> None:
        """Set configuration value with validation"""
        if key in self._validators:
            if not self._validators[key](value):
                raise ValueError(f"Invalid value for {key}: {value}")
        
        old_value = self._config.get(key)
        self._config[key] = value
        
        # Notify observers of changes
        for observer in self._observers:
            observer(key, old_value, value)
    
    def get(self, key: str, default: any = None) -> any:
        """Get configuration value"""
        return self._config.get(key, default)

# Sample 4: Advanced context managers and resource management
class DatabaseConnection:
    """Mock database connection for demonstration"""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
    
    def connect(self):
        print(f"Connecting to {self.connection_string}")
        self.connected = True
    
    def disconnect(self):
        print(f"Disconnecting from {self.connection_string}")
        self.connected = False
    
    def execute(self, query: str):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        return f"Executed: {query}"

class ConnectionPool:
    """Advanced connection pool with context manager support"""
    
    def __init__(self, connection_string: str, pool_size: int = 5):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._pool: List[DatabaseConnection] = []
        self._borrowed: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(self.pool_size):
            conn = DatabaseConnection(self.connection_string)
            conn.connect()
            self._pool.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Context manager for borrowing connections"""
        with self._lock:
            if not self._pool:
                raise RuntimeError("No available connections in pool")
            
            connection = self._pool.pop()
            self._borrowed.add(connection)
        
        try:
            yield connection
        finally:
            with self._lock:
                if connection in self._borrowed:
                    self._borrowed.discard(connection)
                    self._pool.append(connection)
    
    def __del__(self):
        """Cleanup connections when pool is destroyed"""
        for conn in self._pool:
            conn.disconnect()

# Sample 5: Generic classes and advanced typing
@dataclass
class Result(Generic[T]):
    """Generic result type for handling success/failure cases"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        return cls(success=True, data=data)
    
    @classmethod
    def error(cls, error: str) -> 'Result[T]':
        return cls(success=False, error=error)
    
    def map(self, func: callable) -> 'Result':
        """Apply function to data if result is successful"""
        if self.success and self.data is not None:
            try:
                return Result.ok(func(self.data))
            except Exception as e:
                return Result.error(str(e))
        return Result.error(self.error or "No data to map")
    
    def flat_map(self, func: callable) -> 'Result':
        """Apply function that returns Result to data"""
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except Exception as e:
                return Result.error(str(e))
        return Result.error(self.error or "No data to flat_map")

class DataPipeline(Generic[T]):
    """Generic data processing pipeline with result handling"""
    
    def __init__(self):
        self._processors: List[callable] = []
    
    def add_processor(self, processor: callable) -> 'DataPipeline[T]':
        """Add a processing step to the pipeline"""
        self._processors.append(processor)
        return self
    
    def process(self, data: T) -> Result[T]:
        """Process data through the pipeline"""
        result = Result.ok(data)
        
        for processor in self._processors:
            result = result.flat_map(processor)
            if not result.success:
                break
        
        return result

# Sample 6: Advanced decorators with functools
def retry(max_attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """Decorator that retries failed function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    __import__('time').sleep(current_delay)
                    
                    if exponential_backoff:
                        current_delay *= 2
            
        return wrapper
    return decorator

def memoize_with_ttl(ttl_seconds: float = 300):
    """Decorator that memoizes function results with TTL"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
                else:
                    del cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'cache_size': len(cache), 'ttl': ttl_seconds}
        
        return wrapper
    return decorator

# Sample 7: Advanced error handling and custom exceptions
class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error in '{field}': {message}")

class ProcessingError(Exception):
    """Custom exception for processing errors"""
    def __init__(self, stage: str, original_error: Exception):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Error in processing stage '{stage}': {original_error}")

class DataValidator:
    """Advanced data validator with custom error handling"""
    
    @staticmethod
    def validate_email(email: str) -> Result[str]:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        
        if re.match(pattern, email):
            return Result.ok(email)
        else:
            return Result.error(f"Invalid email format: {email}")
    
    @staticmethod
    def validate_age(age: Union[int, str]) -> Result[int]:
        """Validate age with type conversion"""
        try:
            age_int = int(age)
            if 0 <= age_int <= 150:
                return Result.ok(age_int)
            else:
                return Result.error(f"Age must be between 0 and 150, got {age_int}")
        except ValueError:
            return Result.error(f"Age must be a number, got {age}")

# Example usage demonstrating advanced concepts
async def demonstrate_advanced_features():
    """Demonstrate various advanced Python features"""
    
    # Observer pattern
    processor = DataProcessor()
    logger = ProcessingLogger()
    processor.attach(logger)
    
    # Process some data
    test_data = [
        {'id': 1, 'value': 10},
        {'id': 2, 'value': 20},
        {'id': 3, 'value': 30}
    ]
    processor.process_batch(test_data)
    
    # Async processing
    task_manager = AsyncTaskManager(max_concurrent=3)
    urls = ['http://example.com', 'http://test.com', 'http://error.com']
    results = await task_manager.process_urls(urls)
    print(f"Async processing results: {results}")
    
    # Configuration management
    config = ConfigManager()
    config.register_validator('port', lambda x: isinstance(x, int) and 1 <= x <= 65535)
    config.set('port', 8080)
    
    # Data pipeline with generic types
    pipeline = DataPipeline[Dict]()
    pipeline.add_processor(lambda x: Result.ok({**x, 'processed': True}))
    pipeline.add_processor(lambda x: Result.ok({**x, 'timestamp': __import__('time').time()}))
    
    result = pipeline.process({'id': 1, 'name': 'test'})
    print(f"Pipeline result: {result}")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_advanced_features())