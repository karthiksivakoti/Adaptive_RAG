# risk_rag_system/utils/error_handler.py

from typing import Dict, Any, Optional, Callable, TypeVar, Generic, AsyncGenerator
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
from loguru import logger
import traceback
from functools import wraps
import time
import random

T = TypeVar('T')

class RetryConfig(BaseModel):
    """Configuration for retry mechanism"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1

class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_timeout: float = 30.0
    
class CircuitBreakerState(BaseModel):
    """State for circuit breaker"""
    status: str = "closed"  # closed, open, half-open
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

class ErrorCategory(BaseModel):
    """Configuration for error category"""
    name: str
    retryable: bool
    require_cleanup: bool
    log_level: str = "ERROR"
    alert_threshold: Optional[int] = None

class ErrorRegistry:
    """Registry of known error categories"""
    
    def __init__(self):
        self.categories: Dict[str, ErrorCategory] = {
            "connection_error": ErrorCategory(
                name="connection_error",
                retryable=True,
                require_cleanup=False,
                alert_threshold=10
            ),
            "timeout_error": ErrorCategory(
                name="timeout_error",
                retryable=True,
                require_cleanup=False,
                alert_threshold=5
            ),
            "validation_error": ErrorCategory(
                name="validation_error",
                retryable=False,
                require_cleanup=True,
                log_level="WARNING"
            ),
            "resource_error": ErrorCategory(
                name="resource_error",
                retryable=False,
                require_cleanup=True,
                alert_threshold=1
            )
        }
    
    def get_category(self, error: Exception) -> ErrorCategory:
        """Get error category based on exception type"""
        error_type = type(error).__name__
        
        if "Timeout" in error_type:
            return self.categories["timeout_error"]
        elif "Connection" in error_type:
            return self.categories["connection_error"]
        elif "Validation" in error_type:
            return self.categories["validation_error"]
        elif "Resource" in error_type:
            return self.categories["resource_error"]
        
        # Default category
        return ErrorCategory(
            name="unknown_error",
            retryable=False,
            require_cleanup=True
        )

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    async def before_execution(self) -> None:
        """Check circuit breaker state before execution"""
        async with self._lock:
            current_time = datetime.now()
            
            if self.state.status == "open":
                if (current_time - self.state.last_failure_time) > timedelta(
                    seconds=self.config.reset_timeout
                ):
                    # Transition to half-open
                    self.state.status = "half-open"
                    logger.info(f"Circuit {self.name} transitioning to half-open")
                else:
                    raise Exception(f"Circuit {self.name} is open")
                    
            elif self.state.status == "half-open":
                if (current_time - self.state.last_failure_time) > timedelta(
                    seconds=self.config.half_open_timeout
                ):
                    # Allow one request through
                    pass
                else:
                    raise Exception(f"Circuit {self.name} is half-open")
    
    async def on_success(self) -> None:
        """Handle successful execution"""
        async with self._lock:
            self.state.failure_count = 0
            self.state.last_success_time = datetime.now()
            
            if self.state.status == "half-open":
                self.state.status = "closed"
                logger.info(f"Circuit {self.name} closed")
    
    async def on_failure(self, error: Exception) -> None:
        """Handle failed execution"""
        async with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()
            
            if (
                self.state.status == "closed" and
                self.state.failure_count >= self.config.failure_threshold
            ):
                self.state.status = "open"
                logger.warning(f"Circuit {self.name} opened")
            elif self.state.status == "half-open":
                self.state.status = "open"
                logger.warning(f"Circuit {self.name} reopened")

class ErrorHandler:
    """Handles error recovery and retry logic"""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.error_registry = ErrorRegistry()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    def with_retry(self, func: Callable) -> Callable:
        """Decorator for retry logic"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            delay = self.retry_config.initial_delay
            
            for attempt in range(self.retry_config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_error = e
                    category = self.error_registry.get_category(e)
                    
                    if not category.retryable or attempt == self.retry_config.max_retries:
                        raise
                    
                    # Add jitter to delay
                    jitter = self.retry_config.jitter * (2 * random.random() - 1)
                    retry_delay = delay * (1 + jitter)
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s"
                    )
                    
                    await asyncio.sleep(retry_delay)
                    delay = min(
                        delay * self.retry_config.exponential_base,
                        self.retry_config.max_delay
                    )
            
            raise last_error
        return wrapper
    
    def with_circuit_breaker(
        self,
        name: str,
        cleanup_func: Optional[Callable] = None
    ) -> Callable:
        """Decorator for circuit breaker"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                circuit_breaker = await self.get_circuit_breaker(name)
                
                try:
                    await circuit_breaker.before_execution()
                    result = await func(*args, **kwargs)
                    await circuit_breaker.on_success()
                    return result
                except Exception as e:
                    await circuit_breaker.on_failure(e)
                    category = self.error_registry.get_category(e)
                    
                    # Execute cleanup if required
                    if category.require_cleanup and cleanup_func:
                        try:
                            await cleanup_func(*args, **kwargs)
                        except Exception as cleanup_error:
                            logger.error(f"Cleanup failed: {cleanup_error}")
                    
                    # Log error with appropriate level
                    log_func = getattr(logger, category.log_level.lower())
                    log_func(f"Error in {name}: {str(e)}")
                    log_func(f"Traceback: {traceback.format_exc()}")
                    
                    raise
            return wrapper
        return decorator
    
    async def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        async with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(
                    name,
                    self.circuit_breaker_config
                )
            return self.circuit_breakers[name]
    
    async def cleanup(self) -> None:
        """Cleanup error handler resources"""
        # Reset all circuit breakers
        async with self._lock:
            for breaker in self.circuit_breakers.values():
                breaker.state = CircuitBreakerState()
            self.circuit_breakers.clear()

# Example usage
error_handler = ErrorHandler()

# Example of using both retry and circuit breaker
@error_handler.with_retry
@error_handler.with_circuit_breaker("database_operations")
async def database_operation():
    # Database operation code here
    pass