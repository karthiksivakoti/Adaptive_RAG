# risk_rag_system/utils/logging_utils.py

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import sys
from loguru import logger
import json
from datetime import datetime
import threading
from functools import wraps
import time
from datetime import timedelta
import traceback

class LoggingConfig:
    """Configuration for logging setup"""
    
    def __init__(
        self,
        log_dir: Path = Path("./logs"),
        log_level: str = "INFO",
        rotation: str = "1 day",
        retention: str = "1 month",
        format_string: Optional[str] = None
    ):
        self.log_dir = log_dir
        self.log_level = log_level
        self.rotation = rotation
        self.retention = retention
        self.format_string = format_string or (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

class LogManager:
    """Manages logging configuration and utilities"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self._setup_logging()
        self._local = threading.local()
        logger.info("Initialized LogManager")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        # Create log directory
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            format=self.config.format_string,
            level=self.config.log_level,
            colorize=True
        )
        
        # Add file handler for all logs
        logger.add(
            self.config.log_dir / "app.log",
            format=self.config.format_string,
            level=self.config.log_level,
            rotation=self.config.rotation,
            retention=self.config.retention
        )
        
        # Add file handler for errors
        logger.add(
            self.config.log_dir / "error.log",
            format=self.config.format_string,
            level="ERROR",
            rotation=self.config.rotation,
            retention=self.config.retention,
            filter=lambda record: record["level"].name == "ERROR"
        )

    def start_request(self, request_id: str) -> None:
        """Start logging context for a request"""
        self._local.request_id = request_id
        self._local.start_time = time.time()
        logger.info(f"Started request {request_id}")

    def end_request(self) -> None:
        """End logging context for a request"""
        if hasattr(self._local, 'request_id'):
            duration = time.time() - self._local.start_time
            logger.info(
                f"Completed request {self._local.request_id} in {duration:.2f}s"
            )
            delattr(self._local, 'request_id')
            delattr(self._local, 'start_time')

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error with context"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        if hasattr(self._local, 'request_id'):
            error_info["request_id"] = self._local.request_id
        
        # Log to error file
        error_file = self.config.log_dir / "errors" / f"{datetime.now():%Y%m%d}.json"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if error_file.exists():
                with open(error_file, 'r') as f:
                    errors = json.load(f)
            else:
                errors = []
            
            errors.append(error_info)
            
            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving error log: {e}")
        
        # Log to logger
        logger.error(
            "Error occurred: {}\nContext: {}\nTraceback: {}",
            str(error),
            context,
            error_info["traceback"]
        )

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        category: str
    ) -> None:
        """Log metrics to separate metrics file"""
        metrics_file = (
            self.config.log_dir / "metrics" /
            f"{category}_{datetime.now():%Y%m%d}.json"
        )
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = []
            
            # Add timestamp to metrics
            metrics["timestamp"] = datetime.now().isoformat()
            if hasattr(self._local, 'request_id'):
                metrics["request_id"] = self._local.request_id
            
            existing_metrics.append(metrics)
            
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def instrument(self, name: str):
        """Decorator for instrumenting functions with logging"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Log successful execution
                    logger.info(
                        f"{name} completed in {duration:.2f}s"
                    )
                    
                    # Log metrics
                    self.log_metrics(
                        {
                            "function": name,
                            "duration": duration,
                            "status": "success"
                        },
                        "function_metrics"
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    # Log error
                    self.log_error(
                        e,
                        {
                            "function": name,
                            "duration": duration,
                            "args": str(args),
                            "kwargs": str(kwargs)
                        }
                    )
                    
                    # Log metrics
                    self.log_metrics(
                        {
                            "function": name,
                            "duration": duration,
                            "status": "error",
                            "error_type": type(e).__name__
                        },
                        "function_metrics"
                    )
                    
                    raise
                    
            return wrapper
        return decorator

    def get_logs(
        self,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve logs with optional filtering"""
        log_file = self.config.log_dir / "app.log"
        
        if not log_file.exists():
            return []
            
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    
                    # Apply filters
                    if level and log_entry.get("level") != level:
                        continue
                        
                    log_time = datetime.fromisoformat(log_entry["time"])
                    if start_time and log_time < start_time:
                        continue
                    if end_time and log_time > end_time:
                        continue
                        
                    logs.append(log_entry)
                    
                except Exception:
                    continue
                    
        return logs

    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up logs older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in self.config.log_dir.glob("**/*"):
            if log_file.is_file():
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    try:
                        log_file.unlink()
                        logger.info(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        logger.error(f"Error deleting log file {log_file}: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current logger state"""
        return {
            "log_dir": str(self.config.log_dir),
            "log_level": self.config.log_level,
            "rotation": self.config.rotation,
            "retention": self.config.retention
        }