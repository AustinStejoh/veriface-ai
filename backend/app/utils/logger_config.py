"""
Logging Configuration Module
Provides centralized logging setup with file rotation, formatting, and request tracking
"""

import logging
import logging.handlers
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file paths
MAIN_LOG_FILE = os.path.join(LOG_DIR, "app.log")
REQUEST_LOG_FILE = os.path.join(LOG_DIR, "requests.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")
CACHE_LOG_FILE = os.path.join(LOG_DIR, "cache.log")
ASYNC_LOG_FILE = os.path.join(LOG_DIR, "async.log")

# Log format
DETAILED_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

REQUEST_FORMAT = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def setup_logger(name: str, log_file: str, level=logging.INFO, formatter=DETAILED_FORMAT):
    """
    Setup a logger with file rotation handler
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        formatter: Log formatter
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler with rotation
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Console handler for errors
    if level == logging.ERROR:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Initialize loggers for different components
app_logger = setup_logger("app", MAIN_LOG_FILE, logging.INFO, DETAILED_FORMAT)
request_logger = setup_logger("requests", REQUEST_LOG_FILE, logging.INFO, REQUEST_FORMAT)
error_logger = setup_logger("errors", ERROR_LOG_FILE, logging.ERROR, DETAILED_FORMAT)
cache_logger = setup_logger("cache", CACHE_LOG_FILE, logging.INFO, DETAILED_FORMAT)
async_logger = setup_logger("async", ASYNC_LOG_FILE, logging.INFO, DETAILED_FORMAT)


class RequestLogger:
    """Helper class for logging HTTP requests with timing information"""
    
    def __init__(self, endpoint: str, method: str, client_ip: str = None):
        self.endpoint = endpoint
        self.method = method
        self.client_ip = client_ip or "unknown"
        self.start_time = datetime.now()
    
    def log_success(self, status_code: int, response_size: int = 0):
        """Log successful request"""
        elapsed_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        request_logger.info(
            f"✓ {self.method} {self.endpoint} | Status: {status_code} | "
            f"Client: {self.client_ip} | Response: {response_size}B | Duration: {elapsed_ms:.2f}ms"
        )
    
    def log_error(self, status_code: int, error_msg: str = None):
        """Log failed request"""
        elapsed_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        error_logger.error(
            f"✗ {self.method} {self.endpoint} | Status: {status_code} | "
            f"Client: {self.client_ip} | Error: {error_msg} | Duration: {elapsed_ms:.2f}ms"
        )


def log_cache_operation(operation: str, key: str, hit: bool = False, duration_ms: float = 0):
    """Log cache operations"""
    status = "HIT" if hit else "MISS"
    cache_logger.info(f"{operation:8s} | {status:4s} | Key: {key} | Duration: {duration_ms:.2f}ms")


def log_async_task(task_id: str, task_name: str, status: str, duration_ms: float = 0, error: str = None):
    """Log async task operations"""
    if error:
        async_logger.error(f"Task {task_id} ({task_name}) - {status} - Error: {error}")
    else:
        async_logger.info(f"Task {task_id} ({task_name}) - {status} - Duration: {duration_ms:.2f}ms")
