"""
Async Processing Module
Provides async task execution for long-running operations like inference
"""

import threading
import time
import uuid
from typing import Callable, Any, Dict, Optional
from queue import Queue
from dataclasses import dataclass, asdict
from datetime import datetime
from logger_config import async_logger, log_async_task

# Task status constants
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"


@dataclass
class AsyncTask:
    """Represents an async task"""
    task_id: str
    name: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['result_available'] = self.result is not None
        return data


class AsyncTaskManager:
    """Manages async task execution with result tracking"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue: Queue = Queue()
        self._workers = []
        self._shutdown = False
        
        # Start worker threads
        self._start_workers()
        async_logger.info(f"[OK] Async task manager started with {max_workers} workers")
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"AsyncWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self._shutdown:
            try:
                # Get next task with timeout
                task_func = self.task_queue.get(timeout=1)
                if task_func is None:  # Shutdown signal
                    break
                
                task_id, name, func, args, kwargs = task_func
                task = self.tasks.get(task_id)
                
                if task:
                    try:
                        # Update status
                        task.status = TASK_RUNNING
                        task.started_at = datetime.now().isoformat()
                        
                        # Execute task
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        duration_ms = (time.time() - start_time) * 1000
                        
                        # Store result
                        task.result = result
                        task.status = TASK_COMPLETED
                        task.duration_ms = duration_ms
                        task.completed_at = datetime.now().isoformat()
                        
                        log_async_task(task_id, name, TASK_COMPLETED, duration_ms)
                        
                    except Exception as e:
                        # Task failed
                        duration_ms = (time.time() - start_time) * 1000
                        task.status = TASK_FAILED
                        task.error = str(e)
                        task.duration_ms = duration_ms
                        task.completed_at = datetime.now().isoformat()
                        
                        log_async_task(task_id, name, TASK_FAILED, duration_ms, error=str(e))
                
            except:
                # Timeout - continue
                continue
    
    def submit_task(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None
    ) -> str:
        """
        Submit an async task for execution
        
        Args:
            name: Task name
            func: Callable to execute
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        kwargs = kwargs or {}
        
        # Create task object
        task = AsyncTask(
            task_id=task_id,
            name=name,
            status=TASK_PENDING
        )
        
        self.tasks[task_id] = task
        
        # Queue task
        self.task_queue.put((task_id, name, func, args, kwargs))
        
        log_async_task(task_id, name, TASK_PENDING)
        async_logger.info(f"→ Task queued: {task_id} ({name})")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """
        Get task result, optionally waiting for completion
        
        Args:
            task_id: Task ID
            timeout: Max seconds to wait for task completion
        
        Returns:
            Task result or None if not completed
        """
        task = self.get_task(task_id)
        if not task:
            return None
        
        # Wait for completion if timeout specified
        if timeout and task.status != TASK_COMPLETED:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task.status == TASK_COMPLETED:
                    break
                time.sleep(0.1)
        
        return task.result if task.status == TASK_COMPLETED else None
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get task status"""
        task = self.get_task(task_id)
        return task.status if task else None
    
    def wait_for_task(self, task_id: str, timeout: float = None) -> bool:
        """
        Wait for task to complete
        
        Args:
            task_id: Task ID
            timeout: Max seconds to wait
        
        Returns:
            True if completed, False if timeout
        """
        task = self.get_task(task_id)
        if not task:
            return False
        
        timeout = timeout or 300  # 5 minute default
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task.status in (TASK_COMPLETED, TASK_FAILED):
                return True
            time.sleep(0.1)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TASK_COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TASK_FAILED)
        running = sum(1 for t in self.tasks.values() if t.status == TASK_RUNNING)
        pending = sum(1 for t in self.tasks.values() if t.status == TASK_PENDING)
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "queue_size": self.task_queue.qsize(),
            "workers": self.max_workers
        }
    
    def shutdown(self):
        """Shutdown task manager"""
        async_logger.info("[INFO] Shutting down async task manager...")
        self._shutdown = True
        
        # Send shutdown signals to workers
        for _ in self._workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5)
        
        async_logger.info("[OK] Async task manager shutdown complete")


# Global task manager instance
task_manager = AsyncTaskManager(max_workers=4)


def submit_async_task(
    name: str,
    func: Callable,
    args: tuple = (),
    kwargs: dict = None
) -> str:
    """
    Convenience function to submit async task
    
    Example:
        task_id = submit_async_task("inference", predict_image, args=(image_data,))
        result = task_manager.get_task_result(task_id, timeout=30)
    """
    return task_manager.submit_task(name, func, args, kwargs)


class AsyncInference:
    """Context manager for async inference operations"""
    
    def __init__(self, task_name: str = "inference"):
        self.task_name = task_name
        self.task_id = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def execute(self, func: Callable, args: tuple = (), kwargs: dict = None) -> str:
        """
        Execute function asynchronously
        
        Returns:
            Task ID for tracking
        """
        self.task_id = submit_async_task(self.task_name, func, args, kwargs)
        return self.task_id
    
    def get_result(self, timeout: float = None) -> Optional[Any]:
        """Get result if available"""
        if self.task_id:
            return task_manager.get_task_result(self.task_id, timeout)
        return None
    
    def get_status(self) -> Optional[str]:
        """Get task status"""
        if self.task_id:
            return task_manager.get_task_status(self.task_id)
        return None
