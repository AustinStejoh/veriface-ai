# System Design Improvements - Deepfake Detector API

## Overview
Enhanced the backend system architecture with three major improvements:
1. **Redis Caching** - Intelligent prediction caching for performance
2. **Request Logging** - Comprehensive request/response tracking
3. **Async Processing** - Non-blocking inference for scalability

---

## 1. Redis Caching System

### Purpose
Reduces redundant inference computations by caching predictions based on image hash.

### Implementation Details

**Module**: `src/redis_cache.py`

#### Features
- **Lazy Connection**: Automatically attempts Redis connection on first use
- **Graceful Degradation**: App works without Redis (logging warnings instead of failing)
- **Image Hash-Based Keys**: SHA256 hashing ensures consistent cache keys
- **TTL Management**: Configurable time-to-live per cache entry (default: 1 hour)
- **Connection Pooling**: Maintains persistent Redis connection
- **Prefix-Based Operations**: Clear cache by category (e.g., predictions, metrics)

#### Cache Key Prefixes
```python
PREDICTION_CACHE_PREFIX = "pred:"           # Cached predictions
HEATMAP_CACHE_PREFIX = "hm:"                # Cached heatmaps  
ROBUSTNESS_CACHE_PREFIX = "robust:"         # Robustness test results
METRICS_CACHE_PREFIX = "metrics:"           # Model metrics
```

#### API Methods

**`cache.set(key, value, ttl=3600)`**
- Stores serialized JSON in Redis
- Logs operation duration
- Returns: Boolean (success/failure)

**`cache.get(key)`**
- Retrieves value from Redis
- Logs cache hit/miss with timing
- Returns: Deserialized value or None

**`cache.delete(key)`**
- Removes single key from cache

**`cache.clear_prefix(prefix)`**
- Removes all keys matching prefix pattern
- Useful for clearing all predictions: `cache.clear_prefix("pred:")`

**`cache.get_stats()`**
- Returns Redis connection stats
- Memory usage, client connections, uptime

#### Usage Example
```python
from redis_cache import cache, PREDICTION_CACHE_PREFIX

# Check cache
result = cache.get(f"{PREDICTION_CACHE_PREFIX}{image_hash}")
if result:
    print("Cache hit!")
    return result

# Perform computation
result = expensive_operation()

# Store in cache (1 hour)
cache.set(f"{PREDICTION_CACHE_PREFIX}{image_hash}", result, ttl=3600)
```

#### Configuration
To use Redis with this project:
```bash
# Install Redis (if not already installed)
# Option 1: Windows Subsystem for Linux
wsl -u root bash
apt-get install redis-server
redis-server

# Option 2: Docker
docker run -d -p 6379:6379 redis:latest

# Option 3: Windows (requires Memurai or WSL)
```

---

## 2. Request Logging System

### Purpose
Tracks all HTTP requests/responses with detailed timing information for debugging and monitoring.

### Implementation Details

**Module**: `src/logger_config.py`

#### Logger Instances
- **`app_logger`** - General application events
- **`request_logger`** - HTTP request/response tracking
- **`error_logger`** - Error-specific logging
- **`cache_logger`** - Redis cache operations
- **`async_logger`** - Async task operations

#### Log Rotation
- **Max Size**: 10MB per log file
- **Backup Count**: 5 previous versions retained
- **Location**: `logs/` directory (auto-created)

#### Files Generated
```
logs/
├── app.log              # General application logs
├── requests.log         # HTTP request/response logs
├── errors.log           # Error-specific logs
├── cache.log            # Cache operation logs
└── async.log            # Async task operation logs
```

#### Request Logging Middleware

**Before Request** (`@app.before_request`)
```
→ GET /predict from 127.0.0.1
```
- Captures request timestamp and client IP
- Creates RequestLogger instance for tracking

**After Request** (`@app.after_request`)
```
✓ GET /predict | Status: 200 | Client: 127.0.0.1 | Response: 2048B | Duration: 234.56ms
✗ POST /predict | Status: 500 | Client: 127.0.0.1 | Error: Invalid image format | Duration: 12.34ms
```
- Logs HTTP status, response size, and total duration
- Differentiates success (2xx-3xx) from failure (4xx-5xx)

#### Log Format
```
%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s
```
Example:
```
2026-04-20 12:35:42 - app - INFO - [app.py:45] - Model loaded on device: cuda
```

#### Usage Examples

**Log Application Events**
```python
from logger_config import app_logger

app_logger.info("Processing image for deepfake detection")
app_logger.error("Failed to load model: file not found")
```

**Log Cache Operations**
```python
from logger_config import log_cache_operation

log_cache_operation("GET", "pred:abc123def456", hit=True, duration_ms=2.34)
# Output: GET      | HIT  | Key: pred:abc123def456 | Duration: 2.34ms
```

**Log Async Tasks**
```python
from logger_config import log_async_task

log_async_task("task-id-123", "robustness_test", "completed", duration_ms=5234.56)
# Output: Task task-id-123 (robustness_test) - completed - Duration: 5234.56ms
```

---

## 3. Async Processing System

### Purpose
Enables long-running operations (inference, robustness testing) to run asynchronously without blocking HTTP requests.

### Implementation Details

**Module**: `src/async_processor.py`

#### Architecture
- **Worker Pool**: Configurable number of background threads (default: 4)
- **Task Queue**: FIFO queue for pending tasks
- **Task Manager**: Singleton instance tracking all tasks
- **Status Tracking**: Real-time task status updates

#### Task States
```python
TASK_PENDING = "pending"        # Waiting in queue
TASK_RUNNING = "running"        # Currently executing
TASK_COMPLETED = "completed"    # Finished successfully
TASK_FAILED = "failed"          # Encountered error
```

#### Core Components

**`AsyncTask` Dataclass**
```python
@dataclass
class AsyncTask:
    task_id: str                # Unique identifier (UUID)
    name: str                   # Task name (e.g., "prediction")
    status: str                 # Current status
    result: Optional[Any]       # Execution result
    error: Optional[str]        # Error message if failed
    created_at: str             # ISO 8601 timestamp
    started_at: Optional[str]   # Start time
    completed_at: Optional[str] # Completion time
    duration_ms: float          # Total execution time
```

**`AsyncTaskManager` Class**

Methods:
- **`submit_task(name, func, args, kwargs)`** → Task ID
  - Submits function for async execution
  - Returns immediately with task ID
  
- **`get_task(task_id)`** → AsyncTask or None
  - Retrieve task object by ID
  
- **`get_task_result(task_id, timeout)`** → Result or None
  - Blocking call that waits for task completion
  - Optional timeout in seconds
  
- **`get_task_status(task_id)`** → Status string or None
  - Quick status check without waiting
  
- **`wait_for_task(task_id, timeout)`** → Boolean
  - Wait for task completion
  - Returns True if completed, False if timeout
  
- **`get_stats()`** → Dict with task statistics
  - Total tasks, completed, failed, running, pending, queue size

#### Usage Example

**Async Prediction**
```python
from async_processor import submit_async_task, task_manager

# Submit task
task_id = submit_async_task("prediction", perform_inference, args=(image_bytes,))

# Later, check status
task = task_manager.get_task(task_id)
if task.status == "completed":
    result = task.result
elif task.status == "failed":
    error = task.error
```

**API Endpoints**

**`POST /predict?async=true`**
```json
{
  "status": "processing",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Prediction submitted for async processing",
  "check_url": "/task/550e8400-e29b-41d4-a716-446655440000"
}
```

**`GET /task/{task_id}`**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "created_at": "2026-04-20T12:35:42.123456",
  "started_at": "2026-04-20T12:35:42.654321",
  "completed_at": null,
  "duration_ms": 0
}
```

When completed:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "prediction": "fake",
    "confidence": 92.34,
    "inference_time_ms": 45.23
  },
  "duration_ms": 234.56
}
```

---

## 4. Integrated API Endpoints

### Prediction with Caching & Async

**`POST /predict`**
- **Synchronous**: `curl -F "file=@image.jpg" http://localhost:5000/predict`
- **Asynchronous**: `curl -F "file=@image.jpg" http://localhost:5000/predict?async=true`

**Features**:
1. Checks Redis cache before inference
2. Uses SHA256 image hash for cache key
3. Stores result with 1-hour TTL
4. Logs all operations with timing

Response:
```json
{
  "prediction": "real",
  "confidence": 87.45,
  "heatmap": "iVBORw0KGgoAAAANS...",
  "inference_time_ms": 45.23,
  "cached": false,
  "message": "This image is REAL with 87.5% confidence"
}
```

### Robustness Testing with Async

**`POST /robustness-test`**
- **Synchronous**: `curl -F "file=@image.jpg" http://localhost:5000/robustness-test`
- **Asynchronous**: `curl -F "file=@image.jpg" http://localhost:5000/robustness-test?async=true`

### Model Metrics with Caching

**`GET /metrics`**
- Cached for 1 hour
- Includes overall performance, per-class metrics, confusion matrix

### Health Check with System Status

**`GET /health`**
```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true,
  "cache_connected": true,
  "async_workers": 4,
  "pending_tasks": 0,
  "timestamp": 1713606942.123456
}
```

### Cache Management

**`GET /cache/stats`**
```json
{
  "cache": {
    "connected": true,
    "used_memory_human": "2.5M",
    "connected_clients": 1,
    "total_commands_processed": 1245
  },
  "async": {
    "total_tasks": 42,
    "completed": 38,
    "failed": 1,
    "running": 2,
    "pending": 1,
    "workers": 4
  }
}
```

**`POST /cache/clear`**
- Clears all cached predictions
- Returns number of entries removed

### Task Status Checking

**`GET /task/{task_id}`**
- Get status and result of async task
- Polls for updates without re-running computation

---

## Performance Improvements

### Caching Benefits
- **Cache Hit**: ~5ms (vs 45ms full inference)
- **Reduction**: ~90% faster for repeated images
- **Memory Trade-off**: ~10MB per 1000 cached predictions

### Async Benefits
- **Non-blocking**: HTTP request returns immediately (202 status)
- **Scalability**: Multiple inferences can run in parallel
- **Resource Management**: 4 worker threads handle queue

### Robustness Testing
- **Serial Execution**: 7 perturbations × 45ms = ~315ms total
- **Async Option**: Submit, poll for results later
- **Status Tracking**: Monitor progress via `/task/{id}` endpoint

---

## Performance Monitoring

### Check System Status
```bash
curl http://localhost:5000/health
curl http://localhost:5000/cache/stats
```

### View Logs
```bash
# Monitor request logs (real-time)
tail -f logs/requests.log

# View cache operations
grep "cache.log" logs/cache.log | tail -20

# Check for errors
grep ERROR logs/errors.log

# Async task tracking
grep "async" logs/async.log
```

### Load Testing
```bash
# Test prediction caching (same image multiple times)
for i in {1..10}; do
  curl -F "file=@test_image.jpg" http://localhost:5000/predict
done

# Test async processing
curl -F "file=@image.jpg" "http://localhost:5000/predict?async=true"

# Monitor task completion
curl http://localhost:5000/task/{task_id}
```

---

## Configuration

### Environment Variables
Create `.env` file:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
ASYNC_WORKERS=4
CACHE_TTL=3600
```

### Dependencies
```
redis==5.0.0           # Redis Python client
flask==3.0.0           # Web framework
flask-cors==4.0.0      # CORS support
torch==2.11.0          # PyTorch
torchvision==0.26.0    # Vision models
```

---

## Troubleshooting

### Redis Connection Fails
```
[WARN] Redis connection failed: Connection refused
```
- **Solution**: Start Redis server or use Docker
- **Fallback**: App works without Redis (logs warning)

### Unicode Encoding Errors
```
UnicodeEncodeError: 'charmap' codec can't encode
```
- **Fixed**: Using [OK], [ERR], [INFO] instead of emoji
- **Windows Console**: Set environment variable
  ```powershell
  $env:PYTHONIOENCODING="utf-8"
  ```

### Task Hangs
```
GET /task/{id} returns pending forever
```
- **Check**: Worker threads running (see `/health`)
- **Check**: No exceptions in `logs/async.log`
- **Restart**: Kill and restart Flask app

---

## Best Practices

1. **Always use `/predict?async=true` for batch processing**
   - Submit multiple tasks, then poll for results

2. **Check cache stats regularly**
   - Monitor hit ratio: `GET /cache/stats`
   - Clear if needed: `POST /cache/clear`

3. **Monitor logs for errors**
   - Check `logs/errors.log` for exceptions
   - Check `logs/async.log` for task failures

4. **Use health check for monitoring**
   - Verify system status: `GET /health`
   - Monitor worker availability

5. **Set appropriate timeouts**
   - Prediction: 30-60 seconds
   - Robustness test: 2-5 minutes

---

## Future Enhancements

- [ ] Celery integration for distributed task processing
- [ ] Redis pub/sub for real-time task status updates
- [ ] Database persistence for historical logs
- [ ] Metrics export (Prometheus format)
- [ ] Task priority queue
- [ ] Automatic cache eviction policies
- [ ] Advanced analytics dashboard
