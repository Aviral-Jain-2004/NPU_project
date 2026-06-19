# Technical Implementation

## How We Built It

This document describes the technical architecture, implementation details, and design decisions behind the distributed device management system.

## System Architecture

### High-Level Architecture

The system follows a **client-server architecture** with real-time bidirectional communication:

```
┌─────────────────┐         Socket.IO          ┌─────────────────┐
│   Admin Server  │◄──────────────────────────►│  Device Client  │
│   (Flask + SIO) │   WebSocket / Polling      │  (python-sio)   │
└─────────────────┘                            └─────────────────┘
         │                                              │
         │                                              │
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│  AI Integration │                            │  Plugin Engine  │
│  (Gemini/Ollama)│                            │  (Task Queue)   │
└─────────────────┘                            └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│  JSON Storage   │                            │  Metrics Collector│
│  (Reports/Data) │                            │  (psutil)       │
└─────────────────┘                            └─────────────────┘
```

### Component Breakdown

#### 1. Admin Server (`admin/`)

**Technology Stack:**
- **Flask 3.0.2**: Web framework for HTTP endpoints and template rendering
- **Flask-SocketIO 5.3.6**: WebSocket integration for real-time communication
- **eventlet 0.35.2**: Async WebSocket server implementation
- **google-generativeai 0.4.1**: Gemini AI integration for risk scoring and NLP
- **python-dateutil 2.9.0**: Date/time parsing and manipulation

**Key Components:**

- **`app.py`**: Main Flask application with route definitions and Socket.IO setup
- **`Admin_utils/routes.py`**: Core admin routes and Socket.IO event handlers
- **`Admin_utils/groups_routes.py`**: Group management endpoints
- **`Admin_utils/scheduler_routes.py`**: Scan scheduler CRUD operations
- **`core/ai/`**: AI integration modules (risk scoring, charting, anomaly detection)
- **`core/Admin_utils/`**: Device management, history tracking, state management
- **`storage/`**: JSON-based data persistence

**Blueprint Architecture:**
```python
# Modular route organization
admin_bp          # Core admin functionality
groups_bp         # Group management
scheduler_bp      # Scan scheduling
```

#### 2. Device Client (`client/`)

**Technology Stack:**
- **python-socketio 5.16.1**: Socket.IO client implementation
- **python-engineio 4.13.1**: Engine.IO client for transport layer
- **psutil 5.9.8**: System metrics collection (CPU, memory, network)
- **websocket-client 1.9.0**: WebSocket transport support
- **requests 2.32.3**: HTTP client for polling transport

**Key Components:**

- **`client.py`**: Entry point that initializes and runs the client
- **`base_client.py`**: Core client logic with Socket.IO handlers and task queue
- **`plugins/`**: Modular plugin implementations for device operations
- **`core/plugins/executor.py`**: Plugin execution engine
- **`core/logger.py`**: Structured logging system

## Communication Protocol

### Socket.IO Event Flow

**Client Registration:**
```
Client → Server: register {name: "Alice"}
Server → Client: connection established
```

**Heartbeat Mechanism:**
```
Client → Server: heartbeat {name: "Alice"} (every 30s)
Server: Updates last_seen timestamp
Server: Determines online/offline status (90s timeout)
```

**Command Execution:**
```
Server → Client: server_command {
  plugin: "run_all_scans",
  targets: ["Alice"],
  command_id: "uuid",
  parameters: {},
  groups: ["analyst"]
}
Client → Server: plugin_status {
  type: "plugin_status",
  name: "Alice",
  plugin: "run_all_scans",
  phase: "started",
  command_id: "uuid"
}
Client → Server: plugin_result {
  type: "plugin_result",
  name: "Alice",
  plugin: "run_all_scans",
  status: "success",
  exit_code: 0,
  stdout: "...",
  stderr: "",
  groups: ["analyst"],
  command_id: "uuid"
}
```

**Metrics Collection:**
```
Client → Server: client_metrics_batch {
  metrics: [
    {timestamp, client, cpu_percent, memory_percent, user_active, server_latency_ms},
    ...
  ]
}
```

### Transport Layer

**Primary:** WebSocket (low latency, bidirectional)
**Fallback:** HTTP Long-Polling (for restrictive networks)

Configuration:
```python
DEVICE_TRANSPORTS = "websocket,polling"  # Ordered preference
```

## AI Integration

### 1. Risk Scoring (Gemini)

**Implementation:** `core/ai/risk_score.py`

**Process:**
1. After `run_all_scans` completes, parse vulnerability report
2. Extract CVEs, open ports, and security findings
3. Send structured prompt to Gemini API
4. Receive 0-100 risk score with category label
5. Store in `storage/endpoints/<device>/risk_score_latest.json`
6. Append to `risk_score_history.jsonl` for trend analysis

**Prompt Engineering:**
```
Analyze the following security scan results and provide:
1. Risk score (0-100)
2. Risk category (Critical/High/Medium/Low/Clean)
3. Key factors contributing to risk
```

**Scoring Criteria:**
- **80-100 Critical**: Active critical CVEs or dangerous open ports
- **60-79 High**: Several high-severity CVEs or multiple risky ports
- **40-59 Medium**: Moderate findings
- **20-39 Low**: Minor informational issues
- **0-19 Clean**: No significant findings

### 2. Natural Language Queries (Ollama + Gemini)

**Implementation:** `core/ai/executor.py`, `core/ai/planner.py`

**Models Used:**
- **Ollama phi3.5**: Fast intent recognition and parameter extraction
- **Gemini**: Complex query analysis and chart generation

**Query Processing Pipeline:**
```
User Query → Intent Recognition → Parameter Extraction → 
Tool Selection → Execution → Response Formatting
```

**Supported Query Types:**
- Device status queries ("Is Alice online?")
- Risk score queries ("Which device has highest risk?")
- Scan history queries ("Show me scan trend for finance group")
- Chart generation ("Visualize success vs error distribution")

### 3. Adaptive Scan Frequency (Ollama)

**Implementation:** `core/ai/adaptive_scheduler.py`

**Analysis Factors:**
- Average risk score per group
- Vulnerability count volatility
- Recent scan failure rate
- Device availability patterns

**Recommendation Logic:**
```
High risk + high volatility → every_6h
Medium risk + moderate volatility → every_12h
Stable low risk → daily
Very stable → weekly
```

### 4. Chart Intent Recognition

**Implementation:** `core/ai/chart_intent.py`, `core/ai/chart_builder.py`

**Fuzzy Matching:**
- Levenshtein distance for typo tolerance
- Phonetic matching for similar-sounding names
- Keyword extraction for fast-path matching

**Chart Types:**
- Success vs Error Distribution (7-day window)
- Plugin execution trends
- Risk score distribution
- Scan trend visualization

## Plugin System

### Architecture

**Plugin Registry** (`base_client.py`):
```python
CANONICAL_PLUGINS = {
    "disable_camera",
    "disable_microphone",
    "disable_usb",
    "run_all_scans",
    "receive_remediation_plan",
}

PLUGIN_MODULES = {
    "disable_camera": disable_camera,
    "disable_microphone": disable_microphone,
    "disable_usb": disable_usb,
    "run_all_scans": scan_all,
    "receive_remediation_plan": receive_remediation_plan,
}
```

### Plugin Execution Flow

```
Server Command Received → Plugin Validation → 
Task Queue → Worker Thread → Plugin Execution → 
Result Emission → Storage
```

**Thread-Safe Task Queue:**
```python
_task_queue: Queue[Tuple[str, Dict, Optional[str], List[str]]]

def _queue_worker():
    while True:
        plugin, params, cmd_id, groups = _task_queue.get()
        try:
            module = PLUGIN_MODULES.get(plugin)
            result = execute_plugin(module, params, NAME)
            _emit_result(plugin, result, cmd_id, groups)
        finally:
            _task_queue.task_done()
```

### Adding New Plugins

1. Create plugin file in `client/plugins/`
2. Implement required functions
3. Add to `CANONICAL_PLUGINS` set
4. Add module mapping in `PLUGIN_MODULES`
5. Add aliases in `PLUGIN_ALIASES` if needed

## Metrics Collection

### System Metrics Architecture

**Collection Interval:** 5 minutes
**Batch Size:** 5 metrics per batch
**Batch Timeout:** 25 minutes (force send if not full)

**Metrics Collected:**
- CPU utilization percentage
- Memory utilization percentage
- CPU core count
- User activity detection (CPU-based heuristic)
- Network latency to server (socket connection time)
- Platform information

**Implementation** (`base_client.py`):
```python
def _collect_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    user_active = cpu_percent > 5
    server_latency_ms = measure_socket_latency(SERVER_URL)
    return {
        "timestamp": time.time(),
        "client": NAME,
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "user_active": user_active,
        "server_latency_ms": server_latency_ms,
        "platform": platform.system(),
    }
```

### Buffering Strategy

**In-Memory Buffer:**
- Thread-safe buffer using `_metrics_buffer_lock`
- Batches metrics to reduce network overhead

**Disk Buffer:**
- Persists metrics to `storage/buffered_metrics.json` on disconnect
- Syncs buffered metrics on reconnection
- Survives client crashes

## Optimal Time Window Calculation

### Metrics Analyzer

**Implementation:** `core/ai/metrics_analyzer.py`

**Input Data:** `storage/metrics.jsonl` (historical metrics)

**Analysis Algorithm:**
1. Load metrics for specified time window (default 7 days)
2. Group metrics by hour of day
3. Calculate availability score per hour:
   ```
   availability = (metrics with cpu_percent < threshold AND 
                   user_active == false) / total_metrics
   ```
4. Select hour with highest availability above threshold
5. Calculate window duration based on availability pattern

**Scheduling Modes:**
- **per_fleet**: Single optimal window for entire fleet
- **per_client**: Individual optimal windows per device
- **per_group**: Group-specific optimal windows

### Daemon Scheduler

**Implementation:** Background thread in admin server

**Schedule:**
- Run on startup if data is stale (>24 hours old)
- Daily run at 3:00 AM IST

**Trigger:**
```python
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_analysis,
    'cron',
    hour=3,  # 3 AM IST
    timezone=pytz.timezone('Asia/Kolkata')
)
```

## Scheduled Commands

### Command Lifecycle

```
Creation → Pending → Dispatched → Completed/Cancelled
```

**Data Structure** (`storage/scheduled_commands.jsonl`):
```json
{
  "id": "uuid",
  "plugin": "run_all_scans",
  "targets": ["Alice", "Bob"],
  "groups": ["analyst"],
  "parameters": {},
  "scheduled_for": "2024-01-15T03:30:00Z",
  "status": "pending",
  "created_at": "2024-01-14T10:00:00Z",
  "dispatched_at": null,
  "completed_at": null
}
```

### Dispatch Logic

**Implementation:** Background scheduler thread

**Process:**
1. Check every minute for pending commands
2. Compare `scheduled_for` with current time
3. For each due command:
   - Check device online status
   - Dispatch to online devices immediately
   - Mark offline devices as "dispatched" (will execute on connection)
   - Update command status to "dispatched"

**Client Absence Handling:**
- Commands saved with "dispatched" status
- On client connection, server checks for pending commands
- Immediate dispatch if device is target

## Data Storage

### Storage Structure

```
storage/
├── endpoints/
│   ├── <device_name>/
│   │   ├── combined_report_latest.json
│   │   ├── risk_score_latest.json
│   │   ├── risk_score_history.jsonl
│   │   └── remediation_plan.json
├── group_summaries/
│   ├── <group_name>_vulnerabilities.json
│   └── ...
├── logs/
│   ├── logs.jsonl
│   └── scan_all_executions.jsonl
├── groups.json
├── device.jsonl
├── metrics.jsonl
├── optimal_windows.json
└── scheduled_commands.jsonl
```

### JSONL Format

**Why JSONL (JSON Lines):**
- Append-only writes (no file locking issues)
- Efficient for streaming large datasets
- Easy to parse line-by-line
- Supports concurrent writes

**Example:**
```jsonl
{"timestamp": "2024-01-15T10:00:00Z", "event": "register", "payload": {"name": "Alice"}}
{"timestamp": "2024-01-15T10:30:00Z", "event": "heartbeat", "payload": {"name": "Alice"}}
```

## Anomaly Detection

### Baseline Delta Detection

**Implementation:** `core/ai/anomaly_detector.py`

**Process:**
1. Load current scan results
2. Load previous scan results for same device
3. Compare vulnerability fingerprints
4. Identify new vulnerabilities (in current but not in previous)
5. Emit Socket.IO event `new_vulns_detected` if new findings
6. Frontend displays toast notification

**Fingerprint Comparison:**
```python
current_fps = {v['fingerprint'] for v in current_vulns}
previous_fps = {v['fingerprint'] for v in previous_vulns}
new_vulns = current_fps - previous_fps
```

## Remediation Workflow

### Fix Suggestion

**Implementation:** `core/ai/fix_suggester.py`

**Two Modes:**
1. **Ollama (Fast)**: For single vulnerability suggestions
2. **Gemini (Bulk)**: For batch remediation plan generation

**Prompt Structure:**
```
Given vulnerability:
- Description: {description}
- Source: {source}
- Category: {category}
- Severity: {severity}

Provide:
1. Fix command
2. Explanation
3. OSV ID (if applicable)
```

### Remediation Plan Dispatch

**Flow:**
1. User selects vulnerabilities for remediation
2. Admin generates remediation plan with fix commands
3. Plan saved to `storage/endpoints/<device>/remediation_plan.json`
4. If device online, dispatch via Socket.IO
5. Client receives plan and saves (POC: execution not implemented)
6. Admin tracks plan status (pending/delivered/applied)

## Web Interface

### Frontend Technology

**Templates:** Jinja2 templates in `admin/Admin_utils/templates/`

**Key Pages:**
- `index.html`: Home page with command history
- `fleet_administration.html`: Fleet management UI
- `scan_scheduler.html`: Scan scheduling interface
- `scheduling.html`: Optimal windows and scheduled commands
- `manageability_dashboard.html`: Security dashboard

### Dashboard APIs

**Real-Time Updates:**
- Socket.IO events for live status updates
- Polling APIs for periodic data refresh
- JSON file serving for static reports

**Key APIs:**
- `/api/devices/status`: Live online/offline status
- `/api/devices/risk_scores`: AI risk scores
- `/api/scan_trend`: 14-day scan success/failure trend
- `/api/fleet/health`: Fleet health summary
- `/api/fleet/inventory`: Complete device inventory

## Security Considerations

### Transport Security
- Currently uses HTTP (development mode)
- Production deployment should use HTTPS
- Socket.IO supports SSL/TLS encryption

### Plugin Whitelist
- Only canonical plugins can be executed
- Server-side validation before command dispatch
- Client-side validation for defense in depth

### Target-Based Access Control
- Commands specify target devices/groups
- Clients verify they are in target list before execution
- Prevents unauthorized command execution

### Audit Logging
- All command executions logged to `logs.jsonl`
- Device registration/disconnect events in `device.jsonl`
- Scan executions tracked in `scan_all_executions.jsonl`

## Performance Optimizations

### Batching
- Metrics sent in batches of 5
- Reduces network overhead by 80%
- Timeout ensures data freshness

### Caching
- Risk scores cached in memory
- Group summaries cached until next scan
- Device state cached with 90-second TTL

### Async Operations
- Plugin execution in worker threads
- Metrics collection in background thread
- Socket.IO communication non-blocking

### Efficient Storage
- JSONL for append-only logs
- Incremental updates to reports
- Historical data pruning (7-day expiry for remediation plans)

## Deployment Considerations

### Scaling

**Horizontal Scaling:**
- Admin server can be load-balanced
- Socket.IO requires sticky sessions
- Redis adapter for multi-server Socket.IO

**Vertical Scaling:**
- Increase worker processes with eventlet
- Optimize metrics collection interval
- Implement connection pooling

### Monitoring

**Health Checks:**
- `/api/devices/status` endpoint
- Heartbeat monitoring
- Error rate tracking

**Logging:**
- Structured JSON logs
- Log rotation for long-running deployments
- Centralized log aggregation (optional)

### Configuration

**Environment Variables:**
```bash
# Admin Server
FLASK_SECRET=your-secret-key
DEVICE_SERVER_URL=http://localhost:8765

# Client
DEVICE_TRANSPORTS=websocket,polling
```

**Storage Paths:**
- Configurable via environment variables
- Default: `storage/` directory
- Supports external storage mounts

## Development Workflow

### Adding New Features

1. **Feature Planning**: Define requirements and API contracts
2. **Backend Implementation**: Add routes, Socket.IO handlers, business logic
3. **Frontend Integration**: Update templates, add JavaScript handlers
4. **Testing**: Manual testing with multiple clients
5. **Documentation**: Update README and technical docs

### Debugging

**Admin Server Logs:**
- Flask application logs
- Socket.IO connection logs
- AI API call logs

**Client Logs:**
- Connection status
- Plugin execution results
- Metrics collection status

**Common Issues:**
- Connection failures: Check firewall, port availability
- Plugin errors: Verify plugin registration and dependencies
- AI failures: Check API keys and rate limits

## Future Enhancements

### Planned Features
- Multi-admin clustering with Redis
- Role-based access control (RBAC)
- Mobile application (React Native)
- SIEM integration (Splunk, ELK)
- Enhanced remediation with rollback
- Container-based client deployment

### Technical Debt
- Migrate from JSONL to proper database (PostgreSQL)
- Implement proper authentication/authorization
- Add comprehensive unit tests
- Improve error handling and resilience
- Add Prometheus metrics for monitoring

## Conclusion

This distributed device management system demonstrates a modern approach to fleet management using real-time communication, AI-powered insights, and modular architecture. The combination of Flask, Socket.IO, and AI services creates a scalable, intelligent platform for managing distributed device fleets efficiently and securely.
