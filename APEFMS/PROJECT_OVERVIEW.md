# Project Overview

## What This Project Is About

This project is a **distributed device management system** designed for centralized control, monitoring, and security management of fleets of devices across a network. It enables system administrators and security teams to remotely execute commands, monitor device health, assess security vulnerabilities, and automate remediation through an intelligent web-based interface.

## Core Purpose

The system addresses the fundamental challenge of managing distributed device fleets by providing:

- **Centralized Control**: Single web interface to manage multiple devices simultaneously
- **Real-Time Visibility**: Live status monitoring and heartbeat-based online/offline detection
- **Security Assessment**: Automated vulnerability scanning with AI-powered risk scoring
- **Intelligent Scheduling**: Optimal time window calculation for command execution
- **Natural Language Interaction**: AI assistant for querying fleet status and device information

## Key Features

### Real-Time Communication
- Bidirectional Socket.IO communication between admin server and device clients
- 30-second heartbeat mechanism for live device status tracking
- Instant command dispatch and result reporting
- Support for WebSocket and polling transports

### AI-Powered Security
- **Risk Scoring**: Automatic 0-100 scale risk assessment using Gemini AI after each scan
- **Anomaly Detection**: Real-time alerts when new vulnerabilities appear
- **Natural Language Queries**: Ask questions like "Which device has the highest risk score?" or "Show me all offline devices"
- **Adaptive Scan Frequency**: AI-recommended scan intervals based on group risk profiles

### Fleet Management
- **Group Organization**: Organize devices into logical groups (e.g., finance, analyst, production)
- **Targeted Commands**: Execute commands on specific devices, groups, or entire fleet
- **Device Inventory**: Comprehensive view of device status, risk scores, and uptime metrics
- **Execution History**: Track all command executions with detailed results

### Plugin System
- Modular architecture for device operations
- Built-in plugins for security tasks:
  - `disable_camera`: Disable camera functionality
  - `disable_microphone`: Disable microphone
  - `disable_usb`: Disable USB ports
  - `run_all_scans`: Execute comprehensive security scans
- Extensible framework for adding new device operations

### Intelligent Scheduling
- **Optimal Time Windows**: Metrics-based analysis to find best execution times
- **Three Scheduling Modes**: Per-fleet, per-client, or per-group optimization
- **Scheduled Commands**: Queue commands for future execution with automatic dispatch
- **Client Absence Handling**: Commands execute immediately when devices come online

### Security Dashboard
- Vulnerability summaries by group and individual device
- Scan trend visualization (14-day history)
- Risk score distribution across fleet
- Remediation plan generation and dispatch

## Use Cases

### System Administration
- Remote configuration management across device fleets
- Automated software updates and patching
- Device health monitoring and alerting
- Centralized log aggregation and analysis

### Security Operations
- Continuous vulnerability scanning and assessment
- Risk-based prioritization of security issues
- Automated remediation workflow
- Compliance reporting and audit trails

### DevOps Operations
- Fleet-wide command execution
- Scheduled maintenance windows
- Performance monitoring and metrics collection
- Incident response automation

## Target Users

- **System Administrators**: Managing distributed infrastructure
- **Security Teams**: Conducting vulnerability assessments and remediation
- **DevOps Engineers**: Automating fleet operations and maintenance
- **IT Operations**: Monitoring device health and availability

## Business Value

### Operational Efficiency
- Reduce manual device management overhead by 80%+
- Execute fleet-wide commands in seconds instead of hours
- Automated scheduling eliminates manual coordination

### Security Posture
- Continuous visibility into security vulnerabilities
- AI-powered risk assessment prioritizes critical issues
- Automated remediation reduces mean time to remediate (MTTR)

### Cost Savings
- Optimal scheduling reduces downtime impact
- Proactive monitoring prevents costly outages
- Centralized management reduces operational complexity

### Scalability
- Supports fleets of hundreds of devices
- Modular architecture allows easy extension
- Efficient communication protocols minimize network overhead

## System Components

### Admin Server
- Flask-based web application with Socket.IO
- Web interface for fleet administration
- AI integration (Gemini, Ollama) for intelligent features
- RESTful APIs for integration with external systems

### Device Clients
- Lightweight Socket.IO client agents
- Plugin execution engine
- Metrics collection (CPU, memory, latency)
- Task queue for command processing

### Data Storage
- JSON-based storage for reports and configurations
- Device metrics history for optimal window analysis
- Vulnerability reports and risk scores
- Scheduled command queue

## Deployment Model

- **Admin Server**: Central server running on port 8765
- **Device Clients**: Distributed agents connecting to admin server
- **Communication**: Real-time bidirectional Socket.IO over HTTP/WebSocket
- **Scalability**: Horizontal scaling possible with load balancer

## Security Considerations

- Transport encryption via HTTPS (recommended for production)
- Plugin whitelist to prevent unauthorized command execution
- Target-based command routing for access control
- Audit logging for all command executions

## Future Roadmap

- Multi-admin server support with clustering
- Advanced role-based access control (RBAC)
- Integration with SIEM platforms
- Mobile application for on-the-go management
- Enhanced remediation automation with rollback capabilities
