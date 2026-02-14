# AGENTS.md - Video2Log Development Guide

This document contains essential information for agentic coding systems working on the Video2Log repository.

## Development Plan

### Phase 1: Timer + Remote LLM (Current)
**Goal:** Get basic pipeline working - capture image every N seconds, send to remote API, save text description.

```
Timer Trigger → Capture Image → Call Remote LLM API → Save Text Log
     ↑                                                           ↓
   Repeat                                                    (delete image)
```

**Components:**
- `capture_timer.py` - Configurable interval image capture
- `llm_client.py` - HTTP client for OpenAI/Claude API
- `logger.py` - Simple text file logging

### Phase 2: Add Motion Detection
**Goal:** Replace timer with motion detection to trigger captures.

```
Motion Detection → Trigger Capture → Call Remote LLM API → Save Text Log
        ↑                                                            ↓
   Continuous                                                    (delete image)
```

### Phase 3: Raspberry Pi 3 Deployment
**Goal:** Optimize for Pi 3 hardware constraints (1GB RAM, slower CPU).

## Build, Lint, and Test Commands

### Installation & Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run application (Phase 1)
python src/capture_timer.py
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_capture_timer.py

# Run a single test function
pytest tests/test_llm_client.py::test_api_call_success

# Run tests with verbose output
pytest -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Code Style Guidelines

### Python Version
- **Target:** Python 3.9+
- **Phase 1-2 Development:** MacBook (ARM64)
- **Phase 3 Deployment:** Raspberry Pi 3 (1GB RAM)

### Imports
- Standard library first, third-party second, local last
- Organize alphabetically within groups

**Example:**
```python
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import requests

from src.logger import log_event
```

### Naming Conventions
- **Classes:** PascalCase (e.g., `CaptureTimer`, `LLMClient`)
- **Functions/Methods:** snake_case (e.g., `capture_image`, `call_api`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `CAPTURE_INTERVAL`, `API_TIMEOUT`)
- **Private methods:** Prefix with `_` (e.g., `_encode_image`)

### Type Hints
- Use type hints for all function parameters and return types
- Use `Optional[T]` for nullable types (Python 3.9 compatible)

**Example:**
```python
from typing import Optional, Dict

def capture_frame(camera_id: int = 0) -> Optional[np.ndarray]:
    """Capture a single frame from camera."""
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
```

### Error Handling
- Catch specific exceptions, never bare `except:`
- Log errors with full context
- API calls should handle timeouts and retries

**Example:**
```python
import logging

logger = logging.getLogger(__name__)

def call_llm_api(image_path: Path, prompt: str) -> Optional[str]:
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.Timeout:
        logger.error("LLM API timeout after 30s")
        return None
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}", exc_info=True)
        return None
```

### Logging
- Use `logging` module (not print)
- Log levels: DEBUG (dev), INFO (normal), ERROR (failures)

**Example:**
```python
logger.info(f"Captured image: {image_path}")
logger.debug(f"API response time: {elapsed_ms}ms")
logger.error(f"Failed to capture: {e}")
```

## Project Configuration

### Phase 1 Minimal Config (config.yaml)
```yaml
# Timer-based capture settings
capture:
  interval_seconds: 10        # Capture every 10 seconds
  camera_id: 0                # Default camera
  image_format: "jpg"
  temp_dir: "/tmp/video2log"  # Mac: /tmp, Pi: /dev/shm

# LLM API settings (Phase 1 uses remote API)
llm:
  provider: "openai"          # or "claude", "custom"
  api_key: ""                 # From environment or config
  model: "gpt-4o-mini"        # or "claude-3-haiku"
  max_tokens: 100
  timeout_seconds: 30
  prompt: "Describe what you see in this image in one short sentence."

# Logging
logging:
  level: "INFO"
  log_file: "logs/video2log.txt"
```

## File Organization

### Phase 1 Structure
```
video2log/
├── src/
│   ├── capture_timer.py      # Timer-based image capture
│   ├── llm_client.py         # Remote API client
│   ├── logger.py             # Text file logging
│   └── config.py             # Config loader
├── config/
│   └── config.yaml           # User configuration
├── logs/                     # Output text logs
├── tests/
│   ├── test_capture_timer.py
│   └── test_llm_client.py
├── requirements.txt
└── AGENTS.md
```

## Important Notes for Agents

### Phase 1 Priorities
1. **Get it working first** - Simple timer loop, basic API call
2. **No local AI** - Everything runs remotely, Pi just captures and logs
3. **Delete images immediately** - Only text logs persist
4. **Handle API failures gracefully** - Log error, continue loop

### Phase 2 Additions
- Replace timer with OpenCV motion detection
- Add frame differencing logic
- Trigger capture only on significant motion

### Phase 3 Pi 3 Optimizations
- Reduce capture resolution (e.g., 640x480)
- Add memory monitoring
- Consider batching API calls if needed

### Cross-Platform Compatibility
```python
import platform

if platform.system() == "Darwin":  # macOS
    TEMP_DIR = "/tmp/video2log"
    CAMERA_BACKEND = None  # Default
else:  # Linux (Raspberry Pi)
    TEMP_DIR = "/dev/shm/video2log"  # RAM disk
    CAMERA_BACKEND = cv2.CAP_V4L2
```

### API Integration Pattern
```python
class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    def describe_image(self, image_path: Path) -> Optional[str]:
        # 1. Encode image to base64
        # 2. Build API payload
        # 3. POST to API
        # 4. Return description or None on error
        pass
```

### Privacy Reminder
- Never commit API keys to git
- Delete image files immediately after API call
- Only store anonymized text descriptions
