# AGENTS.md - video2log

This file is for agentic coding tools working in this repo. Keep changes minimal, follow existing patterns, and avoid introducing new frameworks unless asked.

## Project Snapshot
- Purpose: video stream/image analysis with OpenCV detectors + LLM descriptions.
- Entry points: `main.py` for CLI; core logic in `src/vision.py`.
- Key config: YAML at `config/config.yaml` (env var expansion via `${VAR}`).

## Build / Run / Lint / Test

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (local)
```bash
# dynamic/static mode and stream input
python main.py --mode dynamic --stream "<stream_url>"

# run once, use config defaults
python main.py --once

# static mode with a local image
python main.py --mode static --input "/path/to/image.jpg"
```

### Lint / Format
- No dedicated lint/format config found (no ruff/flake8/black/pre-commit).
- Keep formatting consistent with existing files (PEP 8 style, 4 spaces).
- If you add tools, update this file with exact commands.

### Tests
- pytest is in requirements; no `tests/` directory found.
- Default test run:
```bash
pytest
```
- Single test (path or node id):
```bash
pytest path/to/test_file.py::TestClass::test_name
```
- If you add tests, keep the above pattern and add any new commands here.

## Code Style Guidelines

### Language and Formatting
- Python 3; follow standard PEP 8 layout.
- Use 4-space indentation, no tabs.
- Keep line length reasonable (existing files are not strictly wrapped).
- Prefer small, readable functions over long procedural blocks.

### Imports
- Order imports as: standard library, third-party, local modules.
- Separate groups with a blank line.
- Example (from project style):
```python
# stdlib
import json
import time
from pathlib import Path

# third-party
import cv2
import numpy as np

# local
from .config import config
from .logger import logger
```

### Typing
- Use type hints for public functions and dataclasses.
- Prefer `Optional[T]` where `None` is a valid value.
- Use `Dict[str, Any]` / `list` when structure is dynamic (LLM outputs).
- In detectors, return `DetectionResult` consistently.

### Naming
- Modules: `snake_case.py`.
- Classes: `PascalCase` (e.g., `VisionProcessor`).
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: prefix `_`.

### Logging
- Use `src/logger.py` for the project logger.
- Use `logger.info` for lifecycle events, `logger.warning` for recoverable issues, `logger.error` for failures.
- Avoid `print` in core modules (except CLI fallback).

### Error Handling
- Prefer specific exception handling around I/O or external calls (OpenCV, requests).
- Log errors and return `None` or a safe default rather than crashing long-running loops.
- In LLM responses, validate JSON payloads and fallback to safe detector choices.

### Configuration
- Use `config.get("key", default)` for settings.
- Config supports `${ENV_VAR}` substitution; do not hardcode secrets.
- Default paths: output in `photos/`, logs in `logs/`.

### OpenCV / Image Handling
- Frames are BGR (`cv2.imread`/`VideoCapture`).
- Resize before LLM calls to reduce token cost.
- When writing new detectors, support `region` with normalized coords via `normalize_region`.

### Detectors
- All detectors extend `BaseDetector`.
- Include `[LLM_DESC]...[/LLM_DESC]` in docstring so LLM can select templates.
- `detect()` should return `DetectionResult` with:
  - `is_suspicious` boolean
  - `confidence` float 0.0–1.0
  - `description` for LLM context
  - `metadata` for structured data
  - `alert_reason` optional string
- Avoid heavy state; if needed, store minimal baseline (see `BlackScreenDetector`).

### LLM Client
- LLM requests go through `src/llm_client.py`.
- Validate API responses; guard against missing keys.
- Use `requests` with timeouts and `raise_for_status()`.

### File/Path Conventions
- Use `pathlib.Path` for filesystem paths.
- Create directories with `mkdir(parents=True, exist_ok=True)`.

## Repository Conventions
- No Cursor or Copilot instruction files found at `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`.
- Keep README and AGENTS in sync when behavior changes.

## Agent Notes
- Do not introduce non-ASCII unless the file already uses it.
- Avoid refactors unless requested; prefer targeted edits.
- If adding new commands (lint/test), update this file.
