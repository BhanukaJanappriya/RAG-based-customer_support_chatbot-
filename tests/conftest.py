"""Shared pytest fixtures and path configuration."""

import sys
from pathlib import Path

# Ensure the project root is importable when running pytest from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))
