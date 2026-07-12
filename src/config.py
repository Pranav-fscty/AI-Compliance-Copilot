"""Configuration module for the compliance-copilot application.

Loads environment variables, validates mandatory configurations, and sets up
the global application logger.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve values with defaults or raise validation errors
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is missing or empty. Please set it in your .env file.")

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Parse logging level
_numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# Setup module-level logger
logging.basicConfig(
    level=_numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger("compliance-copilot")
logger.info("Configuration successfully loaded and logger initialized.")
