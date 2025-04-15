"""
Configuration handling for semantic router.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BertModel:
    """Configuration for BERT model."""
    
    model_id: str
    use_cpu: bool


@dataclass
class RouterConfig:
    """Configuration for the semantic router."""

def load_config(config_path: str) -> RouterConfig:
    """Load configuration from a JSON file."""
