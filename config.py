# -*- coding: utf-8 -*-
"""
Configuration module for Veo Web App
Centralizes all settings with validation and defaults
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_gemini_keys_from_env() -> List[str]:
    """Load all Gemini API keys from environment variables"""
    keys = []
    
    # Check numbered keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
    for i in range(1, 20):
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if key and key.strip() and not key.startswith("your-"):
            keys.append(key.strip())
    
    # Also check single key formats
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        key = os.environ.get(var)
        if key and key.strip() and not key.startswith("your-") and key not in keys:
            keys.append(key.strip())
    
    print(f"[Config] Loaded {len(keys)} Gemini API keys from environment", flush=True)
    return keys


def get_openai_key_from_env() -> Optional[str]:
    """Load OpenAI API key from environment"""
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip() and not key.startswith("sk-your"):
        return key.strip()
    return None


class AspectRatio(str, Enum):
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"


class Resolution(str, Enum):
    HD = "720p"
    FULL_HD = "1080p"


class Duration(str, Enum):
    SHORT = "4"
    MEDIUM = "6"
    LONG = "8"


class PersonGeneration(str, Enum):
    ALLOW_ALL = "allow_all"
    ALLOW_ADULT = "allow_adult"
    DONT_ALLOW = "dont_allow"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    REDO_QUEUED = "redo_queued"  # Waiting for redo generation


class ApprovalStatus(str, Enum):
    PENDING_REVIEW = "pending_review"  # Waiting for user to review
    APPROVED = "approved"              # User accepted the clip
    REJECTED = "rejected"              # User requested redo (in progress)
    MAX_ATTEMPTS = "max_attempts"      # Hit 3 attempts, needs support contact


class ErrorCode(str, Enum):
    # API Errors
    RATE_LIMIT = "RATE_LIMIT_429"
    API_KEY_INVALID = "API_KEY_INVALID"
    API_QUOTA_EXCEEDED = "API_QUOTA_EXCEEDED"
    API_NETWORK_ERROR = "API_NETWORK_ERROR"
    API_TIMEOUT = "API_TIMEOUT"
    
    # Content Filtering
    CELEBRITY_FILTER = "CELEBRITY_RAI_FILTER"
    CONTENT_POLICY = "CONTENT_POLICY_VIOLATION"
    SAFETY_FILTER = "SAFETY_FILTER"
    
    # Image Errors
    IMAGE_INVALID_FORMAT = "IMAGE_INVALID_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    ALL_IMAGES_BLACKLISTED = "ALL_IMAGES_BLACKLISTED"
    
    # Generation Errors
    VIDEO_GENERATION_FAILED = "VIDEO_GENERATION_FAILED"
    PROMPT_TOO_LONG = "PROMPT_TOO_LONG"
    OPENAI_PROMPT_FAILED = "OPENAI_PROMPT_FAILED"
    
    # System Errors
    STORAGE_FULL = "STORAGE_FULL"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    WORKER_CRASHED = "WORKER_CRASHED"
    
    # User Errors
    INVALID_CONFIG = "INVALID_CONFIG"
    NO_IMAGES = "NO_IMAGES"
    NO_DIALOGUE = "NO_DIALOGUE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    
    # Unknown
    UNKNOWN = "UNKNOWN_ERROR"


@dataclass
class AppConfig:
    """Application-wide configuration"""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    uploads_dir: Path = field(default=None)
    outputs_dir: Path = field(default=None)
    data_dir: Path = field(default=None)
    
    # Database
    database_url: str = "sqlite:///./data/jobs.db"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Workers
    max_workers: int = 3  # Concurrent video generation jobs
    worker_poll_interval: float = 1.0
    
    # File limits
    max_upload_size_mb: int = 50
    max_images_per_job: int = 100
    max_dialogue_lines: int = 50
    
    # Cleanup
    keep_uploads_days: int = 7
    keep_outputs_days: int = 30
    
    def __post_init__(self):
        if self.uploads_dir is None:
            self.uploads_dir = self.base_dir / "uploads"
        if self.outputs_dir is None:
            self.outputs_dir = self.base_dir / "outputs"
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        
        # Create directories
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoConfig:
    """Video generation configuration"""
    
    # Video settings
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT
    resolution: Resolution = Resolution.HD
    duration: Duration = Duration.LONG
    
    # Language
    language: str = "English"
    
    # Person generation
    person_generation: PersonGeneration = PersonGeneration.ALLOW_ADULT
    
    # Features
    use_interpolation: bool = True
    use_openai_prompt_tuning: bool = True
    use_frame_vision: bool = True
    timestamp_names: bool = True
    
    # Custom prompt (used when use_openai_prompt_tuning is False)
    custom_prompt: str = ""
    
    # Single image mode (use same image for start/end frames)
    single_image_mode: bool = False
    
    # Retry settings
    max_retries_per_clip: int = 5
    max_image_attempts: int = 15
    max_retries_submit: int = 15  # Try up to 15 times to cycle through all API keys
    poll_interval_sec: int = 10
    
    # Regeneration
    reuse_logged_params: bool = True
    
    # Image selection
    images_sort_key: str = "name"  # "name" or "date"
    images_sort_reverse: bool = False
    skip_first_pairs: int = 0
    skip_last_pairs: int = 0
    max_clips: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Resolution/duration constraints
        if self.resolution == Resolution.FULL_HD and self.duration != Duration.LONG:
            errors.append("1080p resolution requires 8 second duration")
        
        if self.use_interpolation and self.duration != Duration.LONG:
            errors.append("Interpolation requires 8 second duration")
        
        # EU compliance
        if self.person_generation == PersonGeneration.ALLOW_ALL:
            errors.append("'allow_all' not permitted for EU compliance")
        
        return errors


@dataclass
class APIKeysConfig:
    """API keys configuration - loads from environment by default"""
    
    # Google/Gemini keys (multiple for rotation)
    gemini_api_keys: List[str] = field(default_factory=get_gemini_keys_from_env)
    
    # OpenAI key (optional)
    openai_api_key: Optional[str] = field(default_factory=get_openai_key_from_env)
    
    # Key rotation settings
    rotate_keys_on_429: bool = True
    current_key_index: int = 0
    
    def get_current_gemini_key(self) -> Optional[str]:
        """Get current Gemini API key"""
        if not self.gemini_api_keys:
            return None
        
        if self.current_key_index >= len(self.gemini_api_keys):
            self.current_key_index = 0
        
        return self.gemini_api_keys[self.current_key_index]
    
    def rotate_key(self):
        """Rotate to next API key"""
        if self.gemini_api_keys:
            self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
    
    def get_status(self) -> dict:
        """Get status of API keys for admin dashboard"""
        return {
            "gemini_keys_count": len(self.gemini_api_keys),
            "gemini_keys_configured": len(self.gemini_api_keys) > 0,
            "gemini_current_index": self.current_key_index,
            "openai_configured": self.openai_api_key is not None,
        }
    
    def validate(self) -> List[str]:
        """Validate API keys configuration"""
        errors = []
        
        if not self.get_current_gemini_key():
            errors.append("No Gemini API keys configured. Add keys to .env file.")
        
        return errors


# Global API keys instance (loaded from environment)
api_keys_config = APIKeysConfig()


@dataclass
class DialogueLine:
    """Single dialogue line"""
    id: int
    text: str
    
    def validate(self) -> List[str]:
        errors = []
        if self.id < 1:
            errors.append(f"Invalid ID {self.id}: must be positive")
        if not self.text or not self.text.strip():
            errors.append(f"Line {self.id}: text cannot be empty")
        if len(self.text) > 2000:
            errors.append(f"Line {self.id}: text too long (max 2000 chars)")
        return errors


# Veo model configuration
VEO_MODEL = "veo-3.1-fast-generate-preview"
OPENAI_MODEL = "gpt-4.1"

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

# Max image file size (in bytes)
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


# Default prompts and instructions (from original script)
BASE_PROMPT = """
A realistic vertical video that maintains natural continuity.
The visual style, camera setup, and environment are based on the input frames.
The subject's facial expressions, gestures, and body language adapt naturally to communicate the spoken message.
"""

NO_TEXT_INSTRUCTION = (
    "CRITICAL: No text, subtitles, captions, words, letters, numbers, graphics, or overlays "
    "may appear on screen at any time. No burned-in text. No visual text elements whatsoever. "
    "Any text visible in the background (whiteboards, signs) must remain static environmental props."
)

AUDIO_TIMING_INSTRUCTION = (
    "The narrator must stop speaking exactly at 7.0 seconds. "
    "From 7.0 to 8.0 seconds: silence, no words, no breaths, no mouth sounds. "
    "Only natural room tone or ambient sound. "
    "Total duration: 8 seconds (7 seconds speech + 1 second quiet ambience)."
)

AUDIO_QUALITY_INSTRUCTION = (
    "Clean, natural audio recording with professional broadcast quality. "
    "Studio microphone sound, low noise floor, no clipping, no distortion, no metallic or robotic artifacts. "
    "Stable loudness over the whole line, gentle broadcast-style compression, minimal room reverb."
)

PRONUNCIATION_TEMPLATE = (
    "Pronunciation must be native-level {language}, with correct stress on every word. "
    "Avoid foreign accents and avoid moving the stress to the wrong syllable. "
    "Follow standard dictionary stress patterns for {language}. "
    "Use authentic {language} pronunciation with natural rhythm and intonation."
)


# Singleton app config
app_config = AppConfig()