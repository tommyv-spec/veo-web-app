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
    found_vars = []
    
    # Method 1: Scan ALL environment variables for any GEMINI/GOOGLE key patterns
    for var_name, var_value in os.environ.items():
        if var_name.startswith(("GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY")):
            if var_value and var_value.strip() and not var_value.startswith("your-"):
                key = var_value.strip()
                if key not in keys:
                    keys.append(key)
                    found_vars.append(var_name)
    
    # Method 2: Also check single key formats (in case they weren't caught)
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        key = os.environ.get(var)
        if key and key.strip() and not key.startswith("your-") and key.strip() not in keys:
            keys.append(key.strip())
            if var not in found_vars:
                found_vars.append(var)
    
    print(f"[Config] Loaded {len(keys)} Gemini API keys from environment", flush=True)
    if found_vars:
        # Sort and log found variable names (without revealing the keys)
        found_vars.sort()
        print(f"[Config] Found key variables: {found_vars}", flush=True)
    
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
    WAITING_APPROVAL = "waiting_approval"  # Continue mode: waiting for previous clip approval


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
    max_workers: int = 1  # Concurrent video generation jobs (reduced for 512MB RAM limit)
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
    
    # User context (additional info for AI prompt generation)
    user_context: str = ""
    
    # Single image mode (use same image for start/end frames)
    single_image_mode: bool = False
    
    # Retry settings
    max_retries_per_clip: int = 5
    max_image_attempts: int = 15
    max_retries_submit: int = 15  # Try up to 15 times to cycle through all API keys
    poll_interval_sec: int = 10
    
    # Parallel clip generation
    parallel_clips: int = 3  # Number of clips to generate simultaneously
    
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
    
    # Key blocking - maps key index to block timestamp
    blocked_keys: dict = field(default_factory=dict)
    block_duration_hours: int = 12
    
    # Persistence file path
    _blocked_keys_file: Path = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load persisted blocked keys on init"""
        if self._blocked_keys_file is None:
            self._blocked_keys_file = Path(__file__).parent / "data" / "blocked_keys.json"
        self._load_blocked_keys()
    
    def _load_blocked_keys(self):
        """Load blocked keys from disk"""
        try:
            if self._blocked_keys_file and self._blocked_keys_file.exists():
                import json
                from datetime import datetime
                with open(self._blocked_keys_file, 'r') as f:
                    data = json.load(f)
                # Convert ISO strings back to datetime
                self.blocked_keys = {
                    int(k): datetime.fromisoformat(v) 
                    for k, v in data.items()
                }
                print(f"[APIKeys] Loaded {len(self.blocked_keys)} blocked keys from disk", flush=True)
        except Exception as e:
            print(f"[APIKeys] Could not load blocked keys: {e}", flush=True)
            self.blocked_keys = {}
    
    def _save_blocked_keys(self):
        """Save blocked keys to disk"""
        try:
            if self._blocked_keys_file:
                import json
                self._blocked_keys_file.parent.mkdir(parents=True, exist_ok=True)
                # Convert datetime to ISO strings
                data = {
                    str(k): v.isoformat() 
                    for k, v in self.blocked_keys.items()
                }
                with open(self._blocked_keys_file, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"[APIKeys] Could not save blocked keys: {e}", flush=True)
    
    def is_key_blocked(self, key_index: int) -> bool:
        """Check if a key is currently blocked"""
        from datetime import datetime, timedelta
        
        if key_index not in self.blocked_keys:
            return False
        
        block_time = self.blocked_keys[key_index]
        unblock_time = block_time + timedelta(hours=self.block_duration_hours)
        
        if datetime.now() >= unblock_time:
            # Block expired, remove it
            del self.blocked_keys[key_index]
            print(f"[APIKeys] ✅ Key {key_index + 1} unblocked (12h expired)", flush=True)
            self._save_blocked_keys()  # Persist to disk
            return False
        
        return True
    
    def block_key(self, key_index: int):
        """Block a key for 12 hours after hitting 429"""
        from datetime import datetime
        
        self.blocked_keys[key_index] = datetime.now()
        key = self.gemini_api_keys[key_index] if key_index < len(self.gemini_api_keys) else "?"
        key_suffix = key[-8:] if key else "?"
        unblock_time = datetime.now().hour + self.block_duration_hours
        print(f"[APIKeys] 🚫 Key {key_index + 1} (...{key_suffix}) BLOCKED for {self.block_duration_hours}h (429 quota exhausted)", flush=True)
        self._save_blocked_keys()  # Persist to disk
    
    def get_available_key_count(self) -> int:
        """Count how many keys are currently available (not blocked)"""
        available = 0
        for i in range(len(self.gemini_api_keys)):
            if not self.is_key_blocked(i):
                available += 1
        return available
    
    def get_current_gemini_key(self) -> Optional[str]:
        """Get current Gemini API key (skips blocked keys)"""
        if not self.gemini_api_keys:
            return None
        
        if self.current_key_index >= len(self.gemini_api_keys):
            self.current_key_index = 0
        
        # If current key is blocked, find next available
        if self.is_key_blocked(self.current_key_index):
            self._find_next_available_key()
        
        # Check if we have any available keys
        if self.get_available_key_count() == 0:
            print(f"[APIKeys] ⚠️ ALL {len(self.gemini_api_keys)} keys are blocked!", flush=True)
            return None
        
        return self.gemini_api_keys[self.current_key_index]
    
    def _find_next_available_key(self):
        """Find the next non-blocked key"""
        start_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.gemini_api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
            if not self.is_key_blocked(self.current_key_index):
                return
            attempts += 1
        
        # All keys blocked, reset to original
        self.current_key_index = start_index
    
    def rotate_key(self, block_current: bool = False):
        """Rotate to next API key, optionally blocking the current one"""
        if not self.gemini_api_keys:
            return
        
        if block_current:
            self.block_key(self.current_key_index)
        
        self._find_next_available_key()
    
    def get_status(self) -> dict:
        """Get status of API keys for admin dashboard"""
        blocked_info = []
        from datetime import datetime, timedelta
        
        for idx, block_time in self.blocked_keys.items():
            unblock_time = block_time + timedelta(hours=self.block_duration_hours)
            remaining = unblock_time - datetime.now()
            remaining_hours = max(0, remaining.total_seconds() / 3600)
            key = self.gemini_api_keys[idx] if idx < len(self.gemini_api_keys) else "?"
            blocked_info.append({
                "index": idx + 1,
                "key_suffix": key[-8:] if key else "?",
                "blocked_at": block_time.isoformat(),
                "unblocks_at": unblock_time.isoformat(),
                "remaining_hours": round(remaining_hours, 1)
            })
        
        return {
            "gemini_keys_count": len(self.gemini_api_keys),
            "gemini_keys_configured": len(self.gemini_api_keys) > 0,
            "gemini_current_index": self.current_key_index,
            "gemini_available_keys": self.get_available_key_count(),
            "gemini_blocked_keys": len(self.blocked_keys),
            "blocked_details": blocked_info,
            "openai_configured": self.openai_api_key is not None,
        }
    
    def validate(self) -> List[str]:
        """Validate API keys configuration"""
        errors = []
        
        if not self.get_current_gemini_key():
            if self.get_available_key_count() == 0 and len(self.gemini_api_keys) > 0:
                errors.append(f"All {len(self.gemini_api_keys)} Gemini API keys are blocked (quota exhausted). Wait for unblock or add new keys.")
            else:
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