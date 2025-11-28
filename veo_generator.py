# -*- coding: utf-8 -*-
"""
Core Video Generator for Veo Web App

Adapted from the original script with:
- Better error handling
- Progress callbacks
- Database integration
- Structured logging
"""

import os
import re
import time
import random
import hashlib
import mimetypes
import base64
import json
import sys
from functools import lru_cache

def vlog(msg):
    """Log with immediate flush"""
    print(msg, flush=True)
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Any, Callable
from datetime import datetime

# Google GenAI - Optional import (may not be installed)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Video generation disabled.")
    print("         Install with: pip install google-genai")
    print("         (If Rust errors occur, you need to install Rust first)")

from config import (
    VideoConfig, APIKeysConfig, DialogueLine,
    VEO_MODEL, OPENAI_MODEL, SUPPORTED_IMAGE_FORMATS,
    BASE_PROMPT, NO_TEXT_INSTRUCTION, AUDIO_TIMING_INSTRUCTION,
    AUDIO_QUALITY_INSTRUCTION, PRONUNCIATION_TEMPLATE,
    ErrorCode, ClipStatus
)
from error_handler import ErrorHandler, VeoError, error_handler


# ===================== OPENAI INTEGRATION =====================

_openai_client = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_openai_client(api_key: Optional[str] = None):
    """Get or create OpenAI client"""
    global _openai_client
    
    if OpenAI is None:
        return None
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    if _openai_client is None:
        try:
            _openai_client = OpenAI(api_key=api_key)
        except Exception:
            return None
    
    return _openai_client


# ===================== FRAME DESCRIPTION =====================

@lru_cache(maxsize=512)
def describe_frame(image_path: str, openai_key: Optional[str] = None) -> str:
    """Use OpenAI vision to describe a frame"""
    client = get_openai_client(openai_key)
    if client is None:
        return ""
    
    path = Path(image_path)
    if not path.exists():
        return ""
    
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        system_msg = (
            "You are a cinematographer analyzing a video frame. "
            "Describe what you see in 60–80 words for a video generation model. "
            "Focus on: camera angle, shot size, subject appearance (clothing, age, gender), pose, "
            "facial expression, lighting, and background elements. "
            "IMPORTANT: If there is text visible (signs, slides, screens), describe it as 'text' or 'slide with text' "
            "but do NOT mention what language the text is in. The language of visible text is irrelevant - "
            "only describe the visual composition."
        )
        
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this frame visually. Do not mention what language any visible text is in."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ===================== VOICE PROFILE =====================

def generate_voice_profile(
    frame_description: str, 
    language: str,
    openai_key: Optional[str] = None,
    user_context: str = ""
) -> str:
    """Generate voice profile from frame description"""
    vlog(f"[generate_voice_profile] Generating voice profile for language: {language}")
    
    client = get_openai_client(openai_key)
    if client is None or not frame_description:
        profile = get_default_voice_profile(language)
        vlog(f"[generate_voice_profile] Using default profile: {profile}")
        return profile
    
    try:
        system_msg = (
            f"You are a voice casting director. Define voice characteristics for a person who will speak {language}.\n\n"
            f"CRITICAL: The voice MUST be a native {language} speaker with authentic {language} accent and pronunciation.\n"
            f"IMPORTANT: Ignore any text visible in the scene (signs, slides, screens) - the spoken language is {language} regardless of what text appears in the image.\n"
            f"Focus on: gender, age range, voice quality, native {language} accent, pitch, tempo.\n\n"
            f"OUTPUT FORMAT:\nVOICE_PROFILE:\n<2-3 sentences describing the voice as a native {language} speaker>\n"
        )
        
        user_msg = f"FRAME: {frame_description}\nSPOKEN LANGUAGE (not text in image): {language}\nThe speaker must sound like a native {language} speaker. Any text visible in the scene does NOT affect the spoken language."
        
        # Add user context if provided
        if user_context:
            user_msg += f"\n\nADDITIONAL CONTEXT ABOUT THE SPEAKER: {user_context}"
        
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        
        raw = resp.choices[0].message.content.strip()
        if "VOICE_PROFILE:" in raw.upper():
            idx = raw.upper().find("VOICE_PROFILE:")
            profile = raw[idx + len("VOICE_PROFILE:"):].strip()
        else:
            profile = raw
        
        vlog(f"[generate_voice_profile] OpenAI returned: {profile}")
        return profile
        
    except Exception as e:
        vlog(f"[generate_voice_profile] Error: {e}")
        return get_default_voice_profile(language)


def get_default_voice_profile(language: str) -> str:
    """Default voice profile"""
    return (
        f"A natural, conversational {language} voice with clear diction, "
        f"native accent, precise word stress, and professional studio quality."
    )


# ===================== PERFORMANCE MODIFIERS =====================

@lru_cache(maxsize=512)
def generate_performance_modifiers(
    dialogue_line: str, 
    language: str,
    openai_key: Optional[str] = None
) -> str:
    """Analyze dialogue for performance direction"""
    client = get_openai_client(openai_key)
    if client is None:
        return ""
    
    try:
        system_msg = (
            "You are a voice director. Determine how a line should be performed.\n"
            "Describe: emotional tone, energy level, pacing, key words to stress.\n"
            f"Consider natural speech patterns for {language}.\n"
            "OUTPUT FORMAT:\nPERFORMANCE:\n<1-2 sentences>\n"
        )
        
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"LANGUAGE: {language}\nLINE: \"{dialogue_line}\""},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        
        raw = resp.choices[0].message.content.strip()
        if "PERFORMANCE:" in raw.upper():
            idx = raw.upper().find("PERFORMANCE:")
            return raw[idx + len("PERFORMANCE:"):].strip()
        return raw
        
    except Exception:
        return ""


# ===================== VISUAL PROMPT OPTIMIZATION =====================

def optimize_visual_prompt(
    base_prompt: str,
    dialogue_line: str,
    start_frame_desc: str,
    end_frame_desc: str,
    language: str,
    openai_key: Optional[str] = None,
    user_context: str = ""
) -> Tuple[str, str]:
    """Generate optimized visual prompt and gesture notes"""
    client = get_openai_client(openai_key)
    if client is None:
        return base_prompt, ""
    
    try:
        system_msg = (
            "Create video generation prompts.\n"
            "VISUAL_PROMPT: Based on frame descriptions (camera, lighting, setup).\n"
            "GESTURE_NOTES: Based on dialogue (expressions, gestures for the message).\n"
            f"Consider {language} gesture patterns.\n"
            "No text/subtitles on screen.\n"
            "OUTPUT:\nVISUAL_PROMPT:\n<description>\n\nGESTURE_NOTES:\n<notes>\n"
        )
        
        user_msg = (
            f"START FRAME: {start_frame_desc or '[none]'}\n"
            f"END FRAME: {end_frame_desc or '[none]'}\n"
            f"DIALOGUE ({language}): \"{dialogue_line}\"\n"
        )
        
        # Add user context if provided
        if user_context:
            user_msg += f"\nADDITIONAL CONTEXT: {user_context}\n"
        
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        
        raw = resp.choices[0].message.content.strip()
        upper = raw.upper()
        
        if "VISUAL_PROMPT:" in upper and "GESTURE_NOTES:" in upper:
            vp_idx = upper.find("VISUAL_PROMPT:")
            gn_idx = upper.find("GESTURE_NOTES:")
            visual = raw[vp_idx + len("VISUAL_PROMPT:"):gn_idx].strip()
            gesture = raw[gn_idx + len("GESTURE_NOTES:"):].strip()
            return visual, gesture
        
        return raw, ""
        
    except Exception:
        return base_prompt, ""


# ===================== PROMPT BUILDER =====================

def build_prompt(
    dialogue_line: str,
    start_frame_path: Path,
    end_frame_path: Optional[Path],
    clip_index: int,
    language: str,
    voice_profile: str,
    config: VideoConfig,
    openai_key: Optional[str] = None,
) -> str:
    """Build complete JSON prompt for Veo 3.1"""
    
    vlog(f"[build_prompt] Building prompt with language: {language}")
    
    # Get frame descriptions
    start_desc = ""
    end_desc = ""
    
    if config.use_frame_vision and config.use_openai_prompt_tuning:
        start_desc = describe_frame(str(start_frame_path), openai_key)
        if end_frame_path:
            end_desc = describe_frame(str(end_frame_path), openai_key)
    
    # Get visual prompt and gestures
    # Use custom_prompt if AI is disabled, otherwise use AI optimization
    custom_prompt = getattr(config, 'custom_prompt', '') or ''
    
    if custom_prompt and not config.use_openai_prompt_tuning:
        # User provided custom prompt
        visual_prompt = custom_prompt
        gesture_notes = ""
        performance = ""
    elif config.use_openai_prompt_tuning:
        # AI-generated prompt
        user_context = getattr(config, 'user_context', '') or ''
        visual_prompt, gesture_notes = optimize_visual_prompt(
            BASE_PROMPT, dialogue_line, start_desc, end_desc, language, openai_key, user_context
        )
        performance = generate_performance_modifiers(dialogue_line, language, openai_key)
    else:
        # Fallback to base prompt
        visual_prompt = BASE_PROMPT
        gesture_notes = ""
        performance = ""
    
    # Build JSON payload
    prompt_payload = {
        "audio_language": (
            f"CRITICAL AUDIO INSTRUCTION: The speaker MUST speak in {language} with a native {language} accent. "
            f"IGNORE any text visible in the image - the SPOKEN language must be {language} regardless of any on-screen text. "
            f"The voice must sound like a native {language} speaker, NOT influenced by any foreign text visible in the scene."
        ),
        "shot": {
            "description": visual_prompt,
        },
        "subject": {
            "description": "Match input frames.",
            "spoken_language": language,
            "accent": f"Native {language} accent - NOT influenced by any text visible in image",
        },
        "scene": {
            "start_frame_description": start_desc,
            "end_frame_description": end_desc,
            "continuity": "Maintain visual continuity with provided frames."
        },
        "action": {
            "gesture_notes": gesture_notes,
            "performance_direction": performance,
        },
        "audio": {
            "IMPORTANT": f"The speaker's voice must be in {language} only. Any text visible in the image (signs, slides, screens) should NOT affect the spoken language.",
            "language_requirement": f"The speaker MUST use {language} language with authentic native {language} pronunciation.",
            "dialogue": {
                "language": language,
                "accent": f"Native {language} speaker with authentic {language} pronunciation",
                "content": dialogue_line,
                "delivery_notes": performance,
                "voice_profile": voice_profile,
            },
            "constraints": {
                "timing": AUDIO_TIMING_INSTRUCTION,
                "quality": AUDIO_QUALITY_INSTRUCTION,
                "pronunciation": PRONUNCIATION_TEMPLATE.format(language=language),
            },
        },
        "visual_rules": {
            "clean_frame_policy": NO_TEXT_INSTRUCTION,
            "text_in_image_note": "Any text visible in the source image should be kept visually but must NOT influence the spoken audio language.",
        },
        "technical_specifications": {
            "duration_seconds": float(config.duration.value if hasattr(config.duration, 'value') else config.duration),
            "aspect_ratio": config.aspect_ratio.value if hasattr(config.aspect_ratio, 'value') else config.aspect_ratio,
            "resolution": config.resolution.value if hasattr(config.resolution, 'value') else config.resolution,
        },
        "meta": {
            "clip_index": clip_index,
            "language": language,
            "start_frame": start_frame_path.name,
            "end_frame": end_frame_path.name if end_frame_path else None,
        }
    }
    
    prompt_json = json.dumps(prompt_payload, ensure_ascii=False)
    vlog(f"[build_prompt] Voice profile being used: {voice_profile[:200] if voice_profile else 'None'}...")
    vlog(f"[build_prompt] Final prompt (first 500 chars): {prompt_json[:500]}...")
    return prompt_json


# ===================== HELPERS =====================

def list_images(images_dir: Path, config: VideoConfig) -> List[Path]:
    """List and sort images in directory"""
    files = [
        p for p in images_dir.iterdir() 
        if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    ]
    
    if config.images_sort_key == "date":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=config.images_sort_reverse)
    else:
        files.sort(key=lambda p: p.name.lower(), reverse=config.images_sort_reverse)
    
    return files


def get_mime_type(path: Path) -> str:
    """Get MIME type for image"""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error"""
    s = str(exception)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def is_celebrity_error(operation) -> bool:
    """Check if operation failed due to celebrity filter"""
    try:
        op_str = str(operation).lower()
        keywords = ["celebrity", "likenesses", "rai_media_filtered", "filtered_reasons"]
        
        if any(kw in op_str for kw in keywords):
            return True
        
        resp = getattr(operation, "response", None)
        if resp:
            if getattr(resp, "rai_media_filtered_count", 0) > 0:
                return True
            if getattr(resp, "rai_media_filtered_reasons", None):
                return True
    except Exception:
        pass
    
    return False


def get_next_clean_image(
    current_index: int,
    images_list: List[Path],
    blacklist: Set[Path],
    max_attempts: int = 10
) -> Optional[Tuple[int, Path]]:
    """Find next non-blacklisted image"""
    if not images_list:
        return None
    
    total = len(images_list)
    
    for offset in range(1, min(max_attempts + 1, total + 1)):
        new_index = (current_index + offset) % total
        candidate = images_list[new_index]
        
        if candidate not in blacklist:
            return (new_index, candidate)
    
    return None


def generate_output_filename(
    idx: int, 
    start_img: Path, 
    end_img: Optional[Path],
    timestamp: str = ""
) -> str:
    """Generate safe output filename"""
    def slugify(s: str) -> str:
        return re.sub(r"[^\w\-.]+", "_", s).strip("._")
    
    def short_stem(p: Path, n: int = 40) -> str:
        return slugify(p.stem)[:n]
    
    idx_str = str(idx)
    s1 = short_stem(start_img, 40)
    s2 = short_stem(end_img, 40) if end_img else ""
    
    base = f"{idx_str}_{s1}" + (f"_to_{s2}" if s2 else "")
    if timestamp:
        base = f"{base}_{timestamp}"
    
    base = slugify(base)
    if len(base) > 120:
        h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        base = f"{idx_str}_{s1[:40]}" + (f"_to_{s2[:40]}" if s2 else "") + f"_{h}"
    
    return f"{base}.mp4"


# ===================== CALLBACK TYPE =====================

# Progress callback signature: (clip_index, status, message, details)
ProgressCallback = Callable[[int, str, str, Optional[Dict]], None]


# ===================== MAIN GENERATOR CLASS =====================

class VeoGenerator:
    """
    Video generator class with progress callbacks and error handling.
    
    Usage:
        generator = VeoGenerator(config, api_keys)
        generator.on_progress = my_callback_function
        
        for clip in generator.generate_all(images_dir, dialogue_lines, output_dir):
            # clip contains result info
            pass
    """
    
    def __init__(
        self,
        config: VideoConfig,
        api_keys: APIKeysConfig,
        openai_key: Optional[str] = None
    ):
        self.config = config
        self.api_keys = api_keys
        self.openai_key = openai_key
        
        self.blacklist: Set[Path] = set()
        self.voice_profile: Optional[str] = None
        self.client: Optional[genai.Client] = None
        
        # Callbacks
        self.on_progress: Optional[ProgressCallback] = None
        self.on_error: Optional[Callable[[VeoError], None]] = None
        
        # State
        self.cancelled = False
        self.paused = False
    
    def _get_client(self) -> 'genai.Client':
        """Get or create Gemini client"""
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        api_key = self.api_keys.get_current_gemini_key()
        if not api_key:
            raise ValueError("No Gemini API key available")
        return genai.Client(api_key=api_key)
    
    def _rotate_key(self, block_current: bool = True):
        """Rotate to next API key, blocking current one if specified"""
        old_index = self.api_keys.current_key_index
        old_key = self.api_keys.get_current_gemini_key()
        old_suffix = old_key[-8:] if old_key else "?"
        num_keys = len(self.api_keys.gemini_api_keys)
        
        # Block current key for 12 hours if requested
        self.api_keys.rotate_key(block_current=block_current)
        
        new_index = self.api_keys.current_key_index
        new_key = self.api_keys.get_current_gemini_key()
        
        if new_key:
            new_suffix = new_key[-8:]
            available = self.api_keys.get_available_key_count()
            blocked = len(self.api_keys.blocked_keys)
            print(f"[VeoGenerator] 🔄 KEY ROTATION: {old_index+1}(...{old_suffix}) -> {new_index+1}(...{new_suffix})", flush=True)
            print(f"[VeoGenerator]    Available: {available}/{num_keys} keys, Blocked: {blocked}", flush=True)
        else:
            print(f"[VeoGenerator] ⚠️ ALL KEYS BLOCKED! No available keys.", flush=True)
        
        self.client = None  # Force new client
    
    def _emit_progress(
        self, 
        clip_index: int, 
        status: str, 
        message: str,
        details: Dict = None
    ):
        """Emit progress update"""
        if self.on_progress:
            self.on_progress(clip_index, status, message, details)
    
    def _emit_error(self, error: VeoError):
        """Emit error"""
        if self.on_error:
            self.on_error(error)
    
    def generate_single_clip(
        self,
        start_frame: Path,
        end_frame: Optional[Path],
        dialogue_line: str,
        dialogue_id: int,
        clip_index: int,
        output_dir: Path,
        images_list: List[Path],
        current_end_index: int,
    ) -> Dict[str, Any]:
        """
        Generate a single video clip with retry logic.
        
        Returns:
            Dict with keys: success, output_path, end_frame_used, end_index, error
        """
        vlog(f"[VeoGenerator] generate_single_clip called: clip_index={clip_index}, dialogue='{dialogue_line[:50]}...'")
        
        result = {
            "success": False,
            "output_path": None,
            "end_frame_used": None,
            "end_index": current_end_index,
            "error": None,
            "prompt_text": None,
        }
        
        # Check if genai is available
        if not GENAI_AVAILABLE:
            result["error"] = VeoError(
                code=ErrorCode.UNKNOWN,
                message="google-genai package not installed",
                user_message="Video generation SDK not installed. Install with: pip install google-genai",
                details={"hint": "You may need to install Rust first if compilation fails"},
                recoverable=False,
                suggestion="Install the google-genai package to enable video generation"
            )
            return result
        
        # Initialize voice profile on first clip
        if clip_index == 0 or self.voice_profile is None:
            if self.config.use_openai_prompt_tuning:
                start_desc = describe_frame(str(start_frame), self.openai_key)
                user_context = getattr(self.config, 'user_context', '') or ''
                self.voice_profile = generate_voice_profile(
                    start_desc, self.config.language, self.openai_key, user_context
                )
            else:
                self.voice_profile = get_default_voice_profile(self.config.language)
        
        failed_end_frames = []
        attempts = 0
        current_attempt_end_index = current_end_index
        
        while attempts < self.config.max_retries_per_clip:
            if self.cancelled:
                result["error"] = VeoError(
                    code=ErrorCode.UNKNOWN,
                    message="Cancelled by user",
                    user_message="Generation was cancelled.",
                    details={},
                    recoverable=False,
                    suggestion="Start a new job to continue."
                )
                return result
            
            while self.paused:
                time.sleep(1)
                if self.cancelled:
                    return result
            
            attempts += 1
            
            # Determine end frame
            if attempts == 1 and end_frame:
                actual_end_frame = end_frame
                actual_end_index = current_end_index
            else:
                next_result = get_next_clean_image(
                    current_attempt_end_index,
                    images_list,
                    self.blacklist,
                    self.config.max_image_attempts
                )
                
                if next_result is None:
                    self._emit_progress(
                        clip_index, "error",
                        "No clean images available",
                        {"failed_frames": [f.name for f in failed_end_frames]}
                    )
                    
                    if len(failed_end_frames) >= 2:
                        self.blacklist.add(start_frame)
                        result["error"] = VeoError(
                            code=ErrorCode.ALL_IMAGES_BLACKLISTED,
                            message=f"Start frame blacklisted after multiple failures: {start_frame.name}",
                            user_message="Multiple frames failed. The start image may have issues.",
                            details={"start_frame": start_frame.name},
                            recoverable=False,
                            suggestion="Try with different source images."
                        )
                    
                    return result
                
                actual_end_index, actual_end_frame = next_result
                current_attempt_end_index = actual_end_index
                
                if attempts > 1:
                    self._emit_progress(
                        clip_index, "retrying",
                        f"Retry #{attempts}: Using {actual_end_frame.name}",
                        {"attempt": attempts, "end_frame": actual_end_frame.name}
                    )
            
            # Build prompt
            try:
                prompt_text = build_prompt(
                    dialogue_line,
                    start_frame,
                    actual_end_frame,
                    clip_index,
                    self.config.language,
                    self.voice_profile,
                    self.config,
                    self.openai_key,
                )
                result["prompt_text"] = prompt_text
            except Exception as e:
                error = error_handler.classify_exception(e, {"stage": "prompt_building"})
                self._emit_error(error)
                result["error"] = error
                return result
            
            # Prepare images
            try:
                with open(start_frame, "rb") as f:
                    start_bytes = f.read()
                start_image = types.Image(
                    image_bytes=start_bytes,
                    mime_type=get_mime_type(start_frame)
                )
                
                end_image = None
                if actual_end_frame and self.config.use_interpolation:
                    with open(actual_end_frame, "rb") as f:
                        end_bytes = f.read()
                    end_image = types.Image(
                        image_bytes=end_bytes,
                        mime_type=get_mime_type(actual_end_frame)
                    )
            except Exception as e:
                error = error_handler.classify_exception(e, {"stage": "image_loading"})
                self._emit_error(error)
                result["error"] = error
                return result
            
            self._emit_progress(
                clip_index, "generating",
                f"Generating: {start_frame.name} → {actual_end_frame.name if actual_end_frame else 'none'}",
                {"start": start_frame.name, "end": actual_end_frame.name if actual_end_frame else None}
            )
            
            # Submit job with retries
            operation = None
            vlog(f"[VeoGenerator] Attempting to submit clip {clip_index} to Veo API...")
            for submit_attempt in range(1, self.config.max_retries_submit + 1):
                try:
                    client = self._get_client()
                    vlog(f"[VeoGenerator] Got client, preparing config...")
                    
                    # Handle both enum and string config values
                    aspect = self.config.aspect_ratio.value if hasattr(self.config.aspect_ratio, 'value') else self.config.aspect_ratio
                    res = self.config.resolution.value if hasattr(self.config.resolution, 'value') else self.config.resolution
                    dur = self.config.duration.value if hasattr(self.config.duration, 'value') else self.config.duration
                    
                    vlog(f"[VeoGenerator] Config: aspect={aspect}, res={res}, dur={dur}")
                    
                    cfg = types.GenerateVideosConfig(
                        aspect_ratio=aspect,
                        resolution=res,
                        duration_seconds=dur,
                    )
                    
                    if end_image is not None and hasattr(cfg, "last_frame"):
                        cfg.last_frame = end_image
                    
                    try:
                        pg = self.config.person_generation.value if hasattr(self.config.person_generation, 'value') else self.config.person_generation
                        cfg.person_generation = pg
                    except Exception:
                        pass
                    
                    vlog(f"[VeoGenerator] Submitting to Veo API (attempt {submit_attempt})...")
                    operation = client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                        image=start_image,
                        config=cfg,
                    )
                    vlog(f"[VeoGenerator] Submit successful! Operation: {operation.name if hasattr(operation, 'name') else operation}")
                    break
                    
                except Exception as e:
                    vlog(f"[VeoGenerator] Submit error (attempt {submit_attempt}): {type(e).__name__}: {str(e)[:500]}")
                    if is_rate_limit_error(e) and submit_attempt < self.config.max_retries_submit:
                        num_keys = len(self.api_keys.gemini_api_keys)
                        
                        if self.api_keys.rotate_keys_on_429:
                            # Block current key for 12 hours and rotate to next
                            self._rotate_key(block_current=True)
                        
                        # Check if all keys are blocked
                        available_keys = self.api_keys.get_available_key_count()
                        if available_keys == 0:
                            self._emit_progress(
                                clip_index, "all_keys_blocked",
                                f"❌ All {num_keys} API keys are blocked (quota exhausted). Please wait or add new keys.",
                                {"attempt": submit_attempt, "blocked_keys": num_keys}
                            )
                            error = VeoError(
                                code=ErrorCode.RATE_LIMITED,
                                message=f"All {num_keys} API keys are blocked for 12 hours (quota exhausted)",
                                recoverable=False
                            )
                            self._emit_error(error)
                            break
                        
                        blocked_count = len(self.api_keys.blocked_keys)
                        self._emit_progress(
                            clip_index, "rate_limited",
                            f"Key blocked, trying next... ({available_keys} available, {blocked_count} blocked)",
                            {"attempt": submit_attempt, "available": available_keys, "blocked": blocked_count}
                        )
                        
                        continue
                    
                    error = error_handler.classify_exception(e, {"stage": "submit"})
                    self._emit_error(error)
                    failed_end_frames.append(actual_end_frame)
                    break
            
            if operation is None:
                vlog(f"[VeoGenerator] No operation returned, continuing to next attempt...")
                continue
            
            # Poll for completion
            vlog(f"[VeoGenerator] Polling for operation completion...")
            try:
                client = self._get_client()
                poll_count = 0
                while not operation.done:
                    if self.cancelled:
                        return result
                    time.sleep(self.config.poll_interval_sec)
                    operation = client.operations.get(operation)
                    poll_count += 1
                    if poll_count % 10 == 0:
                        vlog(f"[VeoGenerator] Still polling... ({poll_count} polls)")
                vlog(f"[VeoGenerator] Operation completed after {poll_count} polls")
                print(f"[VeoGenerator] Polling complete, checking operation result...", flush=True)
            except Exception as e:
                vlog(f"[VeoGenerator] Polling error: {type(e).__name__}: {str(e)[:200]}")
                error = error_handler.classify_exception(e, {"stage": "polling"})
                self._emit_error(error)
                failed_end_frames.append(actual_end_frame)
                continue
            
            # Check for errors in response
            vlog(f"[VeoGenerator] Checking operation for errors...")
            veo_error = error_handler.classify_veo_operation(
                operation, 
                {"clip_index": clip_index, "end_frame": actual_end_frame.name if actual_end_frame else None}
            )
            
            if veo_error:
                vlog(f"[VeoGenerator] Veo operation error: {veo_error.code} - {veo_error.message}")
                vlog(f"[VeoGenerator] Operation response: {operation}")
                self._emit_error(veo_error)
                
                if veo_error.code == ErrorCode.CELEBRITY_FILTER:
                    if actual_end_frame:
                        self.blacklist.add(actual_end_frame)
                        self._emit_progress(
                            clip_index, "blacklisted",
                            f"Blacklisted: {actual_end_frame.name}",
                            {"image": actual_end_frame.name, "reason": "celebrity_filter"}
                        )
                
                failed_end_frames.append(actual_end_frame)
                
                if len(failed_end_frames) >= 2 and veo_error.code == ErrorCode.CELEBRITY_FILTER:
                    self.blacklist.add(start_frame)
                    result["error"] = veo_error
                    return result
                
                continue
            
            # Success! Download video
            try:
                vlog(f"[VeoGenerator] Success! Downloading video...")
                resp = getattr(operation, "response", None)
                vids = getattr(resp, "generated_videos", None) if resp else None
                
                if not vids or len(vids) == 0:
                    vlog(f"[VeoGenerator] No videos in response! resp={resp}")
                    failed_end_frames.append(actual_end_frame)
                    continue
                
                video = vids[0]
                vlog(f"[VeoGenerator] Got video, saving to disk...")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.timestamp_names else ""
                output_filename = generate_output_filename(
                    dialogue_id, start_frame, actual_end_frame, timestamp
                )
                output_path = output_dir / output_filename
                
                client = self._get_client()
                client.files.download(file=video.video)
                video.video.save(str(output_path))
                vlog(f"[VeoGenerator] Saved to: {output_path}")
                
                result["success"] = True
                result["output_path"] = output_path
                result["end_frame_used"] = actual_end_frame
                result["end_index"] = actual_end_index
                
                self._emit_progress(
                    clip_index, "completed",
                    f"Saved: {output_filename}",
                    {"output": output_filename}
                )
                
                return result
                
            except Exception as e:
                error = error_handler.classify_exception(e, {"stage": "download"})
                self._emit_error(error)
                failed_end_frames.append(actual_end_frame)
                continue
        
        # Exhausted retries
        vlog(f"[VeoGenerator] Exhausted retries for clip {clip_index}. Failed frames: {[f.name for f in failed_end_frames]}")
        if len(failed_end_frames) >= 2:
            self.blacklist.add(start_frame)
        
        result["error"] = VeoError(
            code=ErrorCode.VIDEO_GENERATION_FAILED,
            message=f"Failed after {self.config.max_retries_per_clip} attempts",
            user_message="Video generation failed after multiple retries.",
            details={"attempts": attempts, "failed_frames": [f.name for f in failed_end_frames]},
            recoverable=False,
            suggestion="Try with different images or check your API quota."
        )
        
        vlog(f"[VeoGenerator] Returning error result for clip {clip_index}")
        
        return result
    
    def cancel(self):
        """Cancel generation"""
        self.cancelled = True
    
    def pause(self):
        """Pause generation"""
        self.paused = True
    
    def resume(self):
        """Resume generation"""
        self.paused = False