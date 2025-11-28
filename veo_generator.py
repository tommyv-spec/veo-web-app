# -*- coding: utf-8 -*-
"""
Veo 3.1 Professional Generator
Implements "Enrichment -> Translation -> Routing" Workflow

Architecture:
1. ENRICHMENT: Expand brief user context into forensic details
2. TRANSLATION: Rewrite visual description with context as primary anchor
3. ROUTING: Map enriched details to specific JSON Blueprint slots

This ensures user context is the DOMINANT driver of all generation.
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
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Any, Callable
from datetime import datetime

def vlog(msg):
    """Log with immediate flush"""
    print(msg, flush=True)

# Google GenAI - Optional import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Video generation disabled.")

from config import (
    VideoConfig, APIKeysConfig, DialogueLine,
    VEO_MODEL, OPENAI_MODEL, SUPPORTED_IMAGE_FORMATS,
    BASE_PROMPT, NO_TEXT_INSTRUCTION, AUDIO_TIMING_INSTRUCTION,
    AUDIO_QUALITY_INSTRUCTION, PRONUNCIATION_TEMPLATE,
    ErrorCode, ClipStatus
)
from error_handler import ErrorHandler, VeoError, error_handler

# ===================== OPENAI CLIENT =====================

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


# ===================== STEP 1: ENRICHMENT ENGINE =====================

def process_user_context(
    user_context: str,
    language: str,
    openai_key: Optional[str] = None
) -> dict:
    """
    STEP 1: ENRICHMENT
    
    Takes brief user input (e.g., "he is very angry") and EXPANDS it into
    forensic details for each Expert Network (Visual, Audio, Motion).
    
    This is the KEY function - it transforms simple directions into
    specific, actionable instructions that Veo can follow.
    """
    if not user_context or not user_context.strip():
        return {}

    client = get_openai_client(openai_key)
    if client is None:
        # Fallback: map raw text to all fields
        return {
            "subject_action": user_context,
            "facial_expression": user_context,
            "voice_tone": user_context,
            "delivery_style": user_context,
            "atmosphere": user_context,
            "body_language": user_context,
            "background_action": "",
            "camera_motion": "",
        }

    try:
        system_msg = """You are a Director for Veo 3.1 (Google's AI Video Generator).

Your job is to EXPAND brief user directions into SPECIFIC, FORENSIC details.

The user might say something simple like "he is angry" or "nervous interview".
You must expand this into detailed instructions for EACH aspect of the video.

CRITICAL: Be EXTREMELY SPECIFIC. Don't say "angry expression" - say "furrowed brow, clenched jaw, flared nostrils, intense unblinking stare, flushed cheeks, visible neck tension".

OUTPUT JSON with these fields:
{
  "subject_action": "Precise physical movement (e.g., 'pacing back and forth with clenched fists', 'leaning forward aggressively')",
  "facial_expression": "Detailed facial description (e.g., 'furrowed brow, narrowed eyes, tight lips, jaw clenched, nostrils flared')",
  "voice_tone": "Voice texture and quality (e.g., 'sharp, clipped words, raised volume, slight growl in throat')",
  "delivery_style": "How dialogue is delivered (e.g., 'rapid-fire sentences, emphatic pauses, shouting key words')",
  "body_language": "Posture and gestures (e.g., 'rigid shoulders, pointing finger, forward-leaning stance')",
  "background_action": "What happens in environment (e.g., 'papers flying, door slamming')",
  "camera_motion": "Camera movement (e.g., 'handheld shake, quick zooms on face')",
  "atmosphere": "Lighting and mood (e.g., 'harsh shadows, high contrast, tense energy')"
}

If user didn't mention something, use "natural" or leave empty string."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"USER DIRECTION: {user_context}\nLANGUAGE: {language}\n\nExpand this into detailed video production instructions."}
            ],
            temperature=0.7,  # Higher temp for creative expansion
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        vlog(f"[ENRICHMENT] Expanded '{user_context}' into: {json.dumps(result, indent=2)[:500]}...")
        return result

    except Exception as e:
        vlog(f"[ENRICHMENT] Error: {e}")
        return {
            "subject_action": user_context,
            "facial_expression": user_context,
            "voice_tone": user_context,
            "atmosphere": user_context
        }


# ===================== STEP 2: FRAME ANALYSIS =====================

@lru_cache(maxsize=512)
def describe_frame(image_path: str, openai_key: Optional[str] = None) -> str:
    """Analyze frame for visual context"""
    client = get_openai_client(openai_key)
    if client is None:
        return ""
    
    path = Path(image_path)
    if not path.exists():
        return ""
    
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Describe this video frame in 50 words. Focus on: camera angle, subject appearance, lighting, background. Ignore any text."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ===================== STEP 3: VISUAL TRANSLATION =====================

def build_visual_description(
    base_prompt: str,
    frame_desc: str,
    enriched_context: dict,
    dialogue_line: str,
    language: str,
    openai_key: Optional[str] = None
) -> str:
    """
    STEP 2: TRANSLATION
    
    Rewrites the visual description to ensure USER CONTEXT is the
    DOMINANT anchor. The enriched context OVERRIDES the base prompt.
    """
    client = get_openai_client(openai_key)
    if client is None:
        # Fallback: combine base with context
        action = enriched_context.get('subject_action', '')
        expression = enriched_context.get('facial_expression', '')
        return f"{base_prompt}. {action}. {expression}."
    
    try:
        system_msg = """You are a Cinematographer writing a shot description for Veo 3.1.

Write a SINGLE, highly detailed paragraph (60-80 words) that describes the shot.

CRITICAL RULES:
1. The ENRICHED CONTEXT details MUST be the PRIMARY focus
2. The subject's ACTION and EXPRESSION from context override everything else
3. Include: Subject appearance, Primary Action, Facial Expression, Lighting
4. Write in flowing cinematic prose, not bullet points
5. Embed emotional/physical details directly: "his jaw clenched, veins visible on neck"

The enriched context is the DIRECTOR'S ORDER - it takes priority over the base prompt."""

        user_msg = f"""BASE PROMPT: {base_prompt}
FRAME DESCRIPTION: {frame_desc}
DIALOGUE: "{dialogue_line}"

=== ENRICHED CONTEXT (THIS IS THE PRIORITY) ===
ACTION: {enriched_context.get('subject_action', 'Natural movement')}
EXPRESSION: {enriched_context.get('facial_expression', 'Neutral')}
BODY LANGUAGE: {enriched_context.get('body_language', 'Natural')}
ATMOSPHERE: {enriched_context.get('atmosphere', 'Cinematic')}
CAMERA: {enriched_context.get('camera_motion', 'Smooth')}

Write the visual description with the ENRICHED CONTEXT as the dominant focus."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=250,
        )
        
        result = resp.choices[0].message.content.strip()
        vlog(f"[TRANSLATION] Visual description: {result[:200]}...")
        return result
        
    except Exception as e:
        vlog(f"[TRANSLATION] Error: {e}")
        return base_prompt


# ===================== STEP 4: VOICE PROFILE =====================

def generate_voice_profile(
    frame_description: str, 
    language: str,
    enriched_context: dict,
    openai_key: Optional[str] = None
) -> str:
    """
    Generate voice profile that reflects the enriched context.
    The voice MUST match the emotional direction.
    """
    client = get_openai_client(openai_key)
    
    # Get voice directions from enriched context
    voice_tone = enriched_context.get('voice_tone', '')
    delivery_style = enriched_context.get('delivery_style', '')
    
    if client is None:
        base = f"Native {language} speaker with clear pronunciation."
        if voice_tone:
            return f"{base} Voice quality: {voice_tone}. Delivery: {delivery_style}."
        return base
    
    try:
        system_msg = f"""You are a Voice Casting Director.

Define the voice characteristics for a {language} speaker based on the emotional direction.

CRITICAL: The voice MUST reflect the emotional context provided.
- If context says "angry" → voice should be sharp, loud, intense
- If context says "sad" → voice should be soft, slow, trembling
- If context says "nervous" → voice should be shaky, hesitant, fast

OUTPUT: 2-3 sentences describing the voice quality, tone, and delivery."""

        user_msg = f"""FRAME: {frame_description}
LANGUAGE: {language}

=== VOICE DIRECTION (MUST FOLLOW) ===
VOICE TONE: {voice_tone or 'Natural'}
DELIVERY STYLE: {delivery_style or 'Conversational'}

Describe the voice that matches this direction."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        
        result = resp.choices[0].message.content.strip()
        vlog(f"[VOICE PROFILE] {result[:150]}...")
        return result
        
    except Exception as e:
        vlog(f"[VOICE PROFILE] Error: {e}")
        return f"Native {language} speaker. {voice_tone}. {delivery_style}."


def get_default_voice_profile(language: str, user_context: str = "") -> str:
    """Default voice profile"""
    base = f"Natural {language} voice with clear diction and native accent."
    if user_context:
        return f"{base} Emotional direction: {user_context}."
    return base


# ===================== STEP 5: PROMPT ASSEMBLY (ROUTING) =====================

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
    """
    STEP 3: ROUTING
    
    Routes enriched details into specific Veo 3.1 JSON Blueprint slots.
    Each slot has a specific purpose - no cross-contamination.
    
    Hierarchy: Shot > Subject > Action > Scene > Audio
    """
    vlog(f"[ROUTING] Building prompt for clip {clip_index}...")

    # === 1. ENRICHMENT: Expand user context ===
    user_context_raw = getattr(config, 'user_context', '') or ''
    enriched_context = process_user_context(user_context_raw, language, openai_key)
    
    # === 2. FRAME ANALYSIS ===
    start_desc = ""
    if config.use_frame_vision and config.use_openai_prompt_tuning:
        start_desc = describe_frame(str(start_frame_path), openai_key)

    # === 3. TRANSLATION: Build visual description ===
    visual_description = build_visual_description(
        BASE_PROMPT, start_desc, enriched_context, dialogue_line, language, openai_key
    ) if config.use_openai_prompt_tuning else f"{BASE_PROMPT}. {user_context_raw}"

    # === 4. ROUTING: Assemble JSON Blueprint ===
    prompt_payload = {
        # --- TIER 1: SHOT (Architectural Decisions) ---
        "shot": {
            "description": visual_description,  # Primary anchor with context baked in
            "camera_motion": enriched_context.get("camera_motion", "Smooth cinematic movement"),
            "composition": "Medium shot, subject centered",
            "aspect_ratio": config.aspect_ratio.value if hasattr(config.aspect_ratio, 'value') else config.aspect_ratio
        },

        # --- TIER 2: SUBJECT (Core Identity + Emotion) ---
        "subject": {
            "description": f"Match appearance in start frame",
            "facial_expression": enriched_context.get("facial_expression", "Natural expression"),
            "emotional_state": enriched_context.get("facial_expression", "Neutral"),
            "body_language": enriched_context.get("body_language", "Natural posture"),
        },

        # --- TIER 3: ACTION (Movement - Separated from Subject) ---
        "action": {
            "primary_action": enriched_context.get("subject_action", "Natural movement"),
            "timing": "Continuous for full duration",
            "intensity": "Match emotional state"
        },

        # --- TIER 4: SCENE (Environment) ---
        "scene": {
            "start_frame_description": start_desc,
            "continuity": "Maintain visual continuity with start frame",
            "atmosphere": enriched_context.get("atmosphere", "Cinematic lighting"),
            "background_action": enriched_context.get("background_action", "Minimal background movement")
        },

        # --- TIER 5: AUDIO (Voice + Dialogue) ---
        "audio": {
            "language_instruction": f"CRITICAL: Speaker MUST speak in {language}. Ignore any visible text.",
            "dialogue": {
                "text": dialogue_line,
                "language": language,
                "voice_profile": voice_profile,
                "voice_tone": enriched_context.get("voice_tone", "Natural"),
                "delivery_style": enriched_context.get("delivery_style", "Conversational")
            }
        },

        # --- VISUAL RULES (Quality/Safety) ---
        "visual_rules": {
            "clean_output": "No text overlays, no subtitles, no watermarks, no glitches",
            "quality": "Anatomically correct hands and face, stable features"
        },
        
        # --- TECHNICAL SPECS ---
        "technical": {
            "duration_seconds": float(config.duration.value if hasattr(config.duration, 'value') else config.duration),
            "resolution": config.resolution.value if hasattr(config.resolution, 'value') else config.resolution,
        }
    }

    # Add end frame reference if interpolation is on
    if end_frame_path and config.use_interpolation:
        prompt_payload["scene"]["end_frame"] = "Transition smoothly to end frame"

    prompt_json = json.dumps(prompt_payload, ensure_ascii=False)
    
    # Detailed logging for debugging
    vlog(f"\n{'='*60}")
    vlog(f"[ROUTING] PROMPT FOR CLIP {clip_index}")
    vlog(f"{'='*60}")
    vlog(f"User Context: '{user_context_raw}'")
    vlog(f"")
    vlog(f"ENRICHED DETAILS:")
    vlog(f"  Action: {enriched_context.get('subject_action', 'none')}")
    vlog(f"  Expression: {enriched_context.get('facial_expression', 'none')}")
    vlog(f"  Voice Tone: {enriched_context.get('voice_tone', 'none')}")
    vlog(f"  Delivery: {enriched_context.get('delivery_style', 'none')}")
    vlog(f"  Body Language: {enriched_context.get('body_language', 'none')}")
    vlog(f"  Atmosphere: {enriched_context.get('atmosphere', 'none')}")
    vlog(f"")
    vlog(f"VISUAL DESCRIPTION:")
    vlog(f"  {visual_description[:300]}...")
    vlog(f"")
    vlog(f"FULL JSON PROMPT:")
    vlog(json.dumps(prompt_payload, indent=2, ensure_ascii=False)[:1500])
    vlog(f"{'='*60}\n")
    
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

ProgressCallback = Callable[[int, str, str, Optional[Dict]], None]


# ===================== MAIN GENERATOR CLASS =====================

class VeoGenerator:
    """
    Video generator with Enrichment -> Translation -> Routing workflow.
    
    Usage:
        generator = VeoGenerator(config, api_keys)
        generator.on_progress = my_callback_function
        result = generator.generate_single_clip(...)
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
            raise RuntimeError("google-genai package not installed.")
        
        api_key = self.api_keys.get_current_gemini_key()
        if not api_key:
            raise ValueError("No Gemini API key available")
        return genai.Client(api_key=api_key)
    
    def _rotate_key(self, block_current: bool = True):
        """Rotate to next API key"""
        old_index = self.api_keys.current_key_index
        old_key = self.api_keys.get_current_gemini_key()
        old_suffix = old_key[-8:] if old_key else "?"
        num_keys = len(self.api_keys.gemini_api_keys)
        
        self.api_keys.rotate_key(block_current=block_current)
        
        new_index = self.api_keys.current_key_index
        new_key = self.api_keys.get_current_gemini_key()
        
        if new_key:
            new_suffix = new_key[-8:]
            available = self.api_keys.get_available_key_count()
            blocked = len(self.api_keys.blocked_keys)
            vlog(f"[VeoGenerator] 🔄 KEY ROTATION: {old_index+1}(...{old_suffix}) -> {new_index+1}(...{new_suffix})")
            vlog(f"[VeoGenerator]    Available: {available}/{num_keys}, Blocked: {blocked}")
        else:
            vlog(f"[VeoGenerator] ⚠️ ALL KEYS BLOCKED!")
        
        self.client = None
    
    def _emit_progress(self, clip_index: int, status: str, message: str, details: Dict = None):
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
        """Generate a single video clip with retry logic."""
        
        vlog(f"[VeoGenerator] Generating clip {clip_index}: '{dialogue_line[:50]}...'")
        
        result = {
            "success": False,
            "output_path": None,
            "end_frame_used": None,
            "end_index": current_end_index,
            "error": None,
            "prompt_text": None,
        }
        
        if not GENAI_AVAILABLE:
            result["error"] = VeoError(
                code=ErrorCode.UNKNOWN,
                message="google-genai not installed",
                recoverable=False
            )
            return result
        
        # Initialize voice profile on first clip
        if clip_index == 0 or self.voice_profile is None:
            if self.config.use_openai_prompt_tuning:
                start_desc = describe_frame(str(start_frame), self.openai_key)
                user_context = getattr(self.config, 'user_context', '') or ''
                enriched = process_user_context(user_context, self.config.language, self.openai_key)
                self.voice_profile = generate_voice_profile(
                    start_desc, self.config.language, enriched, self.openai_key
                )
            else:
                self.voice_profile = get_default_voice_profile(self.config.language)
        
        failed_end_frames = []
        attempts = 0
        current_attempt_end_index = current_end_index
        
        while attempts < self.config.max_retries_per_clip:
            if self.cancelled:
                result["error"] = VeoError(code=ErrorCode.UNKNOWN, message="Cancelled", recoverable=False)
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
                    current_attempt_end_index, images_list, self.blacklist, self.config.max_image_attempts
                )
                
                if next_result is None:
                    if len(failed_end_frames) >= 2:
                        self.blacklist.add(start_frame)
                    return result
                
                actual_end_index, actual_end_frame = next_result
                current_attempt_end_index = actual_end_index
            
            # Build prompt using Enrichment -> Translation -> Routing
            try:
                prompt_text = build_prompt(
                    dialogue_line, start_frame, actual_end_frame, clip_index,
                    self.config.language, self.voice_profile, self.config, self.openai_key
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
                result["error"] = error
                return result
            
            self._emit_progress(
                clip_index, "generating",
                f"Generating: {start_frame.name} → {actual_end_frame.name if actual_end_frame else 'none'}",
                {"start": start_frame.name, "end": actual_end_frame.name if actual_end_frame else None}
            )
            
            # Submit to Veo API
            operation = None
            for submit_attempt in range(1, self.config.max_retries_submit + 1):
                try:
                    client = self._get_client()
                    
                    aspect = self.config.aspect_ratio.value if hasattr(self.config.aspect_ratio, 'value') else self.config.aspect_ratio
                    res = self.config.resolution.value if hasattr(self.config.resolution, 'value') else self.config.resolution
                    dur = self.config.duration.value if hasattr(self.config.duration, 'value') else self.config.duration
                    
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
                    
                    vlog(f"[VeoGenerator] Submitting to Veo (attempt {submit_attempt})...")
                    operation = client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                        image=start_image,
                        config=cfg,
                    )
                    vlog(f"[VeoGenerator] Submit OK!")
                    break
                    
                except Exception as e:
                    vlog(f"[VeoGenerator] Submit error: {str(e)[:200]}")
                    if is_rate_limit_error(e) and submit_attempt < self.config.max_retries_submit:
                        if self.api_keys.rotate_keys_on_429:
                            self._rotate_key(block_current=True)
                        
                        available = self.api_keys.get_available_key_count()
                        if available == 0:
                            error = VeoError(code=ErrorCode.RATE_LIMITED, message="All keys blocked", recoverable=False)
                            self._emit_error(error)
                            break
                        
                        blocked = len(self.api_keys.blocked_keys)
                        self._emit_progress(
                            clip_index, "rate_limited",
                            f"Key blocked, trying next... ({available} available, {blocked} blocked)",
                            {"available": available, "blocked": blocked}
                        )
                        continue
                    
                    failed_end_frames.append(actual_end_frame)
                    break
            
            if operation is None:
                continue
            
            # Poll for completion
            try:
                client = self._get_client()
                while not operation.done:
                    if self.cancelled:
                        return result
                    time.sleep(self.config.poll_interval_sec)
                    operation = client.operations.get(operation)
            except Exception as e:
                failed_end_frames.append(actual_end_frame)
                continue
            
            # Check for errors
            veo_error = error_handler.classify_veo_operation(
                operation, {"clip_index": clip_index}
            )
            
            if veo_error:
                self._emit_error(veo_error)
                if veo_error.code == ErrorCode.CELEBRITY_FILTER and actual_end_frame:
                    self.blacklist.add(actual_end_frame)
                failed_end_frames.append(actual_end_frame)
                continue
            
            # Success! Download video
            try:
                resp = getattr(operation, "response", None)
                vids = getattr(resp, "generated_videos", None) if resp else None
                
                if not vids:
                    failed_end_frames.append(actual_end_frame)
                    continue
                
                video = vids[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.timestamp_names else ""
                output_filename = generate_output_filename(dialogue_id, start_frame, actual_end_frame, timestamp)
                output_path = output_dir / output_filename
                
                client = self._get_client()
                client.files.download(file=video.video)
                video.video.save(str(output_path))
                
                result["success"] = True
                result["output_path"] = output_path
                result["end_frame_used"] = actual_end_frame
                result["end_index"] = actual_end_index
                
                self._emit_progress(clip_index, "completed", f"Saved: {output_filename}", {"output": output_filename})
                return result
                
            except Exception as e:
                failed_end_frames.append(actual_end_frame)
                continue
        
        # Exhausted retries
        if len(failed_end_frames) >= 2:
            self.blacklist.add(start_frame)
        
        result["error"] = VeoError(
            code=ErrorCode.VIDEO_GENERATION_FAILED,
            message=f"Failed after {self.config.max_retries_per_clip} attempts",
            recoverable=False
        )
        
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