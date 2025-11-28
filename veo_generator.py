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
    
    Takes brief user input (e.g., "he is very angry" or "news anchor reporting") 
    and EXPANDS it into forensic details for each Expert Network (Visual, Audio, Motion).
    
    This is the KEY function - it transforms simple directions into
    specific, actionable instructions that Veo can follow.
    
    Now also extracts SPEAKER ROLE for voice casting.
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
            "speaker_role": "",
        }

    try:
        system_msg = """You are a Director for Veo 3.1 (Google's AI Video Generator).

Your job is to EXPAND brief user directions into SPECIFIC, REALISTIC details.

The user might say something simple like "he is angry" or "nervous interview" or "fitness coach explaining".
You must expand this into detailed instructions for EACH aspect of the video.

CRITICAL: Be SPECIFIC and REALISTIC. Describe what you would actually SEE in real life.
- Don't say "intense cinematic gaze" - say "narrowed eyes, looking directly at camera"
- Don't say "dramatic tension" - say "shoulders raised, jaw tight"

SPEAKER ROLE EXTRACTION:
Look for any profession, role, or archetype in the user's description:
- "news anchor" → formal, authoritative delivery
- "fitness influencer" → energetic, motivational, upbeat
- "doctor explaining" → calm, reassuring, professional
- "teacher" → clear, patient, educational
- "salesperson" → enthusiastic, persuasive
- "meditation guide" → soft, calm, soothing
If no specific role mentioned, use "natural speaker"

OUTPUT JSON with these fields:
{
  "speaker_role": "The role/profession/archetype (e.g., 'news anchor', 'fitness coach', 'doctor', 'natural speaker')",
  "subject_action": "What the person physically does (e.g., 'leaning forward, pointing finger', 'sitting still, hands folded')",
  "facial_expression": "Realistic facial details (e.g., 'furrowed brow, tight lips, narrowed eyes')",
  "voice_tone": "How the voice sounds - MUST match the speaker_role (e.g., news anchor = 'clear, authoritative, measured', fitness coach = 'energetic, loud, motivational')",
  "delivery_style": "How they speak - MUST match the speaker_role (e.g., news anchor = 'formal pacing, clear enunciation', fitness coach = 'fast-paced, encouraging, punchy')",
  "body_language": "Posture and gestures (e.g., 'arms crossed, leaning back' or 'hands gesturing while talking')",
  "background_action": "What happens in background (e.g., 'nothing, static background')",
  "camera_motion": "Camera movement (e.g., 'static, no movement' or 'slight zoom')",
  "atmosphere": "Lighting (e.g., 'normal indoor lighting' or 'bright daylight')"
}

Keep it REALISTIC and NATURAL. No dramatic or cinematic language."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"USER DIRECTION: {user_context}\nLANGUAGE: {language}\n\nExpand this into detailed video production instructions. Be sure to identify the speaker_role."}
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
            "atmosphere": user_context,
            "speaker_role": "",
        }


# ===================== STEP 2: FRAME ANALYSIS =====================

def analyze_frame(image_path: str, openai_key: Optional[str] = None) -> dict:
    """
    COMPREHENSIVE FRAME ANALYSIS
    
    Extracts EVERYTHING needed from the image automatically:
    - Subject: age, gender, appearance, clothing
    - Role: detected profession/archetype based on visual cues
    - Action: what they're doing, interacting with
    - Objects: microphone, props, items they're holding/using
    - Setting: location, environment, context
    - Mood: apparent emotional state from expression/body language
    - Voice suggestion: appropriate voice based on all of the above
    
    This is the DEFAULT - works without any user input.
    User context can ADD to or OVERRIDE any of these.
    """
    client = get_openai_client(openai_key)
    
    # Default fallback
    default_result = {
        # Subject
        "subject_age": "adult",
        "subject_gender": "neutral",
        "subject_appearance": "person in frame",
        "subject_clothing": "",
        
        # Role & Action
        "apparent_role": "natural speaker",
        "current_action": "speaking to camera",
        "body_language": "neutral posture",
        "facial_expression": "neutral expression",
        
        # Objects & Interaction
        "objects_in_scene": "",
        "objects_interacting_with": "",
        "props": "",
        
        # Setting & Environment
        "setting_location": "indoor",
        "setting_type": "unknown",
        "background_description": "",
        "lighting": "natural lighting",
        "atmosphere": "neutral",
        
        # Voice Suggestions (auto-generated)
        "suggested_voice_tone": "clear, natural, conversational",
        "suggested_delivery": "measured pace, professional",
        "suggested_energy": "moderate",
        
        # Meta
        "visual_description": "",
        "confidence": "low"
    }
    
    if client is None:
        return default_result
    
    path = Path(image_path)
    if not path.exists():
        return default_result
    
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        system_msg = """You are an expert Scene Analyst for video production.

Analyze this frame and extract EVERYTHING needed for video generation.

=== WHAT TO ANALYZE ===

1. SUBJECT (the person):
   - Age range (young adult, middle-aged, older adult, etc.)
   - Gender
   - Appearance (hair, facial features, build)
   - Clothing (what they're wearing - this helps identify role)
   - Facial expression (specific: furrowed brow, slight smile, etc.)
   - Body language (posture, hand position, stance)

2. ROLE DETECTION (based on visual cues):
   Look at clothing + setting + props to determine role:
   - Suit + office = corporate professional / business executive
   - Suit + news desk = news anchor
   - Scrubs + medical setting = doctor / nurse
   - Workout clothes = fitness instructor / athlete
   - Casual + ring light = influencer / vlogger
   - Uniform = specific profession
   - Outdoor + microphone = reporter / street interview
   - Casual street clothes + being interviewed = man on street / interviewee
   
3. ACTION & OBJECTS:
   - What are they doing right now?
   - What objects are visible? (microphone, desk, equipment, props)
   - What are they interacting with or holding?
   - Any relevant props?

4. SETTING & ENVIRONMENT:
   - Where is this? (studio, office, gym, outdoor, home, street, etc.)
   - What's in the background?
   - Lighting quality and type
   - Overall atmosphere/mood of the scene

5. VOICE SUGGESTIONS:
   Based on ALL of the above, suggest:
   - Voice tone that would match this person and role
   - Delivery style appropriate for the context
   - Energy level (calm, moderate, high energy)

=== OUTPUT JSON ===
{
  "subject_age": "specific age range",
  "subject_gender": "male / female",
  "subject_appearance": "brief description of how they look",
  "subject_clothing": "what they're wearing",
  "facial_expression": "specific expression details",
  "body_language": "posture and stance",
  
  "apparent_role": "detected role/profession based on visual cues",
  "current_action": "what they appear to be doing",
  
  "objects_in_scene": "list of visible objects",
  "objects_interacting_with": "what they're holding or using",
  "props": "any notable props",
  
  "setting_location": "indoor / outdoor / studio",
  "setting_type": "specific type (office, gym, street, news studio, etc.)",
  "background_description": "what's behind them",
  "lighting": "lighting description",
  "atmosphere": "mood/feel of the scene",
  
  "suggested_voice_tone": "voice quality that matches this person and role",
  "suggested_delivery": "speaking style appropriate for context",
  "suggested_energy": "low / moderate / high",
  
  "visual_description": "50-word summary of the entire scene",
  "confidence": "high / medium / low - confidence in role detection"
}

Be SPECIFIC. Don't say "professional" - say "news anchor" or "corporate executive".
Don't say "nice clothes" - say "dark blue suit with red tie"."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this frame completely. Extract all details for video generation."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        
        # Log the analysis
        vlog(f"\n[FRAME ANALYSIS] === Auto-detected from image ===")
        vlog(f"  Subject: {result.get('subject_age', '?')} {result.get('subject_gender', '?')}")
        vlog(f"  Clothing: {result.get('subject_clothing', '?')}")
        vlog(f"  Role: {result.get('apparent_role', '?')} (confidence: {result.get('confidence', '?')})")
        vlog(f"  Action: {result.get('current_action', '?')}")
        vlog(f"  Setting: {result.get('setting_type', '?')} ({result.get('setting_location', '?')})")
        vlog(f"  Objects: {result.get('objects_in_scene', '?')}")
        vlog(f"  Expression: {result.get('facial_expression', '?')}")
        vlog(f"  Voice suggestion: {result.get('suggested_voice_tone', '?')}")
        
        return result
        
    except Exception as e:
        vlog(f"[FRAME ANALYSIS] Error: {e}")
        return default_result


# Legacy function for backward compatibility
@lru_cache(maxsize=512)
def describe_frame(image_path: str, openai_key: Optional[str] = None) -> str:
    """Legacy: Simple frame description. Use analyze_frame() for full analysis."""
    analysis = analyze_frame(image_path, openai_key)
    return analysis.get('visual_description', '')


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
    DOMINANT anchor. Focuses on REALISTIC, NATURAL output.
    """
    client = get_openai_client(openai_key)
    if client is None:
        # Fallback: combine base with context
        action = enriched_context.get('subject_action', '')
        expression = enriched_context.get('facial_expression', '')
        return f"{base_prompt}. {action}. {expression}."
    
    try:
        system_msg = """You are writing a shot description for Veo 3.1 video generation.

Write a SINGLE paragraph (50-70 words) describing what happens in the video.

CRITICAL RULES:
1. Write for REALISTIC, NATURAL output - NOT cinematic or dramatic
2. Describe exactly what you see: the person, their expression, their action
3. Use simple, direct language - no film terminology
4. Focus on: What the person looks like, what they're doing, their expression
5. The ENRICHED CONTEXT details MUST be included

DO NOT use words like: cinematic, dramatic, atmospheric, moody, artistic
DO use words like: natural, realistic, authentic, genuine, real

Example good output:
"A middle-aged man in a blue shirt sits at a desk. His brow is furrowed and jaw clenched, showing frustration. He speaks directly to camera with an intense expression, gesturing with his hands occasionally."

Example bad output:
"A dramatic close-up captures the raw intensity of a weathered businessman, shadows dancing across his chiseled features as emotion pours from his soul."
"""

        user_msg = f"""BASE DESCRIPTION: {base_prompt}
FRAME: {frame_desc}
DIALOGUE: "{dialogue_line}"

=== ENRICHED CONTEXT (MUST INCLUDE) ===
ACTION: {enriched_context.get('subject_action', 'Speaking naturally')}
EXPRESSION: {enriched_context.get('facial_expression', 'Natural')}
BODY LANGUAGE: {enriched_context.get('body_language', 'Natural')}

Write a realistic, natural description. No cinematic language."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        
        result = resp.choices[0].message.content.strip()
        vlog(f"[TRANSLATION] Visual description: {result[:200]}...")
        return result
        
    except Exception as e:
        vlog(f"[TRANSLATION] Error: {e}")
        return base_prompt


# ===================== STEP 4: VOICE PROFILE =====================

def generate_voice_profile(
    frame_analysis: dict,
    language: str,
    user_context_enriched: dict,
    openai_key: Optional[str] = None
) -> str:
    """
    Generate voice profile using:
    1. FRAME ANALYSIS (DEFAULT) - auto-detected from image: age, gender, apparent role
    2. USER CONTEXT (OVERRIDE) - when provided, takes priority over auto-detected
    3. Always PROFESSIONAL STUDIO quality
    
    Priority: User context > Frame analysis > Defaults
    """
    client = get_openai_client(openai_key)
    
    # === EXTRACT FROM FRAME ANALYSIS (defaults) ===
    auto_age = frame_analysis.get('subject_age', 'adult')
    auto_gender = frame_analysis.get('subject_gender', 'neutral')
    auto_role = frame_analysis.get('apparent_role', 'natural speaker')
    auto_voice_tone = frame_analysis.get('suggested_voice_tone', 'clear, natural')
    auto_delivery = frame_analysis.get('suggested_delivery', 'measured pace')
    visual_desc = frame_analysis.get('visual_description', '')
    detection_confidence = frame_analysis.get('confidence', 'low')
    
    # === EXTRACT FROM USER CONTEXT (overrides) ===
    user_role = user_context_enriched.get('speaker_role', '')
    user_voice_tone = user_context_enriched.get('voice_tone', '')
    user_delivery = user_context_enriched.get('delivery_style', '')
    
    # === MERGE: User context overrides auto-detected ===
    final_role = user_role if user_role else auto_role
    final_voice_tone = user_voice_tone if user_voice_tone else auto_voice_tone
    final_delivery = user_delivery if user_delivery else auto_delivery
    
    # Studio quality is ALWAYS required - non-negotiable
    studio_quality = (
        "Professional studio recording quality: crystal clear audio, no background noise, "
        "no room reverb, no echo, broadcast-grade microphone, perfect clarity."
    )
    
    # Log the merge decision
    vlog(f"[VOICE CASTING] Auto-detected role: {auto_role} (confidence: {detection_confidence})")
    if user_role:
        vlog(f"[VOICE CASTING] User override role: {user_role} (USING THIS)")
    vlog(f"[VOICE CASTING] Final role: {final_role}")
    
    if client is None:
        base = f"{auto_age} {auto_gender} {language} speaker. {final_role} style. {studio_quality}"
        return f"{base} Voice: {final_voice_tone}. Delivery: {final_delivery}."
    
    try:
        system_msg = f"""You are a Voice Casting Director for professional video production.

Your job: Cast the PERFECT voice for a {language} speaker.

=== ROLE-BASED VOICE PROFILES ===

NEWS ANCHOR / REPORTER:
- Authoritative, measured pace, clear enunciation
- Formal tone, confident, trustworthy, even-paced

FITNESS COACH / INFLUENCER:
- High energy, enthusiastic, loud, motivational
- Fast-paced, punchy, encouraging, upbeat

DOCTOR / MEDICAL PROFESSIONAL:
- Calm, reassuring, clear, precise
- Patient, measured, professional but warm

TEACHER / EDUCATOR:
- Clear, patient, engaging, educational
- Slightly varied pace, encouraging

SALESPERSON / MARKETER:
- Enthusiastic, persuasive, dynamic
- Friendly, approachable, confident

MEDITATION / WELLNESS GUIDE:
- Soft, calm, soothing, slow
- Gentle, peaceful, warm, nurturing

CORPORATE / BUSINESS EXECUTIVE:
- Professional, polished, clear, confident
- Measured pace, authoritative but approachable

VLOGGER / INFLUENCER (CASUAL):
- Natural, conversational, relaxed
- Friendly, authentic, personable

STREET INTERVIEW / MAN ON STREET:
- Natural, unrehearsed quality
- Conversational, authentic, varied pace

=== CRITICAL RULES ===
1. Voice MUST match the subject's age and gender
2. Voice MUST sound like the specified role
3. Scene location does NOT affect voice quality (ignore surroundings)
4. Voice MUST be professional studio quality - ALWAYS
5. Apply any emotional overlay specified

OUTPUT: 4-5 sentences describing the exact voice to use."""

        user_msg = f"""=== SUBJECT FROM FRAME ===
Age: {auto_age}
Gender: {auto_gender}
Visual: {visual_desc}

=== FINAL VOICE CASTING ===
Role: {final_role}
Voice Tone: {final_voice_tone}
Delivery Style: {final_delivery}
Language: {language}

Cast this voice. Remember:
- Match the age/gender exactly
- Sound like a {final_role} would sound
- {studio_quality}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        
        result = resp.choices[0].message.content.strip()
        
        # Ensure studio quality is mentioned
        if "studio" not in result.lower() and "professional" not in result.lower():
            result = f"{result} {studio_quality}"
        
        return result
        
    except Exception as e:
        vlog(f"[VOICE PROFILE] Error: {e}")
        return f"{auto_age} {auto_gender} {language} speaker in {final_role} style. {final_voice_tone}. {final_delivery}. {studio_quality}"


def get_default_voice_profile(language: str, user_context: str = "") -> str:
    """Default voice profile - always studio quality"""
    studio_quality = (
        "Professional studio recording quality: crystal clear audio, no background noise, "
        "no room reverb, broadcast-grade microphone."
    )
    base = f"Natural {language} voice with clear diction and native accent. {studio_quality}"
    if user_context:
        return f"{base} Context: {user_context}."
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
    frame_analysis: Optional[dict] = None,
    user_context_override: Optional[dict] = None,
) -> str:
    """
    PROMPT ASSEMBLY
    
    Builds the final prompt using:
    1. FRAME ANALYSIS (default) - auto-detected from image
    2. USER CONTEXT (override) - adds to or overrides frame analysis
    
    Priority: User context > Frame analysis > Defaults
    """
    vlog(f"[ROUTING] Building prompt for clip {clip_index}...")

    # === 1. GET DEFAULTS FROM FRAME ANALYSIS ===
    if frame_analysis is None:
        frame_analysis = {}
    
    if user_context_override is None:
        user_context_override = {}
    
    # === 2. MERGE: User context overrides frame analysis ===
    def get_value(key, frame_key=None, default=""):
        """Get value with priority: user_context > frame_analysis > default"""
        if frame_key is None:
            frame_key = key
        user_val = user_context_override.get(key, "")
        frame_val = frame_analysis.get(frame_key, "")
        return user_val if user_val else (frame_val if frame_val else default)
    
    # Subject details
    facial_expression = get_value("facial_expression", "facial_expression", "Natural expression")
    body_language = get_value("body_language", "body_language", "Natural posture")
    
    # Action
    current_action = get_value("subject_action", "current_action", "Speaking naturally to camera")
    
    # Scene/atmosphere
    atmosphere = get_value("atmosphere", "atmosphere", "Natural lighting")
    background = get_value("background_action", "background_description", "Static background")
    setting = frame_analysis.get("setting_type", "indoor setting")
    
    # Voice
    voice_tone = get_value("voice_tone", "suggested_voice_tone", "Natural speaking voice")
    delivery_style = get_value("delivery_style", "suggested_delivery", "Natural conversation")
    
    # Camera
    camera_motion = get_value("camera_motion", "camera_motion", "Static or minimal movement")
    
    # === 3. BUILD VISUAL DESCRIPTION ===
    user_context_raw = getattr(config, 'user_context', '') or ''
    
    if config.use_frame_vision and config.use_openai_prompt_tuning:
        start_desc = frame_analysis.get("visual_description", "")
    else:
        start_desc = ""
    
    visual_description = build_visual_description(
        BASE_PROMPT, start_desc, 
        {**frame_analysis, **user_context_override},  # Merged context
        dialogue_line, language, openai_key
    ) if config.use_openai_prompt_tuning else f"{BASE_PROMPT}. {user_context_raw}"

    # === 4. ROUTING: Assemble JSON Blueprint ===
    prompt_payload = {
        # --- TIER 1: SHOT ---
        "shot": {
            "description": visual_description,
            "camera_motion": camera_motion,
            "composition": "Medium shot, subject centered",
            "style": "Realistic, natural, authentic"
        },

        # --- TIER 2: SUBJECT ---
        "subject": {
            "description": "Match appearance in start frame exactly",
            "facial_expression": facial_expression,
            "body_language": body_language,
        },

        # --- TIER 3: ACTION ---
        "action": {
            "primary_action": current_action,
            "objects_interacting_with": frame_analysis.get("objects_interacting_with", ""),
            "movement": "Realistic, natural movements only"
        },

        # --- TIER 4: SCENE ---
        "scene": {
            "setting": setting,
            "start_frame_description": start_desc,
            "continuity": "Maintain exact visual continuity with start frame",
            "lighting": atmosphere,
            "background": background
        },

        # --- TIER 5: AUDIO ---
        "audio": {
            "language_instruction": f"CRITICAL: Speaker MUST speak in {language}. Ignore any visible text.",
            "recording_quality": "Professional studio recording: crystal clear, no background noise, no room reverb, broadcast-grade quality",
            "dialogue": {
                "text": dialogue_line,
                "language": language,
                "voice_profile": voice_profile,
                "voice_tone": voice_tone,
                "delivery_style": delivery_style
            }
        },

        # --- VISUAL RULES ---
        "visual_rules": {
            "style": "Photorealistic, natural, NOT cinematic or dramatic",
            "quality": "No text overlays, no subtitles, no glitches, anatomically correct"
        },
        
        # --- TECHNICAL ---
        "technical": {
            "duration_seconds": float(config.duration.value if hasattr(config.duration, 'value') else config.duration),
            "resolution": config.resolution.value if hasattr(config.resolution, 'value') else config.resolution,
        }
    }

    # Add end frame reference if interpolation is on
    if end_frame_path and config.use_interpolation:
        prompt_payload["scene"]["end_frame"] = "Transition smoothly to end frame"

    # Build the final prompt with voice instructions prominently placed
    # Veo needs voice/audio cues in plain text, not buried in JSON
    
    # Studio quality is ALWAYS required - this is non-negotiable
    studio_quality = "AUDIO QUALITY: Professional studio recording - crystal clear, no background noise, no room echo, broadcast-grade microphone quality."
    
    # Create voice instruction block
    voice_parts = [studio_quality]
    if voice_tone:
        voice_parts.append(f"VOICE TONE: {voice_tone}")
    if delivery_style:
        voice_parts.append(f"DELIVERY: {delivery_style}")
    
    voice_instruction = "\n".join(voice_parts)
    if voice_tone or delivery_style:
        voice_instruction += "\nThe speaker's voice MUST match this emotional direction throughout."
    
    # Combine JSON structure with explicit voice instruction
    prompt_json = json.dumps(prompt_payload, ensure_ascii=False)
    
    # Prepend voice instruction for Veo to catch it (voice cues must be visible at top level)
    final_prompt = f"{voice_instruction}\n\n{prompt_json}"
    
    # Detailed logging for debugging
    vlog(f"\n{'='*60}")
    vlog(f"[ROUTING] PROMPT FOR CLIP {clip_index}")
    vlog(f"{'='*60}")
    vlog(f"")
    vlog(f"=== FROM FRAME ANALYSIS (auto-detected) ===")
    vlog(f"  Role: {frame_analysis.get('apparent_role', 'unknown')}")
    vlog(f"  Setting: {frame_analysis.get('setting_type', 'unknown')}")
    vlog(f"  Action: {frame_analysis.get('current_action', 'unknown')}")
    vlog(f"  Expression: {frame_analysis.get('facial_expression', 'unknown')}")
    vlog(f"  Objects: {frame_analysis.get('objects_interacting_with', 'none')}")
    vlog(f"")
    if user_context_raw:
        vlog(f"=== FROM USER CONTEXT (override) ===")
        vlog(f"  Raw: '{user_context_raw}'")
        vlog(f"  Role override: {user_context_override.get('speaker_role', '(none)')}")
        vlog(f"  Action override: {user_context_override.get('subject_action', '(none)')}")
        vlog(f"  Voice override: {user_context_override.get('voice_tone', '(none)')}")
        vlog(f"")
    vlog(f"=== FINAL VALUES (merged) ===")
    vlog(f"  Action: {current_action}")
    vlog(f"  Expression: {facial_expression}")
    vlog(f"  Body Language: {body_language}")
    vlog(f"  Voice Tone: {voice_tone}")
    vlog(f"  Delivery: {delivery_style}")
    vlog(f"")
    vlog(f"VOICE INSTRUCTION:")
    vlog(f"  {voice_instruction}")
    vlog(f"")
    vlog(f"VISUAL DESCRIPTION:")
    vlog(f"  {visual_description[:300]}...")
    vlog(f"")
    vlog(f"FULL JSON PROMPT:")
    vlog(json.dumps(prompt_payload, indent=2, ensure_ascii=False)[:1500])
    vlog(f"{'='*60}\n")
    
    return final_prompt


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
        self.voice_profile_id: Optional[str] = None  # Short ID for logging
        self.frame_analysis: Optional[dict] = None  # Auto-detected from frame (age, gender, role)
        self.enriched_context: Optional[dict] = None  # From user context (override)
        self.client: Optional[genai.Client] = None
        
        # Callbacks
        self.on_progress: Optional[ProgressCallback] = None
        self.on_error: Optional[Callable[[VeoError], None]] = None
        
        # State
        self.cancelled = False
        self.paused = False
    
    def initialize_voice_profile(self, reference_frame: Path) -> str:
        """
        Initialize voice profile ONCE per job. Must be called before generating clips.
        Returns a voice profile ID for tracking.
        """
        import hashlib
        
        user_context = getattr(self.config, 'user_context', '') or ''
        
        # === STEP 1: ANALYZE FRAME (auto-detect age, gender, role) ===
        # This is the DEFAULT - works even without user context
        if self.config.use_openai_prompt_tuning:
            self.frame_analysis = analyze_frame(str(reference_frame), self.openai_key)
        else:
            self.frame_analysis = {
                "subject_age": "adult",
                "subject_gender": "neutral", 
                "apparent_role": "natural speaker",
                "suggested_voice_tone": "natural, conversational",
                "suggested_delivery": "clear, measured",
                "visual_description": "",
                "confidence": "low"
            }
        
        # === STEP 2: ENRICH USER CONTEXT (if provided - this is the OVERRIDE) ===
        if user_context:
            self.enriched_context = process_user_context(
                user_context, self.config.language, self.openai_key
            )
        else:
            self.enriched_context = {}
        
        # === STEP 3: GENERATE VOICE PROFILE (frame analysis + user override) ===
        if self.config.use_openai_prompt_tuning:
            self.voice_profile = generate_voice_profile(
                self.frame_analysis,  # Auto-detected defaults
                self.config.language,
                self.enriched_context,  # User overrides (if any)
                self.openai_key
            )
        else:
            self.voice_profile = get_default_voice_profile(self.config.language, user_context)
        
        # Create a short ID based on the profile content
        profile_hash = hashlib.md5(self.voice_profile.encode()).hexdigest()[:8]
        self.voice_profile_id = f"VP-{profile_hash.upper()}"
        
        # === LOG EVERYTHING ===
        auto_role = self.frame_analysis.get('apparent_role', 'unknown')
        user_role = self.enriched_context.get('speaker_role', '')
        final_role = user_role if user_role else auto_role
        
        vlog(f"\n{'='*60}")
        vlog(f"[VOICE PROFILE INITIALIZED]")
        vlog(f"{'='*60}")
        vlog(f"Voice ID: {self.voice_profile_id}")
        vlog(f"")
        vlog(f"=== FRAME ANALYSIS (auto-detected) ===")
        vlog(f"  Subject: {self.frame_analysis.get('subject_age', '?')} {self.frame_analysis.get('subject_gender', '?')}")
        vlog(f"  Auto Role: {auto_role} (confidence: {self.frame_analysis.get('confidence', '?')})")
        vlog(f"  Auto Voice: {self.frame_analysis.get('suggested_voice_tone', '?')}")
        vlog(f"  Auto Delivery: {self.frame_analysis.get('suggested_delivery', '?')}")
        vlog(f"")
        if user_context:
            vlog(f"=== USER CONTEXT (override) ===")
            vlog(f"  Raw: '{user_context}'")
            vlog(f"  User Role: {user_role or '(not specified)'}")
            vlog(f"  User Voice: {self.enriched_context.get('voice_tone', '(not specified)')}")
            vlog(f"  User Delivery: {self.enriched_context.get('delivery_style', '(not specified)')}")
            vlog(f"")
        else:
            vlog(f"=== USER CONTEXT: None (using auto-detected) ===")
            vlog(f"")
        vlog(f"=== FINAL VOICE CASTING ===")
        vlog(f"  Final Role: {final_role}")
        vlog(f"")
        vlog(f"Generated Profile:")
        vlog(f"  {self.voice_profile}")
        vlog(f"{'='*60}\n")
        
        return self.voice_profile_id
    
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
        
        # Initialize voice profile if not already done
        # This should be called by the worker before generating clips, but fallback here
        if self.voice_profile is None:
            vlog(f"[WARNING] Voice profile not pre-initialized, initializing now for clip {clip_index}")
            self.initialize_voice_profile(start_frame)
        
        # Log voice ID for this clip
        vlog(f"[Clip {clip_index}] Using Voice ID: {self.voice_profile_id}")
        
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
            
            # Build prompt using frame analysis + user context override
            try:
                prompt_text = build_prompt(
                    dialogue_line, start_frame, actual_end_frame, clip_index,
                    self.config.language, self.voice_profile, self.config, self.openai_key,
                    frame_analysis=self.frame_analysis,  # Auto-detected defaults
                    user_context_override=self.enriched_context  # User overrides (if any)
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
                    # Log which key we're about to use
                    key_index = self.api_keys.current_key_index
                    current_key = self.api_keys.get_current_gemini_key()
                    key_suffix = current_key[-8:] if current_key else "NONE"
                    available = self.api_keys.get_available_key_count()
                    blocked = len(self.api_keys.blocked_keys)
                    vlog(f"[VeoGenerator] Using key {key_index + 1} (...{key_suffix}) - {available} available, {blocked} blocked")
                    
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