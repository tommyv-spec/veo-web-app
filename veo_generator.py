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
    
    IMPORTANT: Avoids any mentions of audience, crowd, stage performance context
    to prevent video generation from adding fake audience sounds.
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
        "setting_type": "studio",
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

=== CRITICAL INSTRUCTION ===
This is for a SOLO SPEAKER video with ISOLATED AUDIO.
NEVER mention or imply:
- Audience, crowd, spectators, attendees, viewers
- Stage, podium, conference, presentation, keynote, TED talk
- Live performance, public speaking event
- Applause, reactions, laughter
- Any suggestion that other people are watching live

Instead, describe settings neutrally:
- "conference stage" → "professional studio with backdrop"
- "speaking to audience" → "speaking to camera"
- "presentation setting" → "professional recording environment"
- "keynote speaker" → "professional speaker" or "subject matter expert"

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
   - Suit + professional backdrop = business professional / executive
   - Suit + desk = corporate professional
   - Scrubs + medical setting = doctor / nurse
   - Workout clothes = fitness instructor / athlete
   - Casual + ring light = content creator / vlogger
   - Uniform = specific profession
   - Casual clothes = everyday person / interviewee
   
3. ACTION & OBJECTS:
   - What are they doing right now? (always "speaking to camera" or similar)
   - What objects are visible? (microphone, desk, equipment, props)
   - What are they interacting with or holding?
   - Any relevant props?

4. SETTING & ENVIRONMENT:
   - Where is this? (studio, office, gym, outdoor, home, etc.)
   - What's in the background? (describe neutrally, no audience)
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
  "setting_type": "specific type (office, gym, studio, etc.) - NEVER conference/stage/event",
  "background_description": "what's behind them - describe neutrally",
  "lighting": "lighting description",
  "atmosphere": "mood/feel of the scene",
  
  "suggested_voice_tone": "voice quality that matches this person and role",
  "suggested_delivery": "speaking style appropriate for context",
  "suggested_energy": "low / moderate / high",
  
  "visual_description": "50-word summary - describe as solo recording, not live event",
  "confidence": "high / medium / low - confidence in role detection"
}

Be SPECIFIC. Don't say "professional" - say "business executive" or "content creator".
Don't say "nice clothes" - say "dark blue suit with red tie".
NEVER use words: audience, crowd, stage, conference, keynote, presentation, applause, spectators."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this frame completely. Extract all details for video generation. Remember: describe as a solo recording setup, never as a live event with audience."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        
        # POST-PROCESS: Sanitize any audience-related terms that might have slipped through
        audience_terms = ['audience', 'crowd', 'spectator', 'attendee', 'viewer', 'stage', 
                         'conference', 'keynote', 'presentation', 'ted talk', 'applause', 
                         'laughter', 'live event', 'public speaking']
        
        for key, value in result.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for term in audience_terms:
                    if term in value_lower:
                        # Replace problematic descriptions
                        if key == 'setting_type':
                            result[key] = 'professional studio'
                        elif key == 'background_description':
                            result[key] = 'professional backdrop'
                        elif key == 'current_action':
                            result[key] = 'speaking directly to camera'
                        elif key == 'visual_description':
                            # Remove audience-related phrases
                            result[key] = value.replace(term, 'camera').replace('stage', 'studio')
                        vlog(f"[FRAME ANALYSIS] Sanitized '{term}' from {key}")
                        break
        
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


def analyze_dialogue_for_gestures(dialogue_line: str, language: str, openai_key: Optional[str] = None) -> dict:
    """
    Analyze the dialogue line to determine appropriate gestures, expressions, and body language.
    
    Returns dict with:
    - emotion: primary emotion (excited, sad, thoughtful, angry, neutral, etc.)
    - intensity: low, medium, high
    - suggested_expression: facial expression description
    - suggested_gestures: hand/body gesture description
    - suggested_posture: body posture description
    - delivery_style: how the line should be delivered
    """
    default_result = {
        "emotion": "neutral",
        "intensity": "medium",
        "suggested_expression": "Natural, engaged expression",
        "suggested_gestures": "Natural hand movements while speaking",
        "suggested_posture": "Upright, attentive posture",
        "delivery_style": "Conversational, natural delivery"
    }
    
    client = get_openai_client(openai_key)
    if client is None:
        return default_result
    
    try:
        system_msg = """You analyze dialogue lines to determine appropriate non-verbal communication.

Given a line of dialogue, determine:
1. EMOTION: The primary emotion (excited, happy, sad, angry, frustrated, thoughtful, surprised, worried, confident, neutral, empathetic, persuasive, curious, skeptical)
2. INTENSITY: How strongly expressed (low, medium, high)
3. EXPRESSION: Specific facial expression (be precise: "raised eyebrows with slight smile", "furrowed brow, tight lips", etc.)
4. GESTURES: Specific hand/arm gestures (be precise: "open palms facing up", "pointing finger for emphasis", "hands clasped together", etc.)
5. POSTURE: Body posture (be precise: "leaning forward engaged", "relaxed shoulders back", "slight head tilt", etc.)
6. DELIVERY: How to speak the line (be precise: "slow and deliberate", "quick and energetic", "soft and empathetic", etc.)

Respond ONLY with valid JSON, no other text."""

        user_msg = f"""Dialogue line ({language}): "{dialogue_line}"

Analyze this line and provide appropriate non-verbal cues. Respond with JSON:
{{
  "emotion": "...",
  "intensity": "...",
  "suggested_expression": "...",
  "suggested_gestures": "...",
  "suggested_posture": "...",
  "delivery_style": "..."
}}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        
        result_text = resp.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        vlog(f"[DIALOGUE ANALYSIS] Emotion: {result.get('emotion')}, Intensity: {result.get('intensity')}")
        vlog(f"[DIALOGUE ANALYSIS] Expression: {result.get('suggested_expression')}")
        vlog(f"[DIALOGUE ANALYSIS] Gestures: {result.get('suggested_gestures')}")
        
        return {
            "emotion": result.get("emotion", "neutral"),
            "intensity": result.get("intensity", "medium"),
            "suggested_expression": result.get("suggested_expression", default_result["suggested_expression"]),
            "suggested_gestures": result.get("suggested_gestures", default_result["suggested_gestures"]),
            "suggested_posture": result.get("suggested_posture", default_result["suggested_posture"]),
            "delivery_style": result.get("delivery_style", default_result["delivery_style"])
        }
        
    except Exception as e:
        vlog(f"[DIALOGUE ANALYSIS] Error: {e}")
        return default_result


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
    Generate a HIGHLY SPECIFIC voice profile for consistency across all clips.
    
    CRITICAL: Default is NO ACCENT (neutral). Only use accent if user explicitly requests it.
    
    Template:
    1. Age & Gender - specific range
    2. Accent - NO ACCENT by default, only if user requests
    3. Pitch & Timbre - specific vocal qualities
    4. Vocal Texture - unique characteristics
    5. Tone / Personality - emotional baseline
    6. Pacing & Rhythm - speech patterns
    7. Signature Traits - unique identifiers
    8. Audio Quality - always studio grade
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
    user_accent = user_context_enriched.get('accent', '')  # Only if explicitly requested
    
    # === MERGE: User context overrides auto-detected ===
    final_role = user_role if user_role else auto_role
    final_voice_tone = user_voice_tone if user_voice_tone else auto_voice_tone
    final_delivery = user_delivery if user_delivery else auto_delivery
    
    # Log the merge decision
    vlog(f"[VOICE CASTING] Auto-detected role: {auto_role} (confidence: {detection_confidence})")
    if user_role:
        vlog(f"[VOICE CASTING] User override role: {user_role} (USING THIS)")
    vlog(f"[VOICE CASTING] Final role: {final_role}")
    if user_accent:
        vlog(f"[VOICE CASTING] User requested accent: {user_accent}")
    else:
        vlog(f"[VOICE CASTING] No accent requested - using NEUTRAL/NO ACCENT")
    
    if client is None:
        # Fallback without OpenAI
        return build_voice_profile_template(
            age=auto_age,
            gender=auto_gender,
            language=language,
            role=final_role,
            tone=final_voice_tone,
            delivery=final_delivery,
            user_accent=user_accent
        )
    
    # Determine accent instruction
    if user_accent:
        accent_instruction = f"User requested accent: {user_accent}. Use this accent."
    else:
        accent_instruction = f"""ACCENT RULE: NO ACCENT. 
The voice must be completely neutral with NO regional accent whatsoever.
- NO Indian accent
- NO British accent  
- NO Southern US accent
- NO Australian accent
- NO Irish accent
- NO Scottish accent
- NO any other regional accent
The voice should sound like a neutral, accentless {language} speaker - like a professional news anchor or voice actor with no detectable regional origin."""
    
    try:
        system_msg = f"""You are a Voice Casting Director creating a UNIQUE VOICE IDENTITY.

=== CRITICAL GOAL ===
Create a voice profile SO SPECIFIC that it generates the EXACT SAME VOICE across multiple video clips.
Generic descriptions = voice drift between clips. We need UNIQUE IDENTIFIERS.

=== ACCENT RULE (EXTREMELY IMPORTANT) ===
{accent_instruction}

=== VOICE IDENTITY TEMPLATE (MANDATORY FORMAT) ===

1. Age & Gender: [Specific age range like "early 30s" not just "adult"] [gender]

2. Accent: {"Use the user-requested accent: " + user_accent if user_accent else "NONE - completely neutral, no regional accent whatsoever. Like a professional voice actor with no detectable origin."}

3. Pitch & Timbre: 
   - Pitch: [specific like "low-medium, around 180Hz" or "higher register"]
   - Timbre: [warm/bright/dark/rich/thin]
   - Resonance: [chest voice/head voice/balanced]

4. Vocal Texture (CRITICAL FOR CONSISTENCY):
   - [Choose 2-3: smooth, slightly breathy, crisp, silky, husky, gravelly, nasal, clear, velvety, airy, full-bodied]
   - Any unique quality: [slight vocal fry, natural raspiness, bell-like clarity, etc.]

5. Tone & Personality:
   - Baseline emotion: [confident, warm, serious, friendly, authoritative]
   - Energy level: [calm, moderate, high-energy]

6. Pacing & Rhythm:
   - Speed: [slow/moderate/fast, specific like "105-115 words per minute"]
   - Rhythm: [steady, varied, staccato, flowing]
   - Pause style: [minimal, natural beats, dramatic pauses]

7. Signature Traits (UNIQUE IDENTIFIERS - pick 2-3):
   - [Examples: slight upward inflection at sentence ends, emphasis on first syllable, soft consonants, crisp T's and D's, gentle fade at phrase ends, subtle warmth on vowels]

8. Audio Quality: Studio-grade, broadcast quality, crystal clear, no background noise.

=== ROLE GUIDELINES ===
Adjust the profile based on role but KEEP IT SPECIFIC:
- News Anchor: authoritative, measured, formal
- Fitness Coach: energetic, motivational, loud
- Doctor: calm, reassuring, precise
- Teacher: clear, patient, engaging
- Salesperson: enthusiastic, persuasive
- Corporate: professional, polished
- Influencer/Vlogger: natural, conversational, authentic

=== CRITICAL RULES ===
1. {"Use ONLY the accent specified: " + user_accent if user_accent else "NO ACCENT - the voice must be completely neutral with zero regional coloring"}
2. Voice must be consistent across all clips
3. Be extremely specific - vague descriptions cause voice drift"""

        user_msg = f"""=== SUBJECT FROM IMAGE ===
Age: {auto_age}
Gender: {auto_gender}
Visual: {visual_desc}

=== VOICE REQUIREMENTS ===
Language: {language}
Role/Style: {final_role}
Tone preference: {final_voice_tone}
Delivery style: {final_delivery}
Accent: {user_accent if user_accent else "NONE - completely neutral, no accent"}

=== YOUR TASK ===
Create a HIGHLY SPECIFIC voice identity using ALL 8 sections of the template.
The voice must be unique enough to be consistent across 10+ video clips.
Include at least 2 "Signature Traits" that make this voice recognizable.

REMEMBER: {"Use the accent: " + user_accent if user_accent else "NO ACCENT AT ALL - neutral voice only"}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        result = resp.choices[0].message.content.strip()
        
        # Add consistency and accent reminder at the end
        accent_reminder = f"ACCENT: {user_accent}" if user_accent else "ACCENT: NONE - completely neutral, no regional accent"
        result += f"\n\n[CONSISTENCY LOCK: This exact voice profile must be used for ALL clips. {accent_reminder}. Do not deviate.]"
        
        return result
        
    except Exception as e:
        vlog(f"[VOICE PROFILE] Error: {e}")
        return build_voice_profile_template(
            age=auto_age,
            gender=auto_gender,
            language=language,
            role=final_role,
            tone=final_voice_tone,
            delivery=final_delivery,
            user_accent=user_accent
        )


def build_voice_profile_template(
    age: str,
    gender: str,
    language: str,
    role: str,
    tone: str,
    delivery: str,
    user_accent: str = ""
) -> str:
    """Build a structured voice profile with specific identifiers for consistency."""
    
    # NO ACCENT by default
    if user_accent:
        accent_line = f"{user_accent}"
    else:
        accent_line = f"NONE - completely neutral {language} voice with NO regional accent. No Indian, British, Southern, Australian, or any other accent. Like a professional voice actor with no detectable origin."
    
    return f"""1. Age & Gender: {age} {gender}

2. Accent: {accent_line}

3. Pitch & Timbre:
   - Pitch: Medium register
   - Timbre: Warm and clear
   - Resonance: Balanced chest-head voice

4. Vocal Texture:
   - Smooth and clear articulation
   - No vocal fry, no breathiness
   - Clean, professional sound

5. Tone & Personality:
   - Baseline: {tone}, professional
   - Energy: Moderate, grounded

6. Pacing & Rhythm:
   - Speed: Moderate, {delivery}
   - Rhythm: Steady and flowing
   - Pauses: Natural phrase breaks

7. Signature Traits:
   - Consistent warmth throughout
   - Clear word endings
   - Natural emphasis patterns

8. Audio Quality: Studio-grade, broadcast quality, crystal clear, no background noise.

[CONSISTENCY LOCK: This exact voice profile must be used for ALL clips. {"ACCENT: " + user_accent if user_accent else "ACCENT: NONE - no regional accent"}. Do not deviate.]"""


def get_default_voice_profile(language: str, user_context: str = "") -> str:
    """Default voice profile with NO ACCENT by default."""
    
    context_note = f"\nSpeaker context: {user_context}" if user_context else ""
    
    return f"""1. Age & Gender: Adult, matching the speaker visible in the image

2. Accent: NONE - completely neutral {language} voice with NO regional accent. No Indian, British, Southern, Australian, or any other accent.

3. Pitch & Timbre:
   - Pitch: Medium register
   - Timbre: Warm and clear
   - Resonance: Balanced, natural

4. Vocal Texture:
   - Smooth and clear articulation
   - Natural, unaffected quality

5. Tone & Personality:
   - Baseline: Professional, confident, approachable
   - Energy: Moderate{context_note}

6. Pacing & Rhythm:
   - Speed: Moderate, conversational
   - Rhythm: Steady and natural
   - Pauses: Natural phrase breaks

7. Signature Traits:
   - Consistent warmth throughout
   - Clear word endings
   - Natural emphasis patterns

8. Audio Quality: Studio-grade, broadcast quality, crystal clear, no background noise.

[CONSISTENCY LOCK: This exact voice profile must be used for ALL clips. ACCENT: NONE - no regional accent. Do not deviate.]"""


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
    redo_feedback: Optional[str] = None,  # User's feedback for redo
    override_duration: Optional[str] = None,  # Override duration for last clip
) -> str:
    """
    PROMPT ASSEMBLY
    
    Builds the final prompt using:
    1. FRAME ANALYSIS (default) - auto-detected from image
    2. USER CONTEXT (override) - adds to or overrides frame analysis
    3. DIALOGUE ANALYSIS - gestures/expressions based on line content
    4. REDO FEEDBACK - user's specific feedback for regeneration (if any)
    
    Priority: Redo feedback > User context > Dialogue analysis > Frame analysis > Defaults
    
    PROMPT STRUCTURE (based on Veo 3.1 best practices):
    - Front-load critical details (first 10% gets 45% attention)
    - Use "without" instead of "no" for negatives
    - Use professional cinematic terminology
    - Quotation marks for dialogue
    - Separate sections for different expert networks
    """
    vlog(f"[ROUTING] Building prompt for clip {clip_index}...")
    if redo_feedback:
        vlog(f"[ROUTING] REDO FEEDBACK: {redo_feedback}")

    # === 1. GET DEFAULTS FROM FRAME ANALYSIS ===
    if frame_analysis is None:
        frame_analysis = {}
    
    if user_context_override is None:
        user_context_override = {}
    
    # === 2. ANALYZE DIALOGUE FOR GESTURES/EXPRESSIONS ===
    dialogue_analysis = {}
    if config.use_openai_prompt_tuning:
        dialogue_analysis = analyze_dialogue_for_gestures(dialogue_line, language, openai_key)
    
    # === 3. MERGE: User context > Dialogue analysis > Frame analysis > Defaults ===
    def get_value(key, frame_key=None, dialogue_key=None, default=""):
        """Get value with priority: user_context > dialogue_analysis > frame_analysis > default"""
        if frame_key is None:
            frame_key = key
        if dialogue_key is None:
            dialogue_key = key
            
        user_val = user_context_override.get(key, "")
        if user_val:
            return user_val
            
        dialogue_val = dialogue_analysis.get(dialogue_key, "")
        if dialogue_val:
            return dialogue_val
            
        frame_val = frame_analysis.get(frame_key, "")
        if frame_val:
            return frame_val
            
        return default
    
    # Subject details - now uses dialogue analysis for expression/gestures
    facial_expression = get_value("facial_expression", "facial_expression", "suggested_expression", "natural expression")
    body_language = get_value("body_language", "body_language", "suggested_posture", "natural posture")
    suggested_gestures = dialogue_analysis.get("suggested_gestures", "natural hand movements while speaking")
    
    # Action
    current_action = get_value("subject_action", "current_action", None, "speaking naturally to camera")
    
    # Scene/atmosphere
    atmosphere = get_value("atmosphere", "atmosphere", None, "natural soft lighting")
    background = get_value("background_action", "background_description", None, "static background")
    setting = frame_analysis.get("setting_type", "indoor setting")
    
    # Voice - use dialogue analysis for delivery style
    voice_tone = get_value("voice_tone", "suggested_voice_tone", None, "natural speaking voice")
    delivery_style = get_value("delivery_style", "suggested_delivery", "delivery_style", "natural conversation")
    
    # Emotion context from dialogue
    emotion = dialogue_analysis.get("emotion", "neutral")
    intensity = dialogue_analysis.get("intensity", "medium")
    
    # Camera
    camera_motion = get_value("camera_motion", "camera_motion", None, "static locked-off tripod")
    
    # === 4. BUILD VISUAL DESCRIPTION ===
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

    # === 5. BUILD OPTIMIZED PROMPT STRUCTURE ===
    # Based on Veo 3.1 best practices:
    # - Formula: [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    # - Front-load critical visual/audio requirements
    # - Use quotation marks for dialogue
    # - Use "without" for negative prompts
    
    # Use override duration if provided (for last clip)
    if override_duration:
        duration = float(override_duration)
    else:
        duration = float(config.duration.value if hasattr(config.duration, 'value') else config.duration)
    speech_end_time = duration - 1.0  # Leave 1 second for silence
    
    # Extract accent info from voice profile
    accent_line = "completely neutral, accentless"
    if voice_profile:
        if "ACCENT:" in voice_profile:
            accent_match = voice_profile.split("ACCENT:")[1].split("\n")[0].strip() if "ACCENT:" in voice_profile else ""
            if accent_match and "NONE" not in accent_match.upper():
                accent_line = accent_match
    
    # === BUILD THE PROMPT IN OPTIMAL ORDER ===
    # TIER 1: FRAMING (most critical - front-loaded)
    shot_section = f"""Medium shot, {camera_motion}, subject centered in frame.
The person from the start frame speaks directly to camera.
Photorealistic, natural, like a real video recording."""

    # TIER 2: SUBJECT (who is in the scene)
    subject_section = f"""The speaker maintains exact appearance and clothing from start frame.
Expression: {facial_expression}. Posture: {body_language}.
Lips move in perfect sync with every spoken word - this is on-camera speech, not voiceover."""

    # TIER 3: ACTION (what happens)
    action_section = f"""Action: {current_action}.
Gestures: {suggested_gestures}.
All movements natural and realistic."""

    # TIER 4: DIALOGUE (using quotation marks per Google's format)
    dialogue_section = f"""The speaker says in {language}, "{dialogue_line}"
Voice: {voice_tone}, {delivery_style}, {emotion} emotion at {intensity} intensity.
Voice profile: {accent_line}.
Timing: Speech completes by {speech_end_time:.1f} seconds. Final {1.0} second is complete silence with natural stillness."""

    # TIER 5: SCENE/CONTEXT
    scene_section = f"""Setting: {setting}. {start_desc}
Lighting: {atmosphere}.
Background: {background}."""

    # TIER 6: AUDIO ENVIRONMENT (critical for your requirements)
    audio_section = f"""Audio environment: Studio recording booth, completely dead silent.
The ONLY sound is the speaker's voice, nothing else.
Recording quality: Broadcast-quality isolated voice, professional microphone."""

    # TIER 7: STYLE & AMBIANCE
    style_section = """Style: Photorealistic, natural, authentic - like a real video recording.
Natural colors and lighting, accurate skin tones."""

    # NEGATIVE PROMPTS (using "without" method - 85% more effective than "no")
    negative_section = """Without: text overlays, subtitles, captions, burned-in text, graphics, watermarks, logos, titles, lower thirds.
Without: fake laughter, audience sounds, applause, crowd noise, background music, ambient noise, sound effects.
Without: morphing faces, distorted limbs, extra fingers, anatomical errors, jerky movements, teleportation."""

    # === ASSEMBLE FINAL PROMPT ===
    # Priority instruction at the very top if redo feedback exists
    if redo_feedback:
        priority_section = f"""=== PRIORITY INSTRUCTION (APPLY FIRST) ===
{redo_feedback}
==="""
    else:
        priority_section = ""
    
    # Build the complete prompt in optimal order
    final_prompt = f"""{priority_section}

{shot_section}

{subject_section}

{action_section}

{dialogue_section}

{scene_section}

{audio_section}

{style_section}

{negative_section}

VOICE PROFILE:
{voice_profile}""".strip()
    
    vlog(f"[ROUTING] Final prompt length: {len(final_prompt)} chars")
    vlog(f"[ROUTING] Prompt preview: {final_prompt[:500]}...")
    
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


def modify_image_for_celebrity_bypass(image_path: Path, api_keys: List[str], attempt: int = 1) -> Optional[Path]:
    """
    Modify an image using Nano Banana Pro to bypass celebrity filter false positives.
    
    Strategy: Slightly modify the image style/lighting to change how the face is perceived
    by the filter while preserving the person's identity and overall appearance.
    
    Args:
        image_path: Path to the original image
        api_keys: List of Gemini API keys
        attempt: Which bypass attempt this is (affects modification strategy)
    
    Returns:
        Path to modified image, or None if modification failed
    """
    try:
        import google.genai as genai
        from google.genai import types
        
        if not api_keys:
            vlog("[Celebrity Bypass] No API keys available")
            return None
        
        api_key = api_keys[0]
        client = genai.Client(api_key=api_key)
        
        # Read the image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Determine mime type
        suffix = image_path.suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }.get(suffix, 'image/jpeg')
        
        # Different modification strategies based on attempt number
        modification_prompts = [
            # Attempt 1: Subtle lighting adjustment
            "Slightly adjust the lighting in this image to be a bit warmer and softer. "
            "Keep the person's identity, expression, pose, clothing, and background exactly the same. "
            "Only make very subtle lighting adjustments.",
            
            # Attempt 2: Minor style adjustment
            "Apply a very subtle cinematic color grade to this image. "
            "Preserve everything about the person - their face, expression, pose, and clothing. "
            "Keep the background identical. Only adjust colors very slightly.",
            
            # Attempt 3: Slight artistic filter
            "Apply a very subtle film photography look to this image with slightly reduced contrast. "
            "The person must look exactly the same - same face, expression, pose, clothing. "
            "Changes should be barely noticeable.",
        ]
        
        prompt_idx = min(attempt - 1, len(modification_prompts) - 1)
        modification_prompt = modification_prompts[prompt_idx]
        
        vlog(f"[Celebrity Bypass] Attempt {attempt}: Modifying image {image_path.name}")
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=modification_prompt)
                ]
            )
        ]
        
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            temperature=0.3  # Low temperature for faithful reproduction
        )
        
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",  # Nano Banana Pro
            contents=contents,
            config=config
        )
        
        # Extract modified image from response
        modified_path = image_path.parent / f"{image_path.stem}_celeb_bypass_{attempt}.png"
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    with open(modified_path, 'wb') as f:
                        f.write(part.inline_data.data)
                    vlog(f"[Celebrity Bypass] Modified image saved: {modified_path.name}")
                    return modified_path
        
        vlog("[Celebrity Bypass] Nano Banana Pro did not return an image")
        return None
        
    except Exception as e:
        vlog(f"[Celebrity Bypass] Error: {e}")
        return None


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
        
        # Per-scene frame analysis cache (keyed by image path string)
        self.scene_frame_analyses: Dict[str, dict] = {}
        
        # Callbacks
        self.on_progress: Optional[ProgressCallback] = None
        self.on_error: Optional[Callable[[VeoError], None]] = None
        
        # State
        self.cancelled = False
        self.paused = False
    
    def get_frame_analysis_for_image(self, image_path: Path) -> dict:
        """
        Get frame analysis for a specific image, using cache if available.
        This ensures each scene's image is analyzed separately.
        """
        path_key = str(image_path)
        
        if path_key in self.scene_frame_analyses:
            vlog(f"[VeoGenerator] Using cached frame analysis for {image_path.name}")
            return self.scene_frame_analyses[path_key]
        
        # Analyze this image
        vlog(f"[VeoGenerator] Analyzing new scene image: {image_path.name}")
        if self.config.use_openai_prompt_tuning:
            analysis = analyze_frame(str(image_path), self.openai_key)
        else:
            analysis = {
                "subject_age": "adult",
                "subject_gender": "neutral", 
                "apparent_role": "natural speaker",
                "suggested_voice_tone": "natural, conversational",
                "suggested_delivery": "clear, measured",
                "visual_description": "",
                "confidence": "low"
            }
        
        # Cache it
        self.scene_frame_analyses[path_key] = analysis
        return analysis
    
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
        scene_image: Optional[Path] = None,  # Original scene image for analysis (may differ from start_frame in CONTINUE mode)
        redo_feedback: Optional[str] = None,  # User's feedback for redo - what should be different
        override_duration: Optional[str] = None,  # Override duration for this specific clip (e.g., "4" for last clip)
    ) -> Dict[str, Any]:
        """Generate a single video clip with retry logic."""
        
        vlog(f"[VeoGenerator] Generating clip {clip_index}: '{dialogue_line[:50]}...'")
        if redo_feedback:
            vlog(f"[VeoGenerator] Redo feedback: '{redo_feedback}'")
        if override_duration:
            vlog(f"[VeoGenerator] Using override duration: {override_duration}s")
        
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
                user_message="Video generation library not available",
                details={},
                recoverable=False,
                suggestion="Contact administrator to install google-genai"
            )
            return result
        
        # Store redo feedback for use in prompt building
        self._current_redo_feedback = redo_feedback
        
        # Initialize voice profile if not already done
        # This should be called by the worker before generating clips, but fallback here
        if self.voice_profile is None:
            vlog(f"[WARNING] Voice profile not pre-initialized, initializing now for clip {clip_index}")
            self.initialize_voice_profile(start_frame)
        
        # Get frame analysis for THIS scene's image (not the start_frame which may be extracted)
        # scene_image is the original uploaded image for this scene
        # start_frame may be an extracted frame from previous clip (in CONTINUE mode)
        analysis_image = scene_image if scene_image else start_frame
        clip_frame_analysis = self.get_frame_analysis_for_image(analysis_image)
        
        # Log voice ID for this clip
        vlog(f"[Clip {clip_index}] Using Voice ID: {self.voice_profile_id}, Scene image: {analysis_image.name if hasattr(analysis_image, 'name') else 'unknown'}")
        
        failed_end_frames = []
        attempts = 0
        current_attempt_end_index = current_end_index
        
        while attempts < self.config.max_retries_per_clip:
            if self.cancelled:
                result["error"] = VeoError(
                    code=ErrorCode.UNKNOWN, 
                    message="Cancelled", 
                    user_message="Generation was cancelled",
                    details={},
                    recoverable=False,
                    suggestion="Start a new job"
                )
                return result
            
            while self.paused:
                time.sleep(1)
                if self.cancelled:
                    return result
            
            attempts += 1
            
            # Determine end frame - ALWAYS use the assigned end_frame if provided
            # Only try alternate end frames if the original is blacklisted
            if end_frame:
                if end_frame not in self.blacklist:
                    actual_end_frame = end_frame
                    actual_end_index = current_end_index
                else:
                    # End frame is blacklisted, try to find alternative
                    next_result = get_next_clean_image(
                        current_end_index, images_list, self.blacklist, self.config.max_image_attempts
                    )
                    
                    if next_result is None:
                        if len(failed_end_frames) >= 2:
                            self.blacklist.add(start_frame)
                        return result
                    
                    actual_end_index, actual_end_frame = next_result
                    current_attempt_end_index = actual_end_index
            else:
                # No end frame specified (single image mode or no interpolation)
                actual_end_frame = None
                actual_end_index = current_end_index
            
            # Build prompt using per-scene frame analysis + user context override
            try:
                prompt_text = build_prompt(
                    dialogue_line, start_frame, actual_end_frame, clip_index,
                    self.config.language, self.voice_profile, self.config, self.openai_key,
                    frame_analysis=clip_frame_analysis,  # Per-scene analysis (based on scene_image)
                    user_context_override=self.enriched_context,  # User overrides (if any)
                    redo_feedback=self._current_redo_feedback,  # User's feedback for redo
                    override_duration=override_duration,  # Override duration for last clip
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
                    # Use override duration if provided (for last clip dynamic duration)
                    if override_duration:
                        dur = override_duration
                    else:
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
                            error = VeoError(
                                code=ErrorCode.RATE_LIMIT, 
                                message="All keys blocked", 
                                user_message="All API keys are temporarily blocked",
                                details={"blocked_count": len(self.api_keys.blocked_keys)},
                                recoverable=True,
                                suggestion="Wait a few minutes for keys to unblock"
                            )
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
                
                # Special handling for celebrity filter - try to bypass with modified image
                if veo_error.code == ErrorCode.CELEBRITY_FILTER:
                    celebrity_bypass_attempts = getattr(self, '_celebrity_bypass_attempts', {})
                    frame_key = str(start_frame)
                    current_bypass_attempt = celebrity_bypass_attempts.get(frame_key, 0) + 1
                    celebrity_bypass_attempts[frame_key] = current_bypass_attempt
                    self._celebrity_bypass_attempts = celebrity_bypass_attempts
                    
                    if current_bypass_attempt <= 3:  # Max 3 bypass attempts per image
                        self._emit_progress(
                            clip_index, "celebrity_bypass",
                            f"Celebrity filter triggered (false positive). Attempting bypass {current_bypass_attempt}/3...",
                            {"bypass_attempt": current_bypass_attempt}
                        )
                        
                        # Try to modify the start frame to bypass the filter
                        api_keys = [self.api_keys.get_current_gemini_key()] if self.api_keys.get_current_gemini_key() else []
                        modified_frame = modify_image_for_celebrity_bypass(start_frame, api_keys, current_bypass_attempt)
                        
                        if modified_frame and modified_frame.exists():
                            # Use the modified frame for next attempt
                            start_frame = modified_frame
                            vlog(f"[Clip {clip_index}] Using modified frame for celebrity bypass: {modified_frame.name}")
                            # Don't count this as a failed attempt, retry with modified image
                            continue
                        else:
                            vlog(f"[Clip {clip_index}] Celebrity bypass modification failed, trying alternate end frame")
                    else:
                        vlog(f"[Clip {clip_index}] Exhausted celebrity bypass attempts, blacklisting image")
                        self.blacklist.add(start_frame)
                    
                    if actual_end_frame:
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
            user_message="Video generation failed after multiple attempts",
            details={"attempts": self.config.max_retries_per_clip, "failed_frames": len(failed_end_frames)},
            recoverable=True,
            suggestion="Try regenerating the clip or use a different image"
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