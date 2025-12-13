"""
Voice Cloning Module using Replicate API (OpenVoice)
- Hosted voice cloning without needing GPU server
- ~$0.05 per voice swap
- Works with Render free tier (just API calls)

Setup:
1. Get API key from https://replicate.com
2. Add REPLICATE_API_TOKEN to environment variables
"""

import os
import asyncio
import httpx
import base64
import tempfile
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
OPENVOICE_MODEL = "chenxwh/openvoice:b31457fa10fea1dd33f5c2f85b0a73d2c3a46e7c66728d8ff0f21ad80744f9c7"

# Supported languages for OpenVoice V2
SUPPORTED_LANGUAGES = ["EN", "ES", "FR", "ZH", "JP", "KR"]


async def upload_to_tmpfiles(file_path: Path) -> str:
    """Upload file to tmpfiles.org and return URL (free, no auth needed)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            response.raise_for_status()
            data = response.json()
            # Convert tmpfiles.org URL to direct download URL
            url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            return url


async def clone_voice_replicate(
    source_audio_url: str,
    reference_voice_url: str,
    language: str = "EN",
    speed: float = 1.0
) -> Optional[str]:
    """
    Clone voice using Replicate's hosted OpenVoice model.
    
    Args:
        source_audio_url: URL to audio that needs voice changed
        reference_voice_url: URL to reference voice sample (10+ seconds recommended)
        language: Language code (EN, ES, FR, ZH, JP, KR)
        speed: Speech speed multiplier (0.5-2.0)
    
    Returns:
        URL to the output audio file, or None if failed
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not set. Get one from https://replicate.com")
    
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Start prediction
    payload = {
        "version": OPENVOICE_MODEL.split(":")[1],
        "input": {
            "audio": source_audio_url,
            "reference_speaker": reference_voice_url,
            "language": language,
            "speed": speed
        }
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create prediction
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        prediction = response.json()
        
        prediction_id = prediction["id"]
        logger.info(f"Started Replicate prediction: {prediction_id}")
        
        # Poll for completion
        max_attempts = 120  # 2 minutes max
        for attempt in range(max_attempts):
            response = await client.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )
            response.raise_for_status()
            prediction = response.json()
            
            status = prediction["status"]
            
            if status == "succeeded":
                output_url = prediction["output"]
                logger.info(f"Voice cloning complete: {output_url}")
                return output_url
            elif status == "failed":
                error = prediction.get("error", "Unknown error")
                logger.error(f"Voice cloning failed: {error}")
                return None
            elif status == "canceled":
                logger.warning("Voice cloning was canceled")
                return None
            
            # Wait before next poll
            await asyncio.sleep(1)
        
        logger.error("Voice cloning timed out")
        return None


async def download_audio(url: str, output_path: Path) -> bool:
    """Download audio file from URL"""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return False


async def voice_swap_video(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    language: str = "EN",
    speed: float = 1.0,
    progress_callback=None
) -> dict:
    """
    Complete voice swap pipeline for video.
    
    Args:
        video_path: Input video file
        reference_voice_path: Reference voice sample (audio file)
        output_path: Output video with swapped voice
        language: Language code
        speed: Speech speed
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    import asyncio
    from audio_processor import extract_audio, replace_audio
    
    if not REPLICATE_API_TOKEN:
        return {"success": False, "error": "REPLICATE_API_TOKEN not configured"}
    
    stats = {
        "success": False,
        "cost_estimate": "$0.05"
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Extract audio from video
        if progress_callback:
            progress_callback("Extracting audio from video...")
        
        source_audio = temp_path / "source_audio.wav"
        if not extract_audio(video_path, source_audio):
            return {"success": False, "error": "Failed to extract audio"}
        
        # Step 2: Upload files to temporary hosting
        if progress_callback:
            progress_callback("Uploading audio files...")
        
        try:
            source_url = await upload_to_tmpfiles(source_audio)
            reference_url = await upload_to_tmpfiles(reference_voice_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to upload files: {e}"}
        
        # Step 3: Call Replicate API
        if progress_callback:
            progress_callback("Processing voice clone (this may take 30-60 seconds)...")
        
        try:
            output_url = await clone_voice_replicate(
                source_url,
                reference_url,
                language=language,
                speed=speed
            )
        except Exception as e:
            return {"success": False, "error": f"Voice cloning failed: {e}"}
        
        if not output_url:
            return {"success": False, "error": "Voice cloning returned no output"}
        
        # Step 4: Download result
        if progress_callback:
            progress_callback("Downloading processed audio...")
        
        cloned_audio = temp_path / "cloned_audio.wav"
        if not await download_audio(output_url, cloned_audio):
            return {"success": False, "error": "Failed to download cloned audio"}
        
        # Step 5: Replace audio in video
        if progress_callback:
            progress_callback("Creating final video...")
        
        if not replace_audio(video_path, cloned_audio, output_path):
            return {"success": False, "error": "Failed to create output video"}
        
        stats["success"] = True
        
    if progress_callback:
        progress_callback("Voice swap complete!")
    
    return stats


def check_replicate_available() -> dict:
    """Check if Replicate API is configured"""
    configured = bool(REPLICATE_API_TOKEN)
    return {
        "available": configured,
        "message": "Replicate API ready" if configured else "Set REPLICATE_API_TOKEN in environment"
    }


# For sync contexts (FastAPI endpoints already in async loop)
def voice_swap_video_sync(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    language: str = "EN",
    speed: float = 1.0,
    progress_callback=None
) -> dict:
    """Synchronous wrapper for voice_swap_video - handles nested event loops"""
    import concurrent.futures
    
    def run_in_thread():
        # Create new event loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(voice_swap_video(
                video_path,
                reference_voice_path,
                output_path,
                language,
                speed,
                progress_callback
            ))
        finally:
            loop.close()
    
    # Run in a separate thread to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=300)  # 5 minute timeout