"""
Voice Cloning Module using Replicate API

Uses HierSpeech++ for direct voice-to-voice conversion ($0.06/run)
- Takes source audio + target voice
- Outputs source speech with target voice characteristics
- Preserves exact timing/prosody from source

Setup:
1. Get API key from https://replicate.com
2. Add REPLICATE_API_TOKEN to environment variables
"""

import os
import asyncio
import httpx
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# HierSpeech++ - Voice-to-Voice conversion
MODEL_NAME = "adirik/hierspeechpp"
MODEL_COST = "$0.06"


async def upload_to_tmpfiles(file_path: Path) -> str:
    """Upload file to tmpfiles.org and return URL (free, no auth needed)"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            response.raise_for_status()
            data = response.json()
            # Convert tmpfiles.org URL to direct download URL
            url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            logger.info(f"Uploaded {file_path.name} -> {url}")
            return url


async def get_model_version(model: str) -> str:
    """Get the latest version of a Replicate model."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"https://api.replicate.com/v1/models/{model}",
            headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
        )
        response.raise_for_status()
        data = response.json()
        version = data.get("latest_version", {}).get("id")
        if not version:
            raise ValueError(f"Could not get version for model {model}")
        logger.info(f"[VoiceCloner] Got model version for {model}: {version[:20]}...")
        return version


async def run_replicate_prediction(model: str, inputs: dict, timeout_seconds: int = 300) -> Optional[str]:
    """Generic function to run a Replicate prediction and wait for result."""
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not set")
    
    version = await get_model_version(model)
    
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    
    payload = {
        "version": version,
        "input": inputs
    }
    
    async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
        # Create prediction
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 422:
            error_detail = response.json()
            logger.error(f"[Replicate] 422 Validation Error: {error_detail}")
            raise ValueError(f"Invalid input: {error_detail}")
        
        if response.status_code == 401:
            raise ValueError("Invalid Replicate API token")
        
        response.raise_for_status()
        prediction = response.json()
        
        logger.info(f"[Replicate] Prediction created: id={prediction.get('id')}, status={prediction.get('status')}")
        
        # Check for immediate result
        if prediction.get("status") == "succeeded":
            return prediction.get("output")
        
        # Poll for completion
        prediction_id = prediction["id"]
        max_attempts = timeout_seconds
        
        for attempt in range(max_attempts):
            await asyncio.sleep(1)
            
            poll_response = await client.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
            )
            poll_response.raise_for_status()
            prediction = poll_response.json()
            
            status = prediction["status"]
            
            if status == "succeeded":
                return prediction.get("output")
            elif status == "failed":
                error = prediction.get("error", "Unknown error")
                raise ValueError(f"Prediction failed: {error}")
            elif status == "canceled":
                return None
            
            if attempt % 15 == 0:
                logger.info(f"[Replicate] Processing... ({attempt}s, status={status})")
        
        raise ValueError(f"Prediction timed out after {timeout_seconds}s")


async def voice_convert(
    source_audio_url: str,
    target_voice_url: str,
    denoise_ratio: float = 0.7
) -> Optional[str]:
    """
    Convert voice using HierSpeech++ (voice-to-voice).
    
    Args:
        source_audio_url: URL to source audio (speech to convert)
        target_voice_url: URL to target voice sample (voice to clone)
        denoise_ratio: Noise reduction (0-1, recommended 0.6-0.8)
    
    Returns:
        URL to the output audio file
    """
    logger.info("[HierSpeech++] Starting voice-to-voice conversion...")
    logger.info(f"  Source: {source_audio_url[:60]}...")
    logger.info(f"  Target: {target_voice_url[:60]}...")
    
    result = await run_replicate_prediction(
        model=MODEL_NAME,
        inputs={
            "input_sound": source_audio_url,
            "target_voice": target_voice_url,
            "denoise_ratio": denoise_ratio,
            "output_sample_rate": 48000
        },
        timeout_seconds=180
    )
    
    logger.info(f"[HierSpeech++] Complete: {result}")
    return result


async def download_audio(url: str, output_path: Path) -> bool:
    """Download audio file from URL"""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"[Download] Audio saved to {output_path}")
            return True
    except Exception as e:
        logger.error(f"[Download] Failed: {e}")
        return False


async def voice_swap_video(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    progress_callback=None
) -> dict:
    """
    Complete voice swap pipeline for video using HierSpeech++.
    
    Args:
        video_path: Input video file
        reference_voice_path: Reference voice sample (the voice to clone)
        output_path: Output video with swapped voice
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    from audio_processor import extract_audio, replace_audio
    
    if not REPLICATE_API_TOKEN:
        return {"success": False, "error": "REPLICATE_API_TOKEN not configured"}
    
    stats = {
        "success": False,
        "model": "HierSpeech++ (Voice-to-Voice)",
        "cost_estimate": MODEL_COST
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Extract audio from video
        if progress_callback:
            progress_callback("Extracting audio...")
        
        source_audio = temp_path / "source_audio.wav"
        if not extract_audio(video_path, source_audio):
            return {"success": False, "error": "Failed to extract audio from video"}
        
        logger.info(f"[VoiceSwap] Extracted source audio: {source_audio.stat().st_size} bytes")
        
        # Step 2: Upload files
        if progress_callback:
            progress_callback("Uploading audio files...")
        
        try:
            source_url = await upload_to_tmpfiles(source_audio)
            target_url = await upload_to_tmpfiles(reference_voice_path)
        except Exception as e:
            logger.error(f"[VoiceSwap] Upload failed: {e}")
            return {"success": False, "error": f"Failed to upload files: {e}"}
        
        # Step 3: Convert voice using HierSpeech++
        if progress_callback:
            progress_callback("Converting voice (30-90 seconds)...")
        
        try:
            output_url = await voice_convert(
                source_audio_url=source_url,
                target_voice_url=target_url
            )
        except Exception as e:
            logger.error(f"[VoiceSwap] Conversion error: {e}")
            return {"success": False, "error": f"Voice conversion failed: {e}"}
        
        if not output_url:
            return {"success": False, "error": "Voice conversion returned no output"}
        
        # Step 4: Download converted audio
        if progress_callback:
            progress_callback("Downloading converted audio...")
        
        converted_audio = temp_path / "converted_audio.mp3"
        if not await download_audio(output_url, converted_audio):
            return {"success": False, "error": "Failed to download converted audio"}
        
        # Step 5: Replace audio in video
        if progress_callback:
            progress_callback("Creating final video...")
        
        if not replace_audio(video_path, converted_audio, output_path):
            return {"success": False, "error": "Failed to create output video"}
        
        stats["success"] = True
        logger.info(f"[VoiceSwap] Complete! Output: {output_path}")
    
    if progress_callback:
        progress_callback("Voice swap complete!")
    
    return stats


def check_replicate_available() -> dict:
    """Check if Replicate API is configured"""
    configured = bool(REPLICATE_API_TOKEN)
    return {
        "available": configured,
        "message": "Replicate API ready" if configured else "Set REPLICATE_API_TOKEN in environment",
        "model": "HierSpeech++ (Voice-to-Voice)",
        "cost": MODEL_COST
    }


# Synchronous wrapper for FastAPI endpoints
def voice_swap_video_sync(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    progress_callback=None
) -> dict:
    """
    Synchronous wrapper for voice_swap_video.
    Handles nested event loops in FastAPI.
    """
    import concurrent.futures
    
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(voice_swap_video(
                video_path,
                reference_voice_path,
                output_path,
                progress_callback
            ))
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=300)  # 5 minute timeout