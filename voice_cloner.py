"""
Voice Cloning Module using Replicate API (HierSpeech++)

HierSpeech++ does ACTUAL voice-to-voice conversion (like OpenVoice ToneColorConverter):
- Takes source audio (speech content to keep)
- Takes target voice (voice to clone)
- Outputs source speech with target voice characteristics

This mimics OpenVoice's converter.convert() functionality:
- audio_src_path -> input_sound
- tgt_se (target speaker embedding) -> target_voice

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

# HierSpeech++ for audio-to-audio voice conversion
# Similar to OpenVoice's ToneColorConverter
HIERSPEECH_MODEL = "adirik/hierspeechpp"

# Supported languages
SUPPORTED_LANGUAGES = ["EN", "ES", "FR", "ZH", "JP", "KR"]


async def upload_file_to_replicate(file_path: Path) -> str:
    """
    Upload file to Replicate's file hosting service.
    Returns a URL that can be used as input to models.
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not set")
    
    # First, create an upload URL
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Get upload URL from Replicate
        response = await client.post(
            "https://api.replicate.com/v1/files",
            headers={
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "filename": file_path.name,
                "content_type": "audio/wav"
            }
        )
        
        if response.status_code == 201:
            data = response.json()
            upload_url = data.get("upload_url")
            file_url = data.get("urls", {}).get("get")
            
            # Upload the file
            with open(file_path, "rb") as f:
                upload_response = await client.put(
                    upload_url,
                    content=f.read(),
                    headers={"Content-Type": "audio/wav"}
                )
                upload_response.raise_for_status()
            
            logger.info(f"Uploaded {file_path.name} to Replicate: {file_url}")
            return file_url
        else:
            # Fallback to tmpfiles.org
            logger.info("Replicate file upload not available, using tmpfiles.org")
            return await upload_to_tmpfiles(file_path)


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
        logger.info(f"[VoiceConvert] Got model version: {version[:20]}...")
        return version


async def voice_convert_replicate(
    source_audio_url: str,
    target_voice_url: str,
    denoise_ratio: float = 0.7,
    output_sample_rate: int = 44100
) -> Optional[str]:
    """
    Convert voice using Replicate's HierSpeech++ model.
    
    This is equivalent to OpenVoice's ToneColorConverter.convert():
    - source_audio_url = audio_src_path (the speech to convert)
    - target_voice_url = reference voice (tgt_se - target speaker embedding)
    
    Args:
        source_audio_url: URL to source audio (speech content to keep)
        target_voice_url: URL to target voice sample (voice to clone)
        denoise_ratio: Noise reduction (0-1, recommended 0.6-0.8)
        output_sample_rate: Output sample rate
    
    Returns:
        URL to the output audio file, or None if failed
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not set. Get one from https://replicate.com")
    
    # First get the model version
    version = await get_model_version(HIERSPEECH_MODEL)
    
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait"  # Wait for result instead of polling
    }
    
    # HierSpeech++ parameters for voice conversion:
    # - input_sound: source audio (like OpenVoice's audio_src_path)
    # - target_voice: voice to clone (like OpenVoice's tgt_se)
    # - NO input_text: this triggers voice conversion mode (not TTS)
    payload = {
        "version": version,
        "input": {
            "input_sound": source_audio_url,      # Source speech (content to keep)
            "target_voice": target_voice_url,      # Target voice to clone into
            "denoise_ratio": denoise_ratio,
            "output_sample_rate": output_sample_rate
        }
    }
    
    logger.info(f"[VoiceConvert] Calling HierSpeech++...")
    logger.info(f"  Source: {source_audio_url[:60]}...")
    logger.info(f"  Target: {target_voice_url[:60]}...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create prediction using /v1/predictions endpoint with version
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload
        )
        
        # Handle errors
        if response.status_code == 422:
            error_detail = response.json()
            logger.error(f"[VoiceConvert] 422 Validation Error: {error_detail}")
            raise ValueError(f"Invalid input: {error_detail}")
        
        if response.status_code == 401:
            logger.error("[VoiceConvert] 401 Unauthorized - check REPLICATE_API_TOKEN")
            raise ValueError("Invalid API token")
        
        response.raise_for_status()
        prediction = response.json()
        
        logger.info(f"[VoiceConvert] Prediction created: id={prediction.get('id')}, status={prediction.get('status')}")
        
        # Check if we got immediate result (with Prefer: wait header)
        if prediction.get("status") == "succeeded":
            output_url = prediction.get("output")
            if output_url:
                logger.info(f"[VoiceConvert] Complete (immediate): {output_url}")
                return output_url
        
        # Poll for completion if not immediate
        prediction_id = prediction["id"]
        logger.info(f"[VoiceConvert] Polling for result... (id={prediction_id})")
        
        max_attempts = 180  # 3 minutes max
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
                output_url = prediction.get("output")
                logger.info(f"[VoiceConvert] Complete: {output_url}")
                return output_url
            elif status == "failed":
                error = prediction.get("error", "Unknown error")
                logger.error(f"[VoiceConvert] Failed: {error}")
                raise ValueError(f"Voice conversion failed: {error}")
            elif status == "canceled":
                logger.warning("[VoiceConvert] Canceled")
                return None
            
            if attempt % 15 == 0:
                logger.info(f"[VoiceConvert] Still processing... ({attempt}s, status={status})")
        
        logger.error("[VoiceConvert] Timed out after 3 minutes")
        return None


async def download_audio(url: str, output_path: Path) -> bool:
    """Download audio file from URL"""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"[VoiceConvert] Downloaded audio to {output_path}")
            return True
    except Exception as e:
        logger.error(f"[VoiceConvert] Download failed: {e}")
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
    
    This mimics OpenVoice's workflow:
    1. Extract audio from video (like extract_audio in your script)
    2. Convert voice using HierSpeech++ (like converter.convert())
    3. Replace audio in video (like replace_audio in your script)
    
    Args:
        video_path: Input video file
        reference_voice_path: Reference voice sample (the voice to clone - like voice_to_clone)
        output_path: Output video with swapped voice
        language: Language code (informational)
        speed: Speech speed (informational)
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    from audio_processor import extract_audio, replace_audio
    
    if not REPLICATE_API_TOKEN:
        return {"success": False, "error": "REPLICATE_API_TOKEN not configured"}
    
    stats = {
        "success": False,
        "cost_estimate": "$0.06"
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Extract audio from video (like your extract_audio function)
        if progress_callback:
            progress_callback("Extracting audio from video...")
        
        source_audio = temp_path / "source_audio.wav"
        if not extract_audio(video_path, source_audio):
            return {"success": False, "error": "Failed to extract audio from video"}
        
        logger.info(f"[VoiceSwap] Extracted source audio: {source_audio.stat().st_size} bytes")
        
        # Step 2: Upload files for processing
        if progress_callback:
            progress_callback("Uploading audio files...")
        
        try:
            source_url = await upload_to_tmpfiles(source_audio)
            target_url = await upload_to_tmpfiles(reference_voice_path)
        except Exception as e:
            logger.error(f"[VoiceSwap] Upload failed: {e}")
            return {"success": False, "error": f"Failed to upload files: {e}"}
        
        # Step 3: Convert voice (like converter.convert() in OpenVoice)
        if progress_callback:
            progress_callback("Converting voice (30-90 seconds)...")
        
        try:
            output_url = await voice_convert_replicate(
                source_audio_url=source_url,
                target_voice_url=target_url,
                denoise_ratio=0.7
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
        
        # Step 5: Replace audio in video (like your replace_audio function)
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
        "message": "Replicate API ready" if configured else "Set REPLICATE_API_TOKEN in environment"
    }


# Synchronous wrapper for FastAPI endpoints
def voice_swap_video_sync(
    video_path: Path,
    reference_voice_path: Path,
    output_path: Path,
    language: str = "EN",
    speed: float = 1.0,
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
                language,
                speed,
                progress_callback
            ))
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=300)  # 5 minute timeout