"""
Audio Processing Module for Veo Web App
- Basic noise reduction and normalization
- Lightweight, works on Render free tier (~10MB extra dependencies)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def run_cmd(cmd: list) -> tuple:
    """Run command and return (returncode, stdout, stderr)"""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=300)
        return p.returncode, out, err
    except Exception as e:
        return -1, "", str(e)


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video as WAV"""
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(audio_path)
    ]
    code, _, err = run_cmd(cmd)
    if code != 0:
        logger.error(f"Failed to extract audio: {err}")
        return False
    return True


def replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Replace audio in video with new audio"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        str(output_path)
    ]
    code, _, err = run_cmd(cmd)
    if code != 0:
        logger.error(f"Failed to replace audio: {err}")
        return False
    return True


def enhance_audio_basic(
    video_path: Path,
    output_path: Path,
    noise_reduction: float = 0.7,
    normalize: bool = True,
    target_db: float = -20.0,
    progress_callback=None
) -> dict:
    """
    Basic audio enhancement for Veo 3 videos.
    
    Args:
        video_path: Input video file
        output_path: Output video file with enhanced audio
        noise_reduction: Noise reduction strength (0-1)
        normalize: Whether to normalize volume
        target_db: Target volume level in dB
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing stats
    """
    try:
        import numpy as np
        import soundfile as sf
        import noisereduce as nr
    except ImportError:
        logger.warning("Audio enhancement libraries not available")
        # Just copy the file if libraries aren't installed
        import shutil
        shutil.copy(video_path, output_path)
        return {"enhanced": False, "reason": "Libraries not installed"}
    
    stats = {
        "enhanced": False,
        "noise_reduction_applied": False,
        "normalized": False
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_in = temp_path / "audio_in.wav"
        audio_out = temp_path / "audio_out.wav"
        
        # Step 1: Extract audio
        if progress_callback:
            progress_callback("Extracting audio...")
        
        if not extract_audio(video_path, audio_in):
            import shutil
            shutil.copy(video_path, output_path)
            return {"enhanced": False, "reason": "Failed to extract audio"}
        
        # Step 2: Load and process audio
        if progress_callback:
            progress_callback("Enhancing audio...")
        
        try:
            data, rate = sf.read(str(audio_in))
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Apply noise reduction
            if noise_reduction > 0:
                data = nr.reduce_noise(
                    y=data, 
                    sr=rate, 
                    prop_decrease=noise_reduction
                )
                stats["noise_reduction_applied"] = True
            
            # Normalize volume
            if normalize:
                rms = np.sqrt(np.mean(data**2))
                if rms > 0:
                    current_db = 20 * np.log10(rms + 1e-10)
                    gain = 10 ** ((target_db - current_db) / 20)
                    data = np.clip(data * gain, -1.0, 1.0)
                    stats["normalized"] = True
            
            # Save processed audio
            sf.write(str(audio_out), data, rate)
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            import shutil
            shutil.copy(video_path, output_path)
            return {"enhanced": False, "reason": str(e)}
        
        # Step 3: Replace audio in video
        if progress_callback:
            progress_callback("Creating enhanced video...")
        
        if not replace_audio(video_path, audio_out, output_path):
            import shutil
            shutil.copy(video_path, output_path)
            return {"enhanced": False, "reason": "Failed to replace audio"}
        
        stats["enhanced"] = True
    
    if progress_callback:
        progress_callback("Audio enhancement complete!")
    
    return stats


def export_audio_only(video_path: Path, output_path: Path, enhance: bool = True) -> bool:
    """
    Export audio from video as WAV file.
    Useful for processing with external tools (ElevenLabs, OpenVoice local).
    
    Args:
        video_path: Input video file
        output_path: Output WAV file
        enhance: Whether to apply basic cleanup first
    
    Returns:
        True if successful
    """
    if not enhance:
        return extract_audio(video_path, output_path)
    
    try:
        import numpy as np
        import soundfile as sf
        import noisereduce as nr
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = Path(temp_dir) / "temp.wav"
            
            if not extract_audio(video_path, temp_audio):
                return False
            
            # Load and clean
            data, rate = sf.read(str(temp_audio))
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Noise reduction
            data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.7)
            
            # Normalize
            rms = np.sqrt(np.mean(data**2))
            if rms > 0:
                gain = 10 ** ((-20 - 20 * np.log10(rms + 1e-10)) / 20)
                data = np.clip(data * gain, -1.0, 1.0)
            
            sf.write(str(output_path), data, rate)
            return True
            
    except ImportError:
        # Fallback to just extraction
        return extract_audio(video_path, output_path)


def import_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """
    Import external audio into video (replace existing audio).
    Use after processing with ElevenLabs or OpenVoice.
    
    Args:
        video_path: Original video file
        audio_path: New audio file (WAV or MP3)
        output_path: Output video with new audio
    
    Returns:
        True if successful
    """
    return replace_audio(video_path, audio_path, output_path)
