# -*- coding: utf-8 -*-
"""
Background Worker for Veo Web App

Handles:
- Job queue processing
- Progress updates
- Error recovery
- Graceful shutdown
"""

import json
import os
import threading
import time
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import traceback

from config import (
    JobStatus, ClipStatus, VideoConfig, APIKeysConfig, 
    DialogueLine, app_config, get_gemini_keys_from_env, get_openai_key_from_env,
    api_keys_config  # Global singleton for persistent key blocking
)
from models import (
    get_db, Job, Clip, JobLog, BlacklistEntry, GenerationLog,
    add_job_log, update_job_progress
)
from veo_generator import VeoGenerator, list_images, GENAI_AVAILABLE, describe_subject_for_continuity
from error_handler import VeoError, error_handler

# Email notification settings
EMAIL_ALERTS_ENABLED = True
ALERT_EMAIL_TO = "kaveno.biz@gmail.com"
# Gmail SMTP settings - requires App Password (not regular password)
# Go to Google Account > Security > 2-Step Verification > App passwords
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")  # Your Gmail address
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # Gmail App Password

# Track sent alerts to avoid spam
_alerts_sent = {
    "low_keys_10": False,
    "no_keys": False,
}

def send_key_alert_email(alert_type: str, available_keys: int, total_keys: int = 0, job_id: str = None):
    """Send email alert when API keys are running low or exhausted."""
    global _alerts_sent
    
    if not EMAIL_ALERTS_ENABLED:
        print(f"[Worker] Email alerts disabled, skipping {alert_type} notification", flush=True)
        return
    
    # Check if we already sent this alert (reset when keys recover)
    if _alerts_sent.get(alert_type, False):
        print(f"[Worker] Already sent {alert_type} alert, skipping", flush=True)
        return
    
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"[Worker] ⚠️ EMAIL ALERT ({alert_type}): {available_keys} keys remaining", flush=True)
        print(f"[Worker] Email not configured. Set SMTP_EMAIL and SMTP_PASSWORD environment variables.", flush=True)
        _alerts_sent[alert_type] = True
        return
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = ALERT_EMAIL_TO
        
        if alert_type == "low_keys_10":
            msg['Subject'] = "⚠️ Veo API Keys Running Low - Only 10 Remaining!"
            body = f"""
⚠️ API KEY WARNING

Your Veo Web App is running low on API keys.

Current Status:
- Available Keys: {available_keys}
- Total Keys: {total_keys}
- Job ID: {job_id or 'N/A'}

Action Required:
Please add more Gemini API keys or wait for rate limits to reset.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        elif alert_type == "no_keys":
            msg['Subject'] = "🚨 URGENT: All Veo API Keys Exhausted!"
            body = f"""
🚨 CRITICAL: ALL API KEYS EXHAUSTED

Your Veo Web App has run out of available API keys.
Generation is PAUSED until keys become available.

Current Status:
- Available Keys: 0
- Total Keys: {total_keys}
- Job ID: {job_id or 'N/A'}

Action Required:
1. Add new Gemini API keys immediately
2. Or wait for rate limits to reset (usually 1 minute to 1 hour)

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            msg['Subject'] = f"Veo API Alert: {alert_type}"
            body = f"API Key Alert: {alert_type}\nAvailable: {available_keys}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"[Worker] ✉️ Email alert sent: {alert_type} to {ALERT_EMAIL_TO}", flush=True)
        _alerts_sent[alert_type] = True
        
    except Exception as e:
        print(f"[Worker] Failed to send email alert: {e}", flush=True)
        # Still mark as sent to avoid repeated failures
        _alerts_sent[alert_type] = True

def reset_key_alerts():
    """Reset alert flags when keys recover."""
    global _alerts_sent
    _alerts_sent["low_keys_10"] = False
    _alerts_sent["no_keys"] = False
    print("[Worker] Key alert flags reset", flush=True)



def get_api_keys_with_fallback(api_keys_json: str = None) -> APIKeysConfig:
    """Get API keys - uses global singleton to persist blocked keys state."""
    global api_keys_config
    
    api_keys_data = json.loads(api_keys_json) if api_keys_json else {}
    gemini_keys = api_keys_data.get("gemini_keys", [])
    openai_key = api_keys_data.get("openai_key")
    
    # If job provides keys, update the global config (but keep blocked state)
    if gemini_keys:
        # Only update if different keys provided
        if gemini_keys != api_keys_config.gemini_api_keys:
            api_keys_config.gemini_api_keys = gemini_keys
            print(f"[Worker] Updated Gemini keys from job: {len(gemini_keys)} keys", flush=True)
    
    if openai_key:
        api_keys_config.openai_api_key = openai_key
    
    # Log current state
    available = api_keys_config.get_available_key_count()
    blocked = len(api_keys_config.blocked_keys)
    print(f"[Worker] API Keys: {available} available, {blocked} blocked", flush=True)
    
    return api_keys_config

class JobWorker:
    """
    Background worker that processes video generation jobs.
    
    Features:
    - Configurable worker pool
    - Real-time progress updates
    - Graceful shutdown
    - Error recovery
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.running_jobs: Dict[str, VeoGenerator] = {}
        self.job_queue: Queue = Queue()
        self.shutdown_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        
        # SSE subscribers (job_id -> list of queues)
        self.subscribers: Dict[str, List[Queue]] = {}
        self.subscribers_lock = threading.Lock()
    
    def start(self):
        """Start the worker"""
        if self.executor is not None:
            return
        
        self.shutdown_event.clear()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start job processor thread
        self.worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.worker_thread.start()
        
        print(f"[Worker] Started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the worker gracefully"""
        print("[Worker] Shutting down...")
        self.shutdown_event.set()
        
        # Cancel all running jobs
        for job_id, generator in list(self.running_jobs.items()):
            generator.cancel()
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        print("[Worker] Shutdown complete")
    
    def _process_jobs(self):
        """Main job processing loop"""
        while not self.shutdown_event.is_set():
            try:
                # Check for pending jobs in database
                self._check_pending_jobs()
                
                # Check for redo requests
                self._check_redo_queue()
                
                time.sleep(app_config.worker_poll_interval)
            except Exception as e:
                print(f"[Worker] Error in job processor: {e}")
                traceback.print_exc()
                time.sleep(5)
    
    def _check_redo_queue(self):
        """Check for clips that need redo and process them"""
        with get_db() as db:
            # Get clips queued for redo
            redo_clips = db.query(Clip).filter(
                Clip.status == ClipStatus.REDO_QUEUED.value
            ).order_by(Clip.id.asc()).limit(5).all()
            
            for clip in redo_clips:
                if len(self.running_jobs) >= self.max_workers:
                    break
                
                # Check if job is already being processed
                if clip.job_id not in self.running_jobs:
                    self._start_redo(clip.job_id, clip.id)
    
    def _start_redo(self, job_id: str, clip_id: int):
        """Start processing a single clip redo"""
        if self.executor is None:
            return
        
        self.executor.submit(self._run_redo, job_id, clip_id)
    
    def _check_pending_jobs(self):
        """Check for and start pending jobs"""
        if len(self.running_jobs) >= self.max_workers:
            return
        
        with get_db() as db:
            # Get pending jobs
            pending = db.query(Job).filter(
                Job.status == JobStatus.PENDING.value
            ).order_by(Job.created_at.asc()).limit(
                self.max_workers - len(self.running_jobs)
            ).all()
            
            for job in pending:
                if job.id not in self.running_jobs:
                    self._start_job(job.id)
    
    def _start_job(self, job_id: str):
        """Start processing a job"""
        if self.executor is None:
            return
        
        self.executor.submit(self._run_job, job_id)
    
    def _run_redo(self, job_id: str, clip_id: int):
        """Run a single clip redo"""
        generator = None
        
        try:
            with get_db() as db:
                clip = db.query(Clip).filter(Clip.id == clip_id).first()
                job = db.query(Job).filter(Job.id == job_id).first()
                
                if not clip or not job:
                    return
                
                # Update clip status
                clip.status = ClipStatus.GENERATING.value
                clip.started_at = datetime.utcnow()
                db.commit()
                
                add_job_log(
                    db, job_id, 
                    f"Starting redo for clip {clip.clip_index + 1} (attempt {clip.generation_attempt}/3)",
                    "INFO", "redo"
                )
                
                # Parse configuration
                config_data = json.loads(job.config_json)
                config = VideoConfig(
                    aspect_ratio=config_data.get("aspect_ratio", "9:16"),
                    resolution=config_data.get("resolution", "720p"),
                    duration=config_data.get("duration", "8"),
                    language=config_data.get("language", "English"),
                    use_interpolation=config_data.get("use_interpolation", True),
                    use_openai_prompt_tuning=config_data.get("use_openai_prompt_tuning", True),
                    use_frame_vision=config_data.get("use_frame_vision", True),
                    max_retries_per_clip=config_data.get("max_retries_per_clip", 5),
                    custom_prompt=config_data.get("custom_prompt", ""),
                    user_context=config_data.get("user_context", ""),
                    single_image_mode=config_data.get("single_image_mode", False),
                )
                
                # Parse API keys (with env fallback)
                api_keys = get_api_keys_with_fallback(job.api_keys_json)
                
                # Get images
                images_dir = Path(job.images_dir)
                output_dir = Path(job.output_dir)
                
                images = list_images(images_dir, config)
                if not images:
                    raise ValueError(f"No images found in {images_dir}")
                
                # Create generator
                generator = VeoGenerator(
                    config=config,
                    api_keys=api_keys,
                    openai_key=api_keys.openai_api_key,
                )
                
                # Set up callbacks
                def on_progress(clip_index, status, message, details):
                    self._handle_progress(job_id, clip_index, status, message, details)
                
                def on_error(error):
                    self._handle_error(job_id, error)
                
                generator.on_progress = on_progress
                generator.on_error = on_error
                
                # Find frames
                start_frame = None
                end_frame = None
                start_index = 0
                end_index = 0
                
                for i, img in enumerate(images):
                    if img.name == clip.start_frame:
                        start_frame = img
                        start_index = i
                    if clip.end_frame and img.name == clip.end_frame:
                        end_frame = img
                        end_index = i
                
                if not start_frame:
                    # Use first image as fallback
                    start_frame = images[0]
                    start_index = 0
                
                # For interpolation: use the stored end_frame, or same image if not set
                if config.use_interpolation:
                    if end_frame:
                        # Use the stored end frame
                        pass
                    elif clip.end_frame:
                        # end_frame name is stored but file not found - try to find it
                        for i, img in enumerate(images):
                            if img.name == clip.end_frame:
                                end_frame = img
                                end_index = i
                                break
                    
                    if not end_frame:
                        # No specific end frame - use same image for interpolation
                        end_frame = start_frame
                        end_index = start_index
                else:
                    # No interpolation - no end frame needed
                    end_frame = None
                    end_index = start_index
                
                # Initialize voice profile for consistency
                voice_id = generator.initialize_voice_profile(start_frame)
                add_job_log(db, job_id, f"Voice Profile for redo: {voice_id}", "INFO", "voice")
                
                # Determine prompt to use
                prompt_text = None
                redo_feedback = clip.redo_reason  # Get user's feedback
                
                if clip.use_logged_params and clip.prompt_text:
                    prompt_text = clip.prompt_text
                    add_job_log(db, job_id, f"Using logged parameters for redo", "INFO", "redo")
                else:
                    add_job_log(db, job_id, f"Using fresh parameters for redo", "INFO", "redo")
                
                if redo_feedback:
                    add_job_log(db, job_id, f"User feedback for redo: {redo_feedback}", "INFO", "redo")
                
                self._broadcast_event(job_id, {
                    "type": "redo_started",
                    "clip_id": clip_id,
                    "clip_index": clip.clip_index,
                    "attempt": clip.generation_attempt,
                    "use_logged_params": clip.use_logged_params,
                    "redo_feedback": redo_feedback,
                })
            
            # Generate clip (outside db context to avoid long transactions)
            # Use start_frame as scene_image for redo (it's the original scene image)
            result = generator.generate_single_clip(
                start_frame=start_frame,
                end_frame=end_frame,
                dialogue_line=clip.dialogue_text,
                dialogue_id=clip.dialogue_id,
                clip_index=clip.clip_index,
                output_dir=output_dir,
                images_list=images,
                current_end_index=end_index,
                scene_image=start_frame,  # For redo, use start_frame as scene image
                redo_feedback=redo_feedback,  # Pass user's feedback
            )
            
            # Update clip with result
            with get_db() as db:
                clip = db.query(Clip).filter(Clip.id == clip_id).first()
                
                if clip:
                    clip.completed_at = datetime.utcnow()
                    
                    if clip.started_at:
                        clip.duration_seconds = (clip.completed_at - clip.started_at).total_seconds()
                    
                    if result["success"]:
                        new_filename = result["output_path"].name if result["output_path"] else None
                        
                        # Add to versions history (avoid duplicates)
                        versions = json.loads(clip.versions_json) if clip.versions_json else []
                        existing_attempts = [v.get('attempt') for v in versions]
                        
                        if clip.generation_attempt not in existing_attempts:
                            versions.append({
                                "attempt": clip.generation_attempt,
                                "filename": new_filename,
                                "generated_at": datetime.utcnow().isoformat(),
                            })
                            clip.versions_json = json.dumps(versions)
                        
                        # Update current output and select new variant
                        clip.status = ClipStatus.COMPLETED.value
                        clip.output_filename = new_filename
                        clip.selected_variant = clip.generation_attempt
                        clip.prompt_text = result.get("prompt_text")
                        clip.approval_status = "pending_review"  # Reset to pending review
                        clip.error_code = None
                        clip.error_message = None
                        
                        if result.get("end_frame_used"):
                            clip.end_frame = result["end_frame_used"].name
                        
                        add_job_log(
                            db, job_id,
                            f"Redo completed for clip {clip.clip_index + 1} (attempt {clip.generation_attempt}/3)",
                            "INFO", "redo"
                        )
                    else:
                        clip.status = ClipStatus.FAILED.value
                        if result["error"]:
                            clip.error_code = result["error"].code.value
                            clip.error_message = result["error"].message
                        
                        add_job_log(
                            db, job_id,
                            f"Redo failed for clip {clip.clip_index + 1}: {result.get('error', 'Unknown error')}",
                            "ERROR", "redo"
                        )
                    
                    db.commit()
                
                self._broadcast_event(job_id, {
                    "type": "redo_completed",
                    "clip_id": clip_id,
                    "clip_index": clip.clip_index,
                    "success": result["success"],
                    "attempt": clip.generation_attempt,
                    "output": result["output_path"].name if result.get("output_path") else None,
                })
                
        except Exception as e:
            error = error_handler.classify_exception(e, {"job_id": job_id, "clip_id": clip_id})
            self._handle_error(job_id, error)
            
            with get_db() as db:
                clip = db.query(Clip).filter(Clip.id == clip_id).first()
                if clip:
                    clip.status = ClipStatus.FAILED.value
                    clip.error_code = error.code.value
                    clip.error_message = error.message
                    db.commit()
    
    def _run_job(self, job_id: str):
        """Run a single job"""
        generator = None
        
        try:
            with get_db() as db:
                job = db.query(Job).filter(Job.id == job_id).first()
                if not job:
                    return
                
                # Update status
                job.status = JobStatus.RUNNING.value
                job.started_at = datetime.utcnow()
                db.commit()
                
                add_job_log(db, job_id, "Job started", "INFO", "system")
                
                # Parse configuration
                config_data = json.loads(job.config_json)
                print(f"[Worker] Job {job_id[:8]}: Raw config_data = {config_data}")
                print(f"[Worker] Job {job_id[:8]}: Language from config = {config_data.get('language')}")
                
                config = VideoConfig(
                    aspect_ratio=config_data.get("aspect_ratio", "9:16"),
                    resolution=config_data.get("resolution", "720p"),
                    duration=config_data.get("duration", "8"),
                    language=config_data.get("language", "English"),
                    use_interpolation=config_data.get("use_interpolation", True),
                    use_openai_prompt_tuning=config_data.get("use_openai_prompt_tuning", True),
                    use_frame_vision=config_data.get("use_frame_vision", True),
                    max_retries_per_clip=config_data.get("max_retries_per_clip", 5),
                    custom_prompt=config_data.get("custom_prompt", ""),
                    user_context=config_data.get("user_context", ""),
                    single_image_mode=config_data.get("single_image_mode", False),
                )
                
                add_job_log(db, job_id, f"Language: {config.language}", "INFO", "config")
                
                # Parse API keys (with env fallback)
                api_keys = get_api_keys_with_fallback(job.api_keys_json)
                
                # Parse dialogue data (new format includes scenes)
                dialogue_raw = json.loads(job.dialogue_json)
                
                # Handle both old format (list) and new format (dict with lines/scenes)
                if isinstance(dialogue_raw, list):
                    # Old format: just a list of lines
                    dialogue_data = dialogue_raw
                    scenes_data = None
                    last_frame_index = None
                else:
                    # New format: {lines: [...], scenes: [...], last_frame_index: ...}
                    dialogue_data = dialogue_raw.get("lines", [])
                    scenes_data = dialogue_raw.get("scenes", None)
                    last_frame_index = dialogue_raw.get("last_frame_index", None)
                
                # Store scenes data for processing
                storyboard_mode = config_data.get("storyboard_mode", False)
                print(f"[Worker] Storyboard mode: {storyboard_mode}", flush=True)
                if scenes_data:
                    print(f"[Worker] Scenes: {json.dumps(scenes_data, indent=2)}", flush=True)
                if last_frame_index is not None:
                    print(f"[Worker] Last frame index: {last_frame_index}", flush=True)
                
                # Get images
                images_dir = Path(job.images_dir)
                output_dir = Path(job.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                images = list_images(images_dir, config)
                if not images:
                    raise ValueError(f"No images found in {images_dir}")
                
                # Log image order for debugging
                print(f"[Worker] Loaded {len(images)} images in order:", flush=True)
                for idx, img in enumerate(images):
                    print(f"  [{idx}] {img.name}", flush=True)
                
                # Create generator
                generator = VeoGenerator(
                    config=config,
                    api_keys=api_keys,
                    openai_key=api_keys.openai_api_key,
                )
                
                # Set up callbacks
                def on_progress(clip_index, status, message, details):
                    self._handle_progress(job_id, clip_index, status, message, details)
                
                def on_error(error: VeoError):
                    self._handle_error(job_id, error)
                
                generator.on_progress = on_progress
                generator.on_error = on_error
                
                # Initialize voice profile ONCE for entire job (use first image as reference)
                voice_id = generator.initialize_voice_profile(images[0])
                add_job_log(db, job_id, f"Voice Profile initialized: {voice_id}", "INFO", "voice")
                
                self.running_jobs[job_id] = generator
            
            # Process clips (pass scenes_data for storyboard mode)
            self._process_clips(job_id, generator, dialogue_data, images, output_dir, scenes_data, last_frame_index)
            
        except Exception as e:
            error = error_handler.classify_exception(e, {"job_id": job_id})
            self._handle_error(job_id, error)
            
            # Only mark as failed if no clips succeeded
            with get_db() as db:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
                    successful = sum(1 for c in clips if c.status == ClipStatus.COMPLETED.value)
                    
                    if successful == 0:
                        # No clips succeeded - mark job as failed
                        job.status = JobStatus.FAILED.value
                    else:
                        # Some clips succeeded - mark as completed with failures
                        job.status = JobStatus.COMPLETED.value
                    
                    job.completed_at = datetime.utcnow()
                    db.commit()
                    
                    add_job_log(
                        db, job_id, 
                        f"Job ended with error: {error.message}", 
                        "ERROR", "system",
                        details=error.to_dict()
                    )
        
        finally:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def _process_clips(
        self,
        job_id: str,
        generator: VeoGenerator,
        dialogue_data: List[Dict],
        images: List[Path],
        output_dir: Path,
        scenes_data: Optional[List[Dict]] = None,
        last_frame_index: Optional[int] = None,
    ):
        """Process all clips for a job - with parallel generation support and scene-aware sequencing"""
        from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
        import subprocess
        
        # Check for single image mode
        single_image_mode = getattr(generator.config, 'single_image_mode', False) or len(images) == 1
        
        images_dir_str = str(images[0].parent)
        
        total_clips = len(dialogue_data)
        completed = 0
        failed = 0
        skipped = 0
        
        # Get parallel clip count from config (default 1 for memory efficiency on free tier)
        parallel_clips = getattr(generator.config, 'parallel_clips', 1)
        
        # Key exhaustion tracking
        no_keys_retries = 0
        max_no_keys_retries = 3
        no_keys_wait_seconds = 300  # 5 minutes
        
        # Log last frame
        if last_frame_index is not None:
            print(f"[Worker] Last frame index set: {last_frame_index} ({images[last_frame_index].name if last_frame_index < len(images) else 'INVALID'})", flush=True)
        
        # === BUILD SCENE-AWARE CLIP STRUCTURE ===
        num_images = len(images)
        use_interpolation = getattr(generator.config, 'use_interpolation', True)
        
        print(f"[Worker] Processing {total_clips} clips with {num_images} images", flush=True)
        print(f"[Worker] Scenes data: {scenes_data}", flush=True)
        
        # Build clip info with scene awareness
        clip_info = []  # List of dicts with all clip metadata
        
        for i, line_data in enumerate(dialogue_data):
            info = {
                "index": i,
                "text": line_data["text"],
                "dialogue_id": line_data["id"],
                "image_idx": line_data.get("start_image_idx", i % num_images) if not single_image_mode else 0,
                "scene_index": line_data.get("scene_index", 0),
                "clip_mode": line_data.get("clip_mode", "blend"),  # 'blend' | 'continue' | 'fresh'
                "scene_transition": line_data.get("scene_transition"),  # 'blend' | 'cut' | None
                "requires_previous": False,  # Will be set below
                "start_frame": None,  # Will be set or calculated
                "end_frame": None,    # Will be set or calculated
            }
            
            # Determine if this clip requires the previous clip to complete first
            # This happens when clip_mode is 'continue' AND it's not the first clip in its scene
            if info["clip_mode"] == "continue" and i > 0:
                prev_scene = dialogue_data[i-1].get("scene_index", 0)
                if prev_scene == info["scene_index"]:
                    # Same scene, continue mode - must wait for previous clip
                    info["requires_previous"] = True
            
            clip_info.append(info)
            print(f"[Worker] Clip {i}: scene={info['scene_index']}, mode={info['clip_mode']}, requires_prev={info['requires_previous']}", flush=True)
        
        # Calculate initial frame assignments
        # 
        # FRAME ASSIGNMENT LOGIC:
        # 
        # For each clip, we need to determine:
        # 1. START FRAME: Where the clip begins
        # 2. END FRAME: Where the clip ends (can be None for no interpolation)
        #
        # The logic depends on clip_mode:
        #
        # BLEND mode (standard):
        #   - Start: assigned image
        #   - End: depends on NEXT clip (allows smooth transitions)
        #
        # CONTINUE mode:
        #   - Start: extracted from previous clip's last frame (set at runtime)
        #   - End: depends on NEXT clip (allows smooth transitions)
        #
        # FRESH mode:
        #   - Start: always original image
        #   - End: NONE (completely standalone clips, no interpolation)
        #
        # END FRAME determination (for BLEND and CONTINUE modes only):
        #   - If LAST clip of video:
        #     - If Last Frame defined: use Last Frame
        #     - Else: None (no interpolation)
        #   - If NEXT clip is in SAME scene:
        #     - None (no end frame, natural continuation)
        #   - If NEXT clip is in DIFFERENT scene:
        #     - If transition = "blend": next scene's image
        #     - If transition = "cut": None
        #
        with get_db() as db:
            for i, info in enumerate(clip_info):
                start_idx = info["image_idx"]
                clip_mode = info["clip_mode"]
                scene_transition = info["scene_transition"]
                scene_index = info["scene_index"]
                
                # Default start: our assigned image
                actual_start_idx = start_idx
                
                # Determine END FRAME based on what comes AFTER this clip
                use_end_frame = False
                actual_end_idx = None
                end_frame_reason = ""
                
                # SINGLE IMAGE MODE: Always use same image as end frame for interpolation
                if single_image_mode and generator.config.use_interpolation:
                    use_end_frame = True
                    actual_end_idx = start_idx  # Same image for smoother motion
                    end_frame_reason = "single image mode, same frame for interpolation"
                else:
                    is_last_clip = (i == len(clip_info) - 1)
                    
                    # Check if we're in auto-cycle mode (no explicit scenes defined)
                    auto_cycle_mode = scenes_data is None or len(scenes_data) == 0
                    
                    # Track if scene transition already determined the end frame
                    scene_transition_handled = False
                    
                    if not is_last_clip:
                        next_info = clip_info[i + 1]
                        next_scene = next_info["scene_index"]
                        next_image_idx = next_info["image_idx"]
                        
                        if auto_cycle_mode:
                            # AUTO-CYCLE MODE: Check if next clip uses a different image
                            if next_image_idx != start_idx:
                                # Different image - blend to it
                                use_end_frame = True
                                actual_end_idx = next_image_idx
                                end_frame_reason = f"auto-cycle: blend to next image {next_image_idx + 1}"
                                scene_transition_handled = True
                        elif next_scene != scene_index:
                            # STORYBOARD MODE: Next clip is in DIFFERENT scene
                            next_transition = next_info["scene_transition"]
                            
                            # If transition is "blend" (or None), use next scene's image (scene transition priority)
                            if next_transition != "cut":
                                use_end_frame = True
                                actual_end_idx = next_info["image_idx"]
                                end_frame_reason = f"scene transition to scene {next_scene} (next scene priority)"
                                scene_transition_handled = True
                            # If transition is "cut", fall through to clip_mode logic below
                    
                    # Apply clip_mode logic if:
                    # - Scene transition didn't handle it (same scene, or different scene with "cut")
                    # - Or it's the last clip
                    if not scene_transition_handled:
                        if is_last_clip and last_frame_index is not None and last_frame_index < len(images):
                            # LAST CLIP with explicit end frame set
                            use_end_frame = True
                            actual_end_idx = last_frame_index
                            end_frame_reason = f"last clip with explicit end frame (image {last_frame_index + 1})"
                        elif clip_mode == "blend":
                            # BLEND mode: Use same image as end frame for interpolation
                            use_end_frame = True
                            actual_end_idx = start_idx  # Same image
                            if is_last_clip:
                                end_frame_reason = "last clip in blend mode: same image for final interpolation"
                            else:
                                end_frame_reason = "blend mode: same image for interpolation"
                        else:
                            # FRESH or CONTINUE mode: No end frame
                            use_end_frame = False
                            if is_last_clip:
                                end_frame_reason = "last clip, no end frame (ends naturally)"
                            elif clip_mode == "fresh":
                                end_frame_reason = "fresh mode, no end frame"
                            else:
                                end_frame_reason = "continue mode, no end frame"
                
                # Set frame names
                start_frame_name = images[actual_start_idx].name
                
                if use_end_frame and actual_end_idx is not None:
                    end_frame_name = images[actual_end_idx].name
                else:
                    end_frame_name = None
                    actual_end_idx = actual_start_idx  # For compatibility, but won't be used
                
                info["start_frame"] = start_frame_name
                info["end_frame"] = end_frame_name
                info["start_idx"] = actual_start_idx
                info["end_idx"] = actual_end_idx if use_end_frame else None
                info["use_end_frame"] = use_end_frame
                
                print(f"[Worker] Clip {i}: {start_frame_name} → {end_frame_name if end_frame_name else 'NONE'} (mode={clip_mode}, reason={end_frame_reason})", flush=True)
                
                # Determine initial status
                # For "continue" mode clips (except first in scene), set to WAITING_APPROVAL
                initial_status = ClipStatus.PENDING.value
                if info["requires_previous"]:
                    initial_status = ClipStatus.WAITING_APPROVAL.value
                    print(f"[Worker] Clip {i}: Set to WAITING_APPROVAL (requires previous clip approval)", flush=True)
                
                # Create clip record
                clip = Clip(
                    job_id=job_id,
                    clip_index=i,
                    dialogue_id=info["dialogue_id"],
                    dialogue_text=info["text"],
                    status=initial_status,
                    start_frame=start_frame_name,
                    end_frame=end_frame_name,
                )
                db.add(clip)
            
            db.commit()
        
        # Build clip_frames list for processing
        clip_frames = []
        for i, info in enumerate(clip_info):
            start_frame = images[info["start_idx"]]
            
            # Only set end_frame if this clip should use interpolation
            if info.get("use_end_frame") and info.get("end_idx") is not None:
                end_frame = images[info["end_idx"]]
            else:
                end_frame = None
            
            clip_frames.append({
                "start_index": info["start_idx"],
                "start_frame": start_frame,
                "end_index": info["end_idx"],
                "end_frame": end_frame,
                "clip_mode": info["clip_mode"],
                "requires_previous": info["requires_previous"],
                "scene_index": info["scene_index"],
                "original_scene_idx": info["image_idx"],  # Original scene image index for subject description
            })
        
        # Log complete frame assignment summary
        print(f"\n{'='*70}", flush=True)
        print(f"[Worker] FRAME ASSIGNMENT SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Total clips: {len(clip_frames)}", flush=True)
        print(f"Last Frame Index: {last_frame_index}", flush=True)
        print(f"", flush=True)
        for i, cf in enumerate(clip_frames):
            mode = cf["clip_mode"]
            req_prev = cf["requires_previous"]
            start = cf["start_frame"].name if hasattr(cf["start_frame"], 'name') else str(cf["start_frame"])
            end = cf["end_frame"].name if cf["end_frame"] and hasattr(cf["end_frame"], 'name') else ("NONE" if cf["end_frame"] is None else str(cf["end_frame"]))
            status = "WAITING_APPROVAL" if req_prev else "PENDING"
            
            print(f"  Clip {i}: [{mode.upper()}] {start} → {end}", flush=True)
            print(f"           requires_previous={req_prev}, status={status}", flush=True)
            if mode == "continue":
                if req_prev:
                    print(f"           → Will extract start frame from clip {i-1} at runtime", flush=True)
                else:
                    print(f"           → First of scene, will use original image", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        # Track completed AND APPROVED clips for 'continue' mode frame extraction
        approved_clip_videos = {}  # clip_index -> video_path (only approved ones)
        completed_clip_videos = {}  # clip_index -> video_path (all completed, for tracking)
        
        # Track subject descriptions per scene for continue mode consistency
        scene_subject_descriptions = {}  # scene_index -> subject description (generated on first clip)
        
        def get_or_generate_subject_description(scene_index: int, scene_image_path: Path) -> str:
            """Get cached subject description or generate for first clip of scene"""
            if scene_index in scene_subject_descriptions:
                return scene_subject_descriptions[scene_index]
            
            # Generate subject description from scene's original image
            print(f"[Worker] Generating subject description for scene {scene_index} from {scene_image_path.name}", flush=True)
            openai_key = get_openai_key_from_env()
            description = describe_subject_for_continuity(str(scene_image_path), openai_key)
            
            if description:
                scene_subject_descriptions[scene_index] = description
                print(f"[Worker] Scene {scene_index} subject: '{description}'", flush=True)
                
                # Log to database
                with get_db() as db:
                    add_job_log(
                        db, job_id,
                        f"📷 Scene {scene_index + 1} subject description: {description}",
                        "INFO", "prompt"
                    )
            else:
                scene_subject_descriptions[scene_index] = ""
                print(f"[Worker] Scene {scene_index}: No subject description generated", flush=True)
            
            return scene_subject_descriptions.get(scene_index, "")
        
        # Queue of pending clip indices (only PENDING status, not WAITING_APPROVAL)
        pending_clips = [i for i, info in enumerate(clip_info) if not info["requires_previous"]]
        waiting_clips = [i for i, info in enumerate(clip_info) if info["requires_previous"]]
        
        print(f"[Worker] Initial queue: {len(pending_clips)} pending, {len(waiting_clips)} waiting for approval", flush=True)
        
        def check_keys_available():
            """Check if any API keys are available"""
            available = generator.api_keys.get_available_key_count()
            return available > 0
        
        def send_no_keys_alert(job_id: str, retry_count: int):
            """Alert admin that keys are exhausted"""
            alert_msg = f"🚨 API KEYS EXHAUSTED - Job {job_id[:8]} paused (retry {retry_count}/{max_no_keys_retries})"
            print(f"\n{'='*60}", flush=True)
            print(alert_msg, flush=True)
            print(f"{'='*60}\n", flush=True)
            
            # Log to database
            with get_db() as db:
                add_job_log(
                    db, job_id,
                    f"⚠️ All API keys exhausted! Waiting {no_keys_wait_seconds}s before retry {retry_count}/{max_no_keys_retries}",
                    "WARNING", "system",
                    details={"keys_status": generator.api_keys.get_status()}
                )
            
            # Broadcast to UI
            self._broadcast_event(job_id, {
                "type": "keys_exhausted",
                "retry_count": retry_count,
                "max_retries": max_no_keys_retries,
                "wait_seconds": no_keys_wait_seconds,
                "message": f"All API keys exhausted. Waiting {no_keys_wait_seconds//60} minutes... (attempt {retry_count}/{max_no_keys_retries})"
            })
            
            # Send email alert
            total_keys = len(generator.api_keys.gemini_keys) if hasattr(generator.api_keys, 'gemini_keys') else 0
            send_key_alert_email("no_keys", 0, total_keys, job_id)
        
        def check_redo_clips():
            """Check for clips queued for redo and return their indices"""
            redo_indices = []
            with get_db() as db:
                redo_clips = db.query(Clip).filter(
                    Clip.job_id == job_id,
                    Clip.status == ClipStatus.REDO_QUEUED.value
                ).all()
                
                for clip in redo_clips:
                    redo_indices.append(clip.clip_index)
                    # Update status to PENDING so it gets processed
                    clip.status = ClipStatus.PENDING.value
                    print(f"[Worker] Clip {clip.clip_index}: REDO_QUEUED → PENDING (integrated into main loop)", flush=True)
                
                if redo_clips:
                    db.commit()
                    add_job_log(
                        db, job_id,
                        f"🔄 {len(redo_clips)} clip(s) queued for redo, adding to processing queue",
                        "INFO", "redo"
                    )
            
            return redo_indices
        
        def extract_frame_from_video(video_path: Path, frame_offset: int = -8) -> Optional[Path]:
            """Extract a frame from video. frame_offset=-8 means 8 frames from the end."""
            try:
                # Use same ffmpeg/ffprobe config as video_processor.py
                # Also check ImageIO_FFMPEG_EXE as fallback (used in some setups)
                ffmpeg_exe = os.environ.get("FFMPEG_BIN") or os.environ.get("ImageIO_FFMPEG_EXE") or "ffmpeg"
                ffprobe_exe = os.environ.get("FFPROBE_BIN", "ffprobe")
                
                # If we have a custom ffmpeg path but not ffprobe, derive ffprobe from ffmpeg path
                if ffmpeg_exe not in ("ffmpeg", None) and ffprobe_exe == "ffprobe":
                    ffmpeg_path = Path(ffmpeg_exe)
                    if ffmpeg_path.exists():
                        # ffprobe should be in same directory
                        probe_name = "ffprobe.exe" if os.name == 'nt' else "ffprobe"
                        derived_probe = ffmpeg_path.parent / probe_name
                        if derived_probe.exists():
                            ffprobe_exe = str(derived_probe)
                
                print(f"[Worker] Using ffprobe: {ffprobe_exe}", flush=True)
                print(f"[Worker] Using ffmpeg: {ffmpeg_exe}", flush=True)
                
                # Get video duration
                probe_cmd = [
                    ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=duration,r_frame_rate",
                    "-of", "csv=p=0", str(video_path)
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.returncode != 0:
                    print(f"[Worker] ffprobe failed: {probe_result.stderr}", flush=True)
                    return None
                
                # Parse duration and fps
                parts = probe_result.stdout.strip().split(',')
                if len(parts) < 2:
                    print(f"[Worker] Could not parse ffprobe output: {probe_result.stdout}", flush=True)
                    return None
                    
                fps_str = parts[0]
                duration_str = parts[1] if len(parts) > 1 else "8"
                
                # Calculate fps from fraction (e.g., "30000/1001" or "30/1")
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str) if fps_str else 30.0
                
                duration = float(duration_str) if duration_str else 8.0
                
                # Calculate timestamp for frame_offset from end
                # frame_offset = -8 means 8 frames before end
                frames_from_end = abs(frame_offset)
                seconds_from_end = frames_from_end / fps
                timestamp = max(0, duration - seconds_from_end)
                
                print(f"[Worker] Extracting frame at {timestamp:.3f}s (fps={fps:.2f}, duration={duration:.2f}s, offset={frame_offset})", flush=True)
                
                # Extract frame
                output_frame = video_path.parent / f"{video_path.stem}_lastframe.jpg"
                extract_cmd = [
                    ffmpeg_exe, "-y", "-ss", str(timestamp), "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "2", str(output_frame)
                ]
                extract_result = subprocess.run(extract_cmd, capture_output=True, text=True)
                
                if extract_result.returncode == 0 and output_frame.exists():
                    print(f"[Worker] Extracted frame to {output_frame.name}", flush=True)
                    return output_frame
                else:
                    print(f"[Worker] ffmpeg frame extraction failed: {extract_result.stderr}", flush=True)
                    return None
                    
            except Exception as e:
                print(f"[Worker] Frame extraction error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return None
        
        def enhance_frame_with_nano_banana(frame_path: Path, original_scene_image: Optional[Path] = None) -> Optional[Path]:
            """
            Enhance an extracted frame using Nano Banana Pro (Gemini 3 Pro Image).
            Upscales and improves quality of the image.
            
            If original_scene_image is provided, also corrects facial features to match
            the original person (fixes AI drift in facial appearance).
            """
            try:
                import google.genai as genai
                from google.genai import types
                import base64
                
                # Get API key from the keys config
                api_keys = get_gemini_keys_from_env()
                if not api_keys:
                    print("[Worker] No Gemini API keys available for Nano Banana Pro enhancement", flush=True)
                    return frame_path  # Return original if no API key
                
                # Use first available key
                api_key = api_keys[0]
                client = genai.Client(api_key=api_key)
                
                # Read the extracted frame
                with open(frame_path, 'rb') as f:
                    frame_bytes = f.read()
                
                # Determine mime type
                suffix = frame_path.suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.webp': 'image/webp'
                }.get(suffix, 'image/jpeg')
                
                print(f"[Worker] Enhancing frame with Nano Banana Pro: {frame_path.name}", flush=True)
                
                # Build the prompt parts
                parts = [
                    types.Part.from_bytes(data=frame_bytes, mime_type=mime_type),
                ]
                
                # If we have original scene image, include it for facial consistency
                if original_scene_image and original_scene_image.exists():
                    print(f"[Worker] Including original scene image for facial consistency: {original_scene_image.name}", flush=True)
                    
                    with open(original_scene_image, 'rb') as f:
                        original_bytes = f.read()
                    
                    original_suffix = original_scene_image.suffix.lower()
                    original_mime = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.webp': 'image/webp'
                    }.get(original_suffix, 'image/jpeg')
                    
                    parts.append(types.Part.from_bytes(data=original_bytes, mime_type=original_mime))
                    
                    prompt_text = (
                        "The first image is an extracted video frame. The second image shows the original person. "
                        "Enhance the first image while correcting the facial features to match the original person in the second image. "
                        "This is NOT a face swap - it's the same person, but the AI video generation may have slightly altered their appearance. "
                        "Correct any facial drift: restore accurate facial structure, skin tone, eye shape, nose shape, and other features to match the original. "
                        "Also upscale to higher resolution, reduce compression artifacts, and improve overall image quality. "
                        "Keep the exact pose, expression, lighting, background, and composition from the first image - only correct the facial features and enhance quality."
                    )
                else:
                    # No reference image - just enhance quality
                    prompt_text = (
                        "Upscale this image to higher resolution while preserving all details. "
                        "Enhance the image quality, reduce any compression artifacts, "
                        "and improve sharpness and clarity. Keep the exact same content, "
                        "colors, and composition - only improve the quality."
                    )
                
                parts.append(types.Part.from_text(text=prompt_text))
                
                contents = [
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ]
                
                # Configure for image output
                config = types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    temperature=0.2  # Low temperature for faithful reproduction
                )
                
                # Generate enhanced image with retry for 503 errors
                max_retries = 3
                response = None
                
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model="gemini-3-pro-image-preview",  # Nano Banana Pro
                            contents=contents,
                            config=config
                        )
                        break  # Success, exit retry loop
                    except Exception as api_error:
                        error_str = str(api_error)
                        if "503" in error_str or "overloaded" in error_str.lower():
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                                print(f"[Worker] Nano Banana Pro overloaded, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})", flush=True)
                                time.sleep(wait_time)
                            else:
                                print(f"[Worker] Nano Banana Pro still overloaded after {max_retries} attempts, using original frame", flush=True)
                                return frame_path
                        else:
                            raise  # Re-raise non-503 errors
                
                if response is None:
                    print("[Worker] Nano Banana Pro: No response received, using original frame", flush=True)
                    return frame_path
                
                # Extract enhanced image from response
                enhanced_path = frame_path.parent / f"{frame_path.stem}_enhanced.png"
                
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Save enhanced image
                            with open(enhanced_path, 'wb') as f:
                                f.write(part.inline_data.data)
                            print(f"[Worker] Enhanced frame saved to {enhanced_path.name}", flush=True)
                            return enhanced_path
                
                print("[Worker] Nano Banana Pro did not return an image, using original frame", flush=True)
                return frame_path
                
            except ImportError as e:
                print(f"[Worker] google-genai SDK not available for enhancement: {e}", flush=True)
                return frame_path
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "overloaded" in error_str.lower():
                    print(f"[Worker] Frame enhancement skipped (Nano Banana Pro overloaded), using original frame", flush=True)
                else:
                    print(f"[Worker] Frame enhancement error: {e}", flush=True)
                return frame_path  # Return original on error
        
        def process_single_clip(clip_index: int):
            """Process a single clip - runs in thread"""
            if generator.cancelled:
                return {"clip_index": clip_index, "success": False, "skipped": True}
            
            line_data = dialogue_data[clip_index]
            dialogue_id = line_data["id"]
            dialogue_text = line_data["text"]
            frames = clip_frames[clip_index]
            
            start_frame = frames["start_frame"]
            end_frame = frames["end_frame"]  # Can be None if no interpolation needed
            start_index = frames["start_index"]
            end_index = frames["end_index"]  # Can be None if no interpolation needed
            clip_mode = frames.get("clip_mode", "blend")
            requires_previous = frames.get("requires_previous", False)
            scene_index = frames.get("scene_index", 0)
            original_scene_idx = frames.get("original_scene_idx", 0)
            
            # For CONTINUE mode clips, inject subject description for visual consistency
            if clip_mode == "continue":
                # Get the original scene image (not extracted frame)
                scene_image_for_desc = images[original_scene_idx] if original_scene_idx < len(images) else images[0]
                subject_desc = get_or_generate_subject_description(scene_index, scene_image_for_desc)
                
                if subject_desc and requires_previous:
                    # Prepend subject description to dialogue for continue clips
                    # Format: "The [subject description] [dialogue]"
                    dialogue_text = f"{subject_desc} {dialogue_text}"
                    print(f"[Worker] Clip {clip_index}: Injected subject description for continuity", flush=True)
            
            # Store the original scene image for facial consistency correction
            original_scene_image = frames["start_frame"]  # The original image for this scene
            
            # Handle "continue" mode - use extracted frame from APPROVED previous clip
            # ONLY if requires_previous is True (meaning previous clip is in SAME scene)
            if clip_mode == "continue" and requires_previous and clip_index > 0:
                prev_idx = clip_index - 1
                prev_video = approved_clip_videos.get(prev_idx)
                print(f"[Worker] Clip {clip_index}: Continue mode check - prev_idx={prev_idx}, approved_clip_videos keys={list(approved_clip_videos.keys())}", flush=True)
                print(f"[Worker] Clip {clip_index}: prev_video={prev_video}", flush=True)
                if prev_video:
                    video_exists = Path(prev_video).exists()
                    print(f"[Worker] Clip {clip_index}: Video exists at path? {video_exists}", flush=True)
                    if video_exists:
                        extracted = extract_frame_from_video(Path(prev_video), frame_offset=-8)
                        if extracted:
                            # Enhance the extracted frame using Nano Banana Pro
                            # Pass the original scene image for facial consistency correction
                            enhanced = enhance_frame_with_nano_banana(extracted, original_scene_image)
                            start_frame = enhanced
                            print(f"[Worker] Clip {clip_index}: Using {'enhanced' if enhanced != extracted else 'extracted'} frame from APPROVED clip {prev_idx}", flush=True)
                        else:
                            print(f"[Worker] Clip {clip_index}: Frame extraction failed, using original image", flush=True)
                    else:
                        print(f"[Worker] Clip {clip_index}: Video file does not exist at {prev_video}, using original image", flush=True)
                else:
                    print(f"[Worker] Clip {clip_index}: Approved previous clip video not found (prev_video is None), using original image", flush=True)
            elif clip_mode == "continue" and not requires_previous:
                # First clip of scene in Continue mode - use original image
                print(f"[Worker] Clip {clip_index}: Continue mode but first of scene, using original image", flush=True)
            
            # Get frame names for logging/database (handle both Path objects and strings)
            def get_frame_name(frame):
                if frame is None:
                    return None
                if hasattr(frame, 'name'):
                    return frame.name
                if hasattr(frame, 'stem'):
                    return Path(frame).name
                return str(frame).split('/')[-1] if '/' in str(frame) else str(frame)
            
            start_frame_name = get_frame_name(start_frame)
            end_frame_name = get_frame_name(end_frame) if end_frame else None
            
            # Update clip status to generating
            with get_db() as db:
                clip = db.query(Clip).filter(
                    Clip.job_id == job_id,
                    Clip.clip_index == clip_index
                ).first()
                
                if clip:
                    clip.status = ClipStatus.GENERATING.value
                    clip.started_at = datetime.utcnow()
                    clip.start_frame = start_frame_name
                    clip.end_frame = end_frame_name
                    db.commit()
            
            # Log exact frame assignment for debugging
            print(f"[Worker] CLIP {clip_index} FRAME ASSIGNMENT:", flush=True)
            print(f"  - start_frame: {start_frame_name} (mode={clip_mode})", flush=True)
            print(f"  - end_frame: {end_frame_name if end_frame_name else 'NONE (no interpolation)'}", flush=True)
            
            self._broadcast_event(job_id, {
                "type": "clip_started",
                "clip_index": clip_index,
                "dialogue_id": dialogue_id,
                "start_frame": start_frame_name,
                "end_frame": end_frame_name,
            })
            
            # Check if start frame is blacklisted (only for Path objects, not extracted frames)
            if hasattr(start_frame, 'exists') and start_frame in generator.blacklist:
                result = self._get_next_clean_start(generator, images, start_index)
                if result:
                    start_index, start_frame = result
                else:
                    with get_db() as db:
                        clip = db.query(Clip).filter(
                            Clip.job_id == job_id,
                            Clip.clip_index == clip_index
                        ).first()
                        if clip:
                            clip.status = ClipStatus.FAILED.value
                            clip.error_code = "ALL_IMAGES_BLACKLISTED"
                            clip.error_message = "No clean images available"
                            db.commit()
                    return {"clip_index": clip_index, "success": False, "failed": True}
            
            # Generate clip
            try:
                print(f"[Worker] Generating clip {clip_index + 1}/{total_clips} in parallel...", flush=True)
                
                # Get the original scene image for prompt analysis
                # This is the uploaded image for this scene (not the extracted frame in CONTINUE mode)
                scene_image = images[start_index] if start_index < len(images) else images[0]
                
                # Calculate dynamic duration for LAST CLIP
                # Last clip picks from 4, 6, or 8 seconds based on expected speech duration
                override_duration = None
                is_last_clip = clip_index == total_clips - 1
                
                if is_last_clip:
                    # Estimate speech duration based on word count
                    word_count = len(dialogue_text.split())
                    language = generator.config.language if hasattr(generator.config, 'language') else 'English'
                    
                    # Words per second by language (approximate)
                    wps_map = {
                        "English": 2.5, "Italian": 2.8, "Spanish": 2.8, "French": 2.5, "German": 2.2,
                        "Portuguese": 2.7, "Dutch": 2.4, "Polish": 2.3, "Russian": 2.2,
                        "Japanese": 3.0, "Korean": 3.0, "Chinese": 3.2, "Arabic": 2.3, "Hindi": 2.6, "Turkish": 2.5
                    }
                    wps = wps_map.get(language, 2.5)
                    estimated_duration = word_count / wps
                    
                    # Pick the duration slightly above the estimated (4, 6, or 8 seconds)
                    if estimated_duration <= 3.5:
                        override_duration = "4"
                    elif estimated_duration <= 5.5:
                        override_duration = "6"
                    else:
                        override_duration = "8"
                    
                    print(f"[Worker] LAST CLIP: {word_count} words, ~{estimated_duration:.1f}s speech → using {override_duration}s duration", flush=True)
                
                # CRITICAL: Log the actual start_frame being used for generation
                actual_start_frame_name = start_frame.name if hasattr(start_frame, 'name') else str(start_frame)
                print(f"[Worker] >>> GENERATING with start_frame: {actual_start_frame_name}", flush=True)
                if clip_mode == "continue" and requires_previous:
                    print(f"[Worker] >>> (This should be an EXTRACTED frame from previous clip, NOT the scene image)", flush=True)
                
                result = generator.generate_single_clip(
                    start_frame=start_frame,
                    end_frame=end_frame,  # Can be None
                    dialogue_line=dialogue_text,
                    dialogue_id=dialogue_id,
                    clip_index=clip_index,
                    output_dir=output_dir,
                    images_list=images,
                    current_end_index=end_index if end_index is not None else start_index,
                    scene_image=scene_image,  # Original scene image for prompt analysis
                    override_duration=override_duration,  # Dynamic duration for last clip
                )
                
                # Log the FULL prompt that was sent to Veo (no truncation)
                if result.get("prompt_text"):
                    full_prompt = result["prompt_text"]
                    with get_db() as db:
                        add_job_log(
                            db, job_id,
                            f"📝 FULL PROMPT for clip {clip_index + 1}:\n{full_prompt}",
                            "INFO", "prompt"
                        )
                
                # Check if failed due to no keys
                if not result["success"]:
                    error = result.get("error")
                    if error and hasattr(error, 'code'):
                        if error.code.value in ["API_KEY_INVALID", "API_QUOTA_EXCEEDED", "RATE_LIMIT_429"]:
                            return {"clip_index": clip_index, "success": False, "no_keys": True, "result": result}
                
            except Exception as gen_error:
                print(f"[Worker] Clip {clip_index} CRASHED: {type(gen_error).__name__}: {str(gen_error)[:200]}", flush=True)
                result = {
                    "success": False,
                    "error": gen_error,
                    "output_path": None,
                    "end_frame_used": None,
                    "end_index": end_index,
                }
            
            # Update clip record
            try:
                with get_db() as db:
                    clip = db.query(Clip).filter(
                        Clip.job_id == job_id,
                        Clip.clip_index == clip_index
                    ).first()
                    
                    if clip:
                        clip.completed_at = datetime.utcnow()
                        
                        if clip.started_at:
                            clip.duration_seconds = (
                                clip.completed_at - clip.started_at
                            ).total_seconds()
                        
                        if result["success"]:
                            new_filename = result["output_path"].name if result.get("output_path") else None
                            
                            versions = [{
                                "attempt": 1,
                                "filename": new_filename,
                                "generated_at": datetime.utcnow().isoformat(),
                            }]
                            clip.versions_json = json.dumps(versions)
                            clip.selected_variant = 1
                            
                            clip.status = ClipStatus.COMPLETED.value
                            clip.approval_status = "pending_review"
                            clip.output_filename = new_filename
                            clip.prompt_text = result.get("prompt_text")
                        else:
                            clip.status = ClipStatus.FAILED.value
                            error_obj = result.get("error")
                            if error_obj:
                                clip.error_code = error_obj.code.value if hasattr(error_obj, 'code') else "UNKNOWN"
                                clip.error_message = str(error_obj.message if hasattr(error_obj, 'message') else error_obj)[:500]
                        
                        db.commit()
                    
                    # Save generation log if successful
                    if result.get("success") and result.get("output_path"):
                        # Safely get frame names (handle None cases for single-image mode)
                        start_frame_name = start_frame.name if start_frame and hasattr(start_frame, 'name') else str(start_frame) if start_frame else "unknown"
                        
                        # For end frame: prefer result's end_frame_used, then end_frame, or fall back to start_frame
                        if result.get("end_frame_used") and hasattr(result["end_frame_used"], 'name'):
                            end_frame_name = result["end_frame_used"].name
                        elif end_frame and hasattr(end_frame, 'name'):
                            end_frame_name = end_frame.name
                        else:
                            end_frame_name = start_frame_name  # Single image mode fallback
                        
                        gen_log = GenerationLog(
                            job_id=job_id,
                            video_id=dialogue_id,
                            images_dir=images_dir_str,
                            start_frame=start_frame_name,
                            end_frame=end_frame_name,
                            dialogue_line=dialogue_text,
                            language=generator.config.language,
                            prompt_text=result.get("prompt_text", ""),
                            video_filename=result["output_path"].name,
                            aspect_ratio=generator.config.aspect_ratio if isinstance(generator.config.aspect_ratio, str) else generator.config.aspect_ratio.value,
                            resolution=generator.config.resolution if isinstance(generator.config.resolution, str) else generator.config.resolution.value,
                            duration=generator.config.duration if isinstance(generator.config.duration, str) else generator.config.duration.value,
                        )
                        db.add(gen_log)
                        db.commit()
            except Exception as db_error:
                print(f"[Worker] DB error updating clip {clip_index}: {db_error}")
            
            self._broadcast_event(job_id, {
                "type": "clip_completed",
                "clip_index": clip_index,
                "success": result["success"],
                "output": result["output_path"].name if result.get("output_path") else None,
            })
            
            return {
                "clip_index": clip_index,
                "success": result["success"],
                "result": result,
            }
        
        # Process clips with queue-based approach
        # Continue while there are pending clips OR waiting clips (approval pending) OR redo clips might exist
        while (pending_clips or waiting_clips) and not generator.cancelled:
            # Check for redo clips and add them to pending
            redo_indices = check_redo_clips()
            if redo_indices:
                # Add redo clips to pending (avoid duplicates)
                for idx in redo_indices:
                    if idx not in pending_clips:
                        pending_clips.append(idx)
                print(f"[Worker] Added {len(redo_indices)} redo clip(s) to pending queue", flush=True)
            
            # Check if keys are available before starting batch
            if not check_keys_available():
                no_keys_retries += 1
                
                if no_keys_retries > max_no_keys_retries:
                    # Max retries reached - fail job with contact support message
                    error_msg = "All API keys exhausted. Please contact support for assistance."
                    
                    with get_db() as db:
                        # Mark remaining clips as failed
                        for clip_idx in pending_clips:
                            clip = db.query(Clip).filter(
                                Clip.job_id == job_id,
                                Clip.clip_index == clip_idx
                            ).first()
                            if clip:
                                clip.status = ClipStatus.FAILED.value
                                clip.error_code = "API_KEYS_EXHAUSTED"
                                clip.error_message = error_msg
                        db.commit()
                        
                        add_job_log(
                            db, job_id,
                            f"❌ Job failed: {error_msg}",
                            "ERROR", "system"
                        )
                    
                    self._broadcast_event(job_id, {
                        "type": "job_failed_no_keys",
                        "message": error_msg,
                        "contact": "Please contact support to resolve this issue."
                    })
                    
                    failed += len(pending_clips)
                    break
                
                # Alert and wait
                send_no_keys_alert(job_id, no_keys_retries)
                
                # Wait with periodic checks (allow cancellation)
                wait_end = time.time() + no_keys_wait_seconds
                while time.time() < wait_end and not generator.cancelled:
                    if check_keys_available():
                        print(f"[Worker] ✅ Keys available again, resuming...", flush=True)
                        with get_db() as db:
                            add_job_log(db, job_id, "✅ API keys available, resuming generation", "INFO", "system")
                        break
                    time.sleep(10)  # Check every 10 seconds
                
                continue  # Re-check keys at top of loop
            
            # Reset retry counter when keys are available
            no_keys_retries = 0
            
            # Determine batch size based on available keys
            available_keys = generator.api_keys.get_available_key_count()
            total_keys = len(generator.api_keys.gemini_keys) if hasattr(generator.api_keys, 'gemini_keys') else 0
            
            # Check for low key alerts
            if available_keys == 0:
                send_key_alert_email("no_keys", available_keys, total_keys, job_id)
            elif available_keys <= 10 and available_keys > 0:
                send_key_alert_email("low_keys_10", available_keys, total_keys, job_id)
            elif available_keys > 15:
                # Keys have recovered - reset alert flags
                if _alerts_sent.get("low_keys_10") or _alerts_sent.get("no_keys"):
                    reset_key_alerts()
            
            # For continue mode clips in waiting_clips, check if previous clip is APPROVED
            newly_ready = []
            still_waiting = []
            
            for clip_idx in waiting_clips:
                prev_idx = clip_idx - 1
                # Check if previous clip is approved
                if prev_idx in approved_clip_videos:
                    # Previous clip is approved, move this to pending
                    newly_ready.append(clip_idx)
                    # Update clip status from WAITING_APPROVAL to PENDING
                    with get_db() as db:
                        clip = db.query(Clip).filter(
                            Clip.job_id == job_id,
                            Clip.clip_index == clip_idx
                        ).first()
                        if clip and clip.status == ClipStatus.WAITING_APPROVAL.value:
                            clip.status = ClipStatus.PENDING.value
                            db.commit()
                            print(f"[Worker] Clip {clip_idx}: Previous approved, moved to PENDING", flush=True)
                else:
                    still_waiting.append(clip_idx)
            
            # Update waiting_clips
            waiting_clips = still_waiting
            
            # Add newly ready clips to pending
            if newly_ready:
                pending_clips.extend(newly_ready)
                print(f"[Worker] {len(newly_ready)} clips now ready after approval", flush=True)
            
            # All pending clips are ready (no dependency check needed anymore - that's handled by waiting_clips)
            ready_clips = pending_clips.copy()
            
            if not ready_clips:
                # No clips ready - check if we're waiting for approvals
                if waiting_clips:
                    # Still have clips waiting for approval - pause job processing
                    print(f"[Worker] {len(waiting_clips)} clips waiting for user approval", flush=True)
                    time.sleep(2)  # Check every 2 seconds for approvals
                    
                    # Check database for any approved clips
                    with get_db() as db:
                        for clip_idx in waiting_clips:
                            prev_idx = clip_idx - 1
                            prev_clip = db.query(Clip).filter(
                                Clip.job_id == job_id,
                                Clip.clip_index == prev_idx
                            ).first()
                            if prev_clip and prev_clip.approval_status == "approved":
                                # Found an approval! Add to approved_clip_videos
                                if prev_idx not in approved_clip_videos:
                                    video_path = None
                                    if prev_clip.output_filename:
                                        video_path = str(output_dir / prev_clip.output_filename)
                                    approved_clip_videos[prev_idx] = video_path
                                    print(f"[Worker] Detected approval for clip {prev_idx}, video_path={video_path}", flush=True)
                    
                    # Also check for redo clips during wait - process them immediately
                    redo_indices = check_redo_clips()
                    if redo_indices:
                        for idx in redo_indices:
                            if idx not in pending_clips:
                                pending_clips.append(idx)
                        print(f"[Worker] Added {len(redo_indices)} redo clip(s) during approval wait", flush=True)
                    
                    continue
                else:
                    # Nothing pending and nothing waiting - check one more time for redo clips
                    redo_indices = check_redo_clips()
                    if redo_indices:
                        for idx in redo_indices:
                            if idx not in pending_clips:
                                pending_clips.append(idx)
                        print(f"[Worker] Added {len(redo_indices)} redo clip(s), continuing processing", flush=True)
                        continue
                    # Still nothing - we're done
                    break
            
            batch_size = min(parallel_clips, available_keys, len(ready_clips))
            
            if batch_size == 0:
                continue
            
            # Get next batch of clips
            batch = ready_clips[:batch_size]
            # Remove processed clips from pending
            pending_clips = [c for c in pending_clips if c not in batch]
            
            print(f"[Worker] Processing batch of {batch_size} clips ({available_keys} keys available, {len(waiting_clips)} awaiting approval)", flush=True)
            
            # Process batch in parallel - but also check for new ready clips dynamically
            with ThreadPoolExecutor(max_workers=parallel_clips) as clip_executor:
                # Track active futures
                futures = {
                    clip_executor.submit(process_single_clip, i): i 
                    for i in batch
                }
                active_count = len(futures)
                
                requeue_clips = []
                
                while futures:
                    # Wait for at least one to complete (with timeout to check for new clips)
                    done_futures = set()
                    for future in list(futures.keys()):
                        if future.done():
                            done_futures.add(future)
                    
                    if not done_futures:
                        # No futures done yet, sleep briefly and check for new ready clips
                        time.sleep(0.5)
                        
                        # Check for redo clips while waiting
                        redo_indices = check_redo_clips()
                        if redo_indices:
                            for idx in redo_indices:
                                if idx not in pending_clips and idx not in [futures[f] for f in futures]:
                                    pending_clips.append(idx)
                            if redo_indices:
                                print(f"[Worker] Added {len(redo_indices)} redo clip(s) while processing batch", flush=True)
                        
                        # Check for newly approved clips
                        newly_ready_in_batch = []
                        still_waiting_in_batch = []
                        for clip_idx in waiting_clips:
                            prev_idx = clip_idx - 1
                            if prev_idx in approved_clip_videos:
                                newly_ready_in_batch.append(clip_idx)
                                with get_db() as db:
                                    clip = db.query(Clip).filter(
                                        Clip.job_id == job_id,
                                        Clip.clip_index == clip_idx
                                    ).first()
                                    if clip and clip.status == ClipStatus.WAITING_APPROVAL.value:
                                        clip.status = ClipStatus.PENDING.value
                                        db.commit()
                                        print(f"[Worker] Clip {clip_idx}: Previous approved, moved to PENDING (during batch)", flush=True)
                            else:
                                # Also check database for approvals
                                with get_db() as db:
                                    prev_clip = db.query(Clip).filter(
                                        Clip.job_id == job_id,
                                        Clip.clip_index == prev_idx
                                    ).first()
                                    if prev_clip and prev_clip.approval_status == "approved":
                                        if prev_idx not in approved_clip_videos:
                                            video_path = None
                                            if prev_clip.output_filename:
                                                video_path = str(output_dir / prev_clip.output_filename)
                                            approved_clip_videos[prev_idx] = video_path
                                            newly_ready_in_batch.append(clip_idx)
                                            print(f"[Worker] Detected approval for clip {prev_idx} during batch, video_path={video_path}", flush=True)
                                    else:
                                        still_waiting_in_batch.append(clip_idx)
                        
                        waiting_clips = still_waiting_in_batch
                        
                        # Add newly ready clips to pending
                        for idx in newly_ready_in_batch:
                            if idx not in pending_clips and idx not in [futures[f] for f in futures]:
                                pending_clips.append(idx)
                        
                        # Submit new clips if we have capacity
                        current_active = len([f for f in futures if not f.done()])
                        available_slots = parallel_clips - current_active
                        
                        if available_slots > 0 and pending_clips:
                            new_batch = pending_clips[:available_slots]
                            pending_clips = [c for c in pending_clips if c not in new_batch]
                            
                            for clip_idx in new_batch:
                                future = clip_executor.submit(process_single_clip, clip_idx)
                                futures[future] = clip_idx
                                print(f"[Worker] Submitted clip {clip_idx} to fill available slot", flush=True)
                        
                        continue
                    
                    # Process completed futures
                    for future in done_futures:
                        clip_index = futures.pop(future)
                        try:
                            result = future.result()
                            
                            if result.get("no_keys"):
                                # Re-queue this clip for later
                                requeue_clips.append(clip_index)
                                print(f"[Worker] Clip {clip_index} failed due to no keys, re-queuing", flush=True)
                            elif result.get("success"):
                                completed += 1
                                # Track completed video for "continue" mode
                                inner_result = result.get("result", {})
                                if inner_result.get("output_path"):
                                    completed_clip_videos[clip_index] = str(inner_result["output_path"])
                                    print(f"[Worker] Tracked completed video for clip {clip_index}: {inner_result['output_path'].name}", flush=True)
                            elif result.get("skipped"):
                                skipped += 1
                            else:
                                failed += 1
                                # For failed clips, still mark as "done" so dependent clips can fall back
                                completed_clip_videos[clip_index] = None
                            
                            # Update job progress
                            with get_db() as db:
                                job = db.query(Job).filter(Job.id == job_id).first()
                                if job:
                                    job.completed_clips = completed
                                    job.failed_clips = failed
                                    job.skipped_clips = skipped
                                    processed = completed + failed + skipped
                                    job.progress_percent = (processed / total_clips) * 100 if total_clips > 0 else 0
                                    db.commit()
                            
                        except Exception as e:
                            print(f"[Worker] Future error for clip {clip_index}: {e}")
                            failed += 1
                            # Mark as done so dependents can proceed
                            completed_clip_videos[clip_index] = None
                
                # Add re-queued clips back to pending
                if requeue_clips:
                    pending_clips = requeue_clips + pending_clips
                    print(f"[Worker] Re-queued {len(requeue_clips)} clips, {len(pending_clips)} pending", flush=True)
        
        # Job completed - calculate status from actual clip data
        actual_completed = 0
        actual_failed = 0
        actual_skipped = 0
        final_status = "unknown"
        
        with get_db() as db:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                # Recalculate stats from actual clips in database
                clips = db.query(Clip).filter(Clip.job_id == job_id).all()
                actual_completed = sum(1 for c in clips if c.status == ClipStatus.COMPLETED.value)
                actual_failed = sum(1 for c in clips if c.status == ClipStatus.FAILED.value)
                actual_skipped = sum(1 for c in clips if c.status == ClipStatus.SKIPPED.value)
                
                job.completed_clips = actual_completed
                job.failed_clips = actual_failed
                job.skipped_clips = actual_skipped
                job.progress_percent = 100.0
                
                if generator.cancelled:
                    job.status = JobStatus.CANCELLED.value
                elif actual_completed == 0 and actual_failed > 0:
                    job.status = JobStatus.FAILED.value
                else:
                    job.status = JobStatus.COMPLETED.value
                
                job.completed_at = datetime.utcnow()
                final_status = job.status
                db.commit()
                
                add_job_log(
                    db, job_id,
                    f"Job completed: {actual_completed} success, {actual_failed} failed, {actual_skipped} skipped",
                    "INFO", "system"
                )
            
            # Save blacklist
            for img_path in generator.blacklist:
                entry = BlacklistEntry(
                    job_id=job_id,
                    image_filename=img_path.name,
                    reason="generation_failed",
                )
                db.add(entry)
            db.commit()
        
        self._broadcast_event(job_id, {
            "type": "job_completed",
            "status": final_status,
            "completed": actual_completed,
            "failed": actual_failed,
            "skipped": actual_skipped,
        })
    
    def _get_next_clean_image(
        self,
        generator: VeoGenerator,
        images: List[Path],
        current_index: int,
    ) -> Optional[tuple]:
        """Get next non-blacklisted image"""
        total = len(images)
        
        for offset in range(1, min(generator.config.max_image_attempts + 1, total + 1)):
            new_index = (current_index + offset) % total
            candidate = images[new_index]
            
            if candidate not in generator.blacklist:
                return (new_index, candidate)
        
        return None
    
    def _get_next_clean_start(
        self,
        generator: VeoGenerator,
        images: List[Path],
        current_index: int,
    ) -> Optional[tuple]:
        """Get next non-blacklisted start frame"""
        return self._get_next_clean_image(generator, images, current_index)
    
    def _handle_progress(
        self,
        job_id: str,
        clip_index: int,
        status: str,
        message: str,
        details: Optional[Dict],
    ):
        """Handle progress update from generator"""
        with get_db() as db:
            add_job_log(
                db, job_id, message,
                level="INFO" if status != "error" else "ERROR",
                category="clip",
                clip_index=clip_index,
                details=details
            )
        
        self._broadcast_event(job_id, {
            "type": "progress",
            "clip_index": clip_index,
            "status": status,
            "message": message,
            "details": details,
        })
    
    def _handle_error(self, job_id: str, error: VeoError):
        """Handle error from generator"""
        with get_db() as db:
            add_job_log(
                db, job_id,
                error.message,
                level="ERROR",
                category="error",
                details=error.to_dict()
            )
        
        self._broadcast_event(job_id, {
            "type": "error",
            "error": error.to_dict(),
        })
    
    # ============ SSE Subscription Management ============
    
    def subscribe(self, job_id: str) -> Queue:
        """Subscribe to job events"""
        event_queue = Queue()
        
        with self.subscribers_lock:
            if job_id not in self.subscribers:
                self.subscribers[job_id] = []
            self.subscribers[job_id].append(event_queue)
            print(f"[Worker] Subscribed to job {job_id[:8]}, total subscribers: {len(self.subscribers[job_id])}", flush=True)
        
        return event_queue
    
    def unsubscribe(self, job_id: str, event_queue: Queue):
        """Unsubscribe from job events"""
        with self.subscribers_lock:
            if job_id in self.subscribers:
                if event_queue in self.subscribers[job_id]:
                    self.subscribers[job_id].remove(event_queue)
                    print(f"[Worker] Unsubscribed from job {job_id[:8]}, remaining: {len(self.subscribers[job_id])}", flush=True)
                if not self.subscribers[job_id]:
                    del self.subscribers[job_id]
    
    def _broadcast_event(self, job_id: str, event: Dict):
        """Broadcast event to all subscribers"""
        print(f"[Worker] Broadcasting event: {event.get('type')} for job {job_id[:8]}", flush=True)
        with self.subscribers_lock:
            if job_id in self.subscribers:
                subscriber_count = len(self.subscribers[job_id])
                print(f"[Worker] Broadcasting to {subscriber_count} subscribers", flush=True)
                for queue in self.subscribers[job_id]:
                    try:
                        queue.put_nowait(event)
                    except Exception as e:
                        print(f"[Worker] Failed to broadcast: {e}", flush=True)
            else:
                print(f"[Worker] No subscribers for job {job_id[:8]}", flush=True)
    
    # ============ Job Control ============
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a running job"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].pause()
            
            with get_db() as db:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = JobStatus.PAUSED.value
                    db.commit()
            
            return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].resume()
            
            with get_db() as db:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = JobStatus.RUNNING.value
                    db.commit()
            
            return True
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status"""
        with get_db() as db:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                return job.to_dict()
        return None


# Singleton worker instance
worker = JobWorker(max_workers=app_config.max_workers)