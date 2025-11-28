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
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import traceback

from config import (
    JobStatus, ClipStatus, VideoConfig, APIKeysConfig, 
    DialogueLine, app_config, get_gemini_keys_from_env, get_openai_key_from_env
)
from models import (
    get_db, Job, Clip, JobLog, BlacklistEntry, GenerationLog,
    add_job_log, update_job_progress
)
from veo_generator import VeoGenerator, list_images, GENAI_AVAILABLE
from error_handler import VeoError, error_handler


def get_api_keys_with_fallback(api_keys_json: str = None) -> APIKeysConfig:
    """Get API keys from job data, falling back to environment variables."""
    api_keys_data = json.loads(api_keys_json) if api_keys_json else {}
    gemini_keys = api_keys_data.get("gemini_keys", [])
    openai_key = api_keys_data.get("openai_key")
    
    # If no keys provided, use from environment
    if not gemini_keys:
        gemini_keys = get_gemini_keys_from_env()
        print(f"[Worker] ✅ Loaded {len(gemini_keys)} Gemini keys from environment", flush=True)
        for i, key in enumerate(gemini_keys):
            print(f"[Worker]    Key {i+1}: ...{key[-8:]}", flush=True)
    if not openai_key:
        openai_key = get_openai_key_from_env()
    
    return APIKeysConfig(
        gemini_api_keys=gemini_keys,
        openai_api_key=openai_key,
    )


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
                
                if not end_frame and len(images) > 1:
                    end_index = (start_index + 1) % len(images)
                    end_frame = images[end_index]
                
                # Determine prompt to use
                prompt_text = None
                if clip.use_logged_params and clip.prompt_text:
                    prompt_text = clip.prompt_text
                    add_job_log(db, job_id, f"Using logged parameters for redo", "INFO", "redo")
                else:
                    add_job_log(db, job_id, f"Using fresh parameters for redo", "INFO", "redo")
                
                self._broadcast_event(job_id, {
                    "type": "redo_started",
                    "clip_id": clip_id,
                    "clip_index": clip.clip_index,
                    "attempt": clip.generation_attempt,
                    "use_logged_params": clip.use_logged_params,
                })
            
            # Generate clip (outside db context to avoid long transactions)
            result = generator.generate_single_clip(
                start_frame=start_frame,
                end_frame=end_frame,
                dialogue_line=clip.dialogue_text,
                dialogue_id=clip.dialogue_id,
                clip_index=clip.clip_index,
                output_dir=output_dir,
                images_list=images,
                current_end_index=end_index,
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
                    single_image_mode=config_data.get("single_image_mode", False),
                )
                
                add_job_log(db, job_id, f"Language: {config.language}", "INFO", "config")
                
                # Parse API keys (with env fallback)
                api_keys = get_api_keys_with_fallback(job.api_keys_json)
                
                # Parse dialogue
                dialogue_data = json.loads(job.dialogue_json)
                
                # Get images
                images_dir = Path(job.images_dir)
                output_dir = Path(job.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
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
                
                def on_error(error: VeoError):
                    self._handle_error(job_id, error)
                
                generator.on_progress = on_progress
                generator.on_error = on_error
                
                self.running_jobs[job_id] = generator
            
            # Process clips
            self._process_clips(job_id, generator, dialogue_data, images, output_dir)
            
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
    ):
        """Process all clips for a job - with parallel generation support"""
        from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
        
        # Check for single image mode
        single_image_mode = getattr(generator.config, 'single_image_mode', False) or len(images) == 1
        
        images_dir_str = str(images[0].parent)
        
        total_clips = len(dialogue_data)
        completed = 0
        failed = 0
        skipped = 0
        
        # Get parallel clip count from config (default 3)
        parallel_clips = getattr(generator.config, 'parallel_clips', 3)
        
        # Key exhaustion tracking
        no_keys_retries = 0
        max_no_keys_retries = 3
        no_keys_wait_seconds = 300  # 5 minutes
        
        with get_db() as db:
            # Create clip records
            for i, line_data in enumerate(dialogue_data):
                clip = Clip(
                    job_id=job_id,
                    clip_index=i,
                    dialogue_id=line_data["id"],
                    dialogue_text=line_data["text"],
                    status=ClipStatus.PENDING.value,
                )
                db.add(clip)
            db.commit()
        
        # Pre-calculate frame assignments for all clips
        clip_frames = []
        for i in range(total_clips):
            if single_image_mode:
                # All clips use the same image
                start_idx = 0
                end_idx = 0
            else:
                # Cycle through images
                start_idx = i % len(images)
                end_idx = (i + 1) % len(images)
            
            clip_frames.append({
                "start_index": start_idx,
                "start_frame": images[start_idx],
                "end_index": end_idx,
                "end_frame": images[end_idx],
            })
        
        # Queue of pending clip indices
        pending_clips = list(range(total_clips))
        
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
            
            # TODO: Add webhook/email alert here
            # self._send_admin_alert(alert_msg, job_id)
        
        def process_single_clip(clip_index: int):
            """Process a single clip - runs in thread"""
            if generator.cancelled:
                return {"clip_index": clip_index, "success": False, "skipped": True}
            
            line_data = dialogue_data[clip_index]
            dialogue_id = line_data["id"]
            dialogue_text = line_data["text"]
            frames = clip_frames[clip_index]
            
            start_frame = frames["start_frame"]
            end_frame = frames["end_frame"]
            start_index = frames["start_index"]
            end_index = frames["end_index"]
            
            # Update clip status to generating
            with get_db() as db:
                clip = db.query(Clip).filter(
                    Clip.job_id == job_id,
                    Clip.clip_index == clip_index
                ).first()
                
                if clip:
                    clip.status = ClipStatus.GENERATING.value
                    clip.started_at = datetime.utcnow()
                    clip.start_frame = start_frame.name
                    clip.end_frame = end_frame.name
                    db.commit()
            
            self._broadcast_event(job_id, {
                "type": "clip_started",
                "clip_index": clip_index,
                "dialogue_id": dialogue_id,
                "start_frame": start_frame.name,
            })
            
            # Check if start frame is blacklisted
            if start_frame in generator.blacklist:
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
                result = generator.generate_single_clip(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    dialogue_line=dialogue_text,
                    dialogue_id=dialogue_id,
                    clip_index=clip_index,
                    output_dir=output_dir,
                    images_list=images,
                    current_end_index=end_index,
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
                        gen_log = GenerationLog(
                            job_id=job_id,
                            video_id=dialogue_id,
                            images_dir=images_dir_str,
                            start_frame=start_frame.name,
                            end_frame=result["end_frame_used"].name if result.get("end_frame_used") else end_frame.name,
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
        while pending_clips and not generator.cancelled:
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
            batch_size = min(parallel_clips, available_keys, len(pending_clips))
            
            if batch_size == 0:
                continue
            
            # Get next batch of clips
            batch = pending_clips[:batch_size]
            pending_clips = pending_clips[batch_size:]
            
            print(f"[Worker] Processing batch of {batch_size} clips ({available_keys} keys available)", flush=True)
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as clip_executor:
                futures = {
                    clip_executor.submit(process_single_clip, i): i 
                    for i in batch
                }
                
                requeue_clips = []
                
                for future in as_completed(futures):
                    clip_index = futures[future]
                    try:
                        result = future.result()
                        
                        if result.get("no_keys"):
                            # Re-queue this clip for later
                            requeue_clips.append(clip_index)
                            print(f"[Worker] Clip {clip_index} failed due to no keys, re-queuing", flush=True)
                        elif result.get("success"):
                            completed += 1
                        elif result.get("skipped"):
                            skipped += 1
                        else:
                            failed += 1
                        
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