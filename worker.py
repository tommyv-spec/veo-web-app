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
    DialogueLine, app_config, get_gemini_keys_from_env, get_openai_key_from_env,
    api_keys_config  # Global singleton for persistent key blocking
)
from models import (
    get_db, Job, Clip, JobLog, BlacklistEntry, GenerationLog,
    add_job_log, update_job_progress
)
from veo_generator import VeoGenerator, list_images, GENAI_AVAILABLE
from error_handler import VeoError, error_handler


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
        
        # Get parallel clip count from config (default 3)
        parallel_clips = getattr(generator.config, 'parallel_clips', 3)
        
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
        # The logic depends on clip_mode and what comes next:
        #
        # BLEND mode (standard):
        #   - Start: assigned image
        #   - End: depends on NEXT clip
        #
        # CONTINUE mode:
        #   - Start: extracted from previous clip (set at runtime)
        #   - End: depends on NEXT clip
        #
        # FRESH mode:
        #   - Start: always original image
        #   - End: depends on NEXT clip
        #
        # END FRAME determination (for ALL modes):
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
                
                is_last_clip = (i == len(clip_info) - 1)
                
                if is_last_clip:
                    # CASE 1: Last clip of the video
                    if last_frame_index is not None and last_frame_index < num_images:
                        # Last Frame is defined - use it
                        use_end_frame = True
                        actual_end_idx = last_frame_index
                        end_frame_reason = f"last clip, using Last Frame (image {last_frame_index + 1})"
                    else:
                        # No Last Frame defined - no end frame
                        use_end_frame = False
                        end_frame_reason = "last clip, no Last Frame defined"
                else:
                    next_info = clip_info[i + 1]
                    next_scene = next_info["scene_index"]
                    
                    if next_scene == scene_index:
                        # CASE 2: Next clip is in SAME scene
                        # No end frame - let it continue naturally
                        use_end_frame = False
                        end_frame_reason = "same scene, natural continuation"
                    else:
                        # CASE 3: Next clip is in DIFFERENT scene
                        next_transition = next_info["scene_transition"]
                        
                        if next_transition == "blend":
                            # Blend to next scene - morph toward next scene's image
                            use_end_frame = True
                            actual_end_idx = next_info["image_idx"]
                            end_frame_reason = f"blend to scene {next_scene}"
                        else:
                            # Cut to next scene - no end frame needed
                            use_end_frame = False
                            end_frame_reason = "cut to next scene"
                
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
            
            # TODO: Add webhook/email alert here
            # self._send_admin_alert(alert_msg, job_id)
        
        def extract_frame_from_video(video_path: Path, frame_offset: int = -8) -> Optional[Path]:
            """Extract a frame from video. frame_offset=-8 means 8 frames from the end."""
            try:
                # Get video duration
                probe_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
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
                    "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
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
                return None
        
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
            
            # Handle "continue" mode - use extracted frame from APPROVED previous clip
            # ONLY if requires_previous is True (meaning previous clip is in SAME scene)
            if clip_mode == "continue" and requires_previous and clip_index > 0:
                prev_video = approved_clip_videos.get(clip_index - 1)
                if prev_video and Path(prev_video).exists():
                    extracted = extract_frame_from_video(Path(prev_video), frame_offset=-8)
                    if extracted:
                        start_frame = extracted
                        print(f"[Worker] Clip {clip_index}: Using extracted frame from APPROVED clip {clip_index - 1}", flush=True)
                    else:
                        print(f"[Worker] Clip {clip_index}: Frame extraction failed, using original image", flush=True)
                else:
                    print(f"[Worker] Clip {clip_index}: Approved previous clip video not found, using original image", flush=True)
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
                result = generator.generate_single_clip(
                    start_frame=start_frame,
                    end_frame=end_frame,  # Can be None
                    dialogue_line=dialogue_text,
                    dialogue_id=dialogue_id,
                    clip_index=clip_index,
                    output_dir=output_dir,
                    images_list=images,
                    current_end_index=end_index if end_index is not None else start_index,
                )
                
                # Log the prompt that was sent to Veo
                if result.get("prompt_text"):
                    prompt_preview = result["prompt_text"][:500] + "..." if len(result.get("prompt_text", "")) > 500 else result.get("prompt_text", "")
                    with get_db() as db:
                        add_job_log(
                            db, job_id,
                            f"📝 Prompt for clip {clip_index + 1}: {prompt_preview}",
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
        # Continue while there are pending clips OR waiting clips (approval pending)
        while (pending_clips or waiting_clips) and not generator.cancelled:
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
                                    print(f"[Worker] Detected approval for clip {prev_idx}", flush=True)
                    
                    continue
                else:
                    # Nothing pending and nothing waiting - we're done
                    break
            
            batch_size = min(parallel_clips, available_keys, len(ready_clips))
            
            if batch_size == 0:
                continue
            
            # Get next batch of clips
            batch = ready_clips[:batch_size]
            # Remove processed clips from pending
            pending_clips = [c for c in pending_clips if c not in batch]
            
            print(f"[Worker] Processing batch of {batch_size} clips ({available_keys} keys available, {len(waiting_clips)} awaiting approval)", flush=True)
            
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