# -*- coding: utf-8 -*-
"""
Veo Web App - Main FastAPI Application

Features:
- REST API for job management
- Server-Sent Events for real-time progress
- File upload handling
- Static file serving
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, 
    BackgroundTasks, Depends, Query
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import (
    app_config, VideoConfig, APIKeysConfig, DialogueLine,
    JobStatus, ClipStatus, SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_SIZE_BYTES, AspectRatio, Resolution, Duration,
    ApprovalStatus, api_keys_config
)
from models import (
    init_db, get_db_session, Job, Clip, JobLog, BlacklistEntry,
    get_job_logs_since, add_job_log
)
from worker import worker
from error_handler import ErrorCode


# ============ Pydantic Models ============

class DialogueLineInput(BaseModel):
    id: int
    text: str


class VideoConfigInput(BaseModel):
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    duration: str = "8"
    language: str = "English"
    use_interpolation: bool = True
    use_openai_prompt_tuning: bool = True
    use_frame_vision: bool = True
    max_retries_per_clip: int = 5
    custom_prompt: str = ""  # User's custom prompt when AI is disabled
    single_image_mode: bool = False  # Use same image for start/end frames


class APIKeysInput(BaseModel):
    gemini_keys: List[str] = []
    openai_key: Optional[str] = None


class CreateJobRequest(BaseModel):
    config: VideoConfigInput
    dialogue_lines: List[DialogueLineInput]
    api_keys: APIKeysInput
    job_id: Optional[str] = None  # Use existing upload job_id if provided


class JobResponse(BaseModel):
    id: str
    status: str
    progress_percent: float
    total_clips: int
    completed_clips: int
    failed_clips: int
    skipped_clips: int
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class ClipResponse(BaseModel):
    id: int
    clip_index: int
    dialogue_id: int
    dialogue_text: str
    status: str
    retry_count: int
    start_frame: Optional[str]
    end_frame: Optional[str]
    output_filename: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]
    # New approval fields
    approval_status: str = "pending_review"
    generation_attempt: int = 1
    attempts_remaining: int = 2
    redo_reason: Optional[str] = None
    versions: List[Dict] = []
    # Variant fields
    selected_variant: int = 1
    total_variants: int = 0


class RedoRequest(BaseModel):
    reason: Optional[str] = None  # Optional reason for redo


class ApprovalResponse(BaseModel):
    clip_id: int
    status: str
    message: str
    attempts_remaining: int


class LogResponse(BaseModel):
    id: int
    created_at: str
    level: str
    category: Optional[str]
    clip_index: Optional[int]
    message: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict] = None


# ============ Application Setup ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    init_db()
    worker.start()
    print("[App] Started")
    
    yield
    
    # Shutdown
    worker.stop()
    print("[App] Shutdown complete")


app = FastAPI(
    title="Veo 3.1 Video Generator",
    description="Web interface for generating videos with Google Veo 3.1",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Static Files ============

# Create static directory if not exists
static_dir = app_config.base_dir / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============ Root / UI ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Veo Web App</h1><p>UI not found. Place index.html in static/</p>")


# ============ Image Upload ============

@app.post("/api/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    job_id: Optional[str] = Form(None),
):
    """
    Upload images for video generation.
    Creates a new job directory if job_id not provided.
    """
    # Create or get job directory
    if job_id is None:
        job_id = str(uuid.uuid4())
    
    job_dir = app_config.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    errors = []
    
    for file in files:
        # Validate file type
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            errors.append({
                "filename": file.filename,
                "error": f"Unsupported format: {ext}",
                "code": ErrorCode.IMAGE_INVALID_FORMAT.value,
            })
            continue
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_IMAGE_SIZE_BYTES:
            errors.append({
                "filename": file.filename,
                "error": f"File too large: {len(content) / 1024 / 1024:.1f}MB",
                "code": ErrorCode.IMAGE_TOO_LARGE.value,
            })
            continue
        
        # Save file
        try:
            filepath = job_dir / file.filename
            with open(filepath, "wb") as f:
                f.write(content)
            uploaded.append({
                "filename": file.filename,
                "size": len(content),
                "path": str(filepath),
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "code": ErrorCode.FILE_WRITE_ERROR.value,
            })
    
    return {
        "job_id": job_id,
        "uploaded": uploaded,
        "errors": errors,
        "total_uploaded": len(uploaded),
        "total_errors": len(errors),
    }


@app.get("/api/upload/{job_id}/images")
async def list_uploaded_images(job_id: str):
    """List images uploaded for a job"""
    job_dir = app_config.uploads_dir / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    images = []
    for f in job_dir.iterdir():
        if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            images.append({
                "filename": f.name,
                "size": f.stat().st_size,
            })
    
    images.sort(key=lambda x: x["filename"])
    
    return {"job_id": job_id, "images": images, "count": len(images)}


@app.delete("/api/upload/{job_id}")
async def delete_uploaded_images(job_id: str):
    """Delete all uploaded images for a job"""
    job_dir = app_config.uploads_dir / job_id
    
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    return {"status": "deleted", "job_id": job_id}


# ============ Job Management ============

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    db: Session = Depends(get_db_session),
):
    """Create a new video generation job"""
    # Use provided job_id (from upload) or generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())
    
    # Validate images exist
    images_dir = app_config.uploads_dir / job_id
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise HTTPException(
            status_code=400,
            detail={"errors": ["No images uploaded. Please upload images first."], "code": ErrorCode.NO_IMAGES.value}
        )
    
    # Create output directory
    output_dir = app_config.outputs_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate config
    config = request.config
    print(f"[main.py] Received config from UI: language={config.language}")
    errors = []
    
    if config.resolution == "1080p" and config.duration != "8":
        errors.append("1080p requires 8 second duration")
    
    if config.use_interpolation and config.duration != "8":
        errors.append("Interpolation requires 8 second duration")
    
    if not request.dialogue_lines:
        errors.append("At least one dialogue line is required")
    
    # Check server-side API keys (not from request)
    if not api_keys_config.gemini_api_keys:
        errors.append("No Gemini API keys configured on server. Contact administrator.")
    
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"errors": errors, "code": ErrorCode.INVALID_CONFIG.value}
        )
    
    # Use server-side API keys (ignore any keys from request)
    api_keys_data = {
        "gemini_keys": api_keys_config.gemini_api_keys,
        "openai_key": api_keys_config.openai_api_key
    }
    
    # Create job record
    config_dict = config.model_dump()
    print(f"[main.py] Creating job with config: language={config_dict.get('language')}")
    
    job = Job(
        id=job_id,
        status=JobStatus.PENDING.value,
        config_json=json.dumps(config_dict),
        dialogue_json=json.dumps([d.model_dump() for d in request.dialogue_lines]),
        api_keys_json=json.dumps(api_keys_data),
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        total_clips=len(request.dialogue_lines),
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    add_job_log(db, job_id, "Job created", "INFO", "system")
    
    return JobResponse(
        id=job.id,
        status=job.status,
        progress_percent=job.progress_percent,
        total_clips=job.total_clips,
        completed_clips=job.completed_clips,
        failed_clips=job.failed_clips,
        skipped_clips=job.skipped_clips,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@app.get("/api/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    db: Session = Depends(get_db_session),
):
    """List all jobs"""
    query = db.query(Job)
    
    if status:
        query = query.filter(Job.status == status)
    
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        JobResponse(
            id=j.id,
            status=j.status,
            progress_percent=j.progress_percent,
            total_clips=j.total_clips,
            completed_clips=j.completed_clips,
            failed_clips=j.failed_clips,
            skipped_clips=j.skipped_clips,
            created_at=j.created_at.isoformat() if j.created_at else None,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
        )
        for j in jobs
    ]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: Session = Depends(get_db_session)):
    """Get job details"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        id=job.id,
        status=job.status,
        progress_percent=job.progress_percent,
        total_clips=job.total_clips,
        completed_clips=job.completed_clips,
        failed_clips=job.failed_clips,
        skipped_clips=job.skipped_clips,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db_session)):
    """Delete a job and its data"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Cancel if running
    if job.status == JobStatus.RUNNING.value:
        worker.cancel_job(job_id)
    
    # Delete files
    images_dir = Path(job.images_dir)
    output_dir = Path(job.output_dir)
    
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Delete database records
    db.delete(job)
    db.commit()
    
    return {"status": "deleted", "job_id": job_id}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db_session)):
    """Cancel a running job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    success = worker.cancel_job(job_id)
    
    if success:
        add_job_log(db, job_id, "Job cancelled by user", "INFO", "system")
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str, db: Session = Depends(get_db_session)):
    """Pause a running job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    success = worker.pause_job(job_id)
    
    if success:
        add_job_log(db, job_id, "Job paused by user", "INFO", "system")
        return {"status": "paused", "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to pause job")


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str, db: Session = Depends(get_db_session)):
    """Resume a paused job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.PAUSED.value:
        raise HTTPException(status_code=400, detail="Job is not paused")
    
    success = worker.resume_job(job_id)
    
    if success:
        add_job_log(db, job_id, "Job resumed by user", "INFO", "system")
        return {"status": "resumed", "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to resume job")


# ============ Clips ============

def deduplicate_versions(versions_json: str) -> list:
    """Deduplicate versions by attempt number, keeping last one"""
    if not versions_json:
        return []
    versions = json.loads(versions_json)
    seen = {}
    for v in versions:
        attempt = v.get("attempt")
        if attempt:
            seen[attempt] = v
    return sorted(seen.values(), key=lambda x: x.get("attempt", 0))

@app.get("/api/jobs/{job_id}/clips", response_model=List[ClipResponse])
async def get_job_clips(job_id: str, db: Session = Depends(get_db_session)):
    """Get all clips for a job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).order_by(Clip.clip_index).all()
    
    return [
        ClipResponse(
            id=c.id,
            clip_index=c.clip_index,
            dialogue_id=c.dialogue_id,
            dialogue_text=c.dialogue_text,
            status=c.status,
            retry_count=c.retry_count,
            start_frame=c.start_frame,
            end_frame=c.end_frame,
            output_filename=c.output_filename,
            error_code=c.error_code,
            error_message=c.error_message,
            approval_status=c.approval_status or "pending_review",
            generation_attempt=c.generation_attempt or 1,
            attempts_remaining=3 - (c.generation_attempt or 1),
            redo_reason=c.redo_reason,
            versions=deduplicate_versions(c.versions_json),
            selected_variant=c.selected_variant or c.generation_attempt or 1,
            total_variants=len(deduplicate_versions(c.versions_json)),
        )
        for c in clips
    ]


# ============ Clip Review & Approval ============

@app.post("/api/clips/{clip_id}/approve", response_model=ApprovalResponse)
async def approve_clip(clip_id: int, db: Session = Depends(get_db_session)):
    """
    Approve a clip - marks it as accepted by the user.
    """
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    if clip.status != ClipStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Can only approve completed clips")
    
    if clip.approval_status == "max_attempts":
        raise HTTPException(status_code=400, detail="Clip has reached max attempts - contact support")
    
    # Update approval status
    clip.approval_status = "approved"
    
    # Update versions history
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    for v in versions:
        if v.get("attempt") == clip.generation_attempt:
            v["approved"] = True
    clip.versions_json = json.dumps(versions)
    
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} approved by user", "INFO", "approval")
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="approved",
        message="Clip has been approved",
        attempts_remaining=3 - clip.generation_attempt
    )


@app.post("/api/clips/{clip_id}/reject", response_model=ApprovalResponse)
async def reject_clip(clip_id: int, db: Session = Depends(get_db_session)):
    """
    Reject a clip without triggering redo.
    User can later choose to redo or leave as rejected.
    """
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    if clip.status != ClipStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Can only reject completed clips")
    
    clip.approval_status = "rejected"
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} rejected by user", "INFO", "approval")
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="rejected",
        message="Clip has been rejected. You can redo it or leave as is.",
        attempts_remaining=3 - clip.generation_attempt
    )


@app.delete("/api/clips/{clip_id}")
async def delete_clip(clip_id: int, db: Session = Depends(get_db_session)):
    """
    Delete a clip and its video file.
    """
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    job_id = clip.job_id
    clip_index = clip.clip_index
    
    # Delete video file if exists
    if clip.output_path:
        try:
            video_path = Path(clip.output_path)
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            print(f"Error deleting video file: {e}", flush=True)
    
    # Delete from database
    db.delete(clip)
    db.commit()
    
    # Update job stats
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        remaining_clips = db.query(Clip).filter(Clip.job_id == job_id).count()
        job.total_clips = remaining_clips
        completed = db.query(Clip).filter(Clip.job_id == job_id, Clip.status == ClipStatus.COMPLETED.value).count()
        job.completed_clips = completed
        if remaining_clips > 0:
            job.progress_percent = int((completed / remaining_clips) * 100)
        db.commit()
    
    add_job_log(db, job_id, f"Clip {clip_index + 1} deleted by user", "INFO", "deletion")
    
    return {"success": True, "message": f"Clip {clip_index + 1} deleted"}


@app.post("/api/clips/{clip_id}/select-variant/{variant_num}")
async def select_clip_variant(clip_id: int, variant_num: int, db: Session = Depends(get_db_session)):
    """
    Select a specific variant for a clip.
    Updates output_filename to point to the selected variant's video.
    """
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    
    # Deduplicate versions by attempt number (keep last one)
    seen = {}
    for v in versions:
        attempt = v.get("attempt")
        if attempt:
            seen[attempt] = v
    versions = list(seen.values())
    versions.sort(key=lambda x: x.get("attempt", 0))
    
    # Save cleaned versions back
    clip.versions_json = json.dumps(versions)
    
    if not versions:
        raise HTTPException(status_code=400, detail="No variants available")
    
    # Check variant is in valid range
    if variant_num < 1 or variant_num > len(versions):
        raise HTTPException(status_code=400, detail=f"Variant must be between 1 and {len(versions)}")
    
    # Find the requested variant
    variant = None
    for v in versions:
        if v.get("attempt") == variant_num:
            variant = v
            break
    
    if not variant:
        raise HTTPException(status_code=404, detail=f"Variant {variant_num} not found")
    
    # Update selected variant and output filename
    clip.selected_variant = variant_num
    clip.output_filename = variant.get("filename")
    clip.approval_status = "pending_review"  # Reset approval when switching
    db.commit()
    
    add_job_log(db, clip.job_id, f"Clip {clip.clip_index + 1} switched to variant {variant_num}", "INFO", "variant")
    
    return {
        "success": True,
        "selected_variant": variant_num,
        "filename": variant.get("filename"),
        "total_variants": len(versions)
    }


@app.post("/api/clips/{clip_id}/redo", response_model=ApprovalResponse)
async def request_clip_redo(
    clip_id: int, 
    request: RedoRequest = None,
    db: Session = Depends(get_db_session)
):
    """
    Request a redo for a clip.
    
    - Attempt 1 → 2: Uses same logged parameters
    - Attempt 2 → 3: Uses fresh parameters (no log)
    - Attempt 3: No more redos allowed, must contact support
    """
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    if clip.status != ClipStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Can only redo completed clips")
    
    # Check attempt limit
    if clip.generation_attempt >= 3:
        clip.approval_status = "max_attempts"
        db.commit()
        raise HTTPException(
            status_code=400, 
            detail={
                "code": "MAX_ATTEMPTS_REACHED",
                "message": "Maximum 3 attempts reached. Please contact support for assistance.",
                "support_email": "support@yourdomain.com"
            }
        )
    
    # Save current version to history before redo (avoid duplicates)
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    existing_attempts = [v.get('attempt') for v in versions]
    
    # Only add if this attempt isn't already saved (avoid duplicates from worker)
    if clip.generation_attempt not in existing_attempts and clip.output_filename:
        versions.append({
            "attempt": clip.generation_attempt,
            "filename": clip.output_filename,
            "generated_at": clip.completed_at.isoformat() if clip.completed_at else None,
            "approved": False,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
        })
        clip.versions_json = json.dumps(versions)
    
    # Increment attempt
    new_attempt = clip.generation_attempt + 1
    clip.generation_attempt = new_attempt
    
    # Determine if we use logged params
    # Attempt 2: use logged params (same settings)
    # Attempt 3: fresh generation (no logged params)
    clip.use_logged_params = (new_attempt == 2)
    
    # Set status for redo queue
    clip.status = ClipStatus.REDO_QUEUED.value
    clip.approval_status = "rejected"
    clip.redo_reason = request.reason if request else None
    
    # Clear previous output (keep in versions history)
    clip.output_filename = None
    clip.error_code = None
    clip.error_message = None
    
    db.commit()
    
    add_job_log(
        db, clip.job_id, 
        f"Clip {clip.clip_index + 1} redo requested (attempt {new_attempt}/3, {'with' if clip.use_logged_params else 'without'} logged params)",
        "INFO", "approval",
        details={"reason": request.reason if request else None, "use_logged_params": clip.use_logged_params}
    )
    
    return ApprovalResponse(
        clip_id=clip.id,
        status="redo_queued",
        message=f"Redo queued (attempt {new_attempt}/3). {'Using same parameters.' if clip.use_logged_params else 'Using fresh parameters.'}",
        attempts_remaining=3 - new_attempt
    )


@app.get("/api/clips/{clip_id}/versions")
async def get_clip_versions(clip_id: int, db: Session = Depends(get_db_session)):
    """Get all generated versions of a clip"""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    versions = json.loads(clip.versions_json) if clip.versions_json else []
    
    # Add current version if completed
    if clip.status == ClipStatus.COMPLETED.value and clip.output_filename:
        versions.append({
            "attempt": clip.generation_attempt,
            "filename": clip.output_filename,
            "generated_at": clip.completed_at.isoformat() if clip.completed_at else None,
            "approved": clip.approval_status == "approved",
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "current": True,
        })
    
    return {
        "clip_id": clip_id,
        "dialogue_id": clip.dialogue_id,
        "total_attempts": clip.generation_attempt,
        "attempts_remaining": 3 - clip.generation_attempt,
        "versions": versions,
    }


@app.post("/api/jobs/{job_id}/cleanup-versions")
async def cleanup_clip_versions(job_id: str, db: Session = Depends(get_db_session)):
    """
    Clean up duplicate versions in all clips of a job.
    Call this to fix clips that have duplicate entries in versions_json.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
    cleaned_count = 0
    
    for clip in clips:
        if not clip.versions_json:
            continue
            
        versions = json.loads(clip.versions_json)
        original_count = len(versions)
        
        # Deduplicate by attempt number
        seen = {}
        for v in versions:
            attempt = v.get("attempt")
            if attempt:
                seen[attempt] = v
        
        cleaned_versions = sorted(seen.values(), key=lambda x: x.get("attempt", 0))
        
        if len(cleaned_versions) < original_count:
            clip.versions_json = json.dumps(cleaned_versions)
            cleaned_count += 1
            print(f"[Cleanup] Clip {clip.clip_index}: {original_count} -> {len(cleaned_versions)} versions", flush=True)
    
    db.commit()
    
    add_job_log(db, job_id, f"Cleaned up versions for {cleaned_count} clips", "INFO", "cleanup")
    
    return {
        "success": True,
        "clips_cleaned": cleaned_count,
        "total_clips": len(clips)
    }


@app.get("/api/jobs/{job_id}/review-status")
async def get_job_review_status(job_id: str, db: Session = Depends(get_db_session)):
    """Get summary of clip approval statuses for a job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
    
    summary = {
        "total": len(clips),
        "pending_review": 0,
        "approved": 0,
        "redo_queued": 0,
        "max_attempts": 0,
        "generating": 0,
        "failed": 0,
    }
    
    for c in clips:
        if c.status == ClipStatus.COMPLETED.value:
            if c.approval_status == "approved":
                summary["approved"] += 1
            elif c.approval_status == "max_attempts":
                summary["max_attempts"] += 1
            else:
                summary["pending_review"] += 1
        elif c.status == ClipStatus.REDO_QUEUED.value:
            summary["redo_queued"] += 1
        elif c.status in [ClipStatus.GENERATING.value, ClipStatus.PENDING.value]:
            summary["generating"] += 1
        elif c.status == ClipStatus.FAILED.value:
            summary["failed"] += 1
    
    summary["all_approved"] = summary["approved"] == summary["total"]
    summary["needs_attention"] = summary["max_attempts"] > 0
    
    return summary


# ============ Logs ============

@app.get("/api/jobs/{job_id}/logs", response_model=List[LogResponse])
async def get_job_logs(
    job_id: str,
    since_id: int = 0,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db_session),
):
    """Get logs for a job (supports polling with since_id)"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = get_job_logs_since(db, job_id, since_id)[:limit]
    
    return [
        LogResponse(
            id=log.id,
            created_at=log.created_at.isoformat() if log.created_at else "",
            level=log.level,
            category=log.category,
            clip_index=log.clip_index,
            message=log.message,
        )
        for log in logs
    ]


# ============ Server-Sent Events ============

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(job_id: str, db: Session = Depends(get_db_session)):
    """
    Stream job events via Server-Sent Events.
    
    Events:
    - progress: Clip progress update
    - clip_started: Clip generation started
    - clip_completed: Clip generation completed
    - error: Error occurred
    - job_completed: Job finished
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        event_queue = worker.subscribe(job_id)
        
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'progress': job.progress_percent})}\n\n"
            
            while True:
                try:
                    # Non-blocking check
                    event = event_queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # Stop streaming if job completed
                    if event.get("type") == "job_completed":
                        break
                        
                except Exception:
                    # Send keepalive
                    yield f": keepalive\n\n"
                    
                    # Check if job is still active
                    from models import get_db
                    with get_db() as check_db:
                        check_job = check_db.query(Job).filter(Job.id == job_id).first()
                        if check_job and check_job.status in [
                            JobStatus.COMPLETED.value,
                            JobStatus.FAILED.value,
                            JobStatus.CANCELLED.value,
                        ]:
                            break
        finally:
            worker.unsubscribe(job_id, event_queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============ Downloads ============

@app.get("/api/jobs/{job_id}/outputs")
async def list_outputs(
    job_id: str, 
    approved_only: bool = False,
    db: Session = Depends(get_db_session)
):
    """
    List generated videos for a job.
    
    If approved_only=True, only returns videos from approved clips (selected variants).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    output_dir = Path(job.output_dir)
    
    if not output_dir.exists():
        return {"job_id": job_id, "videos": [], "count": 0}
    
    videos = []
    
    if approved_only:
        # Only return approved clips' selected variants
        clips = db.query(Clip).filter(
            Clip.job_id == job_id,
            Clip.approval_status == "approved"
        ).order_by(Clip.clip_index).all()
        
        for clip in clips:
            if clip.output_filename:
                filepath = output_dir / clip.output_filename
                if filepath.exists():
                    videos.append({
                        "filename": clip.output_filename,
                        "size": filepath.stat().st_size,
                        "url": f"/api/jobs/{job_id}/outputs/{clip.output_filename}",
                        "clip_index": clip.clip_index,
                        "variant": clip.selected_variant,
                    })
    else:
        # Return all videos
        for f in output_dir.glob("*.mp4"):
            videos.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "url": f"/api/jobs/{job_id}/outputs/{f.name}",
            })
    
    videos.sort(key=lambda x: x.get("clip_index", 0) if approved_only else x["filename"])
    
    return {"job_id": job_id, "videos": videos, "count": len(videos)}


@app.get("/api/jobs/{job_id}/outputs/{filename}")
async def download_output(job_id: str, filename: str, db: Session = Depends(get_db_session)):
    """Download a generated video"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    filepath = Path(job.output_dir) / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=filename,
    )


# ============ Script Splitting ============

class ScriptSplitRequest(BaseModel):
    script: str
    language: str = "English"

# Speaking rates by language (words per second) for natural speech
LANGUAGE_SPEAKING_RATES = {
    "English": 2.5,      # ~150 wpm → 17-18 words per 7 sec
    "Italian": 2.8,      # ~168 wpm → 19-20 words per 7 sec  
    "Spanish": 2.8,      # ~168 wpm → 19-20 words per 7 sec
    "French": 2.5,       # ~150 wpm → 17-18 words per 7 sec
    "German": 2.2,       # ~132 wpm → 15-16 words per 7 sec
    "Portuguese": 2.6,   # ~156 wpm → 18-19 words per 7 sec
    "Dutch": 2.3,        # ~138 wpm → 16-17 words per 7 sec
    "Polish": 2.4,       # ~144 wpm → 17 words per 7 sec
    "Russian": 2.3,      # ~138 wpm → 16-17 words per 7 sec
    "Japanese": 4.0,     # ~240 morae/min → 28 chars per 7 sec
    "Chinese": 3.5,      # ~210 chars/min → 24-25 chars per 7 sec
    "Korean": 3.5,       # Similar to Chinese
    "Arabic": 2.5,       # ~150 wpm → 17-18 words per 7 sec
    "Hindi": 2.6,        # ~156 wpm → 18-19 words per 7 sec
}

TARGET_DURATION_SECONDS = 7

@app.post("/api/split-script")
async def split_script(request: ScriptSplitRequest):
    """
    Split a full script into ~7 second dialogue lines using OpenAI.
    Preserves the EXACT original text - only splits, never rewrites.
    Every line MUST be approximately 7 seconds (enforced via post-processing).
    """
    import os
    
    # Get OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    # Get language-specific rate
    words_per_sec = LANGUAGE_SPEAKING_RATES.get(request.language, 2.5)
    target_words = int(words_per_sec * TARGET_DURATION_SECONDS)
    min_words = max(10, target_words - 5)  # Minimum words per line
    
    # Count total words to estimate expected clips
    total_words = len(request.script.split())
    expected_clips = max(1, round(total_words / target_words))
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        prompt = f"""TASK: Split this script into chunks of EXACTLY ~{target_words} words each.

⚠️ ABSOLUTE REQUIREMENTS:
1. EVERY chunk MUST have AT LEAST {min_words} words (this is ~7 seconds of speech)
2. NEVER create a chunk with less than {min_words} words
3. If a sentence is short, COMBINE it with the next sentence(s) until you reach {min_words}+ words
4. The LAST chunk can be slightly shorter only if all remaining text is less than {min_words} words
5. Preserve EXACT original text - do NOT add, remove, or change any words

ORIGINAL SCRIPT ({total_words} total words):
"{request.script}"

MATH: {total_words} words ÷ {target_words} words = ~{expected_clips} chunks expected

EXAMPLES of what NOT to do:
❌ ["Short sentence.", "Another short one."] - BAD, each under {min_words} words
✅ ["Short sentence. Another short one. And more text here."] - GOOD, combined to reach {min_words}+ words

OUTPUT: JSON array only. Each string MUST have {min_words}+ words.
["chunk with {min_words}+ words here", "another chunk with {min_words}+ words"]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON - handle potential markdown code blocks
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        lines = json.loads(result)
        
        if not isinstance(lines, list) or len(lines) == 0:
            raise ValueError("Invalid response format")
        
        # POST-PROCESSING: Merge any lines that are too short
        merged_lines = []
        buffer = ""
        
        for line in lines:
            if buffer:
                buffer += " " + line.strip()
            else:
                buffer = line.strip()
            
            word_count = len(buffer.split())
            
            # If buffer has enough words, add it to merged_lines
            if word_count >= min_words:
                merged_lines.append(buffer)
                buffer = ""
        
        # Handle remaining buffer
        if buffer:
            if merged_lines:
                # Append to last line if buffer is too short
                buffer_words = len(buffer.split())
                if buffer_words < min_words:
                    merged_lines[-1] = merged_lines[-1] + " " + buffer
                else:
                    merged_lines.append(buffer)
            else:
                # Only one line in total
                merged_lines.append(buffer)
        
        # Clean up whitespace
        merged_lines = [" ".join(line.split()) for line in merged_lines]
        
        # Calculate average duration estimate using language-specific rate
        total_words_result = sum(len(line.split()) for line in merged_lines)
        avg_words = total_words_result / len(merged_lines) if merged_lines else 0
        avg_duration = round(avg_words / words_per_sec, 1)
        
        # Calculate per-line stats
        line_stats = []
        for line in merged_lines:
            wc = len(line.split())
            dur = round(wc / words_per_sec, 1)
            line_stats.append({"words": wc, "duration_sec": dur})
        
        return {
            "success": True,
            "lines": merged_lines,
            "count": len(merged_lines),
            "avg_duration": avg_duration,
            "total_words": total_words_result,
            "target_words_per_line": target_words,
            "min_words_per_line": min_words,
            "language": request.language,
            "line_stats": line_stats
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenAI library not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Script splitting failed: {str(e)}")


# ============ Error Codes Reference ============

@app.get("/api/error-codes")
async def get_error_codes():
    """Get list of all error codes and their meanings"""
    return {
        code.value: {
            "name": code.name,
            "value": code.value,
        }
        for code in ErrorCode
    }


# ============ Health Check ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Check if genai SDK is available
    try:
        from veo_generator import GENAI_AVAILABLE
        sdk_status = "installed" if GENAI_AVAILABLE else "not_installed"
    except:
        sdk_status = "unknown"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "workers": {
            "running_jobs": len(worker.running_jobs),
            "max_workers": worker.max_workers,
        },
        "sdk": {
            "google_genai": sdk_status,
            "message": "Video generation available" if sdk_status == "installed" else "Install google-genai for video generation"
        }
    }


# ============ Admin - API Keys ============

@app.get("/api/admin/keys")
async def get_api_keys_status():
    """
    Check status of API keys configured on the server.
    
    Keys are loaded from .env file:
    - GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    - OPENAI_API_KEY (optional)
    """
    status = api_keys_config.get_status()
    
    # Add masked preview of keys with block status
    masked_keys = []
    for i, key in enumerate(api_keys_config.gemini_api_keys):
        if len(key) > 12:
            masked = f"{key[:8]}...{key[-4:]}"
        else:
            masked = "***"
        
        is_blocked = api_keys_config.is_key_blocked(i)
        blocked_info = None
        if is_blocked and i in api_keys_config.blocked_keys:
            from datetime import datetime, timedelta
            block_time = api_keys_config.blocked_keys[i]
            unblock_time = block_time + timedelta(hours=api_keys_config.block_duration_hours)
            remaining = unblock_time - datetime.now()
            blocked_info = {
                "blocked_at": block_time.isoformat(),
                "unblocks_at": unblock_time.isoformat(),
                "remaining_hours": round(max(0, remaining.total_seconds() / 3600), 1)
            }
        
        masked_keys.append({
            "index": i + 1,
            "masked": masked,
            "is_current": i == api_keys_config.current_key_index,
            "is_blocked": is_blocked,
            "blocked_info": blocked_info
        })
    
    return {
        **status,
        "gemini_keys": masked_keys,
        "openai_masked": f"{api_keys_config.openai_api_key[:8]}...{api_keys_config.openai_api_key[-4:]}" if api_keys_config.openai_api_key else None,
        "config_file": ".env",
        "block_duration_hours": api_keys_config.block_duration_hours,
        "instructions": "Add keys to .env file and restart server to update"
    }


@app.post("/api/admin/keys/unblock/{key_index}")
async def unblock_api_key(key_index: int):
    """
    Manually unblock a specific API key before the 12h timeout.
    key_index is 1-based (1, 2, 3, etc.)
    """
    actual_index = key_index - 1  # Convert to 0-based
    
    if actual_index < 0 or actual_index >= len(api_keys_config.gemini_api_keys):
        raise HTTPException(status_code=400, detail=f"Invalid key index. Must be 1-{len(api_keys_config.gemini_api_keys)}")
    
    if actual_index not in api_keys_config.blocked_keys:
        return {
            "success": True,
            "message": f"Key {key_index} was not blocked",
            "key_index": key_index
        }
    
    del api_keys_config.blocked_keys[actual_index]
    
    return {
        "success": True,
        "message": f"Key {key_index} has been unblocked",
        "key_index": key_index,
        "available_keys": api_keys_config.get_available_key_count(),
        "blocked_keys": len(api_keys_config.blocked_keys)
    }


@app.post("/api/admin/keys/unblock-all")
async def unblock_all_api_keys():
    """
    Unblock all API keys at once.
    """
    blocked_count = len(api_keys_config.blocked_keys)
    api_keys_config.blocked_keys.clear()
    
    return {
        "success": True,
        "message": f"Unblocked {blocked_count} keys",
        "unblocked_count": blocked_count,
        "available_keys": api_keys_config.get_available_key_count()
    }


@app.post("/api/admin/keys/rotate")
async def rotate_api_key(block_current: bool = False):
    """Manually rotate to the next Gemini API key"""
    if not api_keys_config.gemini_api_keys:
        raise HTTPException(status_code=400, detail="No Gemini keys configured")
    
    old_index = api_keys_config.current_key_index
    api_keys_config.rotate_key(block_current=block_current)
    new_index = api_keys_config.current_key_index
    
    return {
        "success": True,
        "previous_index": old_index,
        "current_index": new_index,
        "total_keys": len(api_keys_config.gemini_api_keys)
    }


@app.post("/api/admin/keys/reload")
async def reload_api_keys():
    """
    Reload API keys from .env file without restarting server.
    Useful after updating .env file.
    """
    from config import get_gemini_keys_from_env, get_openai_key_from_env
    from dotenv import load_dotenv
    
    # Reload .env file
    load_dotenv(override=True)
    
    # Update keys
    old_count = len(api_keys_config.gemini_api_keys)
    api_keys_config.gemini_api_keys = get_gemini_keys_from_env()
    api_keys_config.openai_api_key = get_openai_key_from_env()
    api_keys_config.current_key_index = 0  # Reset to first key
    
    new_count = len(api_keys_config.gemini_api_keys)
    
    return {
        "success": True,
        "previous_gemini_count": old_count,
        "current_gemini_count": new_count,
        "openai_configured": api_keys_config.openai_api_key is not None,
        "message": f"Loaded {new_count} Gemini key(s) from .env"
    }


# ============ Main Entry Point ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
    )