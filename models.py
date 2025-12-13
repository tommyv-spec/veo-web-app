# -*- coding: utf-8 -*-
"""
Database models for Veo Web App
Uses SQLAlchemy with SQLite (easily upgradeable to PostgreSQL)
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Boolean, Float, ForeignKey, Enum as SQLEnum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from contextlib import contextmanager

from config import (
    JobStatus, ClipStatus, ErrorCode, app_config,
    AspectRatio, Resolution, Duration, PersonGeneration
)

Base = declarative_base()


class Job(Base):
    """Main job table - one job = one video generation request"""
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True)  # UUID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Status
    status = Column(String(20), default=JobStatus.PENDING.value)
    progress_percent = Column(Float, default=0.0)
    
    # Configuration (stored as JSON for flexibility)
    config_json = Column(Text, nullable=False)
    
    # Dialogue lines (stored as JSON)
    dialogue_json = Column(Text, nullable=False)
    
    # API keys (encrypted in production!)
    api_keys_json = Column(Text, nullable=True)  # Should encrypt in production
    
    # Statistics
    total_clips = Column(Integer, default=0)
    completed_clips = Column(Integer, default=0)
    failed_clips = Column(Integer, default=0)
    skipped_clips = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Upload paths
    images_dir = Column(String(500), nullable=False)
    output_dir = Column(String(500), nullable=False)
    
    # Relationships
    clips = relationship("Clip", back_populates="job", cascade="all, delete-orphan")
    logs = relationship("JobLog", back_populates="job", cascade="all, delete-orphan")
    blacklist = relationship("BlacklistEntry", back_populates="job", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "total_clips": self.total_clips,
            "completed_clips": self.completed_clips,
            "failed_clips": self.failed_clips,
            "skipped_clips": self.skipped_clips,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": json.loads(self.config_json) if self.config_json else {},
            "dialogue_count": len(json.loads(self.dialogue_json)) if self.dialogue_json else 0,
        }


class Clip(Base):
    """Individual clip within a job"""
    __tablename__ = "clips"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    
    # Clip identification
    clip_index = Column(Integer, nullable=False)  # 0-based index
    dialogue_id = Column(Integer, nullable=False)  # ID from dialogue line
    dialogue_text = Column(Text, nullable=False)
    
    # Status
    status = Column(String(20), default=ClipStatus.PENDING.value)
    retry_count = Column(Integer, default=0)
    
    # Frame info
    start_frame = Column(String(255), nullable=True)
    end_frame = Column(String(255), nullable=True)
    
    # Generation parameters (for regeneration)
    prompt_text = Column(Text, nullable=True)
    
    # Output
    output_filename = Column(String(500), nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Error info
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # === NEW: Approval & Redo System ===
    approval_status = Column(String(20), default="pending_review")  # pending_review, approved, rejected, max_attempts
    generation_attempt = Column(Integer, default=1)  # 1, 2, or 3 (max)
    redo_reason = Column(Text, nullable=True)  # User's reason for requesting redo
    use_logged_params = Column(Boolean, default=True)  # True for attempt 2, False for attempt 3
    
    # History of all generated versions (JSON array)
    # Format: [{"attempt": 1, "filename": "...", "generated_at": "...", "approved": false}, ...]
    versions_json = Column(Text, default="[]")
    
    # Currently selected variant (1-based, matches attempt number)
    selected_variant = Column(Integer, default=1)
    
    # Relationship
    job = relationship("Job", back_populates="clips")
    
    def to_dict(self) -> Dict[str, Any]:
        raw_versions = json.loads(self.versions_json) if self.versions_json else []
        
        # Deduplicate versions by attempt number (keep last one for each attempt)
        seen = {}
        for v in raw_versions:
            attempt = v.get("attempt")
            if attempt:
                seen[attempt] = v
        versions = sorted(seen.values(), key=lambda x: x.get("attempt", 0))
        
        # Calculate total variants from deduplicated list
        total_variants = len(versions)
        
        return {
            "id": self.id,
            "clip_index": self.clip_index,
            "dialogue_id": self.dialogue_id,
            "dialogue_text": self.dialogue_text[:100] + "..." if len(self.dialogue_text) > 100 else self.dialogue_text,
            "status": self.status,
            "retry_count": self.retry_count,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "output_filename": self.output_filename,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            # Approval fields
            "approval_status": self.approval_status,
            "generation_attempt": self.generation_attempt,
            "redo_reason": self.redo_reason,
            "attempts_remaining": 3 - self.generation_attempt,
            # Variant fields
            "versions": versions,
            "total_variants": total_variants,
            "selected_variant": self.selected_variant or self.generation_attempt or 1,
        }


class JobLog(Base):
    """Log entries for a job - enables real-time streaming"""
    __tablename__ = "job_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Log info
    level = Column(String(10), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    category = Column(String(50), nullable=True)  # clip, api, system, etc.
    clip_index = Column(Integer, nullable=True)
    
    message = Column(Text, nullable=False)
    details_json = Column(Text, nullable=True)  # Extra context as JSON
    
    # Relationship
    job = relationship("Job", back_populates="logs")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "level": self.level,
            "category": self.category,
            "clip_index": self.clip_index,
            "message": self.message,
            "details": json.loads(self.details_json) if self.details_json else None,
        }


class BlacklistEntry(Base):
    """Blacklisted images for a job"""
    __tablename__ = "blacklist"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Image info
    image_filename = Column(String(255), nullable=False)
    reason = Column(String(50), nullable=True)  # celebrity_filter, generation_failed, etc.
    details = Column(Text, nullable=True)
    
    # Relationship
    job = relationship("Job", back_populates="blacklist")


class GenerationLog(Base):
    """Persistent log of generation parameters for each video"""
    __tablename__ = "generation_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    
    # Video identification
    video_id = Column(Integer, nullable=False)
    
    # Generation parameters
    images_dir = Column(String(500), nullable=False)
    start_frame = Column(String(255), nullable=False)
    end_frame = Column(String(255), nullable=True)
    dialogue_line = Column(Text, nullable=False)
    language = Column(String(50), nullable=False)
    prompt_text = Column(Text, nullable=False)
    video_filename = Column(String(500), nullable=False)
    aspect_ratio = Column(String(10), nullable=False)
    resolution = Column(String(10), nullable=False)
    duration = Column(String(5), nullable=False)
    
    generated_at = Column(DateTime, default=datetime.utcnow)


# Database setup
engine = None
SessionLocal = None


def init_db(database_url: str = None):
    """Initialize database connection"""
    global engine, SessionLocal
    
    if database_url is None:
        database_url = f"sqlite:///{app_config.data_dir / 'jobs.db'}"
    
    engine = create_engine(
        database_url, 
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Run migrations for new columns
    _run_migrations(engine)
    
    return engine


def _run_migrations(engine):
    """Add new columns to existing tables if they don't exist"""
    from sqlalchemy import text
    
    migrations = [
        # Add selected_variant column to clips table
        ("clips", "selected_variant", "ALTER TABLE clips ADD COLUMN selected_variant INTEGER DEFAULT 1"),
    ]
    
    with engine.connect() as conn:
        for table, column, sql in migrations:
            try:
                # Check if column exists
                result = conn.execute(text(f"PRAGMA table_info({table})"))
                columns = [row[1] for row in result]
                
                if column not in columns:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"[Migration] Added column {column} to {table}", flush=True)
            except Exception as e:
                print(f"[Migration] Skipped {column}: {e}", flush=True)


@contextmanager
def get_db() -> Session:
    """Get database session as context manager"""
    if SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session (for FastAPI dependency injection)"""
    if SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions

def add_job_log(
    db: Session,
    job_id: str,
    message: str,
    level: str = "INFO",
    category: str = None,
    clip_index: int = None,
    details: Dict = None
):
    """Add a log entry for a job"""
    log = JobLog(
        job_id=job_id,
        level=level,
        category=category,
        clip_index=clip_index,
        message=message,
        details_json=json.dumps(details) if details else None
    )
    db.add(log)
    db.commit()
    return log


def get_job_logs_since(db: Session, job_id: str, since_id: int = 0) -> List[JobLog]:
    """Get logs for a job since a given ID (for polling)"""
    return db.query(JobLog).filter(
        JobLog.job_id == job_id,
        JobLog.id > since_id
    ).order_by(JobLog.id.asc()).all()


def update_job_progress(db: Session, job_id: str):
    """Recalculate job progress from clips"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return
    
    clips = db.query(Clip).filter(Clip.job_id == job_id).all()
    
    total = len(clips)
    completed = sum(1 for c in clips if c.status == ClipStatus.COMPLETED.value)
    failed = sum(1 for c in clips if c.status == ClipStatus.FAILED.value)
    skipped = sum(1 for c in clips if c.status == ClipStatus.SKIPPED.value)
    
    job.total_clips = total
    job.completed_clips = completed
    job.failed_clips = failed
    job.skipped_clips = skipped
    
    if total > 0:
        job.progress_percent = ((completed + skipped) / total) * 100
    
    db.commit()