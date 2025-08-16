"""SQLAlchemy 2.0 models for Logistics Email AI database."""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, BigInteger, Float, 
    ForeignKey, UniqueConstraint, Index, Text, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Tenant(Base):
    """Tenant model for multi-tenancy."""
    
    __tablename__ = "tenants"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    threads: Mapped[List["Thread"]] = relationship("Thread", back_populates="tenant", cascade="all, delete-orphan")
    emails: Mapped[List["Email"]] = relationship("Email", back_populates="tenant", cascade="all, delete-orphan")
    attachments: Mapped[List["Attachment"]] = relationship("Attachment", back_populates="tenant", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="tenant", cascade="all, delete-orphan")
    labels: Mapped[List["Label"]] = relationship("Label", back_populates="tenant", cascade="all, delete-orphan")
    eval_runs: Mapped[List["EvalRun"]] = relationship("EvalRun", back_populates="tenant", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_tenants_name", "name"),
    )


class Thread(Base):
    """Email thread model."""
    
    __tablename__ = "threads"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    subject_norm: Mapped[Optional[str]] = mapped_column(String(500))
    first_email_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"))
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="threads")
    emails: Mapped[List["Email"]] = relationship("Email", back_populates="thread", foreign_keys="[Email.thread_id]", cascade="all, delete-orphan")
    first_email: Mapped[Optional["Email"]] = relationship("Email", foreign_keys=[first_email_id])
    
    __table_args__ = (
        Index("idx_threads_tenant_last_message", "tenant_id", "last_message_at", postgresql_ops={"last_message_at": "DESC"}),
        Index("idx_threads_tenant_subject", "tenant_id", "subject_norm"),
    )


class Email(Base):
    """Email model."""
    
    __tablename__ = "emails"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("threads.id"), nullable=False)
    message_id: Mapped[str] = mapped_column(String(500), nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(String(500))
    from_addr: Mapped[Optional[str]] = mapped_column(String(255))
    to_addrs: Mapped[Optional[dict]] = mapped_column(JSONB)
    cc_addrs: Mapped[Optional[dict]] = mapped_column(JSONB)
    bcc_addrs: Mapped[Optional[dict]] = mapped_column(JSONB)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    in_reply_to: Mapped[Optional[str]] = mapped_column(String(500))
    references: Mapped[Optional[dict]] = mapped_column(JSONB)
    snippet: Mapped[Optional[str]] = mapped_column(Text)
    has_attachments: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_object_key: Mapped[Optional[str]] = mapped_column(String(500))
    norm_object_key: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="emails")
    thread: Mapped["Thread"] = relationship("Thread", back_populates="emails", foreign_keys=[thread_id])
    attachments: Mapped[List["Attachment"]] = relationship("Attachment", back_populates="email", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="email", cascade="all, delete-orphan")
    email_labels: Mapped[List["EmailLabel"]] = relationship("EmailLabel", back_populates="email", cascade="all, delete-orphan")
    eval_runs: Mapped[List["EvalRun"]] = relationship("EvalRun", back_populates="email", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "message_id", name="uq_emails_tenant_message"),
        Index("idx_emails_tenant_thread_sent", "tenant_id", "thread_id", "sent_at", postgresql_ops={"sent_at": "DESC"}),
        Index("idx_emails_tenant_created", "tenant_id", "created_at", postgresql_ops={"created_at": "DESC"}),
        Index("idx_emails_tenant_attachments", "tenant_id", "has_attachments"),
        Index("idx_emails_tenant_from", "tenant_id", "from_addr"),
        Index("idx_emails_tenant_subject", "tenant_id", "subject"),
    )


class Attachment(Base):
    """Email attachment model."""
    
    __tablename__ = "attachments"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    email_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"), nullable=False)
    filename: Mapped[Optional[str]] = mapped_column(String(500))
    mimetype: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    object_key: Mapped[Optional[str]] = mapped_column(String(500))
    ocr_text_object_key: Mapped[Optional[str]] = mapped_column(String(500))
    ocr_state: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="attachments")
    email: Mapped["Email"] = relationship("Email", back_populates="attachments")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="attachment", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_attachments_tenant_email", "tenant_id", "email_id"),
        Index("idx_attachments_tenant_mimetype", "tenant_id", "mimetype"),
        Index("idx_attachments_tenant_size", "tenant_id", "size_bytes"),
    )


class Chunk(Base):
    """Email content chunk model."""
    
    __tablename__ = "chunks"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    email_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"), nullable=False)
    attachment_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("attachments.id"))
    chunk_uid: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="chunks")
    email: Mapped["Email"] = relationship("Email", back_populates="chunks")
    attachment: Mapped[Optional["Attachment"]] = relationship("Attachment", back_populates="chunks")
    
    __table_args__ = (
        Index("idx_chunks_tenant_email", "tenant_id", "email_id"),
        Index("idx_chunks_tenant_uid", "tenant_id", "chunk_uid"),
        Index("idx_chunks_tenant_attachment", "tenant_id", "attachment_id"),
        Index("idx_chunks_tenant_tokens", "tenant_id", "token_count"),
    )


class Label(Base):
    """Email label model."""
    
    __tablename__ = "labels"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="labels")
    email_labels: Mapped[List["EmailLabel"]] = relationship("EmailLabel", back_populates="label", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_labels_tenant_name"),
        Index("idx_labels_tenant_name", "tenant_id", "name"),
    )


class EmailLabel(Base):
    """Email-label association model."""
    
    __tablename__ = "email_labels"
    
    # Composite primary key
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False, primary_key=True)
    email_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"), nullable=False, primary_key=True)
    label_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("labels.id"), nullable=False, primary_key=True)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant")
    email: Mapped["Email"] = relationship("Email", back_populates="email_labels")
    label: Mapped["Label"] = relationship("Label", back_populates="email_labels")
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "email_id", "label_id", name="uq_email_labels_tenant_email_label"),
        Index("idx_email_labels_tenant_email", "tenant_id", "email_id"),
        Index("idx_email_labels_tenant_label", "tenant_id", "label_id"),
    )


class EvalRun(Base):
    """Evaluation run model."""
    
    __tablename__ = "eval_runs"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    email_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"))
    draft_id: Mapped[Optional[str]] = mapped_column(String(255))
    rubric: Mapped[Optional[str]] = mapped_column(String(100))
    score_grounding: Mapped[Optional[float]] = mapped_column(Float)
    score_completeness: Mapped[Optional[float]] = mapped_column(Float)
    score_tone: Mapped[Optional[float]] = mapped_column(Float)
    score_policy: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="eval_runs")
    email: Mapped[Optional["Email"]] = relationship("Email", back_populates="eval_runs")
    
    __table_args__ = (
        Index("idx_eval_runs_tenant_email", "tenant_id", "email_id"),
        Index("idx_eval_runs_tenant_created", "tenant_id", "created_at", postgresql_ops={"created_at": "DESC"}),
        Index("idx_eval_runs_tenant_scores", "tenant_id", "score_grounding", "score_completeness", "score_tone", "score_policy"),
    )


class DedupLineage(Base):
    """Deduplication lineage tracking model."""
    
    __tablename__ = "dedup_lineage"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("tenants.id"), nullable=False)
    email_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("emails.id"), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    duplicate_type: Mapped[str] = mapped_column(String(20), nullable=False)  # exact, near
    original_content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    similarity_score: Mapped[Optional[float]] = mapped_column(Float)
    dedup_algorithm: Mapped[Optional[str]] = mapped_column(String(50))
    reference_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant")
    email: Mapped["Email"] = relationship("Email")
    
    __table_args__ = (
        Index("idx_dedup_lineage_tenant_hash", "tenant_id", "content_hash"),
        Index("idx_dedup_lineage_tenant_email", "tenant_id", "email_id"),
        Index("idx_dedup_lineage_tenant_type", "tenant_id", "duplicate_type"),
    )
