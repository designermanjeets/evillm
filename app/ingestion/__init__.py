"""Ingestion package for Logistics Email AI."""

from .pipeline import IngestionPipeline, run_ingestion_batch
from .parser import MIMEParser, EmailParser
from .normalizer import ContentNormalizer, HTMLNormalizer
from .attachments import AttachmentProcessor, OCRTaskManager
from .deduplication import DeduplicationEngine, SimHashProcessor
from .chunking import SemanticChunker, ChunkManager
from .threading import ThreadManager, ThreadLinker
from .checkpoints import CheckpointManager, IdempotencyManager
from .metrics import IngestionMetrics, MetricsCollector

__all__ = [
    "IngestionPipeline",
    "run_ingestion_batch",
    "MIMEParser",
    "EmailParser",
    "ContentNormalizer",
    "HTMLNormalizer",
    "AttachmentProcessor",
    "OCRTaskManager",
    "DeduplicationEngine",
    "SimHashProcessor",
    "SemanticChunker",
    "ChunkManager",
    "ThreadManager",
    "ThreadLinker",
    "CheckpointManager",
    "IdempotencyManager",
    "IngestionMetrics",
    "MetricsCollector"
]
