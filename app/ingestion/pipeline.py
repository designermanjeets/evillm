"""Main ingestion pipeline for email processing."""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import structlog
from pathlib import Path

from .models import (
    EmailMetadata, EmailContent, AttachmentInfo, DedupResult, 
    ChunkInfo, ThreadInfo, OCRTask, EmbeddingJob, CheckpointInfo,
    BatchManifest, IngestionMetrics, ProcessingStatus, ErrorType,
    generate_batch_id, generate_chunk_uid, generate_task_id, generate_job_id
)
from .parser import EmailParser
from .normalizer import ContentNormalizer
from .attachments import AttachmentProcessor
from .deduplication import DeduplicationEngine
from .chunking import SemanticChunker
from .threading import ThreadManager
from .checkpoints import CheckpointManager
from .metrics import MetricsCollector
from ..storage import get_storage_client, StoragePathBuilder
from ..storage.metadata import ObjectMetadata, ContentHash
from ..config.ingestion import get_ingestion_settings

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Main ingestion pipeline for processing email batches."""
    
    def __init__(self, tenant_id: str):
        """Initialize ingestion pipeline."""
        self.tenant_id = tenant_id
        self.settings = get_ingestion_settings()
        self.storage_client = get_storage_client()
        
        # Initialize components
        self.email_parser = EmailParser()
        self.content_normalizer = ContentNormalizer()
        self.attachment_processor = AttachmentProcessor()
        self.deduplication_engine = DeduplicationEngine()
        self.semantic_chunker = SemanticChunker()
        self.thread_manager = ThreadManager()
        self.checkpoint_manager = CheckpointManager()
        self.metrics_collector = MetricsCollector()
        
        # Pipeline state
        self.current_batch_id = None
        self.current_batch_manifest = None
        self.processing_stats = {
            'total_emails': 0,
            'processed_emails': 0,
            'failed_emails': 0,
            'quarantined_emails': 0,
            'duplicate_emails': 0
        }
    
    async def process_batch(self, batch_manifest: BatchManifest) -> IngestionMetrics:
        """Process a batch of emails."""
        try:
            self.current_batch_id = batch_manifest.batch_id
            self.current_batch_manifest = batch_manifest
            
            logger.info("Starting batch processing", 
                       batch_id=batch_manifest.batch_id,
                       tenant_id=self.tenant_id,
                       total_files=batch_manifest.total_files)
            
            # Initialize metrics
            self.metrics_collector.start_batch(batch_manifest.batch_id)
            
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                batch_manifest.batch_id, self.tenant_id, batch_manifest.total_files
            )
            
            # Process emails
            await self._process_emails(batch_manifest)
            
            # Finalize batch
            await self._finalize_batch()
            
            # Get final metrics
            metrics = self.metrics_collector.get_metrics()
            
            logger.info("Batch processing completed",
                       batch_id=batch_manifest.batch_id,
                       metrics=metrics.dict())
            
            return metrics
            
        except Exception as exc:
            logger.error("Batch processing failed", 
                        batch_id=batch_manifest.batch_id,
                        exc_info=exc)
            raise
    
    async def _process_emails(self, batch_manifest: BatchManifest):
        """Process individual emails in the batch."""
        # Get email files from manifest or dropbox
        if batch_manifest.file_manifest:
            # Use file paths from manifest
            email_files = [file_info["file_path"] for file_info in batch_manifest.file_manifest]
            logger.info("Processing emails from manifest", file_count=len(email_files))
        else:
            # Fall back to dropbox scanning
            email_files = self._get_email_files(batch_manifest)
            logger.info("Processing emails from dropbox", file_count=len(email_files))
        
        for i, email_file in enumerate(email_files):
            try:
                # Check checkpoint for resume capability
                if await self.checkpoint_manager.should_skip_email(
                    batch_manifest.batch_id, email_file
                ):
                    logger.info("Skipping already processed email", email_file=email_file)
                    continue
                
                # Process single email
                await self._process_single_email(email_file, i)
                
                # Update checkpoint
                await self.checkpoint_manager.update_checkpoint(
                    batch_manifest.batch_id, email_file, i
                )
                
                # Update progress
                self.processing_stats['processed_emails'] += 1
                
            except Exception as exc:
                logger.error("Failed to process email", 
                           email_file=email_file, exc_info=exc)
                self.processing_stats['failed_emails'] += 1
                
                # Continue processing other emails
                continue
    
    async def _process_single_email(self, email_file: str, index: int):
        """Process a single email file."""
        try:
            logger.info("Processing email", email_file=email_file, index=index)
            
            # Parse email
            metadata, attachments, body_content = await self.email_parser.parse_email_file(
                email_file, self.tenant_id, self.current_batch_id
            )
            
            # Normalize content
            email_content = self.content_normalizer.normalize_content(
                body_content, metadata.content_type, metadata.charset
            )
            
            # Process attachments
            processed_attachments = await self._process_attachments(attachments)
            
            # Generate thread and email IDs first
            thread_id = self.thread_manager.generate_thread_id(metadata)
            email_id = self._generate_email_id(metadata.message_id)
            
            # Check for duplicates
            dedup_result = await self._check_duplicates(email_content, processed_attachments)
            
            if dedup_result.is_duplicate:
                logger.info("Duplicate email detected", 
                           message_id=metadata.message_id,
                           duplicate_type=dedup_result.duplicate_type)
                self.processing_stats['duplicate_emails'] += 1
                
                # Persist dedup lineage
                await self.deduplication_engine.persist_dedup_lineage(
                    dedup_result, email_id, self.tenant_id
                )
                return
            
            # Store content in storage
            storage_keys = await self._store_content(email_content, processed_attachments,
                                                   thread_id, email_id, metadata.message_id)
            
            # Persist to database
            email_id = await self._persist_to_database(metadata, email_content, 
                                                     processed_attachments, storage_keys, thread_id)
            
            # Create semantic chunks
            chunks = await self._create_chunks(email_id, email_content, processed_attachments)
            
            # Enqueue embedding jobs
            await self._enqueue_embedding_jobs(chunks)
            
            # Update threading
            await self._update_threading(metadata, email_id)
            
            logger.info("Email processed successfully",
                       message_id=metadata.message_id,
                       email_id=email_id,
                       chunks_created=len(chunks))
            
        except Exception as exc:
            logger.error("Failed to process single email", 
                        email_file=email_file, exc_info=exc)
            raise
    
    async def _process_attachments(self, attachments: List[AttachmentInfo]) -> List[AttachmentInfo]:
        """Process email attachments."""
        processed_attachments = []
        
        for attachment in attachments:
            try:
                # Validate attachment
                if not self._is_valid_attachment(attachment):
                    logger.warning("Invalid attachment, quarantining",
                                 filename=attachment.filename,
                                 content_type=attachment.content_type)
                    await self._quarantine_attachment(attachment)
                    continue
                
                # Extract text if possible
                if attachment.content_type in ['application/pdf', 'application/msword', 
                                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    extracted_text = await self.attachment_processor.extract_text(attachment)
                    if extracted_text:
                        attachment.text_extracted = True
                        attachment.extracted_text = extracted_text
                        attachment.extracted_text_hash = self._calculate_hash(extracted_text.encode('utf-8'))
                
                # Create OCR task if needed
                if attachment.ocr_required:
                    ocr_task = await self._create_ocr_task(attachment)
                    attachment.ocr_task_id = ocr_task.task_id
                
                processed_attachments.append(attachment)
                
            except Exception as exc:
                logger.error("Failed to process attachment",
                           filename=attachment.filename, exc_info=exc)
                await self._quarantine_attachment(attachment)
        
        return processed_attachments
    
    def _is_valid_attachment(self, attachment: AttachmentInfo) -> bool:
        """Check if attachment is valid for processing."""
        # Check mimetype
        if attachment.content_type not in self.settings.allowed_attachment_mimetypes:
            return False
        
        # Check size
        if attachment.content_length > self.settings.max_attachment_size:
            return False
        
        return True
    
    async def _quarantine_attachment(self, attachment: AttachmentInfo):
        """Quarantine invalid attachment."""
        if not self.settings.quarantine_enabled:
            return
        
        try:
            quarantine_path = Path(self.settings.quarantine_path)
            quarantine_path.mkdir(parents=True, exist_ok=True)
            
            quarantine_file = quarantine_path / f"quarantined_{attachment.filename}"
            with open(quarantine_file, 'wb') as f:
                f.write(attachment.content)
            
            logger.info("Attachment quarantined",
                       filename=attachment.filename,
                       quarantine_path=str(quarantine_file))
            
        except Exception as exc:
            logger.error("Failed to quarantine attachment",
                        filename=attachment.filename, exc_info=exc)
    
    async def _check_duplicates(self, email_content: EmailContent, 
                               attachments: List[AttachmentInfo]) -> DedupResult:
        """Check for duplicate content."""
        # Check exact duplicates first
        exact_duplicate = await self.deduplication_engine.check_exact_duplicate(
            email_content.normalized_text_hash
        )
        
        if exact_duplicate:
            return DedupResult(
                content_hash=email_content.normalized_text_hash,
                is_duplicate=True,
                duplicate_type="exact",
                original_content_hash=exact_duplicate
            )
        
        # Check near duplicates if enabled
        if self.settings.enable_near_dup_detection:
            near_duplicate = await self.deduplication_engine.check_near_duplicate(
                email_content.normalized_text
            )
            
            if near_duplicate:
                return DedupResult(
                    content_hash=email_content.normalized_text_hash,
                    is_duplicate=True,
                    duplicate_type="near",
                    original_content_hash=near_duplicate['content_hash'],
                    similarity_score=near_duplicate['similarity_score'],
                    dedup_algorithm=near_duplicate['algorithm']
                )
        
        return DedupResult(
            content_hash=email_content.normalized_text_hash,
            is_duplicate=False,
            duplicate_type="none"
        )
    
    async def _store_content(self, email_content: EmailContent, 
                           attachments: List[AttachmentInfo],
                           thread_id: str, email_id: str, message_id: str) -> Dict[str, str]:
        """Store email content and attachments in storage."""
        storage_keys = {}
        
        try:
            # Store raw email content
            raw_key = StoragePathBuilder.build_email_raw_path(
                self.tenant_id, thread_id, email_id, message_id
            )
            
            raw_metadata = ObjectMetadata(
                tenant_id=self.tenant_id,
                content_sha256=email_content.raw_content_hash,
                content_length=email_content.content_length,
                mimetype="message/rfc822"
            )
            
            await self.storage_client.put_object(
                raw_key, email_content.raw_content, raw_metadata, self.tenant_id
            )
            storage_keys['raw'] = raw_key
            
            # Store normalized text
            norm_key = StoragePathBuilder.build_email_normalized_path(
                self.tenant_id, thread_id, email_id, message_id
            )
            
            norm_metadata = ObjectMetadata(
                tenant_id=self.tenant_id,
                content_sha256=email_content.normalized_text_hash,
                content_length=len(email_content.normalized_text.encode('utf-8')),
                mimetype="text/plain"
            )
            
            await self.storage_client.put_object(
                norm_key, email_content.normalized_text.encode('utf-8'), 
                norm_metadata, self.tenant_id
            )
            storage_keys['normalized'] = norm_key
            
            # Store attachments
            for i, attachment in enumerate(attachments):
                att_key = StoragePathBuilder.build_attachment_path(
                    self.tenant_id, thread_id, email_id, i, attachment.filename
                )
                
                att_metadata = ObjectMetadata(
                    tenant_id=self.tenant_id,
                    content_sha256=attachment.content_hash,
                    content_length=attachment.content_length,
                    mimetype=attachment.content_type
                )
                
                await self.storage_client.put_object(
                    att_key, attachment.content, att_metadata, self.tenant_id
                )
                storage_keys[f'attachment_{i}'] = att_key
                
                # Store extracted text if available
                if attachment.text_extracted and attachment.extracted_text:
                    text_key = StoragePathBuilder.build_ocr_text_path(
                        self.tenant_id, thread_id, email_id, i
                    )
                    
                    text_metadata = ObjectMetadata(
                        tenant_id=self.tenant_id,
                        content_sha256=attachment.extracted_text_hash,
                        content_length=len(attachment.extracted_text.encode('utf-8')),
                        mimetype="text/plain"
                    )
                    
                    await self.storage_client.put_object(
                        text_key, attachment.extracted_text.encode('utf-8'),
                        text_metadata, self.tenant_id
                    )
                    storage_keys[f'attachment_{i}_text'] = text_key
            
            logger.info("Content stored successfully", storage_keys=storage_keys)
            return storage_keys
            
        except Exception as exc:
            logger.error("Failed to store content", exc_info=exc)
            raise
    
    async def _persist_to_database(self, metadata: EmailMetadata, 
                                 email_content: EmailContent,
                                 attachments: List[AttachmentInfo],
                                 storage_keys: Dict[str, str],
                                 thread_id: str) -> str:
        """Persist email data to database with storage object keys."""
        try:
            from ..database.session import get_db_session
            from ..database.models import Email, Attachment, Thread
            
            async with get_db_session() as session:
                # Create or get thread
                thread = await self._get_or_create_thread(session, thread_id, metadata)
                
                # Create email record with storage keys
                email = Email(
                    tenant_id=self.tenant_id,
                    thread_id=thread.id,
                    message_id=metadata.message_id,
                    subject=metadata.subject,
                    from_addr=metadata.from_addr,
                    to_addrs=metadata.to_addrs,
                    cc_addrs=metadata.cc_addrs,
                    bcc_addrs=metadata.bcc_addrs,
                    sent_at=metadata.sent_at,
                    received_at=metadata.received_at,
                    in_reply_to=metadata.in_reply_to,
                    references=metadata.references,
                    snippet=email_content.normalized_text[:200] + "..." if len(email_content.normalized_text) > 200 else email_content.normalized_text,
                    has_attachments=len(attachments) > 0,
                    raw_object_key=storage_keys.get('raw'),
                    norm_object_key=storage_keys.get('normalized')
                )
                
                session.add(email)
                await session.flush()  # Get the email ID
                
                # Create attachment records with storage keys
                for i, attachment in enumerate(attachments):
                    att_key = storage_keys.get(f'attachment_{i}')
                    text_key = storage_keys.get(f'attachment_{i}_text')
                    
                    db_attachment = Attachment(
                        tenant_id=self.tenant_id,
                        email_id=email.id,
                        filename=attachment.filename,
                        mimetype=attachment.content_type,
                        size_bytes=attachment.content_length,
                        object_key=att_key,
                        ocr_text_object_key=text_key,
                        ocr_state="completed" if text_key else "pending"
                    )
                    
                    session.add(db_attachment)
                
                await session.commit()
                
                logger.info("Email persisted to database with storage keys",
                           email_id=email.id,
                           message_id=metadata.message_id,
                           storage_keys=storage_keys)
                
                return email.id
                
        except Exception as exc:
            logger.error("Failed to persist email to database", exc_info=exc)
            raise
    
    async def _get_or_create_thread(self, session, thread_id: str, metadata: EmailMetadata):
        """Get or create thread for email."""
        from ..database.models import Thread
        
        # Try to find existing thread
        thread = await session.get(Thread, thread_id)
        if thread:
            return thread
        
        # Create new thread
        thread = Thread(
            id=thread_id,
            tenant_id=self.tenant_id,
            subject_norm=metadata.subject,
            first_email_id=None,  # Will be set after email creation
            last_message_at=metadata.sent_at or metadata.received_at
        )
        
        session.add(thread)
        await session.flush()
        return thread
    
    def _generate_email_id(self, message_id: str) -> str:
        """Generate a stable email ID from message ID."""
        import hashlib
        return f"email_{hashlib.sha256(message_id.encode()).hexdigest()[:16]}"
    
    async def _create_chunks(self, email_id: str, email_content: EmailContent,
                           attachments: List[AttachmentInfo]) -> List[ChunkInfo]:
        """Create semantic chunks from email content."""
        chunks = []
        
        try:
            # Create chunks from normalized text
            text_chunks = await self.semantic_chunker.create_chunks(
                email_content.normalized_text,
                email_id,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap
            )
            chunks.extend(text_chunks)
            
            # Create chunks from attachment text
            for attachment in attachments:
                if attachment.text_extracted and attachment.extracted_text:
                    att_chunks = await self.semantic_chunker.create_chunks(
                        attachment.extracted_text,
                        email_id,
                        attachment_id=attachment.filename,
                        chunk_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap
                    )
                    chunks.extend(att_chunks)
            
            logger.info("Chunks created successfully",
                       email_id=email_id,
                       total_chunks=len(chunks))
            
            return chunks
            
        except Exception as exc:
            logger.error("Failed to create chunks", email_id=email_id, exc_info=exc)
            raise
    
    async def _enqueue_embedding_jobs(self, chunks: List[ChunkInfo]):
        """Enqueue embedding jobs for chunks."""
        # This is a placeholder - actual job queue integration will be implemented
        # when we add the job queue system
        
        for chunk in chunks:
            embedding_job = EmbeddingJob(
                job_id=generate_job_id(),
                chunk_uid=chunk.chunk_uid,
                tenant_id=self.tenant_id,
                batch_id=self.current_batch_id,
                token_count=chunk.token_count
            )
            
            logger.info("Embedding job enqueued",
                       job_id=embedding_job.job_id,
                       chunk_uid=chunk.chunk_uid)
    
    async def _update_threading(self, metadata: EmailMetadata, email_id: str):
        """Update threading relationships."""
        if not self.settings.threading_enabled:
            return
        
        try:
            thread_info = await self.thread_manager.process_email_threading(
                metadata, email_id
            )
            
            logger.info("Threading updated",
                       email_id=email_id,
                       thread_id=thread_info.thread_id)
            
        except Exception as exc:
            logger.error("Failed to update threading",
                        email_id=email_id, exc_info=exc)
    
    async def _create_ocr_task(self, attachment: AttachmentInfo) -> OCRTask:
        """Create OCR task for attachment."""
        ocr_task = OCRTask(
            task_id=generate_task_id(),
            email_id="temp",  # Will be updated when email is persisted
            attachment_id=attachment.filename,
            content_type=attachment.content_type,
            content_hash=attachment.content_hash,
            content_length=attachment.content_length
        )
        
        logger.info("OCR task created",
                   task_id=ocr_task.task_id,
                   filename=attachment.filename)
        
        return ocr_task
    
    async def _finalize_batch(self):
        """Finalize batch processing."""
        try:
            # Update checkpoint
            await self.checkpoint_manager.complete_checkpoint(
                self.current_batch_id
            )
            
            # Update metrics
            self.metrics_collector.complete_batch()
            
            # Cleanup temporary data
            await self._cleanup_temp_data()
            
            logger.info("Batch finalized", batch_id=self.current_batch_id)
            
        except Exception as exc:
            logger.error("Failed to finalize batch",
                        batch_id=self.current_batch_id, exc_info=exc)
    
    async def _cleanup_temp_data(self):
        """Clean up temporary data."""
        # This is a placeholder for cleanup operations
        pass
    
    def _get_email_files(self, batch_manifest: BatchManifest) -> List[str]:
        """Get list of email files to process."""
        dropbox_path = Path(self.settings.dropbox_path)
        
        if not dropbox_path.exists():
            logger.warning("Dropbox path does not exist", path=str(dropbox_path))
            return []
        
        # Get all .eml files in the dropbox
        email_files = list(dropbox_path.glob("*.eml"))
        email_files.extend(dropbox_path.glob("*.msg"))
        
        # Sort files for deterministic processing
        email_files.sort()
        
        logger.info("Found email files", count=len(email_files))
        return [str(f) for f in email_files]
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        import hashlib
        return hashlib.sha256(content).hexdigest()


async def run_ingestion_batch(manifest: BatchManifest, tenant_id: str) -> IngestionMetrics:
    """Run ingestion batch for a tenant."""
    try:
        # Initialize pipeline
        pipeline = IngestionPipeline(tenant_id)
        
        # Process batch
        metrics = await pipeline.process_batch(manifest)
        
        return metrics
        
    except Exception as exc:
        logger.error("Ingestion batch failed", tenant_id=tenant_id, exc_info=exc)
        raise
