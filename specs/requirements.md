# Logistics Email AI - Requirements Specification

## System Overview
The Logistics Email AI system processes incoming logistics emails to provide intelligent, grounded responses with distance/price calculations and compliance verification.

## EARS Requirements

### EARS-1: Email Ingestion & Storage
**WHEN** an email is received via the ingestion pipeline, **THE SYSTEM SHALL** parse, normalize, and store the email with metadata **WITH** deduplication and idempotent processing.

**Acceptance Criteria:**
- Given: A new email arrives via ingestion endpoint
- When: The email is processed through the pipeline
- Then: Email is stored with normalized text, metadata, and stable chunk_ids
- And: Duplicate emails are detected and skipped
- And: Processing is resumable from checkpoints

### EARS-2: Multi-Tenant Isolation
**WHEN** any operation is performed, **THE SYSTEM SHALL** enforce tenant_id isolation **WITH** no cross-tenant data leakage.

**Acceptance Criteria:**
- Given: Multiple tenants exist in the system
- When: Any data operation is performed
- Then: Only data from the requesting tenant is accessible
- And: No cross-tenant data is returned in search results

### EARS-3: Hybrid Search Retrieval
**WHEN** a search query is submitted, **THE SYSTEM SHALL** return relevant email chunks using hybrid search (BM25 + vector) **WITH** Reciprocal Rank Fusion and optional reranking.

**Acceptance Criteria:**
- Given: A search query is submitted
- When: The system performs retrieval
- Then: Results combine keyword (BM25) and semantic (vector) search
- And: Results are fused using RRF algorithm
- And: Optional cross-encoder reranking is available
- And: All returned chunks include citation metadata

### EARS-4: Grounded Fact Verification
**WHEN** factual claims are made in responses, **THE SYSTEM SHALL** ensure all facts are present in retrieved chunks or deterministic tool outputs **WITH** no numeric hallucinations.

**Acceptance Criteria:**
- Given: A response is generated
- When: The response contains factual claims
- Then: All facts must be traceable to retrieved chunks or tool outputs
- And: Numeric values must come from deterministic calculations, not LLM generation

### EARS-5: Distance & Price Computation
**WHEN** distance or pricing calculations are requested, **THE SYSTEM SHALL** compute values using deterministic tools **WITH** rate cards and geographic data.

**Acceptance Criteria:**
- Given: A request for distance or pricing
- When: The system processes the request
- Then: Calculations use deterministic tools with rate cards
- And: Results are verifiable and repeatable
- And: No LLM-generated numeric values are used

### EARS-6: Compliance & Tone Verification
**WHEN** a response is drafted, **THE SYSTEM SHALL** verify compliance with policies and tone requirements **WITH** automatic redaction if violations are detected.

**Acceptance Criteria:**
- Given: A response draft is generated
- When: The draft is reviewed for compliance
- Then: Policy violations are detected and flagged
- And: Tone requirements are verified
- And: Non-compliant content is automatically redacted

### EARS-7: Performance SLOs
**WHEN** the system is under load, **THE SYSTEM SHALL** maintain performance within specified SLOs **WITH** graceful degradation.

**Acceptance Criteria:**
- Given: System is processing requests
- When: Under normal and peak load
- Then: Ingest throughput ≥100k emails/day
- And: Query p95 ≤4s to first token
- And: Full draft generation ≤8s including tools
- And: Support ≥50 concurrent draft requests

### EARS-8: Observability & Monitoring
**WHEN** system operations occur, **THE SYSTEM SHALL** provide comprehensive observability **WITH** structured logging and metrics.

**Acceptance Criteria:**
- Given: Any system operation
- When: The operation completes
- Then: Structured logs with trace_id are generated
- And: PII is redacted from logs
- And: Performance metrics are recorded
- And: Failure modes are categorized and tracked

### EARS-9: Security & PII Protection
**WHEN** sensitive data is processed, **THE SYSTEM SHALL** protect PII and maintain security **WITH** encryption and access controls.

**Acceptance Criteria:**
- Given: Sensitive data is processed
- When: Data is stored or transmitted
- Then: PII is redacted in logs
- And: Data is encrypted at rest
- And: Access is controlled via signed URLs
- And: Prompt injection is prevented

### EARS-10: Attachment Processing
**WHEN** attachments are present in emails, **THE SYSTEM SHALL** extract text content and create OCR tasks for images **WITH** proper mimetype validation and size limits.

**Acceptance Criteria:**
- Given: An email with attachments arrives
- When: The attachment is processed
- Then: Text-based attachments (DOCX, PDF) have content extracted
- And: Image attachments trigger OCR task creation
- And: Attachment metadata is stored with content hashes
- And: Size and mimetype limits are enforced

### EARS-ING-1: MIME Parsing & Checkpoints
**WHEN** a batch of emails is ingested, **THE SYSTEM SHALL** parse MIME and extract headers/parts **WITH** idempotent checkpoints and resumable progress.

**Acceptance Criteria:**
- Given: A batch of emails is submitted for ingestion
- When: MIME parsing occurs
- Then: Headers (Message-Id, In-Reply-To, References) are extracted
- And: Email parts are identified and separated
- And: Checkpoints are created for resumable processing
- And: Processing can resume from last successful point

### EARS-ING-2: Content Normalization
**WHEN** normalization runs, **THE SYSTEM SHALL** convert HTML to clean text, strip signatures/quoted text, and detect language **WITH** a normalization manifest per email.

**Acceptance Criteria:**
- Given: Raw email content is available (bytes or str)
- When: Normalization is performed
- Then: HTML is converted to clean, readable text
- And: Email signatures are stripped
- And: Quoted text is identified and marked
- And: Language is detected with confidence scores
- And: Normalization manifest records all transformations
- And: Input content is safely handled whether bytes or str
- And: Normalized text is stable for hashing and chunking

### EARS-ING-3: Attachment Processing
**WHEN** attachments are present, **THE SYSTEM SHALL** extract text (DOCX/PDF first) and create OCR tasks for images/PDFs **WITH** allowed mimetypes only.

**Acceptance Criteria:**
- Given: An email contains attachments
- When: Attachment processing occurs
- Then: DOCX and PDF text is extracted
- And: OCR tasks are created for images and PDFs without text layers
- And: Only allowed mimetypes are processed
- And: Attachment metadata includes content hashes and sizes

### EARS-ING-4: Storage Persistence
**WHEN** storing content, **THE SYSTEM SHALL** write raw/normalized/attachment objects to storage **WITH** deterministic paths and content hashes; DB rows MUST reference object keys.

**Acceptance Criteria:**
- Given: Processed email content is ready
- When: Storage operations occur
- Then: Raw emails are stored with deterministic paths (raw_object_key)
- And: Normalized text is stored separately (norm_object_key)
- And: Attachments are stored with proper metadata (attachment.object_key)
- And: OCR text is stored when available (ocr_text_object_key)
- And: Database rows reference storage object keys for all content types
- And: Content hashes are computed and stored for deduplication
- And: All storage paths are tenant-aware and deterministic
- And: Mimetype allowlist is enforced for security

### EARS-ING-5: Deduplication
**WHEN** duplicates are detected (exact hash or near-dup simhash/minhash), **THE SYSTEM SHALL** skip redundant uploads and mark dedup lineage **WITH** a per-batch dedup ratio metric.

**Acceptance Criteria:**
- Given: Multiple emails may contain duplicate content
- When: Deduplication runs
- Then: Exact duplicates (SHA-256) are skipped (no storage upload, no DB row)
- And: Near-duplicates (simhash/minhash ≥ 0.8 threshold) are linked to canonical email_id
- And: Dedup lineage is recorded with reference_count and first_seen_at
- And: Per-batch dedup ratio is computed and exposed (exact + near duplicates / total)
- And: Storage and processing costs are minimized
- And: Metrics track dedup_exact, dedup_near, and overall dedup_ratio
- And: Original content hash is preserved for future dedup decisions
- And: Batch summary shows docs_total, dedup_exact, dedup_near, and dedup_ratio
- And: Dedup lineage references are persisted to database for audit and analysis

### EARS-ING-6: Semantic Chunking
**WHEN** chunking text, **THE SYSTEM SHALL** create semantic chunks **WITH** stable chunk_uids, token_count, and email/attachment mapping; vectors are created later by a separate job.

**Acceptance Criteria:**
- Given: Normalized email text is available
- When: Chunking is performed
- Then: Semantic chunks are created with configurable window/overlap
- And: Stable chunk_uids are generated deterministically
- And: Token counts are estimated
- And: Chunks are mapped to source emails and attachments
- And: Embedding jobs are queued for later processing

### EARS-ING-7: Email Threading
**WHEN** threading is derivable, **THE SYSTEM SHALL** link emails to threads via Message-Id/In-Reply-To/References **WITH** tenant-safe lookups.

**Acceptance Criteria:**
- Given: Email headers contain threading information
- When: Threading analysis occurs
- Then: Emails are linked to threads via Message-Id
- And: Reply chains are identified via In-Reply-To/References
- And: Thread metadata is stored
- And: All lookups are tenant-scoped
- And: Fallback threading uses subject normalization

### EARS-ING-8: Error Handling & Retry
**WHEN** errors occur, **THE SYSTEM SHALL** retry transient failures with exponential backoff; permanent failures SHALL be quarantined **WITH** audit records.

**Acceptance Criteria:**
- Given: Processing errors occur during ingestion
- When: Error handling is triggered
- Then: Transient failures are retried with exponential backoff
- And: Permanent failures are quarantined
- And: Audit records are created for all failures
- And: Processing continues for other emails in batch
- And: Error metrics are collected and exposed

### EARS-ING-9: Metrics & Observability
**WHEN** ingestion completes, **THE SYSTEM SHALL** emit metrics (docs/sec, dedup_ratio, ocr_rate, failure_buckets, lag) **WITH** trace_id propagation.

**Acceptance Criteria:**
- Given: Ingestion batch processing completes
- When: Metrics are collected
- Then: Processing rate (docs/sec) is measured
- And: Deduplication ratio is computed
- And: OCR task creation rate is tracked
- And: Failure buckets categorize error types
- And: Processing lag is measured
- And: Trace IDs propagate through all operations

### EARS-ING-10: Multi-Tenant Isolation
**WHEN** multi-tenant ingestion runs, **THE SYSTEM SHALL** enforce tenant_id at all stages **WITH** no cross-tenant keys/queries.

**Acceptance Criteria:**
- Given: Multiple tenants are processing emails
- When: Ingestion operations occur
- Then: All storage paths include tenant_id prefix
- And: All database queries filter by tenant_id
- And: No cross-tenant data is accessible
- And: Tenant isolation is enforced at all pipeline stages
- And: Storage and database operations are tenant-scoped
**WHEN** emails contain attachments, **THE SYSTEM SHALL** extract and process attachments **WITH** OCR for text extraction where needed.

**Acceptance Criteria:**
- Given: An email contains attachments
- When: The email is processed
- Then: Attachments are extracted and stored
- And: OCR is applied to image/PDF attachments
- And: Text content is indexed for search
- And: Attachments are accessible via secure URLs

### EARS-11: LangGraph Agent Coordination
**WHEN** a complex request is processed, **THE SYSTEM SHALL** coordinate multiple specialized agents **WITH** token and time budget enforcement.

**Acceptance Criteria:**
- Given: A complex request requiring multiple steps
- When: The request is processed
- Then: Coordinator agent plans and orchestrates steps
- And: Token and time budgets are enforced
- And: Human approval gate is available for low-confidence responses
- And: Agent transitions are deterministic

### EARS-12: Evaluation & Quality Gates
**WHEN** responses are generated, **THE SYSTEM SHALL** evaluate quality using automated metrics **WITH** blocking gates for low-quality outputs.

**Acceptance Criteria:**
- Given: A response is generated
- When: Quality evaluation is performed
- Then: Grounding, completeness, tone, and policy scores are computed
- And: Responses below threshold are blocked
- And: Evaluation results are logged and tracked
- And: Golden examples are used for calibration

### EARS-DB-1: Database Connectivity & Health
**WHEN** the system starts up or health checks are performed, **THE SYSTEM SHALL** verify database connectivity and migration status **WITH** health endpoint failure if database is unreachable or migrations are pending.

**Acceptance Criteria:**
- Given: System performs health check
- When: Database is unreachable or migrations pending
- Then: Health endpoint returns non-OK status
- And: Current Alembic revision is exposed via health endpoint

### EARS-DB-2: Email Metadata Persistence
**WHEN** emails are processed, **THE SYSTEM SHALL** persist email metadata with object storage keys **WITH** tenant isolation and referential integrity.

**Acceptance Criteria:**
- Given: Email is processed through ingestion pipeline
- When: Email metadata is stored
- Then: All email fields are persisted with proper tenant_id
- And: Object storage keys are stored for raw and normalized content
- And: Thread relationships are maintained

### EARS-DB-3: Semantic Chunk Storage
**WHEN** email content is chunked for search, **THE SYSTEM SHALL** store semantic chunks with stable chunk_uid **WITH** token counts and metadata preservation.

**Acceptance Criteria:**
- Given: Email content is processed for chunking
- When: Chunks are created
- Then: Each chunk has a stable, unique chunk_uid
- And: Token counts are accurately calculated and stored
- And: Chunks maintain email and attachment relationships

### EARS-DB-4: Multi-Tenant Isolation
**WHEN** any database operation is performed, **THE SYSTEM SHALL** enforce tenant_id isolation **WITH** no cross-tenant data access or joins.

**Acceptance Criteria:**
- Given: Database query is executed
- When: Query involves tenant-specific data
- Then: tenant_id filter is automatically applied
- And: No cross-tenant joins are possible
- And: All tables enforce tenant_id NOT NULL constraint

### EARS-DB-5: Performance Optimization
**WHEN** database queries are executed under load, **THE SYSTEM SHALL** maintain performance within SLOs **WITH** appropriate indexes and query optimization.

**Acceptance Criteria:**
- Given: System is under load (≥50 concurrent requests)
- When: Database queries are executed
- Then: Query p95 response time ≤4s
- And: Appropriate indexes exist for common query patterns
- And: Tenant filtering is optimized

### EARS-DB-6: Migration Management
**WHEN** database schema changes are required, **THE SYSTEM SHALL** provide idempotent Alembic migrations **WITH** rollback capability and version tracking.

**Acceptance Criteria:**
- Given: Database schema needs to be updated
- When: Migration is executed
- Then: Migration is idempotent and safe to re-run
- And: Rollback path is available
- And: Current version is tracked and exposed

### EARS-STO-1: Raw Email Storage
**WHEN** an email is ingested, **THE SYSTEM SHALL** store the raw MIME message in object storage **WITH** deterministic pathing and SHA-256 content hash metadata.

**Acceptance Criteria:**
- Given: Email is ingested through the pipeline
- When: Raw MIME message is processed
- Then: Raw content is stored with deterministic path structure
- And: SHA-256 content hash is calculated and stored as metadata
- And: Object key is stored in database for future retrieval

### EARS-STO-2: Normalized Text Storage
**WHEN** normalization completes, **THE SYSTEM SHALL** store normalized text separately **WITH** linkage to the email row (object_key fields).

**Acceptance Criteria:**
- Given: Email content normalization is completed
- When: Normalized text is generated
- Then: Normalized text is stored separately from raw content
- And: Object key is linked to email database record
- And: Content hash is updated for normalized version

### EARS-STO-3: Attachment Storage
**WHEN** attachments exist, **THE SYSTEM SHALL** store each attachment (and OCR text if present) **WITH** signed-URL retrieval capability and allowed mimetypes only.

**Acceptance Criteria:**
- Given: Email contains attachments
- When: Attachments are processed
- Then: Each attachment is stored with proper mimetype validation
- And: OCR text is stored separately if applicable
- And: Signed URLs are generated for secure retrieval
- And: Only allowed mimetypes are accepted

### EARS-STO-4: Tenant Isolation
**WHEN** storing objects, **THE SYSTEM SHALL** enforce tenant isolation via path prefixes **WITH** deny-by-default bucket policy and server-side encryption configured.

**Acceptance Criteria:**
- Given: Object storage operation is performed
- When: Object is stored
- Then: Tenant isolation is enforced via path prefixes
- And: Bucket policy denies access by default
- And: Server-side encryption is enabled
- And: No cross-tenant data access is possible

### EARS-STO-5: Deduplication
**WHEN** duplicate content is detected (same content hash), **THE SYSTEM SHALL** avoid redundant uploads **WITH** reference counting or pointer reuse.

**Acceptance Criteria:**
- Given: Content with identical hash is detected
- When: Storage operation is attempted
- Then: Redundant upload is avoided
- And: Reference counting is maintained
- And: Storage efficiency is optimized

### EARS-STO-6: Retry Logic
**WHEN** transient failures occur, **THE SYSTEM SHALL** retry with exponential backoff **WITH** bounded attempts and idempotent keys.

**Acceptance Criteria:**
- Given: Transient storage failure occurs
- When: Retry is attempted
- Then: Exponential backoff is applied
- And: Attempts are bounded to prevent infinite loops
- And: Idempotent keys ensure no duplicate operations

### EARS-STO-7: Multipart Upload
**WHEN** large objects (>5MB) are uploaded, **THE SYSTEM SHALL** use multipart upload **WITH** integrity verification.

**Acceptance Criteria:**
- Given: Large object (>5MB) needs to be uploaded
- When: Upload is initiated
- Then: Multipart upload is used
- And: Integrity is verified through checksums
- And: Upload can be resumed if interrupted

### EARS-STO-8: Presigned URLs
**WHEN** queried by API, **THE SYSTEM SHALL** generate presigned URLs **WITH** configurable TTL and audit logging.

**Acceptance Criteria:**
- Given: API request for object access
- When: Presigned URL is generated
- Then: URL has configurable TTL
- And: Access is logged for audit purposes
- And: URL provides secure, time-limited access

### EARS-ING-1: Batch Email Ingestion
**WHEN** a batch of emails is ingested, **THE SYSTEM SHALL** parse MIME and extract headers/parts **WITH** idempotent checkpoints and resumable progress.

**Acceptance Criteria:**
- Given: A batch of emails is submitted for ingestion
- When: The ingestion pipeline processes the batch
- Then: MIME parsing extracts headers and message parts
- And: Idempotent checkpoints are created for resumable progress
- And: Progress can be resumed from any checkpoint

### EARS-ING-2: Content Normalization
**WHEN** normalization runs, **THE SYSTEM SHALL** convert HTML to clean text, strip signatures/quoted text, and detect language **WITH** a normalization manifest per email.

**Acceptance Criteria:**
- Given: Raw email content is processed
- When: Normalization is applied
- Then: HTML is converted to clean, readable text
- And: Signatures and quoted text are stripped
- And: Language is detected and recorded
- And: Normalization manifest is created per email

### EARS-ING-3: Attachment Processing
**WHEN** attachments are present, **THE SYSTEM SHALL** extract text (DOCX/PDF first) and create OCR tasks for images/PDFs **WITH** allowed mimetypes only.

**Acceptance Criteria:**
- Given: Email contains attachments
- When: Attachments are processed
- Then: Text is extracted from DOCX/PDF documents
- And: OCR tasks are created for images and PDFs without text layers
- And: Only allowed mimetypes are processed
- And: Attachment metadata is preserved

### EARS-ING-4: Storage Integration
**WHEN** storing content, **THE SYSTEM SHALL** write raw/normalized/attachment objects to storage **WITH** deterministic paths and content hashes; DB rows MUST reference object keys.

**Acceptance Criteria:**
- Given: Processed email content is ready for storage
- When: Content is stored
- Then: Raw, normalized, and attachment objects are written to storage
- And: Deterministic paths and content hashes are used
- And: Database rows reference storage object keys
- And: All storage operations are tenant-isolated
- And: Object keys are persisted to Email.raw_object_key, Email.norm_object_key, Attachment.object_key, and Attachment.ocr_text_object_key fields
- And: Storage writes occur before database persistence to ensure consistency

### EARS-ING-5: Deduplication
**WHEN** duplicates are detected (exact hash or near-dup simhash/minhash), **THE SYSTEM SHALL** skip redundant uploads and mark dedup lineage **WITH** a per-batch dedup ratio metric.

**Acceptance Criteria:**
- Given: Email content is processed for storage
- When: Duplicates are detected
- Then: Redundant uploads are skipped
- And: Deduplication lineage is recorded
- And: Per-batch dedup ratio is computed and exposed
- And: Storage efficiency is optimized

### EARS-ING-6: Semantic Chunking
**WHEN** chunking text, **THE SYSTEM SHALL** create semantic chunks **WITH** stable chunk_uids, token_count, and email/attachment mapping; vectors are created later by a separate job.

**Acceptance Criteria:**
- Given: Normalized email text is ready for chunking
- When: Text is chunked
- Then: Semantic chunks are created with stable chunk_uids
- And: Token counts are accurately calculated
- And: Chunks maintain email and attachment relationships
- And: Embedding jobs are enqueued for later processing

### EARS-ING-7: Thread Linkage
**WHEN** threading is derivable, **THE SYSTEM SHALL** link emails to threads via Message-Id/In-Reply-To/References **WITH** tenant-safe lookups.

**Acceptance Criteria:**
- Given: Email headers contain threading information
- When: Threading analysis is performed
- Then: Emails are linked to appropriate threads
- And: Thread relationships are preserved in the database
- And: All lookups are tenant-safe
- And: Fallback mechanisms handle missing headers

### EARS-ING-8: Error Handling
**WHEN** errors occur, **THE SYSTEM SHALL** retry transient failures with exponential backoff; permanent failures SHALL be quarantined **WITH** audit records.

**Acceptance Criteria:**
- Given: An error occurs during processing
- When: Error handling is triggered
- Then: Transient failures are retried with exponential backoff
- And: Permanent failures are quarantined
- And: Audit records are created for all failures
- And: Processing can continue for other emails

### EARS-ING-9: Metrics and Observability
**WHEN** ingestion completes, **THE SYSTEM SHALL** emit metrics (docs/sec, dedup_ratio, ocr_rate, failure_buckets, lag) **WITH** trace_id propagation.

**Acceptance Criteria:**
- Given: Email ingestion batch completes
- When: Metrics are collected
- Then: Performance metrics are emitted (docs/sec, dedup_ratio, ocr_rate)
- And: Failure metrics are categorized into buckets
- And: Processing lag is measured and reported
- And: Trace IDs propagate through the entire pipeline

### EARS-ING-10: Multi-Tenant Isolation
**WHEN** multi-tenant ingestion runs, **THE SYSTEM SHALL** enforce tenant_id at all stages **WITH** no cross-tenant keys/queries.

**Acceptance Criteria:**
- Given: Multiple tenants are ingesting emails
- When: Ingestion processing occurs
- Then: Tenant isolation is enforced at all stages
- And: No cross-tenant data access occurs
- And: All storage paths and database queries are tenant-scoped
- And: Security boundaries are maintained
