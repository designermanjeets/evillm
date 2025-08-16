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
**WHEN** email attachments are processed, **THE SYSTEM SHALL** extract text and metadata **WITH** OCR support for images and compliance validation.

**Acceptance Criteria:**
- Given: An email with attachments is processed
- When: The attachment is analyzed
- Then: Text is extracted using native methods or OCR
- And: Metadata is captured (size, type, hash)
- And: Compliance checks are performed
- And: Results are stored with proper indexing

### EARS-RET-1: Embedding Enqueue
**WHEN** content is ingested, **THE SYSTEM SHALL** enqueue chunks for embedding **WITH** batch size, retry, and backpressure controls.

**Acceptance Criteria:**
- Given: New email chunks are created during ingestion
- When: Chunks are ready for embedding
- Then: Chunks are enqueued with tenant-scoped batching
- And: Batch size is configurable per tenant
- And: Retry logic handles provider failures gracefully
- And: Backpressure prevents queue overflow

### EARS-RET-2: Vector Database Storage
**WHEN** embeddings are produced, **THE SYSTEM SHALL** write to a vector DB **WITH** tenant-scoped namespaces and upsert semantics.

**Acceptance Criteria:**
- Given: Embeddings are generated for chunks
- When: Embeddings are ready for storage
- Then: Vectors are written to tenant-scoped namespaces
- And: Upsert semantics prevent duplicates
- And: Metadata includes chunk_id, email_id, and tenant_id
- And: Storage is idempotent and fault-tolerant

### EARS-RET-3: Hybrid Search Execution
**WHEN** a search query arrives, **THE SYSTEM SHALL** run BM25 and vector search **WITH** filters and return citations (chunk→email/attachment).

**Acceptance Criteria:**
- Given: A search query is submitted
- When: The system executes search
- Then: Both BM25 and vector search are performed
- Then: Results include proper citations mapping chunks to emails/attachments
- And: Tenant isolation is enforced on all queries
- And: Filters support date ranges, thread IDs, and content types

### EARS-RET-4: Reciprocal Rank Fusion
**THE SYSTEM SHALL** fuse BM25 and vector results via RRF **WITH** configurable weights and k.

**Acceptance Criteria:**
- Given: BM25 and vector search results
- When: Results need to be combined
- Then: RRF algorithm is applied with configurable parameters
- And: Weights can be tuned per search type
- And: K parameter controls result set size
- And: Fusion improves overall relevance (hit@k)

### EARS-RET-5: Optional Reranker
**Optional reranker SHALL** be supported behind an interface **WITH** a feature flag.

**Acceptance Criteria:**
- Given: Search results are available
- When: Reranking is enabled via feature flag
- Then: Cross-encoder reranker processes top-k results
- And: Reranker interface supports multiple providers
- And: Reranking can be disabled for performance
- And: Reranker results maintain citation integrity

### EARS-AGT-1: Draft Flow Orchestration
**THE SYSTEM SHALL** orchestrate a draft flow via LangGraph sub-agents **WITH** patch-only state, budgets, and streaming output.

**Acceptance Criteria:**
- Given: A draft request is submitted
- When: The system orchestrates the draft flow
- Then: LangGraph coordinates sub-agents in sequence
- And: State updates use patch-only contracts
- And: Token and time budgets are enforced
- And: Output is streamed in real-time
- And: Sub-agent trace is maintained

### EARS-AGT-2: Numeric Verification
**THE SYSTEM SHALL** perform deterministic numeric verification (distance/price/date) and reject ungrounded numerics.

**Acceptance Criteria:**
- Given: A draft contains numeric claims
- When: Numeric verification is performed
- Then: All numerics are verified against retrieved chunks or tools
- And: Ungrounded numerics are rejected
- And: Verification uses deterministic calculations
- And: Verification results are logged and metered

### EARS-AGT-3: Evaluation Gate
**THE SYSTEM SHALL** enforce a fail-closed eval gate (Grounding/Completeness/Tone/Policy) in dev-mode.

**Acceptance Criteria:**
- Given: A draft is completed
- When: Evaluation is performed in dev-mode
- Then: Grounding, completeness, tone, and policy are scored
- And: Drafts below threshold are rejected
- And: Evaluation uses promptfoo/LangSmith integration
- And: Evaluation results are logged and tracked

### EARS-UI-1: Demo UI Core
**THE SYSTEM SHALL** provide a demo UI to upload docs, run search with citations, and stream agent drafts **WITH** tenant selection.

**Acceptance Criteria:**
- Given: A user accesses the demo UI
- When: The UI is loaded
- Then: Upload functionality supports EML/PDF/JPG files
- Then: Search interface shows results with citations
- Then: Draft interface streams agent output
- And: Tenant selection is available and enforced
- And: UI is responsive and accessible

### EARS-UI-2: Auditability & Trace
**THE SYSTEM SHALL** display sub-agent steps/trace and citations for auditability.

**Acceptance Criteria:**
- Given: A draft is generated
- When: The draft is displayed
- Then: Sub-agent execution trace is visible
- And: Citations are clearly linked to source chunks
- And: Processing metrics are displayed
- And: Trace includes timing and decision points
- And: Audit trail is exportable

### EARS-GRAPH-1: Patch-Only State Updates
**WHEN** sub-agents update workflow state, **THE SYSTEM SHALL** apply **patch-only** updates (immutable), **WITH** validation that no duplicate/conflicting keys are produced in a single tick.

**Acceptance Criteria:**
- Given: A sub-agent processes workflow state
- When: The agent returns state updates
- Then: Only changed fields are returned as patches
- And: No duplicate keys are allowed in a single update
- And: Attempts to return whole state objects are rejected
- And: Patch validation prevents conflicting updates

### EARS-GRAPH-2: Deterministic State Reconstruction
**WHEN** the workflow completes, **THE SYSTEM SHALL** reconstruct a typed final state from accumulated patches **WITH** deterministic field precedence (latest-wins) and tenant_id immutability.

**Acceptance Criteria:**
- Given: A workflow completes with multiple patches
- When: Final state is reconstructed
- Then: All patches are merged with latest-wins precedence
- And: Final state is a typed OCRWorkflowState object
- And: tenant_id remains unchanged from initial value
- And: Reconstruction is deterministic and repeatable

### EARS-GRAPH-3: Config-Driven Routing
**THE SYSTEM SHALL** expose a config flag `graph.linear_mode` default true; conditional edges enabled only when `graph.linear_mode=false`.

**Acceptance Criteria:**
- Given: Workflow routing configuration
- When: linear_mode is true (default)
- Then: All nodes execute in linear sequence
- And: No conditional branching occurs
- When: linear_mode is false
- Then: Conditional edges are enabled
- And: Mutually exclusive routing prevents simultaneous execution

### EARS-GRAPH-4: Per-Node Observability
**THE SYSTEM SHALL** emit per-node metrics: patch_size, keys_changed, conflicts_detected, latency; logs **SHALL** include `trace_id` and sub-agent name.

**Acceptance Criteria:**
- Given: A sub-agent executes
- When: The agent completes processing
- Then: Metrics are recorded (patch_size, keys_changed, conflicts, latency)
- And: Logs include trace_id and agent name
- And: PII is redacted from log values
- And: Metrics are available for monitoring dashboards

### EARS-GRAPH-5: Tenant ID Immutability
**THE SYSTEM SHALL** ensure `tenant_id` is constant; attempts to modify **SHALL** fail with a policy error.

**Acceptance Criteria:**
- Given: A sub-agent attempts to modify tenant_id
- When: The patch is validated
- Then: The modification is rejected
- And: A policy violation error is raised
- And: The violation is logged and metered
- And: The workflow continues with original tenant_id
