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
