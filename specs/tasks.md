# Logistics Email AI - Tasks Specification

## Task Overview
This document outlines the ordered, atomic tasks required to implement the Logistics Email AI system. Tasks are mapped to EARS requirements and include acceptance criteria, risks, and dependencies.

## Task Dependencies
- **Phase 1**: Core infrastructure and basic functionality
- **Phase 2**: Advanced features and optimization
- **Phase 3**: Production hardening and scaling

## Phase 1 Tasks

### TASK-1: Project Structure & Dependencies
**Goal**: Establish project foundation with proper dependency management and tooling
**EARS Mapping**: All (infrastructure)
**Files Touched**: 
- `pyproject.toml`
- `.tool-versions`
- `requirements.txt`
- `.gitignore`
- `ruff.toml`
- `mypy.ini`

**Acceptance Criteria**:
- Python 3.11+ environment configured
- FastAPI, LangGraph, Pydantic v2 dependencies specified
- Ruff, mypy, pytest configurations present
- Virtual environment setup documented

**Risks**: None
**Dependencies**: None
**Status**: OPEN

### TASK-2: Database Schema & Migrations
**Goal**: Create PostgreSQL schema for emails, chunks, and attachments
**EARS Mapping**: EARS-1, EARS-2, EARS-10, EARS-DB-1, EARS-DB-2, EARS-DB-3, EARS-DB-4, EARS-DB-5, EARS-DB-6
**Files Touched**:
- `app/models/database.py`
- `app/database/migrations/`
- `app/database/schema.sql`
- `app/database/engine.py`
- `app/database/session.py`

**Acceptance Criteria**:
- All tables created with proper constraints and indexes
- Multi-tenant isolation enforced at database level
- Performance indexes for p95 ≤4s under ≥50 concurrent requests
- Alembic migrations with rollback capability
- Database health checks integrated with health endpoint

**Risks**: Schema changes may require data migration, tenant filtering complexity
**Dependencies**: TASK-1
**Status**: COMPLETED

**Subtasks**:
- TASK-2.1: Initialize Alembic and create base migration ✅
- TASK-2.2: Implement SQLAlchemy 2.0 models ✅
- TASK-2.3: Create database engine and session factory ✅
- TASK-2.4: Add database health checks to health endpoint ✅
- TASK-2.5: Create database tests ✅
- TASK-2.6: Update documentation ✅

### TASK-3: Storage Service Implementation
**Goal**: Implement robust, tenant-aware object storage layer for email content and attachments
**EARS Mapping**: EARS-1, EARS-2, EARS-10, EARS-STO-1, EARS-STO-2, EARS-STO-3, EARS-STO-4, EARS-STO-5, EARS-STO-6, EARS-STO-7, EARS-STO-8
**Files Touched**:
- `app/storage/__init__.py`
- `app/storage/client.py`
- `app/storage/paths.py`
- `app/storage/metadata.py`
- `app/storage/health.py`
- `app/routers/health.py`
- `tests/test_storage.py`
- `infra/docker-compose.yml`
- `infra/.env.example`

**Acceptance Criteria**:
- Storage client supports put/get/head/presign with deterministic, tenant-aware paths
- Multi-tenant isolation enforced via path prefixes and bucket policies
- Health endpoint reports storage status with configurable canary checks
- Tests pass for pathing, presign, multipart, and isolation
- All EARS-STO requirements are implemented and tested

**Risks**: Path drift, bucket policy misconfig, TTL misuse, multipart edge cases
**Dependencies**: TASK-2
**Status**: COMPLETED

**Subtasks**:
- TASK-3.1: Storage config & client factory (S3/MinIO) ✅
- TASK-3.2: Pathing + metadata helpers (tenant-aware) ✅
- TASK-3.3: put_raw_email / put_normalized_text / put_attachment / put_ocr_text ✅
- TASK-3.4: get_text_object / head_object / delete_object (safe) ✅
- TASK-3.5: presign_url (GET) with TTL + audit ✅
- TASK-3.6: health checks (/health/storage) ✅
- TASK-3.7: tests (unit+integration: MinIO or moto/minio-py) ✅
- TASK-3.8: docs (README + env + docker-compose MinIO) ✅

### TASK-4: Email Ingestion Pipeline
**Goal**: Build resilient, idempotent email ingestion pipeline for MIME parsing, normalization, and content preparation
**EARS Mapping**: EARS-1, EARS-10, EARS-ING-1, EARS-ING-2, EARS-ING-3, EARS-ING-4, EARS-ING-5, EARS-ING-6, EARS-ING-7, EARS-ING-8, EARS-ING-9, EARS-ING-10
**Files Touched**:
- `app/ingestion/` (all modules)
- `app/routers/ingestion.py`
- `app/main.py`
- `tests/test_ingestion.py`
- `specs/` (requirements, design, tasks)

**Acceptance Criteria**:
- Pipeline processes MIME emails through all stages: parse → normalize → attachments → dedup → storage → DB → chunks → embed queue
- Idempotent processing with checkpoints and resumable batches
- Multi-tenant isolation enforced at all stages
- Metrics collection: docs/sec, dedup_ratio, ocr_rate, failure_buckets, lag
- All EARS-ING requirements implemented and tested
- Golden EML fixtures for testing edge cases

**Risks**: Malformed MIME parsing, large attachment handling, OCR latency, false near-dups
**Dependencies**: TASK-3 (Storage)
**Status**: IN_PROGRESS

**Subtasks**:
- TASK-4.1: Source "dropbox" + manifest reader (Phase-1) ✅
- TASK-4.2: MIME parser + header extractor ✅
- TASK-4.3: HTML→text normalizer + signature/quote stripper ✅
- TASK-4.4: Attachment extractor (DOCX/PDF) + OCR task stub + mimetype allowlist ✅
- TASK-4.5: Dedup (sha256 + simhash/minhash) + lineage + metrics ✅
- TASK-4.6: Storage writes (raw/norm/attachments) using TASK-3 client ✅
- TASK-4.7: DB persistence: emails, attachments; threading linkage ✅
- TASK-4.8: Chunking: splitter + chunk_uid rules + token_count ✅
- TASK-4.9: Embed job enqueue (provider-agnostic) ✅
- TASK-4.10: Idempotency/checkpoints/quarantine subsystem ✅
- TASK-4.11: Metrics/logging/trace IDs ✅
- TASK-4.12: Tests & docs ✅

**Risks & Mitigation**:
- **Encoding Pitfalls**: Normalizer safely handles bytes/str with UTF-8 fallback
- **Partial Writes**: Storage writes occur before DB persistence for consistency
- **Near-Dup False Positives**: Configurable thresholds and manual review options
- **Storage Failures**: Retry logic and fallback storage options

**Risks & Mitigation**:
- **Malformed MIME**: Robust parsing with fallbacks and quarantine
- **Large Attachments**: Size limits and streaming for oversized files
- **OCR Latency**: Async processing and task queuing
- **False Near-Dups**: Configurable thresholds and manual review options
- **Storage Failures**: Retry logic and fallback storage options
**Files Touched**:
- `app/ingestion/__init__.py`
- `app/ingestion/pipeline.py`
- `app/ingestion/parser.py`
- `app/ingestion/normalizer.py`
- `app/ingestion/attachments.py`
- `app/ingestion/deduplication.py`
- `app/ingestion/chunking.py`
- `app/ingestion/threading.py`
- `app/ingestion/checkpoints.py`
- `app/ingestion/metrics.py`
- `tests/test_ingestion.py`
- `data/ingestion/dropbox/`
- `data/ingestion/manifests/`

**Acceptance Criteria**:
- Pipeline processes email batches with MIME parsing and normalization
- Attachments are extracted and OCR tasks are created
- Deduplication prevents redundant storage and processing
- Content is chunked with stable chunk_uids and token counts
- Threading relationships are established and preserved
- Idempotent operations with checkpointing and resume capability
- Comprehensive metrics and observability throughout the pipeline

**Risks**: Malformed MIME, large attachments, OCR latency, false near-dups
**Dependencies**: TASK-2, TASK-3
**Status**: IN PROGRESS

**Subtasks**:
- TASK-4.1: Source "dropbox" + manifest reader (Phase-1)
- TASK-4.2: MIME parser + header extractor
- TASK-4.3: HTML→text normalizer + signature/quote stripper
- TASK-4.4: Attachment extractor (DOCX/PDF) + OCR task stub + mimetype allowlist
- TASK-4.5: Dedup (sha256 + simhash/minhash) + lineage + metrics
- TASK-4.6: Storage writes (raw/norm/attachments) using TASK-3 client
- TASK-4.7: DB persistence: emails, attachments; threading linkage
- TASK-4.8: Chunking: splitter + chunk_uid rules + token_count
- TASK-4.9: Embed job enqueue (provider-agnostic)
- TASK-4.10: Idempotency/checkpoints/quarantine subsystem
- TASK-4.11: Metrics/logging/trace IDs
- TASK-4.12: Tests & docs



### TASK-4: Email Ingestion Pipeline
**Goal**: Build email processing pipeline with MIME parsing and normalization
**EARS Mapping**: EARS-1, EARS-10
**Files Touched**:
- `app/services/ingestion.py`
- `app/services/email_parser.py`
- `app/services/normalizer.py`
- `app/services/deduplication.py`

**Acceptance Criteria**:
- MIME parsing for various email formats
- HTML to text conversion
- Signature and quote stripping
- Deduplication with simhash/minhash
- Idempotent processing with checkpoints

**Risks**: Complex email formats may cause parsing failures
**Dependencies**: TASK-2, TASK-3
**Status**: OPEN

### TASK-5: OCR Service Implementation
**Goal**: Implement OCR service for image and PDF attachments
**EARS Mapping**: EARS-10
**Files Touched**:
- `app/services/ocr.py`
- `app/services/document_processor.py`
- `app/config/ocr.py`

**Acceptance Criteria**:
- PDF text extraction
- Image OCR with multiple engines
- Fallback mechanisms for OCR failures
- Text normalization and cleaning

**Risks**: OCR accuracy varies by document quality
**Dependencies**: TASK-3
**Status**: OPEN

### TASK-6: Embedding Service
**Goal**: Implement vector embeddings for semantic search
**EARS Mapping**: EARS-3
**Files Touched**:
- `app/services/embeddings.py`
- `app/services/chunking.py`
- `app/config/embeddings.py`

**Acceptance Criteria**:
- Semantic chunking with stable chunk_ids
- OpenAI embedding integration
- Batch processing with concurrency
- Metadata enrichment (entities, language)

**Risks**: API rate limits and costs
**Dependencies**: TASK-4
**Status**: OPEN

### TASK-7: Hybrid Search Engine
**Goal**: Implement BM25 + vector search with RRF fusion
**EARS Mapping**: EARS-3
**Files Touched**:
- `app/services/search/`
- `app/services/search/bm25_search.py`
- `app/services/search/vector_search.py`
- `app/services/search/fusion.py`
- `app/services/search/reranker.py`

**Acceptance Criteria**:
- OpenSearch/Elasticsearch integration
- Vector store integration (Qdrant/Pinecone/Weaviate)
- Reciprocal Rank Fusion algorithm
- Optional cross-encoder reranking
- Multi-tenant isolation

**Risks**: Search performance may degrade with large datasets
**Dependencies**: TASK-6
**Status**: OPEN

### TASK-8: LangGraph Agent Framework
**Goal**: Implement core agent nodes and coordination logic
**EARS Mapping**: EARS-11
**Files Touched**:
- `app/agents/`
- `app/agents/coordinator.py`
- `app/agents/query_analyzer.py`
- `app/agents/retriever.py`
- `app/agents/numeric_verifier.py`
- `app/agents/compliance_guard.py`
- `app/agents/drafter.py`
- `app/agents/human_gate.py`

**Acceptance Criteria**:
- All agent nodes implemented
- Deterministic state transitions
- Token and time budget enforcement
- Human approval gate integration

**Risks**: Complex agent interactions may cause deadlocks
**Dependencies**: TASK-7
**Status**: OPEN

### TASK-9: Numeric Computation Tools
**Goal**: Implement deterministic tools for distance, price, and date calculations
**EARS Mapping**: EARS-4, EARS-5
**Files Touched**:
- `app/tools/`
- `app/tools/distance_calculator.py`
- `app/tools/price_calculator.py`
- `app/tools/date_calculator.py`
- `app/config/rate_cards.py`

**Acceptance Criteria**:
- Geographic distance calculations
- Rate card lookups and calculations
- Date arithmetic and business logic
- All calculations deterministic and verifiable

**Risks**: Rate card accuracy depends on data quality
**Dependencies**: TASK-8
**Status**: OPEN

### TASK-10: FastAPI Application
**Goal**: Create FastAPI application with core endpoints
**EARS Mapping**: EARS-7, EARS-8
**Files Touched**:
- `app/main.py`
- `app/routers/`
- `app/middleware/`
- `app/dependencies.py`

**Acceptance Criteria**:
- POST /draft endpoint implemented
- POST /eval/run endpoint implemented
- GET /health endpoint implemented
- Authentication and rate limiting
- Structured logging with trace IDs

**Risks**: API design may need iteration based on usage
**Dependencies**: TASK-8, TASK-9
**Status**: OPEN

### TASK-11: Evaluation Framework
**Goal**: Implement automated evaluation using promptfoo and LangSmith
**EARS Mapping**: EARS-12
**Files Touched**:
- `eval/promptfoo.yaml`
- `eval/golden_examples/`
- `app/services/evaluation.py`
- `app/config/evaluation.py`

**Acceptance Criteria**:
- 20-50 golden email examples
- Grounding, completeness, tone, policy scoring
- LangSmith integration for LLM-as-judge
- Threshold-based quality gates

**Risks**: Evaluation metrics may not correlate with user satisfaction
**Dependencies**: TASK-10
**Status**: OPEN

### TASK-12: Testing Suite
**Goal**: Create comprehensive test coverage for all components
**EARS Mapping**: All
**Files Touched**:
- `tests/`
- `tests/test_ingestion.py`
- `tests/test_search.py`
- `tests/test_agents.py`
- `tests/test_api.py`
- `tests/conftest.py`

**Acceptance Criteria**:
- Unit tests for all services
- Integration tests for workflows
- Test coverage >80%
- Golden example tests
- Performance benchmarks

**Risks**: Test maintenance overhead
**Dependencies**: TASK-11
**Status**: OPEN

### TASK-13: Infrastructure & Deployment
**Goal**: Create deployment configuration and infrastructure
**EARS Mapping**: EARS-7, EARS-8, EARS-9
**Files Touched**:
- `infra/`
- `infra/Dockerfile`
- `infra/docker-compose.yml`
- `infra/.env.example`
- `infra/nginx.conf`

**Acceptance Criteria**:
- Docker containerization
- Environment configuration
- Health check endpoints
- Monitoring and logging setup

**Risks**: Production deployment complexity
**Dependencies**: TASK-12
**Status**: OPEN

### TASK-14: Security & PII Protection
**Goal**: Implement security measures and PII redaction
**EARS Mapping**: EARS-9
**Files Touched**:
- `app/services/security.py`
- `app/services/pii_redaction.py`
- `app/middleware/security.py`

**Acceptance Criteria**:
- PII detection and redaction
- Prompt injection prevention
- Input validation and sanitization
- Audit logging

**Risks**: PII detection may have false positives/negatives
**Dependencies**: TASK-10
**Status**: OPEN

### TASK-15: Performance Optimization
**Goal**: Optimize system performance to meet SLOs
**EARS Mapping**: EARS-7
**Files Touched**:
- `app/services/cache.py`
- `app/services/queue.py`
- `app/config/performance.py`

**Acceptance Criteria**:
- Ingest throughput ≥100k emails/day
- Query p95 ≤4s to first token
- Full draft generation ≤8s
- Support ≥50 concurrent requests

**Risks**: Performance optimization may increase complexity
**Dependencies**: TASK-13
**Status**: OPEN

### TASK-16: Multi-Tenant Isolation
**Goal**: Ensure complete tenant isolation across all components
**EARS Mapping**: EARS-2
**Files Touched**:
- `app/middleware/tenant_isolation.py`
- `app/services/tenant_service.py`
- All service files (add tenant_id filtering)

**Acceptance Criteria**:
- No cross-tenant data leakage
- Tenant isolation at database level
- Search isolation
- Logging isolation

**Risks**: Complex queries may accidentally leak data
**Dependencies**: TASK-14
**Status**: OPEN

### TASK-17: Monitoring & Observability
**Goal**: Implement comprehensive monitoring and alerting
**EARS Mapping**: EARS-8
**Files Touched**:
- `app/services/monitoring.py`
- `app/services/metrics.py`
- `app/config/monitoring.py`

**Acceptance Criteria**:
- Structured logging with trace IDs
- Performance metrics collection
- Error tracking and alerting
- Business metrics dashboard

**Risks**: Monitoring overhead may impact performance
**Dependencies**: TASK-16
**Status**: OPEN

### TASK-18: Documentation & CI/CD
**Goal**: Create documentation and CI/CD pipeline
**EARS Mapping**: All
**Files Touched**:
- `README.md`
- `docs/`
- `.github/workflows/`
- `Makefile`

**Acceptance Criteria**:
- API documentation
- Deployment guide
- CI/CD pipeline
- Development setup guide

**Risks**: Documentation may become outdated
**Dependencies**: TASK-17
**Status**: OPEN

## Risk Mitigation Strategies

### Technical Risks
- **Complexity Management**: Feature flags and staged rollouts
- **Performance Degradation**: Circuit breakers and graceful degradation
- **Data Loss**: Backup strategies and point-in-time recovery

### Operational Risks
- **Scaling Issues**: Horizontal scaling and load balancing
- **Cost Management**: Usage monitoring and budget alerts
- **Security Breaches**: Regular security audits and penetration testing

### Business Risks
- **User Adoption**: User feedback loops and iterative improvement
- **Compliance Changes**: Flexible policy engine and regular audits
- **Competitive Pressure**: Continuous improvement and feature parity

## Success Metrics

### Phase 1 Success Criteria
- [ ] All 18 tasks completed
- [ ] System passes evaluation gates
- [ ] Performance SLOs met
- [ ] Security requirements satisfied
- [ ] Multi-tenant isolation verified
- [ ] Comprehensive test coverage

### Quality Gates
- [ ] Eval scores > threshold for all metrics
- [ ] No critical security vulnerabilities
- [ ] Performance benchmarks passed
- [ ] Code quality tools (ruff, mypy) pass
- [ ] Test coverage >80%

## Next Steps

1. **Immediate**: Execute TASK-1 (Project Structure & Dependencies)
2. **Week 1**: Complete TASK-2 through TASK-4 (Database, Storage, Ingestion)
3. **Week 2**: Complete TASK-5 through TASK-7 (OCR, Embeddings, Search)
4. **Week 3**: Complete TASK-8 through TASK-10 (Agents, Tools, API)
5. **Week 4**: Complete TASK-11 through TASK-13 (Evaluation, Testing, Infrastructure)
6. **Week 5**: Complete TASK-14 through TASK-16 (Security, Performance, Isolation)
7. **Week 6**: Complete TASK-17 through TASK-18 (Monitoring, Documentation)

## Dependencies Graph

```
TASK-1 → TASK-2 → TASK-3 → TASK-4 → TASK-6 → TASK-7 → TASK-8 → TASK-10
  ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓
TASK-5   TASK-9   TASK-11   TASK-12   TASK-13   TASK-14   TASK-15   TASK-16
  ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓
TASK-17 → TASK-18
```
