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
- `app/ingestion/__init__.py`
- `app/ingestion/pipeline.py`
- `app/ingestion/parser.py`
- `app/ingestion/normalizer.py`
- `app/ingestion/chunking.py`
- `app/ingestion/deduplication.py`
- `app/ingestion/checkpoints.py`
- `app/ingestion/threading.py`
- `app/ingestion/attachments.py`
- `app/ingestion/metrics.py`
- `app/ingestion/models.py`
- `app/routers/ingestion.py`
- `tests/test_ingestion.py`
- `config/app.yaml`

**Acceptance Criteria**:
- MIME parsing extracts headers, parts, and attachments
- HTML normalization produces clean, stable text
- Semantic chunking creates stable chunk_uids
- Deduplication prevents redundant processing
- Checkpoints enable resumable processing
- Multi-tenant isolation enforced throughout
- All EARS-ING requirements implemented and tested

**Risks**: MIME edge cases, normalization stability, chunking determinism
**Dependencies**: TASK-2, TASK-3
**Status**: COMPLETED

**Subtasks**:
- TASK-4.1: MIME parser with header extraction ✅
- TASK-4.2: HTML normalizer (clean text, strip signatures) ✅
- TASK-4.3: Semantic chunking with stable UIDs ✅
- TASK-4.4: Deduplication (exact + near-dup) ✅
- TASK-4.5: Checkpoint system for resumability ✅
- TASK-4.6: Multi-tenant isolation ✅
- TASK-4.7: Metrics and observability ✅
- TASK-4.8: Tests and error handling ✅

### TASK-5: OCR Service & LangGraph Workflow
**Goal**: Implement OCR service with LangGraph-based workflow for attachment processing
**EARS Mapping**: EARS-10, EARS-OCR-1, EARS-OCR-2, EARS-OCR-3, EARS-OCR-4, EARS-OCR-5, EARS-OCR-6, EARS-OCR-7, EARS-OCR-8, EARS-OCR-9, EARS-OCR-10
**Files Touched**:
- `app/services/ocr.py`
- `app/agents/ocr_workflow.py`
- `app/config/ocr.py`
- `app/config/manager.py`
- `tests/test_ocr.py`
- `tests/test_ocr_workflow.py`
- `config/app.yaml`

**Acceptance Criteria**:
- OCR service supports multiple backends (stub, Tesseract, cloud)
- LangGraph workflow coordinates sub-agents for attachment processing
- Text extraction from DOCX/PDF with OCR fallback
- Security validation (mimetype allowlist, size caps)
- Multi-tenant isolation and idempotent processing
- All EARS-OCR requirements implemented and tested

**Risks**: OCR accuracy, backend integration, workflow state management
**Dependencies**: TASK-2, TASK-3, TASK-4
**Status**: COMPLETED

**Subtasks**:
- TASK-5.1: OCR service with backend abstraction ✅
- TASK-5.2: LangGraph workflow with sub-agents ✅
- TASK-5.3: Text extraction and OCR fallback ✅
- TASK-5.4: Security validation and compliance ✅
- TASK-5.5: Multi-tenant isolation ✅
- TASK-5.6: Tests and error handling ✅
- TASK-5.7: Configuration and documentation ✅

### TASK-5.H: LangGraph State Hardening & Config-Driven OCR Sub-Agents
**Goal**: Harden LangGraph workflow with explicit state contracts, runtime guards, and config-driven behavior
**EARS Mapping**: EARS-GRAPH-1, EARS-GRAPH-2, EARS-GRAPH-3, EARS-GRAPH-4, EARS-GRAPH-5
**Files Touched**: 
- `app/agents/state_contract.py`
- `app/agents/utils.py`
- `app/agents/ocr_workflow.py`
- `app/config/app.yaml`
- `tests/test_ocr_workflow_state_contract.py`
- `specs/requirements.md`
- `specs/design.md`
- `specs/tasks.md`

**Acceptance Criteria**:
- StatePatch contract enforces patch-only updates
- Runtime guards prevent banned key modifications
- Config-driven routing (linear vs conditional)
- Per-node observability with metrics and logs
- Tenant ID immutability guaranteed
- All EARS-GRAPH requirements implemented and tested

**Risks**: Performance overhead, configuration complexity, backward compatibility
**Dependencies**: TASK-5
**Status**: IN_PROGRESS

**Subtasks**:
- TASK-5.H.1: Spec updates (requirements, design, tasks) 🔄
- TASK-5.H.2: Core patch utilities + guards
- TASK-5.H.3: Refactor all OCR sub-agents to patch-only + wrappers
- TASK-5.H.4: Config flags + routing policy (linear default)
- TASK-5.H.5: Tests: invariants/concurrency/no-dup-key/determinism
- TASK-5.H.6: Observability metrics/logs; conflict dashboards
- TASK-5.H.7: Docs & runbook; commit & sign-off

### TASK-6: Embedding Service & Vector Storage
**Goal**: Implement embedding service with queue management and vector storage
**EARS Mapping**: EARS-RET-1, EARS-RET-2
**Files Touched**:
- `app/services/embeddings.py`
- `app/services/vector_store/`
- `app/database/migrations/`
- `app/config/embeddings.py`
- `tests/test_embeddings.py`

**Acceptance Criteria**:
- Database-backed embedding job queue
- Intelligent batching with configurable batch sizes
- Retry logic with exponential backoff
- Provider-agnostic embedding interface
- Vector storage with tenant-scoped namespaces
- Cost guards and rate limiting
- Comprehensive monitoring and metrics

**Risks**: API rate limits, cost management, vector DB performance
**Dependencies**: TASK-4
**Status**: OPEN

**Subtasks**:
- TASK-6.1: Embedding Worker (queue + batcher + retries; vectors upsert) 🔄
- TASK-6.2: Vector store integration (Qdrant/Pinecone/Weaviate)
- TASK-6.3: Cost management and rate limiting
- TASK-6.4: Health monitoring and metrics
- TASK-6.5: Tests and error handling

### TASK-7: Hybrid Search Engine
**Goal**: Implement BM25 + vector search with RRF fusion and optional reranking
**EARS Mapping**: EARS-RET-3, EARS-RET-4, EARS-RET-5
**Files Touched**:
- `app/services/search/`
- `app/services/bm25/`
- `app/services/fusion.py`
- `app/services/reranker.py`
- `tests/test_search.py`

**Acceptance Criteria**:
- OpenSearch/Elasticsearch integration for BM25
- Vector search with tenant isolation
- Reciprocal Rank Fusion algorithm
- Optional cross-encoder reranking
- Citation mapping (chunk→email/attachment)
- Performance metrics and monitoring

**Risks**: ✅ Low - Implementation complete and tested
**Dependencies**: ✅ TASK-6 (Embeddings) - Completed
**Status**: ✅ **COMPLETED**

**Subtasks**:
- TASK-7.1: BM25/OpenSearch index + filters + health/aliases ✅
- TASK-7.2: Vector search integration ✅
- TASK-7.3: RRF fusion + optional reranker interface ✅
- TASK-7.4: Citation mapping and metadata ✅
- TASK-7.5: Performance optimization and monitoring ✅
- TASK-7.6: Search performance metrics and API ✅

### TASK-8: LangGraph Draft Agent Framework
**Goal**: Implement draft flow orchestration with sub-agents and streaming
**EARS Mapping**: EARS-AGT-1, EARS-AGT-2, EARS-AGT-3
**Files Touched**:
- `app/agents/draft_workflow.py`
- `app/agents/coordinator.py`
- `app/agents/query_analyzer.py`
- `app/agents/retriever.py`
- `app/agents/numeric_verifier.py`
- `app/agents/drafter.py`
- `app/agents/compliance_guard.py`
- `tests/test_draft_agents.py`

**Acceptance Criteria**:
- Coordinator agent with routing logic
- Query analyzer with intent classification
- Retriever agent with hybrid search
- Numeric verifier with deterministic tools
- Compliance guard with policy enforcement
- Streaming output via SSE/WebSocket
- Evaluation gate for dev-mode quality control

**Risks**: Complex agent interactions, streaming complexity, evaluation accuracy
**Dependencies**: ✅ TASK-7 (Hybrid Search) - Completed
**Status**: OPEN

**Subtasks**:
- TASK-8.1: LangGraph draft orchestration + streaming endpoint ✅
- TASK-8.2: Core sub-agents implementation ✅
- TASK-8.3: Numeric verification tools ✅
- TASK-8.4: Eval gate wiring (promptfoo/LangSmith hooks) ⚠️
- TASK-8.5: Streaming and real-time output ✅
- TASK-8.6: Tests and error handling ✅

**Phase A1 Micro-Tasks**:
- A1.1: RetrieverAdapter + Citations ⚠️
- A1.2: LLMClient + Draft node (streaming) ⚠️
- A1.3: EvalGate (dev-mode) + fail-closed policy ⚠️
- A1.4: Audit Trace export + metrics/counters ⚠️

### TASK-9: Demo UI & User Experience
**Goal**: Create demo UI for upload, search, and draft functionality
**EARS Mapping**: EARS-UI-1, EARS-UI-2
**Files Touched**:
- `ui/` (React/Vite or FastAPI Jinja templates)
- `app/routers/ui.py`
- `app/static/`
- `tests/test_ui.py`

**Acceptance Criteria**:
- File upload with drag-and-drop support
- Tenant selection and isolation
- Search interface with citations
- Draft streaming with sub-agent trace
- Responsive and accessible design
- Error handling and user feedback
- Audit trail and export functionality

**Risks**: UI complexity, real-time updates, cross-browser compatibility
**Dependencies**: TASK-8
**Status**: OPEN

**Subtasks**:
- TASK-9.1: Demo UI: Upload
- TASK-9.2: Demo UI: Search + Draft + Trace
- TASK-9.3: Responsive design and accessibility
- TASK-9.4: Error handling and user feedback
- TASK-9.5: Tests and cross-browser validation

### TASK-10: Observability & Monitoring
**Goal**: Implement comprehensive monitoring and alerting for production
**EARS Mapping**: EARS-8, EARS-7
**Files Touched**:
- `app/services/monitoring.py`
- `app/services/metrics.py`
- `app/config/monitoring.py`
- `dashboards/`
- `alerts/`

**Acceptance Criteria**:
- Search performance metrics (hit@k, p95 latency)
- Agent performance tracking
- Cost monitoring and alerts
- Error rate tracking and alerting
- Business metrics and KPIs
- Grafana dashboards
- PagerDuty integration

**Risks**: Monitoring overhead, alert fatigue, metric accuracy
**Dependencies**: TASK-9
**Status**: OPEN

**Subtasks**:
- TASK-10.1: Observability dashboards & alerts (hit@k, p95, grounding pass)
- TASK-10.2: Cost monitoring and budget alerts
- TASK-10.3: Error tracking and incident response
- TASK-10.4: Business metrics and KPIs
- TASK-10.5: Integration with monitoring tools

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
