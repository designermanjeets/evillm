# Logistics Email AI

Intelligent email processing system for logistics with grounded responses, distance/price calculations, and compliance verification.

## Overview

The Logistics Email AI system processes incoming logistics emails to provide intelligent, grounded responses with:
- **Hybrid Search**: BM25 + vector search with Reciprocal Rank Fusion
- **Multi-Agent Coordination**: LangGraph-based workflow orchestration
- **Grounded Responses**: All facts verified against retrieved chunks or deterministic tools
- **Multi-Tenant Isolation**: Complete tenant separation and security
- **Performance SLOs**: 100k emails/day ingest, <4s query response, 50+ concurrent requests

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  LangGraph      │    │   Hybrid        │
│                 │    │  Agents         │    │   Search        │
│  • /draft      │◄──►│  • Coordinator  │◄──►│  • BM25         │
│  • /eval/run   │    │  • Retriever    │    │  • Vector       │
│  • /health     │    │  • Drafter      │    │  • RRF Fusion   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Object        │    │   Vector        │
│   Database      │    │   Storage       │    │   Store         │
│                 │    │                 │    │                 │
│  • Emails      │    │  • Raw emails   │    │  • Embeddings   │
│  • Chunks      │    │  • Attachments  │    │  • Metadata     │
│  • Metadata    │    │  • OCR text     │    │  • Similarity   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Core Capabilities
- **Email Ingestion**: MIME parsing, HTML→text conversion, signature stripping
- **Attachment Processing**: PDF/image OCR, text extraction, secure storage
- **Deduplication**: Exact hash + simhash/minhash for near-duplicate detection
- **Semantic Chunking**: Stable chunk_ids with metadata preservation
- **Hybrid Retrieval**: BM25 + vector search with RRF fusion and optional reranking

## Ingestion Pipeline

The email ingestion pipeline processes emails through multiple stages with resilience, idempotency, and multi-tenant isolation.

### Pipeline Flow
```
Source → MIME Parse → Normalize → Attachments/OCR → Dedup → Storage → DB → Chunks → Embed Queue
```

### Usage

#### CLI Interface
```bash
# Ingest from dropbox folder
python -m app.cli dropbox /path/to/emails --tenant tenant1

# Ingest from manifest file
python -m app.cli manifest /path/to/manifest.json --tenant tenant1

# Ingest with custom batch size
python -m app.cli dropbox /path/to/emails --tenant tenant1 --batch-size 50
```

#### API Endpoints
```bash
# Start ingestion
POST /ingestion/ingest
{
  "source_type": "dropbox",
  "source_path": "/path/to/emails",
  "batch_size": 100,
  "tenant_id": "tenant1"
}

# Check batch status
GET /ingestion/status/{batch_id}

# List recent batches
GET /ingestion/batches?limit=10&offset=0

# Retry failed emails
POST /ingestion/retry/{batch_id}
```

#### Sample Manifest Format
```json
{
  "batch_id": "sample_batch_001",
  "source_type": "manifest",
  "file_manifest": [
    {
      "file_path": "/path/to/email1.eml",
      "file_name": "email1.eml",
      "file_size": 1024
    }
  ],
  "processing_options": {
    "batch_size": 100,
    "enable_ocr": true,
    "dedup_threshold": 0.8,
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

### Features
- **Idempotent Processing**: Safe re-runs with checkpoints and resumable batches
- **Multi-Tenant Isolation**: All operations scoped to tenant_id
- **Error Handling**: Retry logic with exponential backoff and quarantine for permanent failures
- **Metrics Collection**: docs/sec, dedup_ratio, ocr_rate, failure_buckets, lag
- **Checkpointing**: Resume processing from last successful point
- **Deduplication**: SHA-256 exact matching + simhash/minhash near-duplicate detection
- **Threading**: Email thread detection via Message-Id, In-Reply-To, References headers

### Metrics to Watch
- **Processing Rate**: emails/second throughput
- **Deduplication Ratio**: percentage of duplicate emails detected
- **OCR Rate**: percentage of attachments requiring OCR
- **Failure Buckets**: categorized error types and counts
- **Processing Lag**: time from email arrival to completion
- **Storage Usage**: object storage consumption per tenant

### AI Agents (LangGraph)
- **Coordinator**: Workflow orchestration and budget management
- **Query Analyzer**: Intent classification and search plan generation
- **Retriever**: Hybrid search execution and result fusion
- **Numeric Verifier**: Deterministic distance/price/date calculations
- **Compliance Guard**: Policy verification and tone checking
- **Drafter**: Response generation with citations
- **Human Gate**: Optional review for low-confidence responses

### Security & Compliance
- **Multi-Tenant Isolation**: Complete data separation
- **PII Protection**: Automatic detection and redaction
- **Prompt Injection Prevention**: Input validation and sanitization
- **Audit Logging**: Comprehensive operation tracking

### Performance & Monitoring
- **SLOs**: 100k emails/day, <4s response, 50+ concurrent
- **Observability**: Structured logging, metrics, tracing
- **Health Checks**: Kubernetes-ready probes
- **Circuit Breakers**: Graceful degradation patterns

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL, OpenSearch, Qdrant (or use Docker)

### Database & Migrations

#### Setting up the Database
1. **Environment Configuration**: Set `DATABASE_URL` in your `.env` file:
   ```bash
   DATABASE_URL=postgresql+asyncpg://postgres:evillm_password@localhost:5432/evillm
   ```

2. **Running Migrations**: Use Alembic to manage database schema:
   ```bash
   # Apply all pending migrations
   alembic upgrade head
   
   # Check current migration status
   alembic current
   
   # Rollback to previous migration
   alembic downgrade -1
   
   # View migration history
   alembic history
   ```

3. **Database Health Checks**: The system automatically checks database health:
   - Connection status
   - Migration status
   - Pending migrations
   - Database information

#### Local Development Database
For local development, use the provided Docker Compose setup:
```bash
# Start all services including PostgreSQL
make docker-up

# Check database health
curl http://localhost:8000/health/detailed

# Run tests against local database
make test
```

### Storage Service

#### Setting up Object Storage
1. **Environment Configuration**: Configure storage settings in your `.env` file:
   ```bash
   # For local MinIO development
   STORAGE_PROVIDER=minio
   STORAGE_ENDPOINT_URL=http://localhost:9000
   STORAGE_ACCESS_KEY_ID=minioadmin
   STORAGE_SECRET_ACCESS_KEY=minioadmin
   STORAGE_BUCKET_NAME=evillm
   
   # For production AWS S3
   STORAGE_PROVIDER=s3
   STORAGE_ENDPOINT_URL=https://s3.amazonaws.com
   STORAGE_REGION=us-east-1
   STORAGE_ACCESS_KEY_ID=your_access_key
   STORAGE_SECRET_ACCESS_KEY=your_secret_key
   ```

2. **Local MinIO Setup**: Use Docker Compose for local development:
   ```bash
   # Start MinIO service
   docker-compose -f infra/docker-compose.yml up -d minio
   
   # Access MinIO console at http://localhost:9001
   # Login with minioadmin/minioadmin
   ```

3. **Storage Health Checks**: Monitor storage service health:
   ```bash
   # Check storage health
   curl http://localhost:8000/health/detailed
   
   # Storage health includes:
   # - Bucket accessibility
   # - Optional canary tests (non-prod)
   # - Configuration information
   ```

#### Storage Operations
The storage service provides:
- **Raw Email Storage**: MIME messages with deterministic paths
- **Normalized Text**: Processed email content
- **Attachment Storage**: Files with OCR text support
- **Presigned URLs**: Secure, time-limited access
- **Multi-tenant Isolation**: Path-based tenant separation
- **Content Deduplication**: SHA-256 hash-based deduplication

#### Example Usage
```python
from app.storage import get_storage_client, StoragePathBuilder, ObjectMetadata, ContentHash

# Get storage client
client = get_storage_client()

# Build storage path
path = StoragePathBuilder.build_email_raw_path(
    tenant_id="tenant-123",
    thread_id="thread-456", 
    email_id="email-789",
    message_id="msg-abc"
)

# Create metadata
content_hash = ContentHash.from_content(b"email content")
metadata = ObjectMetadata(
    tenant_id="tenant-123",
    content_sha256=content_hash.value,
    content_length=len(b"email content"),
    mimetype="message/rfc822"
)

# Store object
success = await client.put_object(
    key=path,
    data=b"email content",
    metadata=metadata,
    tenant_id="tenant-123"
)

# Generate presigned URL
url = await client.generate_presigned_url(
    key=path,
    tenant_id="tenant-123",
    ttl=900  # 15 minutes
)
```

### Local Development
```

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd evillm
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -e .
   ```

2. **Environment configuration**:
   ```bash
   cp infra/env.example .env
   # Edit .env with your configuration
   ```

3. **Start services with Docker**:
   ```bash
   cd infra
   docker-compose up -d
   ```

4. **Run the application**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Docker Deployment

```bash
# Build and run all services
docker-compose -f infra/docker-compose.yml up -d

# View logs
docker-compose -f infra/docker-compose.yml logs -f app

# Stop services
docker-compose -f infra/docker-compose.yml down
```

## API Usage

### Generate Response Draft

```bash
curl -X POST "http://localhost:8000/draft" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant-id" \
  -d '{
    "query": "What is the shipping cost from NYC to LA for 500kg?",
    "tenant_id": "your-tenant-id",
    "context": {
      "email_thread_id": "thread-123",
      "business_rules": {"priority": "standard"}
    }
  }'
```

### Run Evaluation

```bash
curl -X POST "http://localhost:8000/eval/run" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant-id" \
  -d '{
    "response": "The shipping cost is $2,500 for standard delivery.",
    "ground_truth": "Standard shipping from NYC to LA for 500kg costs $2,500",
    "tenant_id": "your-tenant-id"
  }'
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4

# Vector Store
VECTOR_STORE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Search Engine
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200

# Storage
STORAGE_PROVIDER=minio
STORAGE_ENDPOINT=localhost:9000
```

### Service Configuration

- **PostgreSQL**: Email metadata and relationships
- **MinIO/S3**: Raw email and attachment storage
- **OpenSearch**: BM25 keyword search
- **Qdrant**: Vector similarity search
- **Redis**: Caching and job queues

## Development

### Project Structure

```
evillm/
├── app/                    # Application code
│   ├── agents/            # LangGraph agent nodes
│   ├── config/            # Configuration and settings
│   ├── database/          # Database models and migrations
│   ├── middleware/        # FastAPI middleware
│   ├── routers/           # API endpoints
│   ├── services/          # Business logic services
│   └── tools/             # Deterministic computation tools
├── eval/                  # Evaluation framework
├── infra/                 # Infrastructure and deployment
├── specs/                 # Requirements and design specifications
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy app/

# Formatting
ruff format .
```

## Evaluation Framework

### Quality Metrics

- **Grounding**: Factual accuracy and source verification
- **Completeness**: Query coverage and response thoroughness
- **Tone**: Professional communication appropriateness
- **Policy**: Compliance with business rules and regulations

### Evaluation Tools

- **promptfoo**: Automated test cases and golden examples
- **LangSmith**: LLM-as-judge evaluation and calibration
- **Custom Metrics**: Business-specific scoring algorithms

## Performance Tuning

### Search Optimization

- **Chunking Strategy**: Semantic boundaries with overlap
- **Index Tuning**: BM25 parameters and vector dimensions
- **Caching**: Redis-based result and embedding caching
- **Async Processing**: Concurrent search and generation

### Scaling Considerations

- **Horizontal Scaling**: Multiple worker instances
- **Database Sharding**: Tenant-based data distribution
- **Load Balancing**: Request distribution and health checks
- **Monitoring**: Performance metrics and alerting

## Security

### Multi-Tenant Isolation

- **Database Level**: Row-level security and tenant filtering
- **Search Level**: Index isolation and query filtering
- **Storage Level**: Bucket and path-based separation
- **API Level**: Middleware validation and header enforcement

### Data Protection

- **Encryption**: At-rest and in-transit encryption
- **PII Detection**: Automatic identification and redaction
- **Access Control**: Role-based permissions and audit trails
- **Input Validation**: Prompt injection prevention and sanitization

## Monitoring & Observability

### Metrics Collection

- **Application Metrics**: Request rates, response times, error rates
- **Business Metrics**: Email volume, search quality, user satisfaction
- **Infrastructure Metrics**: Resource utilization, service health
- **Custom Metrics**: Tenant-specific performance indicators

### Logging & Tracing

- **Structured Logging**: JSON format with correlation IDs
- **Distributed Tracing**: Request flow across services
- **Error Tracking**: Exception monitoring and alerting
- **Audit Logging**: Security and compliance event tracking

## Contributing

### Development Workflow

1. **Spec-Driven**: All changes start with specification updates
2. **Task-Based**: Implementation follows defined task sequences
3. **Quality Gates**: Code must pass tests and evaluations
4. **Review Process**: All changes require review and approval

### Code Standards

- **Python**: Type hints, docstrings, ruff/mypy compliance
- **FastAPI**: Async endpoints, proper error handling
- **Testing**: >80% coverage, unit + integration tests
- **Documentation**: Keep specs as source of truth

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: [Project Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Security**: [Security Policy](link-to-security)

## Roadmap

### Phase 1 (Current)
- [x] Project foundation and specifications
- [x] Basic FastAPI application structure
- [ ] Database schema and migrations
- [ ] Storage service implementation
- [ ] Email ingestion pipeline
- [ ] Hybrid search engine
- [ ] LangGraph agent framework
- [ ] Evaluation framework

### Phase 2
- [ ] Advanced OCR and document processing
- [ ] Performance optimization and scaling
- [ ] Advanced evaluation metrics
- [ ] Production deployment automation

### Phase 3
- [ ] Multi-language support
- [ ] Advanced compliance features
- [ ] Machine learning model integration
- [ ] Enterprise features and integrations
