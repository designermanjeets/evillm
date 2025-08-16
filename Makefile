# Makefile for Logistics Email AI
# Common development tasks and shortcuts

.PHONY: help install test lint format clean docker-up docker-down run dev

# Default target
help:
	@echo "Logistics Email AI - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies"
	@echo "  setup       Setup development environment"
	@echo ""
	@echo "Development:"
	@echo "  run         Run the application"
	@echo "  dev         Run with auto-reload"
	@echo "  test        Run tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up   Start all services with Docker"
	@echo "  docker-down Stop all Docker services"
	@echo ""
	@echo "Quality:"
	@echo "  check       Run all quality checks"
	@echo "  clean       Clean up generated files"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate  Run database migrations"
	@echo "  db-reset    Reset database"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval        Run promptfoo evaluation"

# Setup and installation
install:
	@echo "Installing dependencies..."
	pip install -e .
	pip install -e ".[dev,test]"

setup: install
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then \
		cp infra/env.example .env; \
		echo "Created .env file from template"; \
		echo "Please edit .env with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Development commands
run:
	@echo "Starting Logistics Email AI..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	@echo "Starting Logistics Email AI in development mode..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-fast:
	@echo "Running fast tests only..."
	pytest tests/ -v -m "not slow"

# Code quality
lint:
	@echo "Running linting checks..."
	ruff check .
	mypy app/

format:
	@echo "Formatting code..."
	ruff format .
	ruff check --fix .

check: lint test
	@echo "All quality checks passed!"

# Docker commands
docker-up:
	@echo "Starting Docker services..."
	cd infra && docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	cd infra && docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	cd infra && docker-compose logs -f

docker-build:
	@echo "Building Docker image..."
	cd infra && docker-compose build

# Database commands
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	alembic upgrade head

db-reset: ## Reset database (WARNING: Destructive)
	@echo "WARNING: This will delete all data"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "Dropping all tables..."; \
		alembic downgrade base; \
		echo "Recreating schema..."; \
		alembic upgrade head; \
		echo "Database reset complete"; \
	else \
		echo "Database reset cancelled"; \
	fi

db-status: ## Check database migration status
	@echo "Checking database migration status..."
	alembic current
	alembic heads

db-create-migration: ## Create a new migration file
	@read -p "Enter migration description: " desc; \
	alembic revision --autogenerate -m "$$desc"

# Evaluation
eval:
	@echo "Running promptfoo evaluation..."
	# TODO: Install promptfoo when needed
	@echo "Please install promptfoo: npm install -g promptfoo"
	@echo "Then run: promptfoo eval -c eval/promptfoo.yaml"

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "Cleanup complete!"

# Health checks
health:
	@echo "Checking service health..."
	@if command -v curl >/dev/null 2>&1; then \
		curl -s http://localhost:8000/health/ | jq . || echo "Service not running"; \
	else \
		echo "curl not available"; \
	fi

# Development workflow
dev-setup: setup docker-up
	@echo "Development environment setup complete!"
	@echo "Services are starting up..."
	@echo "Run 'make dev' to start the application"

# Production-like testing
prod-test:
	@echo "Running production-like tests..."
	docker-compose -f infra/docker-compose.yml up -d
	sleep 30  # Wait for services to be ready
	make test
	docker-compose -f infra/docker-compose.yml down

# Performance testing
perf-test:
	@echo "Running performance tests..."
	# TODO: Implement performance testing
	@echo "Performance testing not yet implemented"

# Security checks
security-check:
	@echo "Running security checks..."
	# TODO: Implement security scanning
	@echo "Security scanning not yet implemented"

# Documentation
docs:
	@echo "Building documentation..."
	# TODO: Implement when mkdocs is set up
	@echo "Documentation building not yet implemented"

# Release preparation
release-check: check security-check
	@echo "Release checks completed successfully!"

# Quick development cycle
dev-cycle: format lint test
	@echo "Development cycle completed!"

# Show current status
status:
	@echo "Logistics Email AI - Current Status"
	@echo "=================================="
	@echo "Python version: $(shell python --version 2>/dev/null || echo 'Python not found')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Docker not found')"
	@echo "Docker Compose: $(shell docker-compose --version 2>/dev/null || echo 'Docker Compose not found')"
	@echo "Environment file: $(shell if [ -f .env ]; then echo 'Present'; else echo 'Missing'; fi)"
	@echo "Services: $(shell if docker ps --format '{{.Names}}' | grep -q 'evillm'; then echo 'Running'; else echo 'Stopped'; fi)"
