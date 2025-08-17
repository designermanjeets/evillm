# EvilLLM Makefile - Demo and Development Targets

.PHONY: help install test lint clean demo-seed demo-run demo-open demo-smoke strict-scan collect-metrics

# Default target
help:
	@echo "EvilLLM Development and Demo Targets"
	@echo "===================================="
	@echo ""
	@echo "Development:"
	@echo "  install        Install dependencies"
	@echo "  test           Run all tests"
	@echo "  lint           Run linting and formatting"
	@echo "  clean          Clean up temporary files"
	@echo ""
	@echo "Demo:"
	@echo "  demo-seed      Seed demo data and create fixtures"
	@echo "  demo-run       Start the application server"
	@echo "  demo-open      Open demo UI in browser"
	@echo "  demo-smoke     Run strict mode verification and smoke tests"
	@echo ""
	@echo "Verification:"
	@echo "  strict-scan    Run strict mode scanner (no fallbacks)"
	@echo "  collect-metrics Collect metrics snapshot"
	@echo ""
	@echo "Examples:"
	@echo "  make demo-seed && make demo-run && make demo-open"
	@echo "  make demo-smoke  # Full verification"

# Development targets
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short
	@echo "✅ Tests completed"

lint:
	@echo "Running linting..."
	black app/ tests/ --check
	flake8 app/ tests/ --max-line-length=88
	@echo "✅ Linting completed"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "✅ Cleanup completed"

# Demo targets
demo-seed:
	@echo "🌱 Seeding demo data..."
	python -m scripts.demo_seed
	@echo "✅ Demo data seeded"

demo-run:
	@echo "🚀 Starting EvilLLM application..."
	@echo "📱 Application will be available at: http://localhost:8000"
	@echo "📚 API docs: http://localhost:8000/docs"
	@echo "🔍 Debug endpoints: http://localhost:8000/debug/"
	@echo ""
	@echo "Press Ctrl+C to stop the server"
	@echo ""
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

demo-open:
	@echo "🌐 Opening demo UI in browser..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:8000/ui/; \
		open http://localhost:8000/ui/upload; \
		open http://localhost:8000/ui/search; \
		open http://localhost:8000/ui/draft; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8000/ui/; \
		xdg-open http://localhost:8000/ui/upload; \
		xdg-open http://localhost:8000/ui/search; \
		xdg-open http://localhost:8000/ui/draft; \
	else \
		echo "Please manually open these URLs in your browser:"; \
		echo "  Main: http://localhost:8000/ui/"; \
		echo "  Upload: http://localhost:8000/ui/upload"; \
		echo "  Search: http://localhost:8000/ui/search"; \
		echo "  Draft: http://localhost:8000/ui/draft"; \
	fi

demo-smoke: strict-scan collect-metrics
	@echo "🧪 Running smoke tests..."
	pytest tests/strict_mode/ -v --tb=short
	@echo "✅ Demo smoke tests completed"

# Verification targets
strict-scan:
	@echo "🔍 Running strict mode scanner..."
	python scripts/strict_scan.py
	@echo "✅ Strict mode scan completed"

collect-metrics:
	@echo "📊 Collecting metrics snapshot..."
	python scripts/collect_metrics.py
	@echo "✅ Metrics collection completed"

# Quick verification
verify: strict-scan
	@echo "🔒 Strict mode verification completed"
	@echo "📊 Run 'make collect-metrics' to collect performance data"
	@echo "🧪 Run 'make demo-smoke' for full verification"

# Health check
health:
	@echo "🏥 Checking application health..."
	@if curl -s http://localhost:8000/health/ready >/dev/null; then \
		echo "✅ Application is healthy and running"; \
	else \
		echo "❌ Application is not responding"; \
		echo "   Run 'make demo-run' to start the server"; \
	fi

# Strict mode report
strict-report:
	@echo "📋 Getting strict mode configuration..."
	@curl -s -H "X-Tenant-ID: demo-tenant" http://localhost:8000/debug/strict-report | python -m json.tool

# Demo workflow (all-in-one)
demo: demo-seed demo-run
	@echo ""
	@echo "🎉 Demo setup completed!"
	@echo "📱 Application is running at http://localhost:8000"
	@echo "🌐 Run 'make demo-open' to open UI in browser"
	@echo "🧪 Run 'make demo-smoke' to verify functionality"

# Development workflow
dev: install lint test
	@echo "✅ Development environment ready"

# Full verification workflow
verify-all: strict-scan collect-metrics demo-smoke
	@echo ""
	@echo "🎯 Full verification completed!"
	@echo "📊 Check /reports/ for detailed results"
	@echo "🔒 Strict mode: VERIFIED (no fallbacks)"
	@echo "📱 Demo: READY for client presentation"

# Helpers
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import fastapi, pydantic, structlog; print('✅ Core dependencies available')"
	@echo "✅ Dependency check completed"

check-config:
	@echo "⚙️ Checking configuration..."
	@python -c "from app.config.settings import get_settings; s = get_settings(); print(f'✅ Config loaded: strict_mode={s.security.strict_mode}')"
	@echo "✅ Configuration check completed"

# Quick start for new developers
quickstart: check-deps check-config strict-scan
	@echo ""
	@echo "🚀 Quick start completed!"
	@echo "📱 Run 'make demo-run' to start the application"
	@echo "🌐 Run 'make demo-open' to open the UI"
	@echo "🧪 Run 'make demo-smoke' to verify everything works"

# Production readiness check
prod-check: strict-scan test lint
	@echo ""
	@echo "🏭 Production readiness check completed!"
	@echo "🔒 Strict mode: VERIFIED"
	@echo "🧪 Tests: PASSED"
	@echo "✨ Code quality: VERIFIED"
	@echo "🚀 Ready for production deployment"

# Show current status
status:
	@echo "📊 EvilLLM Status Report"
	@echo "========================"
	@echo ""
	@echo "🔍 Strict Mode Scanner:"
	@python scripts/strict_scan.py >/dev/null 2>&1 && echo "  ✅ PASS" || echo "  ❌ FAIL"
	@echo ""
	@echo "🏥 Application Health:"
	@curl -s http://localhost:8000/health/ready >/dev/null 2>&1 && echo "  ✅ RUNNING" || echo "  ❌ STOPPED"
	@echo ""
	@echo "📁 Reports Available:"
	@ls -la reports/ 2>/dev/null || echo "  No reports directory found"
	@echo ""
	@echo "🎯 Next Steps:"
	@echo "  - Run 'make demo-run' to start the application"
	@echo "  - Run 'make demo-smoke' for full verification"
	@echo "  - Check /reports/ for detailed results"
