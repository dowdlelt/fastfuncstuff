.PHONY: check check-types check-lint fix install test clean help

help:
	@echo "Available commands:"
	@echo "  make check       - Run all checks (types + linting)"
	@echo "  make check-types - Run pyright type checker"
	@echo "  make check-lint  - Run ruff linter"
	@echo "  make fix         - Auto-fix linting issues"
	@echo "  make install     - Install package in editable mode"
	@echo "  make test        - Run pytest tests"
	@echo "  make clean       - Remove generated files"

check: check-types check-lint

check-types:
	@echo "🔍 Running pyright type checker..."
	@pyright fastfuncsim/ bin/ examples/ || (echo "⚠️  Type errors found" && exit 1)
	@echo "✅ Type checking passed!"

check-lint:
	@echo "🔧 Running ruff linter..."
	@ruff check fastfuncsim/ bin/ examples/
	@echo "✅ Linting passed!"

fix:
	@echo "🔧 Auto-fixing linting issues..."
	@ruff check --fix fastfuncsim/ bin/ examples/
	@ruff format fastfuncsim/ bin/ examples/
	@echo "✅ Auto-fixes applied!"

install:
	@echo "📦 Installing fastfuncsim in editable mode..."
	@pip install -e .
	@echo "✅ Installation complete!"

test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v
	@echo "✅ Tests complete!"

clean:
	@echo "🧹 Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .coverage .pytest_cache/ dist/ build/
	@echo "✅ Cleanup complete!"
