.PHONY: check lint fix install test clean help

help:
	@echo "Available commands:"
	@echo "  make check   - Run type checker and linter"
	@echo "  make lint    - Run ruff linter only"
	@echo "  make fix     - Auto-fix linting issues and format"
	@echo "  make install - Install package in editable mode"
	@echo "  make test    - Run pytest test suite"
	@echo "  make clean   - Remove generated files"

check: lint
	ty check fastfuncsim/

lint:
	ruff check fastfuncsim/ examples/

fix:
	ruff check --fix fastfuncsim/ examples/
	ruff format fastfuncsim/ examples/

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/ dist/ build/
