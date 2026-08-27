.PHONY: check lint fix install test clean help completions

help:
	@echo "Available commands:"
	@echo "  make check   - Run type checker and linter"
	@echo "  make lint    - Run ruff linter only"
	@echo "  make fix     - Auto-fix linting issues and format"
	@echo "  make install - Install package in editable mode"
	@echo "  make test    - Run pytest test suite"
	@echo "  make clean   - Remove generated files"
	@echo "  make completions - Regenerate shell completions for the ffs_* tools"

check: lint
	ty check fastfuncstuff/

lint:
	ruff check fastfuncstuff/ examples/

fix:
	ruff check --fix fastfuncstuff/ examples/
	ruff format fastfuncstuff/ examples/

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

completions:
	@# Uses THIS environment's python on purpose: the tool list comes from its
	@# installed metadata, so generating from another env writes a partial set.
	python -m fastfuncstuff.cli.completion -shell fish -o $(HOME)/.config/fish/completions
	python -m fastfuncstuff.cli.completion -shell bash -o $(HOME)/.local/share/bash-completion/completions
	python -m fastfuncstuff.cli.completion -shell zsh  -o $(HOME)/.zfunc

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/ dist/ build/
