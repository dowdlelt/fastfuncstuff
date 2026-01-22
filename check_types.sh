#!/bin/bash
# Type checking and linting script
# Run this before committing to catch import/type errors

set -e

echo "🔍 Running type checks and linting..."
echo ""

# Check if in fastfuncsim directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this from the fastfuncsim root directory"
    exit 1
fi

# Run pyright on key files
echo "📊 Pyright (type checking)..."
pyright fastfuncsim/ bin/ examples/ || {
    echo ""
    echo "⚠️  Pyright found type errors (see above)"
    echo "   Focus on reportCallIssue and reportArgumentType errors"
    echo ""
}

# Run ruff for linting
echo ""
echo "🔧 Ruff (linting)..."
ruff check fastfuncsim/ bin/ examples/ --select=F,E --ignore=E501,E731 || {
    echo ""
    echo "⚠️  Ruff found linting issues (see above)"
    echo ""
}

echo ""
echo "✅ Type checking complete!"
echo ""
echo "To fix automatically fixable issues: ruff check --fix fastfuncsim/ bin/ examples/"
