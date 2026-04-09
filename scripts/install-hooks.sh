#!/bin/bash
# Install git hooks for HebrewScribe development.
# Run once after cloning: bash scripts/install-hooks.sh

set -e
HOOK_DIR="$(git rev-parse --git-dir)/hooks"

# pre-push: run E2E smoke test before every push
cat > "$HOOK_DIR/pre-push" << 'EOF'
#!/bin/bash
# pre-push hook: run E2E smoke test before pushing.
# Catches integration regressions that unit tests miss.
# Skips gracefully if faster-whisper is not installed.
# To bypass in an emergency: git push --no-verify

set -e
echo "Running E2E smoke tests..."
python3 -m pytest tests/ -m slow --tb=short -q 2>/dev/null \
    || python -m pytest tests/ -m slow --tb=short -q 2>/dev/null
echo "E2E tests passed."
EOF
chmod +x "$HOOK_DIR/pre-push"

echo "Git hooks installed."
