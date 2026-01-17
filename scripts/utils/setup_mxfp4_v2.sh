#!/bin/bash
# Setup script for MXFP4 v2 branches
# Creates fresh branches from upstream for systematic feature porting

set -e

DATE=$(date +%Y%m%d)
VLLM_DIR="${HOME}/projects/vllm"
FLASHINFER_DIR="${HOME}/projects/flashinfer"
MXFP4_DIR="${HOME}/projects/ai/mxfp4"

echo "=== MXFP4 v2 Branch Setup ==="
echo "Date: $DATE"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_clean() {
    local dir=$1
    local name=$2
    cd "$dir"
    if [[ -n $(git status --porcelain) ]]; then
        echo -e "${YELLOW}Warning: $name has uncommitted changes${NC}"
        git status --short
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Step 1: Backup existing branches
echo "=== Step 1: Backup existing branches ==="

echo "Backing up FlashInfer mxfp4_wip..."
cd "$FLASHINFER_DIR"
git fetch origin
if git rev-parse mxfp4_wip >/dev/null 2>&1; then
    git tag -f "mxfp4_wip_backup_$DATE" mxfp4_wip
    echo -e "${GREEN}Tagged: mxfp4_wip_backup_$DATE${NC}"
else
    echo -e "${YELLOW}No mxfp4_wip branch found${NC}"
fi

echo "Backing up vLLM mxfp4_wip..."
cd "$VLLM_DIR"
git fetch origin
git fetch upstream
if git rev-parse mxfp4_wip >/dev/null 2>&1; then
    git tag -f "mxfp4_wip_backup_$DATE" mxfp4_wip
    echo -e "${GREEN}Tagged: mxfp4_wip_backup_$DATE${NC}"
else
    echo -e "${YELLOW}No mxfp4_wip branch found${NC}"
fi

echo "Backing up mxfp4 repo..."
cd "$MXFP4_DIR"
git add -A 2>/dev/null || true
git commit -m "Backup before mxfp4_v2 setup" 2>/dev/null || true
git tag -f "mxfp4_backup_$DATE"
echo -e "${GREEN}Tagged: mxfp4_backup_$DATE${NC}"

echo ""
echo "=== Step 2: Create fresh branches ==="

# FlashInfer: fresh from upstream/main
echo "Creating FlashInfer mxfp4_v2 from upstream/main..."
cd "$FLASHINFER_DIR"
git fetch upstream
check_clean "$FLASHINFER_DIR" "FlashInfer"

if git rev-parse mxfp4_v2 >/dev/null 2>&1; then
    echo -e "${YELLOW}mxfp4_v2 already exists. Delete and recreate? [y/N]${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -D mxfp4_v2
    else
        echo "Keeping existing mxfp4_v2"
    fi
fi

git checkout -b mxfp4_v2 upstream/main 2>/dev/null || git checkout mxfp4_v2
FLASHINFER_SHA=$(git rev-parse HEAD)
echo -e "${GREEN}FlashInfer mxfp4_v2 at: $FLASHINFER_SHA${NC}"

# vLLM: from PR with MXFP4 gating work
echo ""
echo "Creating vLLM mxfp4_v2 from PR #31740..."
cd "$VLLM_DIR"
git fetch upstream
git fetch upstream pull/31740/head:pr-31740 2>/dev/null || echo "PR already fetched"
check_clean "$VLLM_DIR" "vLLM"

if git rev-parse mxfp4_v2 >/dev/null 2>&1; then
    echo -e "${YELLOW}mxfp4_v2 already exists. Delete and recreate? [y/N]${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -D mxfp4_v2
    else
        echo "Keeping existing mxfp4_v2"
    fi
fi

git checkout -b mxfp4_v2 pr-31740 2>/dev/null || git checkout mxfp4_v2
VLLM_SHA=$(git rev-parse HEAD)
echo -e "${GREEN}vLLM mxfp4_v2 at: $VLLM_SHA${NC}"

echo ""
echo "=== Step 3: Verify setup ==="

echo "FlashInfer branch: $(cd $FLASHINFER_DIR && git branch --show-current)"
echo "FlashInfer SHA: $FLASHINFER_SHA"
echo ""
echo "vLLM branch: $(cd $VLLM_DIR && git branch --show-current)"
echo "vLLM SHA: $VLLM_SHA"

echo ""
echo "=== Step 4: Record baseline SHAs ==="

BASELINE_FILE="$MXFP4_DIR/docs/BASELINE_SHAS.md"
cat > "$BASELINE_FILE" << EOF
# Baseline Git SHAs for MXFP4 v2

Created: $(date -Iseconds)

## Fresh Branches (mxfp4_v2)

| Repo | Branch | SHA | Source |
|------|--------|-----|--------|
| FlashInfer | mxfp4_v2 | \`$FLASHINFER_SHA\` | upstream/main |
| vLLM | mxfp4_v2 | \`$VLLM_SHA\` | PR #31740 |

## Backup Tags

| Repo | Tag | Date |
|------|-----|------|
| FlashInfer | mxfp4_wip_backup_$DATE | $DATE |
| vLLM | mxfp4_wip_backup_$DATE | $DATE |
| mxfp4 | mxfp4_backup_$DATE | $DATE |

## Verification

\`\`\`bash
# Verify FlashInfer
cd ~/projects/flashinfer && git log -1 --oneline

# Verify vLLM
cd ~/projects/vllm && git log -1 --oneline

# Check branches
cd ~/projects/flashinfer && git branch --show-current
cd ~/projects/vllm && git branch --show-current
\`\`\`
EOF

echo -e "${GREEN}Baseline SHAs recorded in: $BASELINE_FILE${NC}"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Rebuild docker container: docker compose -f docker-compose.dev.yml build"
echo "  2. Start container: docker compose -f docker-compose.dev.yml up -d"
echo "  3. Run baseline benchmark: scripts/benchmark_matrix.py --config baseline"
echo "  4. Start porting features from mxfp4_wip branches"
echo ""
echo -e "${GREEN}Ready for MXFP4 v2 development!${NC}"
