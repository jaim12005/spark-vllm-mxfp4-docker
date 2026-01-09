# Baseline Git SHAs for MXFP4 v2

Created: 2026-01-09

## Fresh Branches (mxfp4_v2)

| Repo | Branch | SHA | Source |
|------|--------|-----|--------|
| FlashInfer | mxfp4_v2 | `bd2b033f` | upstream/main |
| vLLM | mxfp4_v2 | `77bf5a554` | PR #31740 |

## Backup Tags

| Repo | Tag | Date |
|------|-----|------|
| FlashInfer | mxfp4_wip_backup_20260109 | 2026-01-09 |
| vLLM | mxfp4_wip_backup_20260109 | 2026-01-09 |
| mxfp4 | mxfp4_backup_20260109 | 2026-01-09 |

## Verification

```bash
# Verify FlashInfer
cd ~/projects/flashinfer && git log -1 --oneline

# Verify vLLM
cd ~/projects/vllm && git log -1 --oneline

# Check branches
cd ~/projects/flashinfer && git branch --show-current
cd ~/projects/vllm && git branch --show-current
```

## Restoring Previous Work

To restore the previous `mxfp4_wip` work:

```bash
# FlashInfer
cd ~/projects/flashinfer
git checkout mxfp4_wip  # Or: git checkout mxfp4_wip_backup_20260109

# vLLM
cd ~/projects/vllm
git checkout mxfp4_wip  # Or: git checkout mxfp4_wip_backup_20260109
```

## PR #31740 Details

The vLLM mxfp4_v2 branch is based on PR #31740 which contains:
- SM121 (GB10) support with MXFP4 quantization
- FlashInfer MoE integration
- Environment variable gating for kernel selection

To update to latest PR:
```bash
cd ~/projects/vllm
git fetch upstream pull/31740/head:pr-31740-latest
git merge pr-31740-latest  # Or rebase
```
