# Integration Test Report: GitHub ↔ Linear Webhook (WEN-165)

## Purpose

Verify GitHub PR / Review / Merge events sync to Linear Issue WEN-165 via the official Linear-GitHub integration.

## Prerequisites

- [ ] Linear **Settings → Integrations → GitHub** connected to target organization/repository
- [ ] Code access enabled for Review visibility (organization settings)
- [ ] Test repository within authorized scope
- [ ] Personal GitHub account connected in Linear (Settings → Connected accounts)

## Test Results

| Step | GitHub Action | Expected Linear Behavior | Actual Result | Status |
|------|--------------|-------------------------|---------------|--------|
| 1 | Create PR linked to WEN-165 | PR link appears in Issue Activity | | |
| 2 | Mark PR Ready for review | Issue → **In Review** (if configured) | | |
| 3 | Submit Review (Approve/Request Changes) | Review visible in Issue | | |
| 4 | Merge PR (with `Fixes WEN-165`) | Issue → **Done** | | |

## Artifacts

- **Test Repository**: HybridArena
- **PR URL**: (TBD after PR creation)
- **Test Date**: 2026-05-23
- **Tester**: (name)

## Notes

- This is a one-time integration test issue. After verification, the issue can be closed or deleted.
- If status doesn't update, edit the PR description on GitHub (add a space, then save) to trigger webhook replay.
- Default status mapping may vary based on workspace configuration.

## Decision Log

- **2026-05-23**: Using `Fixes WEN-165` in PR description to verify full lifecycle (PR creation → merge → issue auto-close).
