# Integration Test Report: GitHub ↔ Linear Webhook (WEN-165)

## Purpose

Verify GitHub PR / Review / Merge events sync to Linear Issue WEN-165 via the official Linear-GitHub integration.

## Prerequisites

- [x] Linear **Settings → Integrations → GitHub** connected to target organization/repository
- [ ] Code access enabled for Review visibility (organization settings)
- [x] Test repository within authorized scope
- [ ] Personal GitHub account connected in Linear (Settings → Connected accounts)

## Test Results

| Step | GitHub Action | Expected Linear Behavior | Actual Result | Status |
|------|--------------|-------------------------|---------------|--------|
| 1 | Create PR linked to WEN-165 | PR link appears in Issue Activity | PR automatically linked to Issue; status changed to "In Progress" | ✅ PASS |
| 2 | Mark PR Ready for review | Issue → **In Review** (if configured) | (Pending user verification) | ⏳ PENDING |
| 3 | Submit Review (Approve/Request Changes) | Review visible in Issue | (Pending user verification) | ⏳ PENDING |
| 4 | Merge PR (with `Fixes WEN-165`) | Issue → **Done** | (Pending user verification) | ⏳ PENDING |

## Artifacts

- **Test Repository**: HybridArena
- **PR URL**: https://github.com/xwt-personal-org/HybridArena/pull/11
- **Test Date**: 2026-05-23
- **Tester**: (name)

## Notes

- This is a one-time integration test issue. After verification, the issue can be closed or deleted.
- If status doesn't update, edit the PR description on GitHub (add a space, then save) to trigger webhook replay.
- Default status mapping may vary based on workspace configuration.

## 记录区（测试时填写）

* **测试仓库**：HybridArena (xwt-personal-org/HybridArena)
* **PR URL**：https://github.com/xwt-personal-org/HybridArena/pull/11
* **观察到的状态变化**：
  - PR 创建后，Linear Issue 状态自动从 Todo 变为 "In Progress"
  - PR 链接自动出现在 Issue 的 Attachments 中
  - 分支名 `w2030298/wen-165-集成测试-github-linear-webhook-联调验证` 自动关联
* **异常/未同步步骤**：
  - Review 和 Merge 事件需要用户手动验证

## Decision Log

- **2026-05-23**: Using `Fixes WEN-165` in PR description to verify full lifecycle (PR creation → merge → issue auto-close).
