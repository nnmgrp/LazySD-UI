---
name: code-auditor
description: >-
  Performs rigorous code quality, security, and edge-case audits on modified files or targeted modules.
  Detects vulnerabilities (path traversal, injection, XSS), unhandled exceptions, race conditions,
  API contract drift, and resource leaks. Trigger when asked to audit, review, verify code safety,
  or perform a pre-commit security check.
---

# Code & Security Auditor

A specialized audit runbook for conducting deep, adversarial code reviews. Act as an independent auditor: assume inputs are untrusted, networks can fail, and edge cases will occur.

---

## Golden Rules

1. **Target the Delta First**: Prioritize changes (`git diff --staged` or `git diff HEAD~1`) before auditing surrounding context.
2. **Evidence-Based Findings**: Every flagged issue must cite the exact file path, line numbers, and a concrete failure/exploit scenario. No vague generalities.
3. **Actionable Fixes**: Provide a direct, drop-in replacement snippet for every Critical or Warning issue.

---

## 6-Stage Audit Checklist

### Stage 1: Security & Injection Hazards
- **Path Traversal**:
  - Python: Check any `Path(p)`, `os.path.join()`, or `open()` taking user/network input. Ensure `.resolve()` is validated against base directory (e.g. `path.resolve().is_relative_to(base)`).
  - Node/JS: Check `path.join()`, `fs.readFile()`, or sendFile.
- **Command / Shell Injection**:
  - Python: Verify `subprocess.run()`, `Popen()`. Ensure `shell=False` and args are passed as list, not formatted strings.
  - Windows specific: Check quoting of paths containing spaces or quotes.
- **Frontend XSS & DOM Injection**:
  - Check `innerHTML`, `outerHTML`, `v-html`, or `dangerouslySetInnerHTML`.
  - Ensure any user-controlled string is escaped using `escapeHtml()` or inserted via `textContent`.
- **Permission & Role Bypass**:
  - Verify auth tokens, guest checks, and endpoint scopes. Ensure caller identity cannot be spoofed.

### Stage 2: Runtime Stability & Exception Handling
- **Silent Failures**: Flag any `except Exception: pass` or empty `catch {}` that hides critical errors without logging.
- **Null / Undefined Dereferencing**:
  - JS/TS: Flag unguarded deep property chains (e.g. `obj.a.b` without `obj?.a?.b`).
  - Python: Flag `dict[key]` without prior check or `.get(key)` where the key might be absent.
- **Type Conversions**:
  - Flag unvalidated `int(val)` or `float(val)` that throw `ValueError` on empty string, `null`, or NaN.

### Stage 3: Concurrency & State Consistency
- **Race Conditions**: Shared mutable state modified across asynchronous handlers or multiple threads.
- **Stale State**: In frontend UI, check if state callbacks can execute after component unmount or after active model/tab has changed.
- **Deadlocks & Timeout Gaps**: Ensure subprocesses, HTTP requests, and locks always have explicit timeouts.

### Stage 4: Resource & Memory Management
- **Resource Leaks**:
  - Unclosed file handles or database connections.
  - Uncancelled timers (`setInterval`, `schedule`) or unremoved event listeners.
- **GPU / Memory Bloat**:
  - PyTorch: Check for missing `torch.no_grad()` or `torch.inference_mode()` during generation.
  - WebGL / Canvas: Ensure textures, FBOs, and shaders are destroyed on resize or model swap.

### Stage 5: API & Contract Consistency
- **Frontend / Backend Sync**:
  - Verify that query parameters or JSON body fields sent by frontend match what the backend endpoint expects.
  - Check default values: ensure booleans aren't sent as string `"false"` (which evaluates to truthy).
- **Silent Parameter Dropping**:
  - Trace request payloads through worker classes to ensure options (e.g. `cfg`, `noise`, `vae`, `seed`) actually reach the CLI or engine instead of being ignored.

---

## Audit Report Format

Structure audit results strictly using this template:

```markdown
## 🛡️ Code & Security Audit Report

### Summary
- **Files Audited**: `<list of files>`
- **Overall Verdict**: `[PASSED] | [PASSED WITH WARNINGS] | [BLOCKED - ISSUES FOUND]`

---

### 🔴 Critical Issues (Must Fix Before Merge/Run)
*(Issues that cause crashes, data corruption, security exploits, or contract breaks)*

- **[File & Line]**: `path/to/file.py#L42-L48`
  - **Issue**: Short description of the vulnerability/bug
  - **Exploit / Failure Scenario**: What input or condition triggers it
  - **Fix**:
    ```python
    # Corrected code snippet
    ```

---

### 🟡 Warnings & Edge Cases
*(Issues that can fail under non-ideal conditions, silent errors, or minor memory leaks)*

- **[File & Line]**: `path/to/file.js#L112`
  - **Issue**: Detailed explanation
  - **Recommended Fix**: Guidance or snippet

---

### 🟢 Suggestions & Code Health
*(Cleanliness, idiomatic improvements, performance optimizations)*
- **[File & Line]**: Note and suggestion
```
