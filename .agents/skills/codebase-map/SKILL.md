---
name: codebase-map
description: >-
  Builds or incrementally updates a compact, token-efficient REPO_MAP.md (≤250 lines) external memory index.
  Extracts directory roles, key exported signatures, run scripts, and critical data flows using zero-token CLI discovery.
  Trigger when asked to map or index the codebase, explore repository architecture, or onboard to a project.
---

# High-Density Codebase Indexer

Your goal is to build or maintain a compact, high-signal `REPO_MAP.md` (hard limit: 250 lines, ~1500 tokens) at the root of the workspace. This file acts as an external memory index so agents can jump directly to the right files without exploratory context-burning searches.

---

## Golden Rules

1. **CLI-First, Zero Blind Full-File Reads**: Never read entire source files to discover structure. Use `git ls-files`, `git grep`, or PowerShell `Select-String` to extract contracts and signatures.
2. **Signatures Only**: Record only exported types, classes, public methods, schemas, and routes. Never include implementation details or function bodies.
3. **Hard Token Budget (≤ 250 Lines)**: If `REPO_MAP.md` exceeds 250 lines, prune utility/helper modules and preserve core domain boundaries.
4. **Ignore Build Artifacts**: Completely ignore `node_modules/`, `dist/`, `build/`, `.next/`, `venv/`, `__pycache__/`, `target/`, `.git/`, test coverage, lockfiles, and minified bundles.

---

## Execution Protocol

### Step 1: Shell Discovery (Zero LLM Token Scanning)

Run fast CLI commands to inspect the ecosystem without loading file bodies into the context window.

**1. Ecosystem & Manifest Discovery**:
```bash
# Check available manifests
git ls-files "package.json" "pyproject.toml" "Cargo.toml" "go.mod" "requirements.txt" "vite.config.*" "tsconfig.json"
```

Read **only** the detected package manifest (`package.json`, `pyproject.toml`, etc.) to extract:
- Run scripts (`test`, `build`, `dev`, `lint`, `start`)
- Main entry point
- Core runtime dependencies

**2. Top-Level Directory Roles**:
```bash
# Windows PowerShell
git ls-files | ForEach-Object { ($_ -split '[\\/]', 2)[0] } | Sort-Object -Unique

# Linux / Bash
git ls-files | awk -F/ '{if (NF>1) print $1"/"$2; else print $1}' | sort -u | head -n 50
```

---

### Step 2: Extract Signatures (Contract-Level Scan)

Use `git grep` (available on all platforms with Git installed, including Windows) to extract public interfaces without opening source files.

**TypeScript / JavaScript**:
```bash
git grep -n -E "^(export (default )?(class|interface|type|enum|const [A-Z]|function [a-zA-Z]))" -- "*.ts" "*.tsx" "*.js" "*.jsx"
```

**Python**:
```bash
git grep -n -E "^(class [a-zA-Z0-9_]+|def [a-zA-Z0-9_]+\(|async def [a-zA-Z0-9_]+\()" -- "*.py"
```

**Rust**:
```bash
git grep -n -E "^(pub (fn|struct|enum|trait|type)|fn [a-z])" -- "*.rs"
```

**Go**:
```bash
git grep -n -E "^(func |type [A-Z]|struct |interface )" -- "*.go"
```

**API Routes & Endpoints**:
```bash
git grep -n -E "(@(Get|Post|Put|Delete|Patch|Router)|router\.(get|post|put|delete)|app\.(get|post|use)\(|@app\.(get|post|route))"
```

---

### Step 3: Generate or Overwrite `REPO_MAP.md`

Format the output strictly according to this template at the root of the workspace:

```markdown
# REPO MAP

> **Commands**: `dev`: `<cmd>` | `build`: `<cmd>` | `test`: `<cmd>` | `lint`: `<cmd>`  
> **Ecosystem**: <runtime/framework> (e.g. React + Tauri + TypeScript, FastAPI + PyTorch)

## 1. System Topology & Directory Roles
- `path/to/domain/`: Core business logic [1 line]
- `path/to/api/`: Routing, serialization, controllers [1 line]
- `path/to/ui/` or `frontend/`: UI views, state management, components [1 line]
- `path/to/infra/`: DB drivers, external API clients, shared helpers [1 line]

## 2. Key Symbols & Contract Registry
Format: `path/to/file.ext` -> key signatures / exports only
- `src/services/auth.ts` -> `AuthService`, `verifySession()`, `TokenPayload`
- `backend/app.py` -> Endpoints: `POST /api/generate`, `GET /api/models`
- `src/db/models.py` -> Models: `User`, `Workspace`, `Session`

## 3. Architecture & Data Flow
1. **Entry & Routing**: User action / HTTP request -> Router / Middleware -> Controller
2. **Business Domain**: Controller -> Service / Handler -> Domain Entities
3. **State & Persistence**: Store (Redux/Zustand) / DB Driver / Storage Engine

## 4. Gotchas, Quirks & Conventions
- Path aliases: e.g. `@/` resolves to `./src/`
- Required environment variables or local services (e.g. `CUDA_VISIBLE_DEVICES`, local daemon)
- Tricky conventions or non-obvious design choices
```

---

## Incremental Maintenance Mode

When invoked after code changes, or when asked to update the index:

1. Inspect modified modules:
   ```bash
   git status -s
   git diff --name-only HEAD~3
   ```
2. Identify which sections in `REPO_MAP.md` are affected.
3. Update **only** the touched lines (files, classes, or endpoints).
4. Do **not** rewrite unaffected sections.
5. Re-check the hard budget (total lines must remain ≤ 250).

---

## Quality Checklist Before Finishing

- [ ] File is named exactly `REPO_MAP.md` and placed at workspace root.
- [ ] Total line count is strictly ≤ 250 lines.
- [ ] No function bodies or code snippets included.
- [ ] Scripts section contains real project commands.
- [ ] All listed paths exist in the repository.
