# vyre-frontend-rust Design

GPU-first Rust compiler frontend for Vyre.

## Layer Position

- **This crate** (`vyre-frontend-rust`): Layer 1 thick driver. Owns the full compilation pipeline.
- **`vyre-libs::parsing::rust`**: Layer 0 reusable substrate. Lexer + parser only.
- **`vyre-lower` downstream analysis layer**: Dataflow and borrow-like checks used after parsing.

Layer 0 cannot depend on Layer 1. Downstream analysis integration lives here.

## Conventions (copied from `vyre-frontend-c`)

We follow the C frontend's conventions to prevent divergence:

| Convention | C frontend | Rust frontend |
|---|---|---|
| API module | `api/mod.rs` + `api/entrypoints.rs` | ✅ `api/mod.rs` + `api/entrypoints.rs` |
| Pipeline stages | One duty per file under `pipeline/` | ✅ `pipeline/lexer_dispatch.rs`, etc. |
| Pipeline orchestrator | `pipeline.rs` wires stages | ✅ `pipeline.rs` wires stages |
| Error messages | `"description. Fix: suggestion."` | ✅ Matched in `RustFrontendError` |
| Object format | `VYRECOB2` v7 sections | 🔄 Stub; must converge before v0.1.0 |
| Oracle/parity | `api/parity.rs` with `ParityFact` | 🔄 Basic `oracle/mod.rs`; will expand |
| Scratch buffers | `thread_local!` `RefCell` for GPU dispatch | 🔄 Not yet needed |

## Extraction Points

When a 3rd language frontend (Go, Python, or a new one) reaches pipeline parity, extract `vyre-frontend-core` containing:

1. **Pipeline orchestration** — `compile_unit()` stage sequencing
2. **Lexer dispatch framework** — GPU probe → plan → fallback
3. **Object writer** — VYRECOB2 section builder
4. **Parity infrastructure** — `ParityFact`, `ParityFinding`, release gating
5. **Backend selection** — CUDA vs WGPU dispatch

Marked in code with `// TODO(vyre-frontend-core): ...`.

## What Is Language-Specific (never extract)

- Grammar / parser rules
- Semantic analysis (type system, borrow check, trait resolution)
- ABI layout rules
- Lowering to Vyre IR

## Nano-Subset

The v0.1.0 nano-subset supports:

- Functions: `fn name(params) -> ret { body }`
- Let bindings: `let x: i32 = expr;`, `let mut x: i32 = expr;`
- If/else: `if cond { then } else { else }`
- Return: `return expr;`
- Types: `i32`, `bool`, `&T`, `&mut T`
- Expressions: literals, identifiers, binary ops (`+ - * / < ==`), unary (`! - * &`)

Anything outside this subset is rejected at parse time.
