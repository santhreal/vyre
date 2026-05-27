# `lower/` vs `emit/` naming convention

Established by audit cleanup A11 (2026-04-30).

## Rule

| Name | Lives in | Purpose |
|---|---|---|
| `lower/` | `vyre-foundation/src/lower/` | Substrate-side lowering: vyre IR → backend-IR / dialect-specific IR. Backend-agnostic. Substrate decides what to lower; backends decide how to emit. |
| `emit/` | `vyre-driver-<backend>/src/emit/` | Backend-specific final emit: backend-IR → final source (WGSL string), bytecode (PTX/SPIR-V), or binary. Backend-specific. |
| `codegen/` | `vyre-driver-cuda/src/codegen/` | CUDA convention; equivalent to `emit/` for the cuda backend. Kept for nvcc/PTX-tooling familiarity. |
| `lowering.rs` | `vyre-driver/src/backend/lowering.rs` | Cross-backend lowering trait boundary (`LowerableOp`, `TargetGenCtx`). Lives at the driver layer because it defines the contract every backend implements. |

## Renamings done in A11

- `vyre-driver-wgpu/src/lowering/` → `vyre-driver-wgpu/src/emit/` (the
  naga_emit subdir + WGSL emission code is final emit, not substrate
  lowering).

## Why

Before A11, `lower/` and `lowering/` co-existed across crates with no
documented division. Two different concepts shared similar names. The
A11 convention disambiguates: substrate-side transformation = `lower/`;
backend-final-output = `emit/` (or `codegen/` for CUDA).

## Where to put new code

- New cross-backend IR transformation that vyre IR enters and a backend-
  IR exits → `vyre-foundation/src/lower/`.
- New backend-specific emission code (WGSL string, PTX text, SPIR-V
  binary, naga IR construction) → `vyre-driver-<backend>/src/emit/`.
- New trait that backends implement → `vyre-driver/src/backend/`.
