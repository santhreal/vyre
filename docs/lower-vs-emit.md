# Lowering vs emission ownership

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

Established by audit cleanup A11 (2026-04-30) and updated for the current crate
split (`vyre-lower`, `vyre-emit-*`).

## Rule

| Name | Lives in | Purpose |
|---|---|---|
| Foundation substrate lower | `vyre-foundation/src/lower/` | Semantic, backend-agnostic IR transforms that still produce Vyre IR or shared analysis facts |
| Neutral lowering crate | `vyre-lower` | Backend-neutral descriptors, pre-emission shaping, and shared lowering helpers |
| Emitter crates | `vyre-emit-naga`, `vyre-emit-ptx`, `vyre-emit-spirv`, `vyre-emit-metal` | Target text/binary construction from lowered programs |
| Driver emit glue | `vyre-driver-<backend>/src/emit/` (where present) | Backend-local integration of emitter output with device objects |
| CUDA codegen alias | `vyre-driver-cuda/src/codegen/` | CUDA-facing name for emission integration |
| Shared lowering traits | `vyre-driver` backend contracts | Cross-backend lowering/capability contracts without concrete dialects |

## Pipeline placement

```text
Program (foundation)
  -> IR-pure optimizer (foundation)
  -> vyre-lower descriptors / shaping
  -> vyre-emit-* target bytes
  -> concrete driver acquires device objects and dispatches
```

Artifact freeze (`vyre-megakernel`) can capture neutral structure before or
beside target emission. It does not replace emitters.

## Historical note

The WGPU emitter previously lived under a directory named `lowering`. Target
emission now lives in `vyre-emit-naga` plus `vyre-driver-wgpu` integration.
Do not recreate a backend-local `lowering/` tree for final shader text.

## Where to put new code

- New cross-backend IR transformation that stays in Vyre IR →
  `vyre-foundation` optimizer/lower substrate.
- New shared pre-emission analysis used by multiple emitters → `vyre-lower`.
- New WGSL/Naga/PTX/SPIR-V/Metal encoding → the matching `vyre-emit-*` crate.
- New device object binding after bytes exist → concrete `vyre-driver-*`.
- New trait every backend implements → `vyre-driver` neutral contracts.

## Related docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`megakernel-wiring.md`](megakernel-wiring.md)
- [`docs/CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml)
