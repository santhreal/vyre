# Optimization Taxonomy

Applies to Vyre 0.7.2.

This taxonomy names the accepted optimization classes and their owners. Add a
class here before you add a new optimization subsystem.

## Layer 1: semantic IR optimization

Layer 1 transforms `Program`, `Node`, or `Expr` while preserving semantics for
every backend.

| Class | Home | Required proof |
|---|---|---|
| Arithmetic and algebraic simplification | `vyre-foundation/src/optimizer/passes/algebraic/` | Before/after IR plus reference equivalence |
| Loop transformation | `vyre-foundation/src/optimizer/passes/loops/` | Scope, boundary, and adversarial trip-count tests |
| Memory transformation | `vyre-foundation/src/optimizer/passes/memory/` | Alias, effect, and exact-output tests |
| Fusion and CSE | `vyre-foundation/src/optimizer/passes/fusion_cse/` | Effect safety and cost-gate evidence |
| Cleanup | `vyre-foundation/src/optimizer/passes/cleanup/` | Structural and observable-equivalence tests |
| Shared facts | `vyre-foundation/src/optimizer/fact_substrate/` | Invalidation and single-source fact tests |

Layer 1 does not inspect a concrete backend name or emit target code.

## Shared driver policy

Neutral launch, binding, validation, capability, and device-identity policy
lives in `vyre-driver/src/`. Shared driver code may define neutral traits and
records. It does not import a concrete driver SDK.

## Layer 2: concrete lowering strategy

Target-dependent choices stay in the owning driver crate:

| Owner | Examples | Required proof |
|---|---|---|
| `vyre-driver-cuda` | PTX instruction selection, module cache, streams, resident buffers | PTX checks and live CUDA conformance |
| `vyre-driver-wgpu` | Naga/WGSL emission, pipeline layout, readback, portable dispatch | Emission checks and WGPU dispatch tests |
| `vyre-driver-metal` | Metal emission and Apple device dispatch | Metal emission and device tests |
| `vyre-driver-spirv` | SPIR-V emission and validation | SPIR-V validation and parity tests |

A concrete driver does not duplicate a Layer 1 semantic rewrite.

## Runtime megakernel policy

Persistent protocol, scheduling, recovery, and I/O ownership lives under
`vyre-runtime/src/megakernel/`. The runtime consumes backend capabilities and
handles. Concrete drivers do not reimplement runtime scheduling policy.
