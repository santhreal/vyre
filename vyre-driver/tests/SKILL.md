# tests/SKILL.md  -  vyre-driver

## Purpose

`vyre-driver` is the substrate-agnostic backend machinery: the `VyreBackend`
trait and its conservative defaults, the linked-backend registry, the pipeline
cache key, dispatch policy, and the driver error surface. Every concrete driver
crate depends on it. It names no concrete backend and holds no shader knowledge.

## Critical invariants

- `VyreBackend`, `CompiledPipeline`, and `PendingDispatch` are sealed through
  `crate::backend::private::Sealed`. A crate outside this workspace cannot
  implement them.
- Every capability query and lifecycle hook has a conservative default. A
  backend that overrides nothing reports no optional capability, succeeds on
  every hook, and inherits `default_supported_ops()` for `supported_ops`.
- The linked-backend registry freezes once. `registered_backends` hands back one
  `&'static` slice, so repeated queries return the same allocation rather than
  leaking a rebuilt one.
- `acquire_preferred_dispatch_backend` is GPU-only and fails closed. The CPU
  reference backend is reachable only through `acquire` by explicit id, and the
  failure message never advertises a fallback.
- `PipelineCacheKey` separates pipeline identity. It carries a format `version`
  so a stale key misses instead of falsely hitting, and a private field so
  additive fields are not a downstream break.
- `BackendError` and `ErrorCode` are matched downstream. Renaming or removing a
  variant breaks consumers.

## Adversarial surface

- A backend that reports every capability as available but errors on every
  `dispatch`. The default trait paths must still compose.
- A `PipelineCacheKey` whose `version` is not the current one. Lookup must miss.
- `DispatchConfig` with a near-zero or `u64::MAX` timeout.
- Policy inputs at the extremes of their integer domains. The policies compute
  in `u128` and must saturate to a verdict instead of overflowing.

## Cross-crate contracts

- `VyreBackend` is implemented by the concrete driver crates.
- `registered_target_operation_facets` joins the semantic catalog with each
  linked target registration. `vyre-reference` consumes it.
- `SemanticOperation`, `TargetOperationFacet`, and `OperationRegistration` are
  owned by `vyre-foundation::operation`. Each driver crate submits its
  registration through `inventory`.
- `BackendError` and `ErrorCode` surface to every consumer.

## What NOT to test here

- Concrete backend lowering or dispatch. That belongs to the owning driver crate.
- Wire format. That is `vyre-foundation/tests`.
- Operation semantics. Those are `vyre-spec/tests` and `vyre-reference/tests`.

## Running

```bash
./cargo_full test -p vyre-driver
./cargo_full test -p vyre-driver --test backend_trait_contract
./cargo_full test -p vyre-driver --test backend_registry
./cargo_full test -p vyre-driver --test async_dispatch_contract
```
