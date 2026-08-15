# tests/SKILL.md, vyre-reference

One test file per contract. A file is named for the contract it proves, and
the directory has no catch-all target. `docs/testing/TESTING.toml` holds the
workspace-level default command, hardware expectation and failure behavior
for every package.

## Purpose

`vyre-reference` is the pure-Rust host interpreter for IR programs and the
parity oracle every backend is compared against. Every backend's conformance
proof is anchored against the bytes this crate produces for a witnessed
input. Determinism is the contract.

## Critical invariants

- Determinism. The same `Program`, the same inputs and the same reference
  version produce the same output bytes. No hash-map iteration order, no
  random ordering, no thread-scheduler dependence.
- Order independence where the model allows it. The interpreter can replay a
  program with the workgroup and invocation step order forward, reversed or
  rotated. A program whose result depends on which order was chosen is a
  program the model does not permit.
- Zero unsafe. The crate declares `#![forbid(unsafe_code)]`, so an `unsafe`
  block cannot be added without removing that line.
- No panic on a validated program. If `validate(p)` succeeds, execution
  returns a structured `ReferenceError` for a runtime condition such as an
  out-of-bounds access, and never aborts.
- Every diagnostic names the correction. A resolution failure says which
  registration is missing and what to add.

## Adversarial surface

- Arithmetic at the defined edges. `expr_adversarial_proptest` pins unsigned
  division by zero as the maximum value and signed modulo by zero as an
  error, and `saturating_binops_contract` pins the saturating forms.
- Subnormal inputs. `subnormal_contract` and the subnormal case in
  `adversarial_gaps` pin canonical results for square root, sine and cosine
  rather than whatever the host happens to produce.
- Degenerate programs. `adversarial_empty` and the no-buffer case in
  `adversarial_gaps` prove a program with no work and a program with no
  buffers both execute and return.
- Value width edges. The `value_*_property_contracts` and the generated
  `value_extend_bytes_width` and `value_write_bytes_width` matrices pin
  narrowing, widening and byte layout at every declared width.
- Atomics under contention. `atomic_law_property_contracts` and
  `atomic_oracle_contract` pin the algebraic laws and the serialization the
  oracle fixes.
- Lane identity. `subgroup_collectives_are_lane_identified` and
  `subgroup_edge_contract` prove a collective result names which lane
  produced it.

## Active coverage

- Workgroup execution runs on the hashmap interpreter with persistent locals,
  which makes a subgroup snapshot cheap.
- `run_storage_graph` is covered by `storage_graph_generated_adversarial`,
  which builds 32768 generated acyclic graphs from a fixed seed and compares
  the oracle against an independent recursive shadow evaluator.
- 11 `sweep_*_oracle_matrix` targets enumerate a dimension exhaustively
  instead of sampling it, and the crate declares each one in `Cargo.toml` so
  a new dimension is a visible manifest change.
- Dual-reference coverage is enforced by the registry tests for the bitwise
  primitives that publish an independent second reference.

## Cross-crate contracts

- `vyre-driver::shadow::ReferenceExecutor` wires this interpreter into the
  driver without creating a driver-to-reference dependency cycle.
- The crate consumes `vyre_foundation::Program` and the `vyre_foundation::ir`
  types.
- `Expr::Call` is resolved through `OperationRegistry::global()` and the CPU
  body through `reference_fn(op_id)`. No evaluator matches on an op-id
  string.
- `dual_op_ids`, `resolve_dual` and `DualReferenceFacet` publish which
  operations carry a second independent reference.
- The `Value` output is consumed by the conform runners and by the
  byte-identity proofs in every backend.

## Bench targets

The crate declares no bench target. Interpreter throughput is not a shipped
property: the interpreter is the oracle, and it is allowed to be slow.

## Fuzz targets

The crate declares no fuzz target. Coverage of the input space is enumerated
by the `sweep_*_oracle_matrix` targets and the property tests rather than
sampled, because the oracle has to be right on every point of a dimension,
not on a random subset.

## What NOT to test here

- Device dispatch. That belongs to the concrete driver crates.
- Wire format. That belongs to `vyre-foundation` tests.
- Op metadata. That belongs to `vyre-spec` tests.

## Running

```bash
./cargo_full test -p vyre-reference
./cargo_full test -p vyre-reference --all-features
```
