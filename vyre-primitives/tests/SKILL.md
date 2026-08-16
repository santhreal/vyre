# tests/SKILL.md, vyre-primitives

One test file per contract. A file is named for the contract it proves, and
the directory has no catch-all target. `docs/testing/TESTING.toml` holds the
workspace-level default command, hardware expectation and failure behavior
for every package.

## Purpose

`vyre-primitives` owns operations that cannot be composed. Marker types are
always on and carry no dependencies. Category C hardware intrinsics sit behind
the `hardware` feature and are the ops that need a dedicated backend emitter
arm and a dedicated reference arm. The remaining domain directories, one Cargo
feature each, export `fn(..) -> Program` builders that are compositions: they
belong in `vyre-libs` and the crate README records them as a live defect. Reuse
count is not an admission criterion, so a test written here pins a builder that
is on its way out, not a placement decision.

The path is the interface. A caller writes
`vyre_libs::text::char_class::char_class(..)`, so the composition chain
is visible at the call site, and a test is named for the builder it pins.

## Critical invariants

- A builder is infallible and traps in IR. An invalid user-controlled shape
  becomes an explicit `Node::Trap` in the returned `Program`, never a host
  panic, because generated dialect code composes these builders without a
  fallible seam.
- A marker is a unit struct with no data. The reference interpreter and the
  backend emitters dispatch on the type, so a marker never carries a string
  that could drift from the id its operation is registered under.
- Unsafe is denied crate-wide. The single audited exception is
  `wire::fill_le_words_into`, which carries an `#[allow(unsafe_code)]` and a
  safety proof. `deny` rather than `forbid` exists only so that one
  annotated site compiles.
- A builder's IR agrees with its own `cpu_ref` bit for bit. The parity
  targets under Coverage shape own that proof.

## Adversarial surface

One `adversarial_*` target per domain pins that domain's rejection and
boundary behavior: bitset membership, bitset reduction and bitset ops;
decode; fixpoint; graph, graph ops, CSR validation and reachability
fixpoint; the frontier queue clear path; hash; label; matching; math; NFA;
reduce by gather, scatter, histogram, radix sort and segment reduce; and
text by char class, UTF-8 validation, UTF-8 shape counts, line index and
byte histogram.

## Coverage shape

- The `adversarial_*` targets pin rejection and boundary behavior per domain.
- The `proptest_*` targets pin algebraic laws over a domain's `cpu_ref`, such
  as the sum and count laws for reductions. They are gated on `cpu-parity`
  alongside their domain feature.
- The `*_ir_parity` and `*_signed_parity` targets drive the built `Program`
  through `reference_eval` and assert bit-exact equality against the shipped
  `cpu_ref` for the same op. The comparison is exact, not tolerance-based,
  because both sides run the same operation order.
- The `sweep_*_oracle_matrix` targets enumerate a dimension exhaustively
  instead of sampling it. The `*_volume_*` variants run an independent
  reference against the shipped `cpu_ref` over a hostile corpus at volume,
  and each one carries a note that it must not be weakened to a shape-only
  assertion.
- `shared_owner_closure` reads source text and therefore takes no domain
  feature. Gating it on one would narrow the walk to the modules that feature
  enables, and a class is not closed over a subset of the tree.

## Cross-crate contracts

- The markers are consumed by `vyre-reference` for dispatch and by the
  conform runners to enumerate primitives.
- The Tier 2.5 builders are consumed by the Tier 3 dialect crates, which
  depend on `vyre-primitives` and enable only the domains they need.
- The crate builds against `vyre-foundation` and `vyre-spec` and no backend.
  Backend crates appear only in dev-dependencies, for the parity tests that
  compare a builder against a device.

## Bench targets

The crate declares one bench, `wire_throughput`, with `harness = false`. It
locks the little-endian `cast_slice` pack and unpack path against a naive
per-word baseline at 1 KiB, 1 MiB and 100 MiB. The path is bandwidth-bound,
so the ratio is the regression gate: reintroducing a per-word copy makes the
bench plateau back down to scalar speed.

## Fuzz targets

The crate declares no fuzz target. The input space of a builder is covered by
the `sweep_*_oracle_matrix` enumeration and the `proptest_*` laws, which is
the stronger claim for a total function over a small dimension.

## What NOT to test here

- Concrete backend dispatch. That belongs to the owning driver crate.
- Product-library behavior composed from these builders. That belongs to the
  Tier 3 crate.
- Registry admission policy. That belongs to `vyre-driver` and
  `vyre-registry-link`.

## Running

```bash
./cargo_full test -p vyre-primitives --all-features
./cargo_full test -p vyre-primitives --features "hash,inventory-registry" --test integration
./cargo_full bench -p vyre-primitives --bench wire_throughput
```
