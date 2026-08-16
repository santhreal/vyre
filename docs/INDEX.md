<!-- Generated from docs/DOCS.toml by xtask docs-check. Do not edit. -->
# Documentation Authority and Lifecycle

Source: [`docs/DOCS.toml`](DOCS.toml).

Each active page declares its audience, owner, authority source, kind, and
generation mode. Generated pages also declare the generator. Superseded and
archived pages remain lifecycle evidence and are excluded from navigation.

## Documentation owners

| Owner | Authority |
| --- | --- |
| `architecture` | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| `benchmark` | [`optimization/BENCH_TARGETS.toml`](optimization/BENCH_TARGETS.toml) |
| `docs-governance` | [`DOCS.toml`](DOCS.toml) |
| `historical` | [`DOCS.toml`](DOCS.toml) |
| `operation-registry` | [`../vyre-foundation/src/operation.rs`](../vyre-foundation/src/operation.rs) |
| `optimization` | [`optimization/OWNERSHIP.toml`](optimization/OWNERSHIP.toml) |
| `public-facade` | [`../vyre/src/lib.rs`](../vyre/src/lib.rs) |
| `release-tooling` | [`../xtask/src/release/release_docs.rs`](../xtask/src/release/release_docs.rs) |
| `runtime` | [`../vyre-runtime/src/lib.rs`](../vyre-runtime/src/lib.rs) |
| `testing` | [`testing/TESTING.toml`](testing/TESTING.toml) |

## Lifecycle counts

- current: 2.
- generated: 38.
- superseded: 0.
- archived: 33.

## Pages

| Status | Audience | Owner | Kind | Page | Authority | Generation |
| --- | --- | --- | --- | --- | --- | --- |
| `current` | `extension` | `architecture` | `ownership` | `ARCHITECTURE.md` | self | manual |
| `generated` | `extension` | `architecture` | `ownership` | `CRATE_GRAPH.md` | [CRATE_OWNERSHIP.toml](CRATE_OWNERSHIP.toml) | generated: [../scripts/crate_ownership.py](../scripts/crate_ownership.py) |
| `generated` | `contributor` | `docs-governance` | `governance` | `INDEX.md` | [DOCS.toml](DOCS.toml) | generated: [../xtask/src/docs/docs_check.rs](../xtask/src/docs/docs_check.rs) |
| `generated` | `extension` | `architecture` | `ownership` | `OWNERSHIP.md` | [CRATE_OWNERSHIP.toml](CRATE_OWNERSHIP.toml) | generated: [../scripts/crate_ownership.py](../scripts/crate_ownership.py) |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/ARCHITECTURE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/CONVENTIONS.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/CPU_GPU_CONVERGENCE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/GATE_CLOSURE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/HOT_PATH_PROOFS.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/MATH_PRIMITIVES_PLACEMENT.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/MIGRATION.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/OPS.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/PER_OP_SURFACE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/PRIMITIVES.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/PUBLISH_GATE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/README.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/RECURSION_THESIS.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/RELEASE_1_0_GATE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/RELEASE_ENGINEERING.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/TESTING_PROGRAM.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/VISION.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/dialect-cookbook.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/AlgebraicLaw.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/EnforceGate.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/ExprVisitor.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/Lowerable.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/MutationClass.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/frozen-traits/VyreBackend.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/lego-block-rule.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/library-tiers.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/occ.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/parity/three_substrate.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/parsing-and-frontends.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/primitives-tier.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/santh-standard.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/stability.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/0.7-2026-08-15/wire-format-0.6-reservations.md` | self | manual |
| `current` | `extension` | `architecture` | `ownership` | `lego-block-rule.md` | self | manual |
| `generated` | `contributor` | `testing` | `testing` | `testing/structure-gate.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-aot.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-bench.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-conform-spec.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-conform.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-debug.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver-cuda.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver-metal.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver-reference.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver-spirv.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver-wgpu.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-driver.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-emit-metal.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-emit-naga.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-emit-ptx.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-emit-spirv.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-foundation.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-grammar-gen.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-libs.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-lints.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-lower.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-macros.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-megakernel.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-pass-engine.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-primitives.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-reference.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-registry-link.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-runtime.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-safetensors.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-spec.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-test-support.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/xtask-evidence.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/xtask-registry.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/xtask.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
