<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. Do not edit. -->
# Documentation Authority and Lifecycle

Source: [`docs/DOCS.toml`](DOCS.toml).

Each active page declares its audience, owner, authority source, kind, and
generation mode. Generated pages also declare the generator. Superseded and
archived pages remain lifecycle evidence and are excluded from navigation.

## Documentation owners

| Owner | Authority |
| --- | --- |
| `architecture` | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| `benchmark` | [`PERF.md`](PERF.md) |
| `docs-governance` | [`DOCUMENTATION_GOVERNANCE.md`](DOCUMENTATION_GOVERNANCE.md) |
| `foundation` | [`../vyre-foundation/src/lib.rs`](../vyre-foundation/src/lib.rs) |
| `frontend` | [`../vyre-frontend-c/src/lib.rs`](../vyre-frontend-c/src/lib.rs) |
| `historical` | [`DOCS.toml`](DOCS.toml) |
| `operation-registry` | [`../vyre-foundation/src/operation.rs`](../vyre-foundation/src/operation.rs) |
| `optimization` | [`optimization/README.md`](optimization/README.md) |
| `public-facade` | [`../vyre/src/lib.rs`](../vyre/src/lib.rs) |
| `release-tooling` | [`../scripts/release_docs.py`](../scripts/release_docs.py) |
| `runtime` | [`../vyre-runtime/src/lib.rs`](../vyre-runtime/src/lib.rs) |
| `testing` | [`testing/TESTING.toml`](testing/TESTING.toml) |

## Cargo-derived workspace facts

- Workspace packages: 34.
- Shipped library, binary, and example targets: 72.
- Source: `cargo metadata --no-deps --format-version 1`.

## Lifecycle counts

- current: 42.
- generated: 71.
- superseded: 34.
- archived: 18.

## Pages

| Status | Audience | Owner | Kind | Page | Authority | Generation |
| --- | --- | --- | --- | --- | --- | --- |
| `current` | `extension` | `architecture` | `ownership` | `ARCHITECTURE.md` | self | manual |
| `generated` | `user` | `public-facade` | `guide` | `CLI.md` | [CLI.toml](CLI.toml) | generated: [../scripts/cli_docs.py](../scripts/cli_docs.py) |
| `superseded` | `contributor` | `historical` | `history` | `CONVENTIONS.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `CPU_GPU_CONVERGENCE.md` | self | manual |
| `generated` | `extension` | `architecture` | `ownership` | `CRATE_GRAPH.md` | [CRATE_OWNERSHIP.toml](CRATE_OWNERSHIP.toml) | generated: [../scripts/crate_ownership.py](../scripts/crate_ownership.py) |
| `current` | `contributor` | `docs-governance` | `governance` | `DOCUMENTATION_COVERAGE.md` | self | manual |
| `current` | `contributor` | `docs-governance` | `governance` | `DOCUMENTATION_GOVERNANCE.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `ERROR_SURFACE.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `GATE_CLOSURE.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `HOT_PATH_PROOFS.md` | self | manual |
| `generated` | `contributor` | `docs-governance` | `governance` | `INDEX.md` | [DOCS.toml](DOCS.toml) | generated: [../scripts/docs_manifest.py](../scripts/docs_manifest.py) |
| `archived` | `contributor` | `historical` | `history` | `LAW7_ORGANIZATION.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `MATH_FRONTIER.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `MATH_PRIMITIVES_PLACEMENT.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `MIGRATION.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `OPS.md` | self | manual |
| `current` | `extension` | `optimization` | `optimization` | `OPTIMIZATION_ARCHITECTURE.md` | self | manual |
| `generated` | `extension` | `architecture` | `ownership` | `OWNERSHIP.md` | [CRATE_OWNERSHIP.toml](CRATE_OWNERSHIP.toml) | generated: [../scripts/crate_ownership.py](../scripts/crate_ownership.py) |
| `current` | `release` | `benchmark` | `release` | `PERF.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `PER_OP_SURFACE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `PREDICATE_EXPR_DUALITY.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `PRIMITIVES.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `PUBLISH_GATE.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `RECURSION_THESIS.md` | self | manual |
| `current` | `release` | `release-tooling` | `release` | `RELEASE.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `RELEASE_1_0_GATE.md` | self | manual |
| `generated` | `release` | `release-tooling` | `release` | `RELEASE_CHECKLIST.md` | [../release/release-train.toml](../release/release-train.toml) | generated: [../scripts/release_docs.py](../scripts/release_docs.py) |
| `superseded` | `contributor` | `historical` | `history` | `RELEASE_ENGINEERING.md` | self | manual |
| `current` | `extension` | `runtime` | `lifecycle` | `RUNTIME_PIPELINE.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `SUBSTRATE_RFCS.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `TESTING_PROGRAM.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `THESIS.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `VISION.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/HEURISTIC_TO_MATH_TRACKER.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/INNOVATION_SWEEP.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/JULES_PRIMITIVE_MANIFEST.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/MICRO_FLAW_LOG.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/MIGRATION_0.6_TO_0.7.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/NAGA_CRITICAL_HOLES.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/README.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/UX_SWEEP.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `archive/vision-2026-04-27-essay.md` | self | manual |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/README.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/bitset.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/core.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/decode.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/fixpoint.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/geom.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/graph.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/hardware.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/hash.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/io.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/label.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/logical.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/matching.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/math.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/mem.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/nn.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/opt.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/optim.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/parsing.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/predicate.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/quant.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/reduce.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/representation.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/scan.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/security.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/substrate.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/text.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/vfs.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `generated` | `extension` | `operation-registry` | `reference` | `catalog/visual.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/catalog.rs](../xtask/src/catalog.rs) |
| `current` | `user` | `public-facade` | `guide` | `code-style.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `consumer-integration.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `consumer-showcase.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `dialect-cookbook.md` | self | manual |
| `current` | `extension` | `foundation` | `reference` | `error-codes.md` | [../vyre-foundation/src/validate/validation_error.rs](../vyre-foundation/src/validate/validation_error.rs) | manual |
| `current` | `user` | `public-facade` | `guide` | `faq.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/AlgebraicLaw.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/EnforceGate.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/ExprVisitor.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/Lowerable.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/MutationClass.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `frozen-traits/README.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `frozen-traits/VyreBackend.md` | self | manual |
| `generated` | `extension` | `operation-registry` | `reference` | `generated/OP_INVENTORY.md` | [generated/OP_SCHEMA.json](generated/OP_SCHEMA.json) | generated: [../xtask/src/list_ops.rs](../xtask/src/list_ops.rs) |
| `current` | `extension` | `operation-registry` | `reference` | `generated/README.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `inventory-contract.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `ir-semantics.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `legacy/README.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `lego-block-rule.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `library-tiers.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `lower-vs-emit.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `megakernel-wiring.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `memory-model.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `migration-vyre-ops-to-intrinsics.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `observability.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `occ.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `op-naming.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `ops-catalog.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `optimization/LEGACY_DOCS.md` | self | manual |
| `generated` | `extension` | `optimization` | `reference` | `optimization/PASSES.md` | [../vyre-foundation/src/optimizer.rs](../vyre-foundation/src/optimizer.rs) | generated: [../xtask/src/optimization_docs.rs](../xtask/src/optimization_docs.rs) |
| `current` | `contributor` | `optimization` | `optimization` | `optimization/README.md` | self | manual |
| `current` | `contributor` | `optimization` | `optimization` | `optimization/START_HERE.md` | self | manual |
| `current` | `contributor` | `optimization` | `optimization` | `optimization/TAXONOMY.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `parity/three_substrate.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `parsing-and-frontends.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `primitives-tier.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `reference-interpreter-witness-limits.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `region-chain.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `release/v0.4.1.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `release/v0.4.2.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `release/v0.7.0.md` | self | manual |
| `archived` | `contributor` | `historical` | `history` | `release/v0.7.1.md` | self | manual |
| `generated` | `release` | `release-tooling` | `release` | `release/v0.7.2.md` | [../release/release-train.toml](../release/release-train.toml) | generated: [../scripts/release_docs.py](../scripts/release_docs.py) |
| `current` | `user` | `public-facade` | `guide` | `rfcs/0001-region-inline-pass.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `rfcs/0002-autodiff-ir-transform.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `rfcs/0003-datatype-quantized.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `rfcs/0004-collective-ops.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `rfcs/0005-persistent-megakernel.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `santh-standard.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `scanning-a-corpus-the-right-way.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `semver-policy.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `stability.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `support.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `targets.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `test-layout.md` | self | manual |
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
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-frontend-c.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-frontend-rust.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-grammar-gen.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-intrinsics.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-libs.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-lints.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-lower.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-macros.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-megakernel.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-primitives.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-reference.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-runtime.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-scan.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-self-substrate.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-spec.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre-test-support.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/vyre.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `generated` | `contributor` | `testing` | `testing` | `testing/xtask.md` | [testing/TESTING.toml](testing/TESTING.toml) | generated: [../scripts/testing_guides.py](../scripts/testing_guides.py) |
| `current` | `user` | `public-facade` | `guide` | `threat-model.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `trust-model.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `vyre-libs-features.md` | self | manual |
| `superseded` | `contributor` | `historical` | `history` | `wire-format-0.6-reservations.md` | self | manual |
| `current` | `user` | `public-facade` | `guide` | `wire-format.md` | self | manual |
