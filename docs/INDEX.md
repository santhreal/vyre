# Documentation Index

Last verified: 2026-07-30

This file is the routing table for the public documentation set. It is
maintained by hand, not generated. `scripts/check_docs_index.sh` reads the
filesystem, not git's index: it fails when a document that exists under `docs/`
and is not gitignored has no row here, when a row points at a file that does not
exist on disk, and when a row points at a gitignored file that no reader outside
the authoring working copy can open. Add a row whenever you add a public
document, and drop the row when the document is deleted. A document that exists
but has not been committed yet is still a real document, so index it. Working
notes matching the ignored `*PLAN*`, `*STATUS*`, `*ROADMAP*`, `*AUDIT*`,
`*BACKLOG*` and `AGENT_*` name patterns are not public documentation and must
not be listed.

Twenty six rows were removed on 2026-07-30: twenty five pointed at gitignored
working notes, one at a deleted file. If one of those documents is later made
public, restore its row. Do NOT broaden the ignore pattern to re-admit it, and
do NOT loosen this contract to a warning. The failure mode being guarded against
is a routing table that lists documents only the author can open, which reads as
complete and is not, and the way that happens is someone relaxing the rule to
clear a red gate instead of deciding whether the document belongs in public.

Consumer names inside archived documents are historical context only. Take
current implementation guidance from rows marked `current` or `generated`.

Status values:

- `current`: active architecture, contract, release, or contributor guidance.
- `generated`: produced from source/evidence; regenerate instead of hand-editing content.
- `superseded`: retained for traceability; a newer document owns the active contract.
- `archived`: historical context only; do not use as implementation guidance.

| Status | Last verified | Modified | Document |
|---|---:|---:|---|
| `current` | 2026-05-26 | 2026-05-26 | [docs/ARCHITECTURE.md](ARCHITECTURE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/bitset.md](catalog/bitset.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/fixpoint.md](catalog/fixpoint.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/graph.md](catalog/graph.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/hash.md](catalog/hash.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/intrinsics.md](catalog/intrinsics.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/label.md](catalog/label.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/logical.md](catalog/logical.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/matching.md](catalog/matching.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/math.md](catalog/math.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/nn.md](catalog/nn.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/parsing.md](catalog/parsing.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/predicate.md](catalog/predicate.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/README.md](catalog/README.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/reduce.md](catalog/reduce.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/representation.md](catalog/representation.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/security.md](catalog/security.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/text.md](catalog/text.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/catalog/vfs.md](catalog/vfs.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/decode.md](catalog/decode.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/geom.md](catalog/geom.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/opt.md](catalog/opt.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/optim.md](catalog/optim.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/quant.md](catalog/quant.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/scan.md](catalog/scan.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/substrate.md](catalog/substrate.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/catalog/visual.md](catalog/visual.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/code-style.md](code-style.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/consumer-integration.md](consumer-integration.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/CONVENTIONS.md](CONVENTIONS.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/CPU_GPU_CONVERGENCE.md](CPU_GPU_CONVERGENCE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/CRATE_GRAPH.md](CRATE_GRAPH.md) |
| `current` | 2026-05-28 | 2026-05-28 | [docs/consumer-showcase.md](consumer-showcase.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/dialect-cookbook.md](dialect-cookbook.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/DOCUMENTATION_GOVERNANCE.md](DOCUMENTATION_GOVERNANCE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/error-codes.md](error-codes.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/ERROR_SURFACE.md](ERROR_SURFACE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/faq.md](faq.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/AlgebraicLaw.md](frozen-traits/AlgebraicLaw.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/EnforceGate.md](frozen-traits/EnforceGate.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/ExprVisitor.md](frozen-traits/ExprVisitor.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/Lowerable.md](frozen-traits/Lowerable.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/MutationClass.md](frozen-traits/MutationClass.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/README.md](frozen-traits/README.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/frozen-traits/VyreBackend.md](frozen-traits/VyreBackend.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/GATE_CLOSURE.md](GATE_CLOSURE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/HOT_PATH_PROOFS.md](HOT_PATH_PROOFS.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/inventory-contract.md](inventory-contract.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/ir-semantics.md](ir-semantics.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/LAW7_ORGANIZATION.md](LAW7_ORGANIZATION.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/lego-block-rule.md](lego-block-rule.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/library-tiers.md](library-tiers.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/lower-vs-emit.md](lower-vs-emit.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/MATH_FRONTIER.md](MATH_FRONTIER.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/MATH_PRIMITIVES_PLACEMENT.md](MATH_PRIMITIVES_PLACEMENT.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/megakernel-wiring.md](megakernel-wiring.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/memory-model.md](memory-model.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/migration-vyre-ops-to-intrinsics.md](migration-vyre-ops-to-intrinsics.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/MIGRATION.md](MIGRATION.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/observability.md](observability.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/occ.md](occ.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/op-naming.md](op-naming.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/ops-catalog.md](ops-catalog.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/OPS.md](OPS.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/optimization/README.md](optimization/README.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/optimization/START_HERE.md](optimization/START_HERE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/optimization/TAXONOMY.md](optimization/TAXONOMY.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/OPTIMIZATION_ARCHITECTURE.md](OPTIMIZATION_ARCHITECTURE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/OWNERSHIP.md](OWNERSHIP.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PARADIGM_SHIFT_TRAJECTORY.md](PARADIGM_SHIFT_TRAJECTORY.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/parity/three_substrate.md](parity/three_substrate.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/parsing-and-frontends.md](parsing-and-frontends.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PER_OP_SURFACE.md](PER_OP_SURFACE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PERF.md](PERF.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PREDICATE_EXPR_DUALITY.md](PREDICATE_EXPR_DUALITY.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/primitives-tier.md](primitives-tier.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PRIMITIVES.md](PRIMITIVES.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/PUBLISH_GATE.md](PUBLISH_GATE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/README.md](README.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RECURSION_THESIS.md](RECURSION_THESIS.md) |
| `current` | 2026-07-29 | 2026-07-29 | [docs/reference-interpreter-witness-limits.md](reference-interpreter-witness-limits.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/region-chain.md](region-chain.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RELEASE.md](RELEASE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/release/v0.4.2.md](release/v0.4.2.md) |
| `current` | 2026-07-25 | 2026-07-25 | [docs/release/v0.7.0.md](release/v0.7.0.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RELEASE_1_0_GATE.md](RELEASE_1_0_GATE.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RELEASE_ENGINEERING.md](RELEASE_ENGINEERING.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/rfcs/0001-region-inline-pass.md](rfcs/0001-region-inline-pass.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/rfcs/0002-autodiff-ir-transform.md](rfcs/0002-autodiff-ir-transform.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/rfcs/0003-datatype-quantized.md](rfcs/0003-datatype-quantized.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/rfcs/0004-collective-ops.md](rfcs/0004-collective-ops.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/rfcs/0005-persistent-megakernel.md](rfcs/0005-persistent-megakernel.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/RUST_COMPILER_BUILDOUT.md](RUST_COMPILER_BUILDOUT.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/santh-standard.md](santh-standard.md) |
| `current` | 2026-07-03 | 2026-07-03 | [docs/scanning-a-corpus-the-right-way.md](scanning-a-corpus-the-right-way.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/semver-policy.md](semver-policy.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/stability.md](stability.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/SUBSTRATE_RFCS.md](SUBSTRATE_RFCS.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/support.md](support.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/targets.md](targets.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/test-layout.md](test-layout.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/TESTING_PROGRAM.md](TESTING_PROGRAM.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-aot.md](testing/vyre-aot.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-bench.md](testing/vyre-bench.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-conform-enforce.md](testing/vyre-conform-enforce.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-conform-generate.md](testing/vyre-conform-generate.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-conform-runner.md](testing/vyre-conform-runner.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-conform-spec.md](testing/vyre-conform-spec.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-core.md](testing/vyre-core.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-debug.md](testing/vyre-debug.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-driver-cuda.md](testing/vyre-driver-cuda.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-driver-reference.md](testing/vyre-driver-reference.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-driver-spirv.md](testing/vyre-driver-spirv.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-driver-wgpu.md](testing/vyre-driver-wgpu.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-driver.md](testing/vyre-driver.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-emit-naga.md](testing/vyre-emit-naga.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-emit-ptx.md](testing/vyre-emit-ptx.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-emit-spirv.md](testing/vyre-emit-spirv.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-foundation.md](testing/vyre-foundation.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-frontend-c.md](testing/vyre-frontend-c.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-frontend-rust.md](testing/vyre-frontend-rust.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-harness.md](testing/vyre-harness.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-intrinsics.md](testing/vyre-intrinsics.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-libs.md](testing/vyre-libs.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-lints.md](testing/vyre-lints.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-lower.md](testing/vyre-lower.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-macros.md](testing/vyre-macros.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-primitives.md](testing/vyre-primitives.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-reference.md](testing/vyre-reference.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-runtime.md](testing/vyre-runtime.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-self-substrate.md](testing/vyre-self-substrate.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-spec.md](testing/vyre-spec.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/vyre-test-harness.md](testing/vyre-test-harness.md) |
| `current` | 2026-06-03 | 2026-06-03 | [docs/testing/xtask.md](testing/xtask.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/THESIS.md](THESIS.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/threat-model.md](threat-model.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/trust-model.md](trust-model.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/VISION.md](VISION.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/vyre-libs-features.md](vyre-libs-features.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/VYRE_BENCH_META_HARNESS_PRD.md](VYRE_BENCH_META_HARNESS_PRD.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/wire-format-0.6-reservations.md](wire-format-0.6-reservations.md) |
| `current` | 2026-05-26 | 2026-05-26 | [docs/wire-format.md](wire-format.md) |
| `generated` | 2026-05-26 | 2026-05-26 | [docs/generated/OP_INVENTORY.md](generated/OP_INVENTORY.md) |
| `generated` | 2026-05-26 | 2026-05-26 | [docs/generated/README.md](generated/README.md) |
| `superseded` | 2026-05-26 | 2026-05-26 | [docs/optimization/LEGACY_DOCS.md](optimization/LEGACY_DOCS.md) |
| `superseded` | 2026-05-26 | 2026-05-26 | [docs/release/v0.4.1.md](release/v0.4.1.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/HEURISTIC_TO_MATH_TRACKER.md](archive/HEURISTIC_TO_MATH_TRACKER.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/INNOVATION_SWEEP.md](archive/INNOVATION_SWEEP.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/JULES_PRIMITIVE_MANIFEST.md](archive/JULES_PRIMITIVE_MANIFEST.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/MICRO_FLAW_LOG.md](archive/MICRO_FLAW_LOG.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/MIGRATION_0.6_TO_0.7.md](archive/MIGRATION_0.6_TO_0.7.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/NAGA_CRITICAL_HOLES.md](archive/NAGA_CRITICAL_HOLES.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/ROADMAP_APPEND_ONLY_2026-05-22.md](archive/ROADMAP_APPEND_ONLY_2026-05-22.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/UX_SWEEP.md](archive/UX_SWEEP.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/archive/vision-2026-04-27-essay.md](archive/vision-2026-04-27-essay.md) |
| `archived` | 2026-05-26 | 2026-05-26 | [docs/legacy/PERF_ROADMAP_2026-05-01.md](legacy/PERF_ROADMAP_2026-05-01.md) |
| `current` | 2026-07-29 | 2026-07-29 | [docs/archive/README.md](archive/README.md) |
| `current` | 2026-07-29 | 2026-07-29 | [docs/legacy/README.md](legacy/README.md) |
| `current` | 2026-07-12 | 2026-07-12 | [docs/GPU_OOM_SEGMENTATION.md](GPU_OOM_SEGMENTATION.md) |
| `current` | 2026-07-12 | 2026-07-12 | [docs/SUBGROUP_REDUCE_GENERALIZATION.md](SUBGROUP_REDUCE_GENERALIZATION.md) |
| `current` | 2026-07-12 | 2026-07-12 | [docs/optimization/XTASK_COMMAND_MATRIX.md](optimization/XTASK_COMMAND_MATRIX.md) |

## Non-public internals

`.internals/` is maintainer working material and is excluded from the repository, so it does not ship and is not linked from any published document. Nothing in the table above depends on it. Its layout is whatever the maintainer is holding locally rather than a documented structure, so this file names no subdirectories: an earlier revision claimed active plans under `.internals/plans/` and archives under `.internals/archive/` and `.internals/archived-plans/`, and none of those three paths exist on disk. Naming a private path is a claim that rots unobserved, because no gate can check a path that is excluded by design.
