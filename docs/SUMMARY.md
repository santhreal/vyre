<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. -->
# Summary

- [Documentation authority and lifecycle](INDEX.md)

# Documentation authority

- [Documentation Governance](DOCUMENTATION_GOVERNANCE.md)
- [Documentation coverage](DOCUMENTATION_COVERAGE.md)

# Architecture and ownership

- [Vyre Crate Graph](CRATE_GRAPH.md)
- [Vyre Crate Ownership](OWNERSHIP.md)
- [Vyre architecture](ARCHITECTURE.md)

# Lifecycle and extension contracts

- [Runtime pipeline](RUNTIME_PIPELINE.md)

# Optimization

- [Optimization Taxonomy](optimization/TAXONOMY.md)
- [Optimization architecture](OPTIMIZATION_ARCHITECTURE.md)
- [Optimizer pass reference](optimization/PASSES.md)
- [Start Here for Optimization Work](optimization/START_HERE.md)
- [Vyre optimization control plane](optimization/README.md)

# User workflows

- [Command-line interfaces](CLI.md)
- [Downstream analyzer integration](consumer-integration.md)
- [Error Surface Contract](ERROR_SURFACE.md)
- [Frozen trait snapshots](frozen-traits/README.md)
- [Getting Support](support.md)
- [Linked Registration Contract](inventory-contract.md)
- [Lowering vs emission ownership](lower-vs-emit.md)
- [Megakernel wiring](megakernel-wiring.md)
- [Named external integration](consumer-showcase.md)
- [Operation catalog](ops-catalog.md)
- [RFC 0001  -  Region inline pass](rfcs/0001-region-inline-pass.md)
- [RFC 0002  -  Reverse-mode autodiff as an IR transform](rfcs/0002-autodiff-ir-transform.md)
- [RFC 0003  -  DataType::Quantized](rfcs/0003-datatype-quantized.md)
- [RFC 0004  -  Collective ops (AllReduce, AllGather, ReduceScatter)](rfcs/0004-collective-ops.md)
- [Region chain  -  the compositional back-pointer invariant](region-chain.md)
- [Scanning a corpus the right way](scanning-a-corpus-the-right-way.md)
- [Test layout convention](test-layout.md)
- [Trust Model](trust-model.md)
- [Vyre Code Style](code-style.md)
- [Vyre FAQ](faq.md)
- [Vyre Targets](targets.md)
- [Vyre Thesis](THESIS.md)
- [What the reference interpreter can and cannot witness](reference-interpreter-witness-limits.md)
- [vyre IR statement semantics](ir-semantics.md)
- [vyre Memory Model](memory-model.md)
- [vyre Semver Policy](semver-policy.md)
- [vyre Wire Format (VIR0)](wire-format.md)
- [vyre observability](observability.md)
- [vyre threat model](threat-model.md)
- [vyre-libs Feature Matrix](vyre-libs-features.md)
- [vyre-libs op naming](op-naming.md)

# API and operation reference

- [Generated operation documentation](generated/README.md)
- [Vyre operation catalog](catalog/README.md)
- [Vyre operation inventory](generated/OP_INVENTORY.md)
- [`bitset` operations](catalog/bitset.md)
- [`core` operations](catalog/core.md)
- [`decode` operations](catalog/decode.md)
- [`fixpoint` operations](catalog/fixpoint.md)
- [`geom` operations](catalog/geom.md)
- [`graph` operations](catalog/graph.md)
- [`hardware` operations](catalog/hardware.md)
- [`hash` operations](catalog/hash.md)
- [`io` operations](catalog/io.md)
- [`label` operations](catalog/label.md)
- [`logical` operations](catalog/logical.md)
- [`matching` operations](catalog/matching.md)
- [`math` operations](catalog/math.md)
- [`mem` operations](catalog/mem.md)
- [`nn` operations](catalog/nn.md)
- [`opt` operations](catalog/opt.md)
- [`optim` operations](catalog/optim.md)
- [`parsing` operations](catalog/parsing.md)
- [`predicate` operations](catalog/predicate.md)
- [`quant` operations](catalog/quant.md)
- [`reduce` operations](catalog/reduce.md)
- [`representation` operations](catalog/representation.md)
- [`scan` operations](catalog/scan.md)
- [`security` operations](catalog/security.md)
- [`substrate` operations](catalog/substrate.md)
- [`text` operations](catalog/text.md)
- [`vfs` operations](catalog/vfs.md)
- [`visual` operations](catalog/visual.md)

# Testing and conformance

- [Testing `vyre-aot`](testing/vyre-aot.md)
- [Testing `vyre-bench`](testing/vyre-bench.md)
- [Testing `vyre-conform-spec`](testing/vyre-conform-spec.md)
- [Testing `vyre-conform`](testing/vyre-conform.md)
- [Testing `vyre-debug`](testing/vyre-debug.md)
- [Testing `vyre-driver-cuda`](testing/vyre-driver-cuda.md)
- [Testing `vyre-driver-metal`](testing/vyre-driver-metal.md)
- [Testing `vyre-driver-reference`](testing/vyre-driver-reference.md)
- [Testing `vyre-driver-spirv`](testing/vyre-driver-spirv.md)
- [Testing `vyre-driver-wgpu`](testing/vyre-driver-wgpu.md)
- [Testing `vyre-driver`](testing/vyre-driver.md)
- [Testing `vyre-emit-metal`](testing/vyre-emit-metal.md)
- [Testing `vyre-emit-naga`](testing/vyre-emit-naga.md)
- [Testing `vyre-emit-ptx`](testing/vyre-emit-ptx.md)
- [Testing `vyre-emit-spirv`](testing/vyre-emit-spirv.md)
- [Testing `vyre-foundation`](testing/vyre-foundation.md)
- [Testing `vyre-frontend-c`](testing/vyre-frontend-c.md)
- [Testing `vyre-frontend-rust`](testing/vyre-frontend-rust.md)
- [Testing `vyre-grammar-gen`](testing/vyre-grammar-gen.md)
- [Testing `vyre-intrinsics`](testing/vyre-intrinsics.md)
- [Testing `vyre-libs`](testing/vyre-libs.md)
- [Testing `vyre-lints`](testing/vyre-lints.md)
- [Testing `vyre-lower`](testing/vyre-lower.md)
- [Testing `vyre-macros`](testing/vyre-macros.md)
- [Testing `vyre-megakernel`](testing/vyre-megakernel.md)
- [Testing `vyre-primitives`](testing/vyre-primitives.md)
- [Testing `vyre-reference`](testing/vyre-reference.md)
- [Testing `vyre-runtime`](testing/vyre-runtime.md)
- [Testing `vyre-scan`](testing/vyre-scan.md)
- [Testing `vyre-self-substrate`](testing/vyre-self-substrate.md)
- [Testing `vyre-spec`](testing/vyre-spec.md)
- [Testing `vyre-test-support`](testing/vyre-test-support.md)
- [Testing `vyre`](testing/vyre.md)
- [Testing `xtask`](testing/xtask.md)

# Performance and release

- [Build Performance and Optimization](PERF.md)
- [Vyre 0.7.2 release notes](release/v0.7.2.md)
- [Vyre release checklist](RELEASE_CHECKLIST.md)
- [Vyre release process](RELEASE.md)

# Extension contracts

- [vyre Error Codes](error-codes.md)
