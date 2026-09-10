<!-- Generated from docs/DOCS.toml by xtask docs-check. -->
# Summary

- [Documentation authority and lifecycle](INDEX.md)

# Architecture and ownership

- [Crate boundaries](architecture/crates.md)
- [LEGO-block rule](lego-block-rule.md)
- [Launch geometry is a lowering decision](design/geometry-lowering.md)
- [Parsing](architecture/parsing.md)
- [The artifact is the output type](architecture/artifact.md)
- [Tile values in the IR](design/tile-values.md)
- [Vyre Crate Graph](CRATE_GRAPH.md)
- [Vyre Crate Ownership](OWNERSHIP.md)
- [Vyre architecture](ARCHITECTURE.md)
- [Whole-program compile search](architecture/compile-search.md)

# Lifecycle and extension contracts

- [Add a backend](extending/backend.md)
- [Add an operation from outside the workspace](extending/operation.md)

# User workflows

- [Compile a graph to an artifact](guide/first-program.md)
- [Install](guide/install.md)
- [Run an artifact on a device](guide/backends.md)

# API and operation reference

- [Diagnostics](reference/diagnostics.md)
- [Numeric contracts](reference/numeric-contracts.md)
- [Program wire format](reference/wire-format.md)
- [The operation registry](reference/operations.md)
- [Value contracts](reference/values.md)

# Testing and conformance

- [Conformance](conformance/program.md)
- [Testing `structure-gate`](testing/structure-gate.md)
- [Testing `vyre-alloc-probe`](testing/vyre-alloc-probe.md)
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
- [Testing `vyre-libs-analysis`](testing/vyre-libs-analysis.md)
- [Testing `vyre-libs-bitset`](testing/vyre-libs-bitset.md)
- [Testing `vyre-libs-builder`](testing/vyre-libs-builder.md)
- [Testing `vyre-libs-decode`](testing/vyre-libs-decode.md)
- [Testing `vyre-libs-device`](testing/vyre-libs-device.md)
- [Testing `vyre-libs-encoding`](testing/vyre-libs-encoding.md)
- [Testing `vyre-libs-fixpoint`](testing/vyre-libs-fixpoint.md)
- [Testing `vyre-libs-graph`](testing/vyre-libs-graph.md)
- [Testing `vyre-libs-hash`](testing/vyre-libs-hash.md)
- [Testing `vyre-libs-math`](testing/vyre-libs-math.md)
- [Testing `vyre-libs-nn`](testing/vyre-libs-nn.md)
- [Testing `vyre-libs-parsing`](testing/vyre-libs-parsing.md)
- [Testing `vyre-libs-pattern`](testing/vyre-libs-pattern.md)
- [Testing `vyre-libs-reasoning`](testing/vyre-libs-reasoning.md)
- [Testing `vyre-libs-reduce`](testing/vyre-libs-reduce.md)
- [Testing `vyre-libs-rule`](testing/vyre-libs-rule.md)
- [Testing `vyre-libs-scheduling`](testing/vyre-libs-scheduling.md)
- [Testing `vyre-libs-security`](testing/vyre-libs-security.md)
- [Testing `vyre-libs-solvers`](testing/vyre-libs-solvers.md)
- [Testing `vyre-libs-text`](testing/vyre-libs-text.md)
- [Testing `vyre-libs-vfs`](testing/vyre-libs-vfs.md)
- [Testing `vyre-libs-visual`](testing/vyre-libs-visual.md)
- [Testing `vyre-libs`](testing/vyre-libs.md)
- [Testing `vyre-lints`](testing/vyre-lints.md)
- [Testing `vyre-lower`](testing/vyre-lower.md)
- [Testing `vyre-macros`](testing/vyre-macros.md)
- [Testing `vyre-megakernel`](testing/vyre-megakernel.md)
- [Testing `vyre-pass-engine`](testing/vyre-pass-engine.md)
- [Testing `vyre-primitives`](testing/vyre-primitives.md)
- [Testing `vyre-reference`](testing/vyre-reference.md)
- [Testing `vyre-registry-link`](testing/vyre-registry-link.md)
- [Testing `vyre-runtime`](testing/vyre-runtime.md)
- [Testing `vyre-safetensors`](testing/vyre-safetensors.md)
- [Testing `vyre-spec`](testing/vyre-spec.md)
- [Testing `vyre-test-support`](testing/vyre-test-support.md)
- [Testing `vyre`](testing/vyre.md)
- [Testing `xtask-evidence`](testing/xtask-evidence.md)
- [Testing `xtask-registry`](testing/xtask-registry.md)
- [Testing `xtask`](testing/xtask.md)

# Performance and release

- [Release](release/process.md)
