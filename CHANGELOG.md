# Changelog

All user-visible changes to `vyre` are documented in this file. Format follows Keep a Changelog.

## [Unreleased]

### Added

- Consolidated the workspace into one user-facing facade (`vyre`), one IR crate (`vyre-ir`), one generic composition library (`vyre-libs`), one whole-program megakernel compiler (`vyre-compiler`), one multi-target code emitter (`vyre-emit`), one persistent device runtime (`vyre-runtime`), and one CPU differential oracle (`vyre-reference`).
- Added persistent megakernel task-graph compilation with producer-consumer stage fusion, device-side work queues, and in-kernel fixpoint convergence across CUDA, Vulkan/wgpu, SPIR-V, and Metal backends.
- Added the end-to-end megakernel workload suite in `vyre-bench` covering graph traversal, parsing, pattern scanning, iterative dataflow, math and simulation, and neural inference.
- Added authenticated artifact admission (`ArtifactSession`), resident workspace pools, and `io_uring` host-to-device streaming in `vyre-runtime`.
- Added specialized portfolio compilation (`compile_specialized_portfolio`) over target capability and resource axes.
- Added zero-copy canonical wire framing and deterministic content digests for `Program` and `ProgramGraph` in `vyre-ir`.

### Changed

- Moved `SemanticExecutor` and `KernelDescriptor` into `vyre-ir` so dependency edges flow strictly one way from `vyre-ir` to `vyre`.
- Merged per-domain library crates (`vyre-libs-*` and `vyre-safetensors`) into feature-selected modules of `vyre-libs`.
- Merged pass execution, lowering, compiler analyses, megakernel scheduling, and ahead-of-time bundle generation into `vyre-compiler`.
- Merged target emitters (`ptx`, `naga`, `spirv`, `metal`) into `vyre-emit` and backend drivers (`cuda`, `wgpu`, `metal`, `spirv`) into `vyre-runtime`.
- Folded heap allocation counters into `vyre-bench`, conformance harnesses into `vyre-test-support`, and repository verification plus artifact inspection subcommands into `tools/xtask`.

### Fixed

- Eliminated host buffer copies across wire decoding, resident buffer binding, and persistent megakernel dispatch.
- Enforced explicit errors when a requested float lowering mode or target capability is unsupported by the selected backend.
- Clamped subgroup ballot lane reads to declared buffer extents on partial trailing tiles.
- Replaced recorded-commit comparison checks in CI with self-contained workspace verification.

### Removed

- Removed unreferenced security predicate, relation analyzer, and detection rule-engine pipelines.
- Removed test-only branch compaction, frontier partitioning, multi-corpus batching, bitset compression, and matroid scheduler planners along with unused telemetry counters.
- Removed two-commit release evidence artifacts, unreleased fragment files, redundant crate-level documentation snapshots, and standalone example/consumer scaffolding.

## [0.6.5] - 2026-07-13

### Added

- Added stream-ordered `cuMemPool` device allocation, per-kernel occupancy telemetry, and roofline bandwidth measurement in `vyre-runtime` and `vyre-bench`.
- Added device-sharded graph frontier expansion, paged corpus scanning, and resident literal/region scan pipelines in `vyre-libs`.
- Added distinct diagnostics for unsupported regular-expression constructs including backreferences, bounded alternations, and captures.
- Added grid-aware reference evaluation and instruction-count tracking in `vyre-reference`.

### Changed

- Executed multi-device sharded pattern scans concurrently across participating GPUs.
- Unified fixed-point arithmetic helpers onto a single implementation in `vyre-ir` and `vyre-libs`.

## [0.6.4] - 2026-06-23

### Added

- Added cooperative grid-barrier convergence loops (`persistent_fixpoint_grid`) for multi-workgroup iterative workloads.

### Fixed

- Validated cooperative grid launch dimensions against device occupancy limits prior to kernel submission.
- Emitted or rejected `Node::Return` explicitly in PTX code generation instead of dropping trailing return nodes.
- Sized writable workspace buffers from declared element counts across validation, runtime, and `vyre-reference`.

## [0.6.0] - 2026-04-19

### Added

- Introduced whole-program `CompileRequest`, `Artifact`, and content-addressed target module bundles.

### Changed

- Separated immutable compiler artifacts from stateful runtime device sessions.

## [0.5.0] - 2026-04-19

### Added

- Added SPIR-V and Metal shader emission targets alongside PTX and WGSL.
- Added differential parity harnesses comparing GPU backend outputs against `vyre-reference`.

### Fixed

- Preserved SSA result identifiers across loop unrolling and induction-variable rebinding in lowering.

## [0.4.0] - 2026-04-17

### Added

- Initial release of the typed GPU IR, Naga WGSL emitter, CUDA PTX emitter, and CPU reference interpreter.
