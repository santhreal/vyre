# Vyre Open-Work Closure  -  2026-04-29

Scope: every open-work marker found in the 2026-04-29 sweep, including
source comments, conformance skips, docs, and stale audit findings.

Status legend:
- `open`: implementation or truth cleanup still required.
- `in_progress`: owned in this closure pass.
- `fixed`: implementation landed or stale claim corrected.

## Source implementation items

| Status | Area | Required closure |
|---|---|---|
| completed | `vyre-libs::security` conform fixtures | Security op exemptions removed; scoped ops carry real `test_inputs` + `expected_output` pairs. |
| completed | C parser conform skips | C parser skip branches removed from the scoped runner files. |
| completed | `vyre-primitives::predicate::size_argument_of` | Current implementation proved as the contract: reverse `CALL_ARG` traversal over size-argument candidates, with stale text removed. |
| completed | `vyre-driver-wgpu` capability comments | f16/bf16 comments now describe the current ABI and support contract. |
| completed | `vyre-driver::pipeline` disk cache comments | Replaced stale cache language now that on-disk cache modules exist, and ensured docs/tests describe the current implementation. |
| completed | runtime backend/router persistence | Router comments now describe current persistence behavior rather than open work. |
| completed | `vyre-reference` hash references | Removed stale return-language; current hash references are described as real primitive references. |
| completed | `vyre-foundation` crate-level `dead_code` allowance | Removed the broad dialect-sweep allowance. |
| completed | self-substrate source claims | Stale higher-order comments rewritten to current shipped contracts. |
| completed | graph/dataflow/security primitive module claims | Stale graph/dataflow/security primitive claims resolved in source or marked stale with proof in the scoped audit files. |
| completed | WGPU megakernel dispatch | `WgpuMegakernelDispatcher` implements `MegakernelDispatch`; GPU tests cover Naga emission, CAS lowering, shutdown lifecycle, raw queue validation, and trait-backed dispatch. |
| completed | WGPU raw WGSL dispatch cache | Backend-owned raw WGSL pipeline cache added and cleared on device generation changes; GPU cache-reuse test passed. |
| completed | optimizer scheduler / Region DCE | Scheduler invalidation no longer creates false same-iteration missing-requirement failures; DCE now descends into `Node::Region` bodies while propagating live-ins. |

## Docs and audit truth items

| Status | Area | Required closure |
|---|---|---|
| completed | `docs/CPU_GPU_CONVERGENCE.md` | Replaced closed/exempted wording with generated-catalog truth and source-change findings. |
| completed | `docs/PER_OP_SURFACE.md` | Replaced stale parser/security inventory with generated-catalog truth. |
| completed | `docs/observability.md` | Removed exporter roadmap from the current observability contract. |
| completed | `docs/INNOVATION_SWEEP.md` / `docs/UX_SWEEP.md` | Converted open rows into source-change findings and removed release-version promises. |
| completed | `docs/GATE_CLOSURE.md` | Replaced nonexistent gate-script names with actual scripts or current verified commands. |
| completed | `docs/targets.md` / `docs/ARCHITECTURE.md` | Replaced stale target language with dispatch/emission/contract-target status. |
| completed | `docs/BENCHMARKS.md` / `benches/RESULTS.md` | Corrected stale benchmark notes and source-required ledger rows. |
| completed | stale graph audits | `reachable_program` and `toposort_program` exist; `PHASE7_GRAPH.md` and `VYRE_PRIMITIVES_GAPS.md` now record the fixed status. |
| completed | stale performance inventory | Reclassified rows that called open source work “fixed” as `open` or `partial`. |
| completed | stale conformance audits | Security exemption findings updated after fixtures and discipline-gate cleanup. |

## Gates

| Status | Gate |
|---|---|
| completed | source open-work marker script |
| completed | scoped marker grep over graph/dataflow/security source and named audit files |
| completed | targeted conformance tests for modified op fixtures |
| completed | targeted crate tests for modified implementation files |
| completed | scoped WGPU megakernel GPU tests on RTX 5090 |

## Source-required claims left open

These audit rows still require source changes; they were not converted
into doc-only closure:

- Scoped `audits/PHASE2_DECODE.md` vyre-libs rows are fixed for stored-block
  inflate naming, fused decode scan replay, hex lookup-table decode, base64
  length validation, and structured streaming-fuse errors. Cross-project rows
  under `surgec`, `keyhog`, `encodex`, and external `ziftsieve` remain outside
  this decode/parser ownership pass.
- Scoped `audits/PHASE4_PARSE.md` parser rows are fixed for grammar-generator
  LR smoke output, DFA max-munch default, SGGC payload checksum,
  object-like macro preprocessing, and recursive-descent transition lookup.
- Scoped `audits/PHASE5_ASTWALK.md`, `audits/VYRE_TESTING_GAPS.md`,
  `audits/PHASE6_DATAFLOW.md`, `audits/VYRE_CONFORMANCE_GAPS.md`,
  `audits/PHASE7_GRAPH.md`, and `audits/VYRE_PRIMITIVES_GAPS.md`
  graph/dataflow/security primitive rows are fixed or marked stale with
  source proof.
- Scoped `audits/VYRE_BACKEND_WGPU.md`, `audits/PHASE10_DIFF.md`, and
  `docs/megakernel-wiring.md` WGPU/megakernel rows are fixed or marked
  stale with source/test proof. Surgec watch/diff rows remain outside the
  WGPU/megakernel source ownership for this pass.
- Scoped `audits/VYRE_OPTIMIZER.md`, `audits/VYRE_IR_HOTSPOTS.md`, and
  `audits/VYRE_SCHED_TRANSFORM_DEEPER.md` scheduler / Region-transform
  rows are fixed or marked stale with source/test proof.
