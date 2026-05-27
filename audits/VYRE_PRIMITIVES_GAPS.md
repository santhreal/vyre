# VYRE PRIMITIVES GAPS  -  Scoped Closure Status

Date: 2026-04-29

This file records the graph/dataflow/security primitive claims from the
earlier primitive-gap review. Broader non-security primitive work belongs
in a separate audit so this pass does not take ownership of unrelated
matching, parsing, VFS, or optimizer surfaces.

## Graph/Dataflow/Security Claims

| Status | Claim | Source proof |
|---|---|---|
| fixed | Prefix scan primitive was absent. | `vyre-primitives/src/math/prefix_scan.rs` provides inclusive and exclusive scan builders plus CPU references and tests. |
| fixed | Segment reduce primitive was absent. | `vyre-primitives/src/reduce/segment_reduce.rs` provides `segment_reduce_sum` with harness fixture and tests. |
| fixed | Histogram primitive was absent. | `vyre-primitives/src/text/byte_histogram.rs` and `vyre-primitives/src/reduce/histogram.rs` provide byte and value histograms. |
| fixed | Gather/scatter primitives were marker-only. | `vyre-primitives/src/reduce/gather.rs` and `vyre-primitives/src/reduce/scatter.rs` provide Program builders, CPU references, and tests. |
| fixed | Persistent BFS primitive was absent. | `vyre-primitives/src/graph/persistent_bfs.rs` and `persistent_bfs_step.rs` provide multi-step frontier expansion and harness coverage. |
| fixed | `reachable` lacked a GPU Program. | `vyre-primitives/src/graph/reachable.rs` now provides `reachable_program`. |
| fixed | `toposort` lacked a GPU Program. | `vyre-primitives/src/graph/toposort.rs` now provides `toposort_program`. |
| fixed | `size_argument_of` carried a fused-kernel caveat. | `vyre-primitives/src/predicate/size_argument_of.rs` now states the concrete reverse-`CALL_ARG` traversal contract and its CPU reference mirrors the GPU traversal. |
| fixed | Security shims had duplicate or vacuous semantics. | `flows_to` and `taint_flow` use `FLOWS_TO_MASK`; `sanitized_by` fuses `bitset_and_not` before traversal; fixtures exercise non-vacuous propagation. |

## Out Of This Pass

The earlier review also discussed general-purpose matching markers,
VFS helpers, and parser node fragments. They are outside this
graph/dataflow/security closure pass and should be tracked in their
own audit if the user asks for that surface next.
