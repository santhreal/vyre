# Whole-program compile search

```text
validated ProgramGraph
  -> planning facts and dependency edges
  -> candidate generation, bounded by SearchBudget
  -> cost evaluation per candidate
  -> one selected plan
  -> immutable Artifact
```

`vyre-megakernel` owns this seam. Its input is a validated typed
`ProgramGraph`, immutable `ExternalFacts` and an explicit `SearchBudget`.
Its output is one versioned immutable `Artifact` plus optional
`TargetPayload` values in an `ArtifactEnvelope`. Device admission,
materialization, submission, queues, residency and recovery consume that
product and do not alter artifact identity.

The current artifact schema is `ARTIFACT_SCHEMA_VERSION = 7`.

## Legality before cost

A fusion candidate is checked for legality before anything measures or
scores it. `analyze_fusion_pair` returns `FusionDecision::Legal` or
`Rejected(reason)`, and every rejection reason carries a stable code:

| Code | Reason |
|---|---|
| `MKL001_UNKNOWN_GRAPH_MEMBER` | a referenced node or value is absent from the graph |
| `MKL002_NOT_PRODUCER_CONSUMER` | the value does not connect the proposed producer and consumer |
| `MKL003_LIFECYCLE_BOUNDARY` | the value crosses an invocation or retained-state boundary |
| `MKL004_MULTIPLE_CONSUMERS` | more than one node consumes the value |
| `MKL005_WORKGROUP_MISMATCH` | the programs declare different workgroup geometry |
| `MKL006_SYNCHRONIZATION_BOUNDARY` | the geometries differ and one program reasons about the size of its own workgroup |
| `MKL007_DEPENDENCY_CYCLE` | contracting the proposed group would create a dependency cycle |

`FusionRejectionReason` is `#[non_exhaustive]`. A rejection is recorded in
the artifact rather than dropped, so a plan that looks unfused says why.

A barrier is not a rejection on its own. Two programs that declare the same
workgroup and synchronize inside it fuse into one kernel, because the merge
concatenates the arms at that geometry and every barrier is already
workgroup-uniform. That is what lets a score pass and a value pass over one
workgroup tile compile to a single dispatch.

## The unfused baseline is always a candidate

`explore` seeds the candidate set with `CandidatePlan::baseline`, one group
per node. Fusion has to earn its place against that baseline on cost, and
an exhausted budget degrades to the baseline instead of to nothing.

## The budget bounds the search, not the answer

`SearchBudget` carries `max_candidates`, `max_cpu_work`,
`max_target_compilations`, `max_measurements` and `max_elapsed_ns`. Every
edge considered spends CPU work, and the loop stops when the budget is
spent. Candidates past `max_candidates` are not pushed. There is no
implicit budget and no unbounded search.

The compiler reports what it spent in `SearchWork`: candidates explored,
CPU work, target compilations, measurements and elapsed nanoseconds.

## The cost model is open

`CostBreakdown` has eleven fields and no hidden term. The unit is nanoseconds
of expected device time.

| Field | Meaning |
|---|---|
| `semantic_work` | sum of semantic IR nodes in the complete graph |
| `launches` | number of generated kernel launches |
| `materializations` | number of values crossing generated-kernel boundaries |
| `materialized_bytes` | bytes those crossing values move |
| `live_value_peak` | largest per-invocation live value count in any one group |
| `shared_scratch_bytes` | largest shared scratch any one group declares, unioned by buffer name |
| `occupancy_passes_peak` | largest number of resident passes any one group needs |
| `launch_ns` | launch term |
| `materialization_ns` | materialized-traffic term |
| `occupancy_ns` | occupancy term |
| `total` | sum of the three terms, minimized by selection |

A materialization is counted per data dependency whose producer and consumer
land in different groups, which is exactly the value that has to round-trip
through memory. `semantic_work` is recorded as evidence and excluded from
`total`: it is the same for every candidate over one graph.

A launch is priced at the device's measured per-launch overhead, and at a
recorded floor of 4224 nanoseconds when the device reports none. Traffic is
priced at 3788 bytes per nanosecond. Both figures come from
`foundation.elementwise.add.1m` in `vyre-bench/snapshots`, and both are
constants in `vyre-megakernel/src/cost.rs`. Reproducing a selection needs the
graph, the budget and the device facts, and nothing else.

## Candidate order is deterministic

Candidates are sorted by their group assignment and deduplicated by it, so
two runs over the same graph produce the same candidate set in the same
order. Selection over a deterministic set is what makes an artifact digest
identify a compile rather than a run.

## Unmeasured is recorded as unmeasured

A selection the compiler reached analytically is recorded as analytic. It
is not presented as a measured winner and it is not called autoroute. The
compiler will not claim a clock produced a number no clock produced.
