# Whole-program compile search

```text
validated ProgramGraph
  -> validated schedule-free LogicalProgramGraph
  -> planning facts and dependency edges
  -> versioned neutral schedule transforms, bounded by SearchBudget
  -> cost evaluation per validated candidate
  -> validated SelectedPlan with exact phase schedule
  -> immutable Artifact
```

`vyre-megakernel` owns this seam. It validates a typed `ProgramGraph` into a
schedule-free `LogicalProgramGraph` before planning reads it. Immutable
`ExternalFacts` and an explicit `SearchBudget` complete the input. The output
is one versioned immutable `Artifact` plus optional authenticated
`TargetPayload` values in an `ArtifactEnvelope`. Device admission,
materialization, submission, queues, residency and recovery consume that
product and do not alter artifact identity.

## Logical domain boundary

`LOGICAL_ALGORITHM_VERSION = 2` authenticates the schedule-free domain
contract. Each graph node produces one structured logical region. Its typed
extents come from constants or symbolic dimensions on a graph value and resolve
before search. The region records parallel, sequential, reduction or retained
state semantics, a row-major index map and tensor layout, disjointness and
retained-state aliases, read/write and synchronization effects, producer
dependencies and an overflow-checked point bound. Missing or zero bindings,
unresolved runtime extents, overflowing bounds, dependency cycles and
incompatible aliases reject the logical stage.

Library registrations are checked as a registry-derived set. Each registered
composition must build a `ProgramGraph` and a complete logical region without a
separate library-specific domain path.

Ordinary library programs use `LogicalIndex`, `LogicalTileId`,
`LogicalWithinTileId`, and `LogicalBarrier`. They contain no physical
invocation, workgroup, local, or barrier markers before schedule selection.
The linked operation registry checks this closure for every library-tier
registration.

## Selected schedule boundary

`SCHEDULE_IR_VERSION = 1` authenticates the backend-neutral schedule. The
foundation schema represents phase fission, fusion, tiling, splitting,
reordering, vectorization, lane through device-partition mappings, memory
placement, prefetch, bounded pipelines and queues, recomputation, dispatch cuts,
synchronization and asymmetric joins. Each transform is applied transactionally
after typed preconditions pass. The record includes source phases and logical
regions, the complete prior-schedule identity as inverse provenance, and checked
resource increments.

Artifact validation replays every transform from immutable source phases.
Changed preconditions, provenance, phase geometry, ordering or resource bounds
reject the artifact before `vyre-lower` applies the selected phase workgroup and
constructs a `PhysicalKernel`.

`lower_scheduled` maps logical domain, tile, within-tile, and barrier markers to
physical invocation, workgroup, local, and barrier IR before descriptor
construction. `lower_physical` rejects any unresolved logical marker.

The current artifact schema is `ARTIFACT_SCHEMA_VERSION = 9`.

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

`FusionRejectionReason` and `TopologyRejectionReason` are `#[non_exhaustive]`. A rejection is recorded in
the artifact rather than dropped, so a plan that looks unfused says why.

Execution topologies are also validated before scoring:

| Code | Reason |
|---|---|
| `MKL010_INSUFFICIENT_CONCURRENT_QUEUES` | target device does not report or support required concurrent queues |
| `MKL011_INSUFFICIENT_COMPUTE_UNITS` | target device does not report or support required compute units |
| `MKL012_UNENFORCEABLE_SPATIAL_MASKING` | spatial masking requested on a target without enforceable spatial partitioning capability |
| `MKL013_REQUIRES_COOPERATIVE_LAUNCH` | bounded resident queue or device-wide join requested on a device without cooperative launch |
| `MKL014_RESOURCE_CONFLICT` | RAW/WAR/WAW hazard or resource alias between concurrent arms |
| `MKL015_CONTROL_DEPENDENCY_OR_EFFECT` | cross-arm control dependency or effect that cannot be satisfied by concurrent queues |
| `MKL016_ILLEGAL_ASYMMETRIC_JOIN` | asymmetric or divergent join across resident boundary without cooperative join or GridSync cut |
| `MKL017_NO_INDEPENDENT_CONCURRENCY` | candidate has no independent arms to execute concurrently |
| `MKL018_OCCUPANCY_EXCEEDED` | occupancy or scratch budget exceeded for resident execution |

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
