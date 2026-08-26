# Whole-program compile search

```text
validated ProgramGraph
  -> validated schedule-free LogicalProgramGraph
  -> planning facts and dependency edges
  -> schedule grammar derivations, bounded by SearchBudget
  -> constraint propagation over every derived candidate
  -> cost evaluation per admitted candidate
  -> validated SelectedPlan with exact phase schedule and search certificate
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

`SCHEDULE_IR_VERSION = 2` authenticates the backend-neutral schedule. The
foundation schema represents phase fission, fusion, tiling, splitting,
reordering, vectorization, lane through device-partition mappings, memory
placement, prefetch, bounded pipelines and queues, recomputation, dispatch cuts,
synchronization and asymmetric joins. Each transform is applied transactionally
after typed preconditions pass. The record includes source phases and logical
regions, the complete prior-schedule identity as inverse provenance, and checked
resource increments.

Tiling and splitting rewrite the axis nest of their phase. A factor that divides
an axis extent replaces that axis with an outer axis of the quotient extent and
inserts an inner axis of the factor directly after it, taking the next free axis
index in the same logical region. Schema 2 records that nest, so a tiled phase is
distinguishable from an untiled one by schedule identity.

Artifact validation replays every transform from immutable source phases.
Changed preconditions, provenance, phase geometry, ordering or resource bounds
reject the artifact before `vyre-lower` applies the selected phase workgroup and
constructs a `PhysicalKernel`.

`lower_scheduled` maps logical domain, tile, within-tile, and barrier markers to
physical invocation, workgroup, local, and barrier IR before descriptor
construction. `lower_physical` rejects any unresolved logical marker.

The current artifact schema is `ARTIFACT_SCHEMA_VERSION = 11`. Schema 11 records
the selected launch of every entry point: the entry dependency order, logical
coverage, grid, workgroup, vector width, pipeline roles, ring slots, barrier
phases, dynamic shared bytes, launch resource intent, and persistence, together
with the workspace plan a runtime allocates, the grammar derivation that produced
the plan, and the certificate of the search that selected it. A target payload
states the same geometry or admission rejects it, and emission carries the
recorded launch rather than reporting one.

## Candidate generation is a grammar

`SCHEDULE_GRAMMAR_VERSION = 1` authenticates candidate generation. Search does
not hold a catalog of kernel shapes. It derives candidates from the baseline by
applying productions of a versioned grammar, one production per schedule
transform. A production reads phase axes and planning facts only. Device
capabilities are absent from the grammar on purpose: a production proposes every
structure the schedule IR can express, and constraint propagation eliminates the
ones the authenticated target facts do not grant, so the certificate reports a
family as considered and eliminated instead of reporting a smaller search.

| Code | Production | Derives |
|---|---|---|
| `MKP001_FUSION` | Fusion | one generated kernel out of two phases |
| `MKP002_FISSION` | Fission | a phase boundary inside one generated kernel |
| `MKP003_LAUNCH_WIDTH` | LaunchWidth | a workgroup shape for one phase |
| `MKP004_SPATIAL_PARTITION` | SpatialPartition | a compute-unit partition of one phase |
| `MKP005_PERSISTENT_QUEUE` | PersistentQueue | a bounded resident queue for one phase |
| `MKP006_PIPELINE` | Pipeline | producer/consumer overlap with ring slots and role groups |
| `MKP007_DISPATCH_CUT` | DispatchCut | a dispatch boundary between two phases |
| `MKP008_ASYMMETRIC_JOIN` | AsymmetricJoin | a join of independent arms into one consumer |
| `MKP009_SYNCHRONIZATION` | Synchronization | a barrier phase at a named scope |
| `MKP010_MEMORY_PLACEMENT` | MemoryPlacement | a value placed in workgroup or invocation storage |
| `MKP011_PREFETCH` | Prefetch | a bounded prefetch distance for one value |
| `MKP012_RECOMPUTATION` | Recomputation | a value recomputed instead of materialized |
| `MKP013_TILING` | Tiling | a tiled axis nest for one phase |
| `MKP014_AXIS_SPLIT` | AxisSplit | one axis split by an exact factor |
| `MKP015_VECTORIZATION` | Vectorization | a vector width for one axis |
| `MKP016_AXIS_MAPPING` | AxisMapping | an axis mapped to a hardware level |
| `MKP017_AXIS_REORDER` | AxisReorder | a permutation of the axis nest |

`ScheduleProduction::deriving` maps every `ScheduleTransform` variant onto
exactly one production through an exhaustive match, so a transform added to the
foundation schema cannot compile until a production derives it.

Production order is expansion priority: kernel organization first, then
concurrency and ordering, then storage, then intra-phase loop shape. A bounded
search therefore spends its budget on the families that change program structure
before the families that change one number inside a phase.

The same semantic graph produces structurally different kernel graphs on
different devices, because ranking and elimination read the device facts. A
device that reports no cooperative launch keeps no resident candidate; a device
that reports eight compute units and four queues admits partitioned and
concurrent organizations of the same graph. A device that pays for every launch
selects fewer, larger generated kernels than one that pays for resident state.

## Constraint propagation before code generation

Every derived candidate is eliminated or admitted before anything is generated
for it. Elimination is a proof, not a price: an illegal candidate is rejected,
never scored. Each eliminated family records a stable reason.

| Code | Reason |
|---|---|
| `MKC001_NUMERICAL` | the transform changes what a width-observing phase computes |
| `MKC002_DEPENDENCE` | the grouping would contract a dependency cycle |
| `MKC003_ALIAS_OR_EFFECT` | an alias or effect between concurrent arms is unsatisfiable |
| `MKC004_BARRIER_VISIBILITY` | a barrier phase or proxy is not visible to every participant |
| `MKC005_PIPELINE_CAPACITY` | ring slots exceed the storage the device grants |
| `MKC006_OCCUPANCY` | the launch exceeds the invocations per workgroup the device grants |
| `MKC007_SCRATCH` | a phase declares more shared scratch than the device grants |
| `MKC008_WORKSPACE` | cross-group workspace is not addressable at its alignment |
| `MKC009_PROGRESS` | forward progress needs a capability the device does not report |
| `MKC010_OBJECTIVE_DOMINATED` | a proved bound is no better than the best proved candidate |
| `MKC011_TARGET_FACTS` | an operand needs a capability no authenticated fact reports |
| `MKC012_REPRESENTATION` | the artifact cannot represent the derived organization |
| `MKC013_SCHEDULE_LEGALITY` | a typed schedule precondition failed |
| `MKC014_EMISSION` | the target compiler rejected the plan before measurement |

`PruneReason::ALL` is the variant space, and the certificate records one row per
eliminated family with its count, so a plan that looks unfused says which
constraint removed the alternative.

Fusion and topology legality keep their own codes, and the constraint classes map
onto them rather than restating them:

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

`explore` seeds the derivation with `CandidatePlan::baseline_for`, one group per
node at its declared launch width and no specialization. Every production has to
earn its place against that baseline on cost, and a cost tie is broken toward the
shorter derivation, so an accepted transform is one that paid for itself. An
exhausted budget degrades to the best proved candidate instead of to nothing.

## The budget bounds the search, not the answer

`SearchBudget` carries `max_candidates`, `max_cpu_work`,
`max_target_compilations`, `max_measurements` and `max_elapsed_ns`. Every
derivation considered spends CPU work, and expansion stops when a bound is
reached. Candidates past `max_candidates` are not pushed. There is no
implicit budget and no unbounded search.

The compiler reports what it spent in `SearchWork`: candidates explored,
CPU work, target compilations, measurements and elapsed nanoseconds. The
certificate additionally records the grammar version, the depth the expansion
reached, one row per derived family with its admitted count, one row per
eliminated family with its reason, and whether a bound stopped the search. That
record is what reproduces a compile without re-running it.

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

Expansion is a breadth-first worklist over a canonically ordered production set,
and a candidate already derived is dropped by a structural key rather than
re-expanded. Two runs over the same graph, facts, device and budget therefore
derive the same candidates in the same order and record the same certificate.
Selection over a deterministic set is what makes an artifact digest identify a
compile rather than a run.

## Evaluation is a bounded ladder

Fidelity and cost rise together, so a level runs only on what the level below it
kept.

| Level | What it answers | What it spends |
|---|---|---|
| symbolic bound | can any descendant of this candidate beat the incumbent | CPU work |
| analytic cost | how the admitted candidates rank | CPU work |
| emission | which ranked plans the target compiler builds | `max_target_compilations` |
| measurement | which built plan is fastest on the device | `max_measurements` |

`compile_measured` emits the top `max_target_compilations` ranked plans. A plan
the target compiler rejects is eliminated with `MKC014_EMISSION` charged to the
production that derived it, and the ladder continues with the next ranked plan,
so one unbuildable plan does not fail the compilation. Each plan that emitted is
launched `max_measurements` times and the lowest median device time wins, which
can select a plan the analytic model ranked behind another. A compilation where
no ranked plan emitted fails with the rejection instead of returning a plan the
target cannot build.

## Unmeasured is recorded as unmeasured

A selection the compiler reached analytically is recorded as analytic. It
is not presented as a measured winner and it is not called autoroute. The
compiler will not claim a clock produced a number no clock produced.
