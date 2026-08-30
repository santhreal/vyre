# Whole-program compile search

```text
validated ProgramGraph
  -> validated schedule-free LogicalProgramGraph
  -> planning facts and dependency edges
  -> schedule grammar derivations, bounded by SearchBudget
  -> constraint propagation over every derived candidate
  -> cost evaluation per admitted candidate
  -> objective ordering over the legal Pareto frontier
  -> validated SelectedPlan with exact phase schedule and search certificate
  -> immutable Artifact
```

`vyre-megakernel` owns this seam. It validates a typed `ProgramGraph` into a
schedule-free `LogicalProgramGraph` before planning reads it. Immutable
`ExternalFacts`, a typed `CompileObjective` and an explicit `SearchBudget`
complete the input. The output
is one versioned immutable `Artifact` plus optional authenticated
`TargetPayload` values in an `ArtifactEnvelope`. Device admission,
materialization, submission, queues, residency and recovery consume that
product and do not alter artifact identity.

## Logical domain boundary

`LOGICAL_ALGORITHM_VERSION = 4` authenticates the schedule-free domain
contract. Each graph node produces one structured logical region. Its typed
extents come from constants or symbolic dimensions on a graph value and resolve
before search. The region records parallel, sequential, reduction or retained
state semantics, a row-major index map and tensor layout, disjointness and
retained-state aliases, read/write and synchronization effects, producer
dependencies and an overflow-checked point bound. Missing or zero bindings,
unresolved runtime extents, overflowing bounds, dependency cycles and
incompatible aliases reject the logical stage.

Each region also states how it may be cut and what it exchanges, without naming
a device. `LogicalPartitionFacts` lists the axes a shard may split, each with
its exact bound and what splitting it means: elementwise points are
independent, reduction points combine associatively, sequence points are
ordered, spatial points may read a point another shard holds, routed points are
assigned by the data. A region that reads a value it also writes has spatial
axes; a region whose updates land at data-dependent locations has routed axes.
The facts also state whether every participant may hold the whole region; a
region that advances retained state or updates at data-dependent locations is
not replicable.

A region states the exact packed bytes it writes, and each dependence states the
bytes of the values that induce it, so a placement that moves a value between
devices prices the value's own contract.

`LogicalExchange` states one semantic exchange: the region, the kind
(all-reduce, all-gather, reduce-scatter, broadcast or point-to-point), the
participant group, the combining operator when the kind combines, the graph
values moved in operand order, and the exact payload bytes of one participant's
contribution. Those bytes come from the value contract the graph connects, so an
exchange over an unconnected buffer reports no values and places nothing.
Devices, mesh coordinates and transports are target facts, and choosing among
them is schedule selection.

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

`PHYSICAL_SCHEDULE_VERSION = 1` authenticates the projection the same call
attaches to the verified kernel. A target reads the frozen phase, logical
coverage, workgroup, vector width, axis mapping levels, pipeline role groups,
ring slots, synchronization boundaries with their alternating parity, persistent
queue capacity and checked resource ceiling from it, rather than inferring any of
them from the op stream. `lower_physical` attaches no projection, so a program
lowered without a selected schedule states no frozen fact instead of stating a
default one, and a target that needs one rejects the kernel. Emission compares
the projection against the artifact's recorded geometry for the same node and
refuses a disagreement, so the recorded launch and the compiled module cannot
drift apart.

A matrix multiply-accumulate states a tile extent triple and one typed fragment
per operand: element type, tile orientation, the invocations the fragment is
distributed across, and the access map of the staging storage when the fragment
is not register-resident. Operand arity and result span are derived from those
facts, so the verifier checks the operand list against the declaration instead of
against a literal, and a tile no target has an instruction for is statable,
verifiable and rejected at emission rather than unstatable. Instruction
selection belongs to the target: a backend compares the declared form against
the native forms it emits, and rejects the rest instead of reinterpreting them.

An asynchronous transfer states the scope its result becomes readable at, the
slot of the bounded stage ring it occupies, and the fence an ordinary read needs
after it lands. The ring slot is assigned at the lowering boundary from the
selected pipeline depth, and its wait takes the slot of the transfer it
completes, so a wait states how much of the ring may stay in flight instead of
meaning that everything issued has landed. The verifier rejects a wait that
pairs with no transfer, a wait whose slot no transfer filled, a slot outside its
own ring, and a fence narrower than the stated visibility. The transfer
mechanism and the wait form belong to the target: a backend selects a bulk or
scalar transfer and the fence instruction, and rejects a transaction it has no
form for, including a wait its native form completes out of order and a fence
wider than its dialect places.

The physical storage layout states what one workgroup allocates.
`STORAGE_LAYOUT_VERSION = 1` covers one region per workgroup-scoped and
invocation-private binding, each carrying its byte span, its alignment, the op
span it is live across, and the offset a deterministic first-fit plan gave it.
Two regions share bytes only when their lifetimes are disjoint, a region touched
inside a loop or a branch is live across the whole enclosing construct, and a
declared region nothing reads is live for the whole kernel. The layout also
states the peak simultaneously live result ids one invocation holds and the peak
register-resident matrix fragment width, which are physical counts over lowered
SSA and not the semantic pressure estimate the whole-program cost model reads.
A workgroup pool larger than the selected schedule's shared bound, or a register
count above its register bound, is rejected at the lowering boundary; a bound of
zero is an unstated bound and is not enforced. An op that addresses a slot the
binding layout does not declare is rejected by the neutral verifier, because the
region it names states no size, no class and no lifetime. Bank permutation,
vector width and the transfer mechanism that fills a region belong to the
target, which chooses them under these offsets.

The lowering boundary also compares what the program states against what the
lowered kernel performs. Each side states its effect on caller-visible storage
per binding name: reads, writes, and read-modify-writes. A write the program
performs and the kernel does not, a write the kernel performs and the program
does not, a read-modify-write only one side performs, and a read of storage no
expression reads are all rejected before the kernel leaves the boundary. A read
the program performs may disappear, because a value nothing consumes is
eliminable. Workgroup-scoped and invocation-private storage is outside the
comparison: lowering creates it, so no semantic buffer names it. The diagnostic
sidecar is excluded by name for the same reason.

Every capacity a neutral analysis reads arrives as a stated device fact.
`AnalysisFacts` carries the shared-memory bank count, the per-workgroup shared
capacity, and the constant capacity a target reported, and holds no default for
any of them. An analysis whose capacity is unstated does not run: the audit
returns that section absent, the constant promotion leaves the binding in global
memory, and no recommendation is produced. A default recorded in the neutral
crate would be correct for the device it was copied from and silently wrong for
the next one, while a report computed from it is indistinguishable from a
measured finding.

The current artifact schema is `ARTIFACT_SCHEMA_VERSION = 14`. Schema 14 adds the
identity of the graph before any symbolic binding resolved it, so a guarded set
of variants compiled over several extents of one graph proves it is one product,
and states every digest at a fixed width so an artifact's canonical byte length
does not move with the content of a hash. Schema 13 added the objective the plan
was selected under and the width of the legal Pareto frontier it was selected
from. The schema records the selected launch of every entry point: the entry
dependency order, logical coverage, grid, workgroup, vector width, pipeline
roles, ring slots, barrier phases, dynamic shared bytes, launch resource intent,
and persistence, together with the allocation and layout plan a runtime
allocates and binds, the
grammar derivation that produced the plan, and the certificate of the search that
selected it. A target payload states the same geometry or admission rejects it,
and emission carries the recorded launch rather than reporting one.

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

## Frontier topology is selected from a measured sample

A traversal phase also states how its active set is walked.
`select_frontier_topology` reads one measured sample (frontier density, dispatch
cost, readback bytes), the graph shape, the memory budget, the launch overhead,
the fusion pressure the caller measured, and whether the target grants a
device-wide barrier. It returns one `FrontierTopology` together with the three
basis-point figures the selection was made from.

The bands are read in one order: memory red zone, fused wave, subgroup-sparse,
sparse, block-dense, dense, and hybrid for the band that remains.

| Topology | Selected when |
|---|---|
| `SparseFrontier` | memory pressure reaches the red zone, whatever the density; otherwise density at or below 0.125 |
| `FusedWave` | fusion pressure, launch pressure and readback bytes are each at or above their bound, and memory pressure is clear of the red zone |
| `SubgroupSparseFrontier` | density at or below 0.03125 and average degree at or below the sparse degree bound, so one subgroup owns the active nodes |
| `BlockDenseFrontier` | density at or above 0.85 with a dense average degree |
| `DenseFrontier` | density at or above 0.70 with a dense average degree |
| `HybridFrontier` | the transition band no other row admits |

`SparseFrontier` is the baseline: `FrontierTopology::baseline` states it,
`fallback_baseline` returns to it, and a memory budget in the red zone selects it
before any density band is read. A target that grants no device-wide barrier
contributes a fusion pressure of zero, so `FusedWave` is unreachable there rather
than selected and rejected later.

`select_frontier_topology_stable` applies hysteresis against the previous
topology, so a density sitting on a band edge does not alternate between two
topologies across waves.

`FrontierTopologyDecision::stable_explanation` renders the selection as one
`megakernel-topology-v1` line carrying the topology, the three figures, and one
stable reason code per topology.

## Constraint propagation before code generation

Every derived candidate is eliminated or admitted before anything is generated
for it. Elimination is a proof, not a price: an illegal candidate is rejected,
never scored. Each eliminated family records a stable reason.

| Code | Reason |
|---|---|
| `MKC001_NUMERICAL` | the transform changes what the program computes: a width-observing phase reshaped, or a reduction reordered that does not reassociate |
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
| `MKC015_OBJECTIVE_BOUND` | an aggregated figure exceeds a hard bound the objective states |
| `MKC016_SCHEDULE_REQUIREMENT` | the candidate is outside the schedule family the caller required |

`PruneReason::ALL` is the variant space, and the certificate records one row per
eliminated family with its count, so a plan that looks unfused says which
constraint removed the alternative.

`MKC001_NUMERICAL` reads the budget the caller stated. Without one, every
candidate that reorders a rounding accumulation is eliminated. With one, a
candidate is admitted where the reordered contract fits the declared measure, so
stating a budget is what makes a tree reduction, a spatial partition and a
resident queue reachable over floating point. See
[numeric contracts](../reference/numeric-contracts.md).

## A caller may require one schedule family

`CompileRequest::requiring_schedule` states the family the selected plan must
exercise: `RequiredSchedule::Baseline` for a plan that applies no production, or
`RequiredSchedule::Production(p)` for a plan whose derivation applies `p` at
least once. Every candidate is still derived and every legality decision is
still recorded, so a requirement narrows what may be selected and not what is
searched. A candidate outside the family is eliminated with
`MKC016_SCHEDULE_REQUIREMENT`.

A family no legal candidate reaches fails the compile with
`MKC045_REQUIRED_SCHEDULE_UNREACHABLE`, and `is_required_schedule_unreachable`
reads that refusal by code. A single-node graph has no producer-consumer pair to
contract, so requiring a fusion of one is answered with the refusal rather than
with the baseline.

Conformance uses this to run one semantic graph under each family of
`CONFORMANCE_SCHEDULES` and check one declared numeric contract across all of
them. Without it a case written to check a tiled schedule checks whichever
family the objective ranked first.

`ScheduleAgreement` states which contract those outputs hold to.
`ScheduleAgreement::Exact` admits byte equality alone.
`ScheduleAgreement::Float32Ulps` admits a bounded unit-in-last-place distance
per finite lane and refuses a sign change, a class change, and any non-finite
lane that is not bit-identical, because no unit-in-last-place bound expresses
one. `check_schedule_agreement` compares every reached family against the
unspecialized baseline and refuses a run whose baseline produced nothing, so no
other family is promoted to reference.

## A declared dialect schema version is enforced

`CompileRequest::declaring_dialect_version` states the dialect schema version a
program was built against. A dialect left undeclared is compiled at its
registered version, which is what a caller rebuilding against the current
schema has migrated to. A declared version is held to: below the dialect's
supported floor, above its schema version, and a call to an operation
introduced after it all fail validation with `MKC046_SEMANTIC_VERSION_SKEW`,
before any candidate is derived.

Every registered dialect's own declarations are admitted on the same path. A
dialect whose supported floor exceeds its schema version admits no declarable
version, and an operation declaring a version its dialect has not reached is
unreachable at every version a caller could declare. Both are refused before a
program compiles rather than surfacing as a plan that cannot be selected.

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

## Reordering a reduction needs a law

A spatial partition, a persistent queue, a pipeline, an asymmetric join, an axis
remap, an axis reorder and a recomputation all change the order in which
invocations combine into a shared location. That order is unobservable only when
the combine is associative and commutative.

`ScheduleTransform::combine_order` states, per transform, whether the order it
produces differs from the order the program states. `SetWorkgroup` answers
conditionally: freezing a phase at a shape its own regions declared reshapes
nothing.

`algebraic_reordering::reordering_class` answers the other half from the program:
`NoCombine` when no combine crosses invocations, `Reassociable` when every
combine is registered associative and commutative for its element type, and
`Ordered` otherwise. Operator laws come from the algebraic law registry under
`vyre.combine.exact.*` and `vyre.combine.rounding.*` ids, so an extension
operator that registers its own laws is answered without being named, and an
operator with no registered law is ordered. Which operator applies which combine
is `CombineKind` in `vyre-spec`, and which IR variant combines at all is
`visit::expr_combine` and `visit::node_combine`: three exhaustive matches with no
catch-all arm, so a new operator or variant fails to compile rather than
defaulting to reorderable.

A candidate whose transform changes the order of a phase covering an `Ordered`
node is eliminated as `MKC001_NUMERICAL`. Integer and bitwise reductions keep
every reordering production; a floating-point reduction keeps fusion, fission,
dispatch cuts, synchronization, memory placement and prefetch, so the graph still
compiles and the baseline is still ranked.

## The unfused baseline is always a candidate

`explore` seeds the derivation with `CandidatePlan::baseline_for`, one group per
node at its declared launch width and no specialization. Every production has to
earn its place against that baseline on the objective's metric vector, and a tie
on every ordering metric is broken toward the shorter derivation, so an accepted transform is one that paid for itself. An
exhausted budget degrades to the best proved candidate instead of to nothing.

## A declared law derives candidates of its own

The grammar derives schedules. The declared laws derive programs, and both feed
the same candidate set.

`law_candidates::derive_law_alternatives` runs once per graph node before any
expansion. `optimizer::law_saturation::derive_program_alternative` offers every
expression of the node's program to the combine laws and keeps the smaller
derived term; `optimizer::region_law::derive_region_alternatives` composes the
region law families and returns each equivalent program with the chain of law
names it was reached through. Both run under `LawDerivationBudget`, two law
chains deep and eight alternatives wide by default, so the derivation is bounded
before the search is.

An alternative replaces one node's program, so it changes that node's measured
work and nothing about grouping, launch width or topology. The candidate carries
the derived measurements in `CandidatePlan::law_facts`, and `cost::evaluate`
prices it against those rather than against the baseline's, so a law that removes
work is charged for the program it produced.

Permission has two halves. Region-law grants come from the request: a law whose
pass declares a numerical contract the caller did not grant is never applied, so
a bit-exact request derives only bit-exact alternatives. The combine law set
composes the request with the element type, because exactness is a property of
the type: an integer combine reads the exact laws under any error budget, and a
rounding combine reads them only where the request grants reassociation within a
budget.

`SearchCertificate::law_derived` names the chain behind every admitted
law-derived candidate, `law_pruned` names the chains constraint propagation
eliminated and why, and `law_budget_reached` states whether a derivation stopped
with laws still to compose. `SelectedPlan::law_derivation` names the chains of
the plan that was selected, and is empty for a plan that states the programs as
written.

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

## What a compile optimizes is stated

A ranking with no stated objective is a ranking against whichever scalar the
cost model happened to total. Two callers then disagree about what optimal
meant, and a cache cannot tell a latency artifact from a throughput one, because
neither artifact states which it is. Every production compile therefore carries
a `CompileObjective`, and it is a constructor argument rather than a default.

The record states a primary ordered metric, up to four tie breakers, the
workload arrangements it optimizes for with a permille weight each, whether
those combine by weighted mean or worst case, the risk statistic a measured
comparison reads, the horizon one-time device cost is amortized over, hard
bounds per metric, and the retained-artifact portfolio policy.

| Metric | Unit | Ranks a candidate |
|---|---|---|
| `Latency` | nanoseconds | yes |
| `Throughput` | nanoseconds per launch | yes |
| `ColdStart` | nanoseconds | yes |
| `PeakMemory` | bytes | yes |
| `Energy` | microjoules | yes |
| `ArtifactBytes` | bytes | no |
| `VariantCount` | artifacts | no |
| `CompileWork` | work units | no |
| `MeasurementWork` | measurements | no |

Every metric is stated so lower is better, throughput included: it is
steady-state nanoseconds per launch, which orders the same way as launches per
second for one fixed graph and removes the need for a second comparison
direction.

A metric that only has a figure after emission, or after a whole portfolio is
assembled, cannot rank a candidate: ranking would have to invent the figure.
Those metrics are admissible as bounds, checked where the figure is real. The
artifact byte ceiling is one of them, and it is the only place that ceiling
lives; a request that states none is refused.

A metric priced by a calibrated target fact is refused when the device reported
none. `Throughput` and `ColdStart` need the persistent setup cost
`DeviceFacts::with_launch_costs` supplies. `Energy` needs an energy rate no
target in this tree reports, so an energy objective fails with the fact named
rather than being ranked against a guess.

Ranking keeps the legal Pareto frontier before it orders anything: a candidate
no better than another on every ordering metric cannot win under that objective
whatever it is later measured at, so it is dominated and never measured. The
selected plan records the frontier width, so a reader can tell a selection the
tie breakers had to decide from one the legal set decided on its own.

A bound is a refusal rather than a preference. A candidate whose aggregated
figure exceeds one is pruned with `MKC015_OBJECTIVE_BOUND`, and a compile whose
whole legal set exceeds one fails with the bound it came nearest to meeting.

| Code | Failure |
|---|---|
| `MKC029_INVALID_OBJECTIVE` | the record is internally inconsistent |
| `MKC030_MISSING_CALIBRATED_FACT` | a stated metric needs a fact the device withheld |
| `MKC031_OBJECTIVE_BOUND_VIOLATED` | every legal candidate exceeds a stated bound |
| `MKC032_PORTFOLIO_COVERAGE_UNSATISFIED` | one artifact cannot satisfy the stated coverage policy |

The whole record is `Copy` and serializable, and it participates in request,
artifact, cache and measurement identity by value. Changing any field changes
the request digest, so no compile can reuse a decision another objective made.

## The retained artifact set is one decision

The portfolio policy states how many artifacts a compile retains and which
workload classes each has to serve. `CoveragePolicy::Single` retains one
artifact for every stated class; `CoveragePolicy::EveryWorkloadClass` retains a
set in which each class is served by some artifact, bounded by the policy's
artifact ceiling, the `VariantCount` bound, and an aggregate byte ceiling.

`compile` and `compile_measured` emit one artifact and refuse a policy one
artifact cannot satisfy with `MKC032_PORTFOLIO_COVERAGE_UNSATISFIED`, naming
`compile_portfolio`. `compile_portfolio` and `compile_portfolio_measured` return
an `ArtifactPortfolio`: the retained artifacts plus the artifact index each
stated class is served by.

Selection is joint rather than per class. Optimizing each class alone maximizes
the retained set, and every retained artifact costs compile work, bytes, and
load time. A workload profile holds at most four classes, so every partition of
the stated classes the policy admits is enumerated, each part is compiled once
under the objective narrowed to that part with its weights restated to a
thousand permille, and whole partitions are ordered by the objective read over
every stated class. Ties go to the smaller set and then the smaller aggregate
byte count, so two sets the objective cannot separate are separated by what they
cost to keep. Each retained artifact records the narrowed objective it was
selected under, so a runtime holding several can tell which arrangement each
serves.

Partitions are enumerated as restricted growth strings, so each set partition is
enumerated once instead of once per relabelling of its parts, and the assignment
one compile reports is canonical.

## What a variant is selected by is stated

A workload class states how a launch is arranged. It does not state what shape
the data has, so the retained set above cannot say which artifact is correct for
which input. That is what the specialization contract states, at
`SPECIALIZATION_SCHEMA_VERSION = 1`.

A contract declares axes and a domain for each. An axis is a typed fact: a
symbolic graph dimension, the layout or density class of a graph value, retained
state, launch batch, concurrency, a constant's content identity, an authenticated
target capability, or a target resource fact. No axis carries a caller-supplied
display name, so a model name or a family branch cannot enter a compiler,
backend, artifact, or runtime signature; application information reaches the
compiler as the configuration digest and as graph identity.

A `VariantGuard` is a conjunction of terms over those axes plus a precedence. Two
proofs decide whether a guard set is usable, and both are computed rather than
asserted. Guards that are not provably disjoint must carry distinct precedence.
Guard bounds cut each axis domain into cells, and every cell must be admitted by
some guard or served by a generic remainder; a gap with an unsupported remainder
is `MKC036_GUARD_COVERAGE_GAP`.

`compile_specialized_portfolio` scores every subset of the proposed guards the
variant ceiling admits, including the empty subset, so the unspecialized baseline
stays in the candidate set and a variant that buys less than it costs is not
retained. Each retained variant is compiled from the request narrowed by its
guard, so its schedule may differ structurally from the generic one. The figure
of a set is each member's figure weighted by the part of the domain it serves.

A `PortfolioEnvelope` seals the contract, every variant with its guard, and the
remainder as one authenticated product for one target identity. Members must
agree on the graph before any binding resolved it, on the objective, on the
compiler version, and on the artifact schema. Decoding re-runs both proofs, so an
edited guard set does not decode.

A runtime admits the whole set for one required payload format and one
authenticated target identity, then selects by evaluating guards over trusted
facts. Facts outside the declared domain are refused with
`MKC039_UNSUPPORTED_WORKLOAD` rather than served by the remainder, which was
compiled for that domain. Selection returns a member that was compiled and scored
before admission; nothing after the compile alters a schedule.

## The cost model is open

`CostBreakdown` has twenty-three fields and no hidden term. Each one states its
unit and where its weight came from in `vyre-megakernel/src/cost/provenance.rs`,
and `CostBreakdown::term` reads that row by field name. A field with no row
fails `every_cost_field_states_its_unit_and_provenance`, which derives the field
set from the serialized shape rather than from a list.

Fifteen fields are evidence, recorded and excluded from `total`.

| Field | Unit | Meaning |
|---|---|---|
| `semantic_work` | count | sum of semantic IR nodes in the complete graph |
| `launches` | count | generated kernel launches |
| `materializations` | count | values crossing generated-kernel boundaries |
| `materialized_bytes` | bytes | bytes those crossing values move |
| `live_value_peak` | registers | largest per-invocation live value count in any one group |
| `shared_scratch_bytes` | bytes | largest shared scratch any one group declares, unioned by buffer name |
| `occupancy_passes_peak` | count | largest number of resident passes any one group needs |
| `planned_peak_bytes` | bytes | bytes the allocation plan holds at once under this grouping |
| `instructions` | count | scalar instructions the graph states |
| `tensor_ops` | count | tile statements the graph states |
| `barriers` | count | workgroup barriers the graph states |
| `grid_syncs` | count | stated grid rendezvous, plus one per resident-partition stage boundary |
| `divergent_regions` | count | lane-gated regions the graph states |
| `spill_registers_peak` | registers | worst group's live values above the full-occupancy register budget |
| `cache_resident_bytes` | bytes | replayed bytes the device-wide cache holds |
| `reported_spill_bytes` | bytes | target-reported local spill across every launched invocation |

Eight fields are charged, in nanoseconds of expected device time.

| Field | Priced from |
|---|---|
| `launch_ns` | per-launch overhead fact, or a recorded floor of 4224 ns |
| `materialization_ns` | materialized bytes at the bandwidth fact |
| `occupancy_ns` | replayed bytes the cache does not serve, at the bandwidth fact |
| `instruction_ns` | scalar instructions at the reported instruction rate |
| `tensor_ns` | tile statements at the reported matrix-engine rate |
| `synchronization_ns` | barriers and grid rendezvous at their reported costs |
| `divergence_ns` | idle lanes of each gated region at the instruction rate |
| `total` | sum of every charged term, minimized by selection |

A materialization is counted per data dependency whose producer and consumer
land in different groups, which is exactly the value that has to round-trip
through memory. `semantic_work` is the same for every candidate over one graph.

A term whose rate the device does not report is charged zero rather than
estimated: the count stays as evidence and the nanoseconds stay out of the
total. The register budget an occupancy pass divides is the registers per
invocation at full occupancy, which a device reports as its per-compute-unit
register file divided by its per-compute-unit invocation limit; the
architectural per-invocation ceiling is a separate fact, and exceeding it has no
execution to price. Launch overhead falls back to a recorded floor of 4224
nanoseconds and bandwidth to 3788 bytes per nanosecond, both from
`foundation.elementwise.add.1m` in `vyre-bench/snapshots`. Reproducing a
selection needs the graph, the budget and the device facts, and nothing else.

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
| emitted resources | what each built plan allocates on the device | one module query per built plan |
| measurement | which built plan is fastest on the device | `max_measurements` |

`compile_measured` emits the top `max_target_compilations` ranked plans. A plan
the target compiler rejects is eliminated with `MKC014_EMISSION` charged to the
production that derived it, and the ladder continues with the next ranked plan,
so one unbuildable plan does not fail the compilation.

Each plan that emitted is then asked what it allocated. Registers and spill are
assigned by the target compiler, so no estimate derived from the IR can state
them and only the loaded module holds them. `FinalistEvaluator::resources`
returns one record per payload entry with registers and local-memory spill per
invocation and statically declared shared bytes. A reported figure replaces the
model's estimate for that term, spill is priced as traffic over the invocations
the authenticated geometry launches, and the finalists are re-ranked on the
result before any measurement is spent. A plan whose reported registers exceed
the device's architectural limit is eliminated with the emission reason: that is
a launch the hardware cannot run, not a price. A backend whose API reports none
of it returns default records, which leaves every finalist ranked on the
estimate.

## Measurement runs one versioned protocol

A device time is noisy. The same entry point launched twice returns two numbers,
and the difference is queue occupancy, clock state and cache state rather than a
property of the program. `MeasurementProtocol` version 1 states every rule the
comparison depends on, and the artifact records the protocol beside the samples
taken under it.

| Rule | Version 1 | Why it is stated |
|---|---|---|
| warmup launches | 2 | module load and first-touch allocation are not the schedule |
| repetitions per round | 1 | one sample per candidate per round |
| rounds | 3 to 64 | samples spread across the session make drift visible |
| trim | 200 permille, slow end only | device noise is one-sided |
| estimator | trimmed median | one stalled launch cannot move it |
| uncertainty | median absolute deviation scaled to sigma | states how far apart two estimates must be |
| stopping rule | every estimate inside 30 permille | spends launches until a comparison is possible, no longer |
| equivalence band | 20 permille of the incumbent | a smaller difference is the device |

`max_measurements` bounds the launches one candidate receives, warmup included,
and the protocol is fitted to that budget before the session starts: rounds
shrink before repetitions do. Every candidate is sampled in every round, and the
visit order rotates by one position per round, so no candidate is always charged
for whatever the device does at the start of a round.

The winner is the lowest trimmed-median estimate. A later candidate takes the
selection from an earlier one only by clearing the equivalence band and the
combined uncertainty of both estimates, so a candidate the analytic ranking put
first keeps the selection when the measured difference is not evidence. This is
what makes two runs of the same search on the same device select the same
artifact. Passing the previous artifact's record to
`CompileRequest::with_recorded_measurement` extends that across compilations: an
authenticated winner is kept unless a challenger clears the band.

Two figures decide whether an earlier record has authority over a later session:
the protocol version and the calibrated fact-set version the ranking priced
with, which `DeviceFacts::with_calibration_version` records and the measurement
record carries. When both match, the earlier winner stands and only the band can
move it. When either differs, the two sessions are incomparable and the later one
selects freely, because the rules or the priced figures changed. A recalibration
that leaves the version at its previous value therefore never takes effect on a
recorded selection.

The record retains every candidate's raw samples in measurement order, its
identity, its analytic rank, the cost the model predicted for it, and the
estimate its samples reduce to. Prediction error is the signed difference between
the predicted and measured figures in permille; it recalibrates versioned fact
sets and never changes a selection. Beside the samples the record retains what
the device was doing: the clock, thermal and power state the backend reported,
and the drift the session observed between its first and last counted round,
which is available on a backend that reports no state at all.

A compilation where no ranked plan emitted fails with the rejection instead of
returning a plan the target cannot build.

## Unmeasured is recorded as unmeasured

A selection the compiler reached analytically is recorded as analytic. It
is not presented as a measured winner and it is not called autoroute. The
compiler will not claim a clock produced a number no clock produced.
