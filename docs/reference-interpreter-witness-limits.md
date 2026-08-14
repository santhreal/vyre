# What the reference interpreter can and cannot witness

Applies to Vyre 0.7.2.

`vyre-reference` is the CPU oracle that most of this tree's correctness evidence
rests on. This document states, as a property of the instrument rather than of any
one kernel, which classes of concurrency defect it can expose and which it cannot.

The short version, and the rule to remember:

> `reference_eval` can witness a defect that survives **whole-loop workgroup
> reordering**. It cannot witness a defect that requires **interleaving two
> workgroups inside a single iteration**. So "no `reference_eval` divergence,
> therefore sound" is invalid for any persistent-loop kernel.

## Why the limit exists

`reference_eval` executes one workgroup at a time, and runs each workgroup's
invocations to completion before starting the next. `reference_eval_lane_reversed`
does the same thing with the workgroup order inverted. Both are legal schedules,
which is what makes them useful: real hardware is free to pick either, and nothing
in the IR orders one workgroup against another.

For a kernel whose whole fixpoint loop lives **inside** the kernel, this has a
consequence that is easy to miss. Workgroup 0 runs its entire loop, every
iteration of it, before workgroup 1 begins. The two groups never occupy the same
iteration at the same time. Any defect whose window is "group 0 is at iteration
`i` while group 1 is still finishing iteration `i`" is therefore unreachable, in
both workgroup orders, no matter how the fixture is built.

That window is exactly where the interesting shared-flag races live.

## The two classes, with worked examples

### Witnessable: defects that survive whole-loop reordering

These show up as a plain disagreement between `reference_eval` and
`reference_eval_lane_reversed`, or between either and a CPU oracle. Both come from
one group finishing everything before another starts, which is a schedule the
interpreter produces natively.

- **Never-set.** A group whose compare-and-publish gate excludes every one of its
  lanes cannot signal at all, and a flag nobody sets reads identically to a flag
  that cleared because everything converged. `bellman_shortest_path` at 4 nodes
  and 257 edges: workgroup 1 owns edge 256, relaxes `next_dist[3]` correctly, but
  its compare is gated `t < n_nodes`, which is `256 < 4`, so it has no lane that
  can publish or flag. Wrong output **and** a false convergence verdict in the
  canonical forward order. This one is deterministic, not a race.
- **Unpublished state.** A group runs its whole loop before another group produced
  the value it was supposed to copy, sees no change, reads a zero flag and retires
  for good. `bellman_shortest_path` at 257 nodes, `sinkhorn_iterate` at
  `m = 257`, and the `persistent_fixpoint` wrapper at 257 words all diverge this
  way under reversed workgroup order.

All of these were caught with `reference_eval` and are locked by tests today.

### Not witnessable: defects needing within-iteration interleaving

These require two groups to be inside the same iteration at once, so no workgroup
order the interpreter can produce will expose them. A green run says nothing about
them.

- **Lost set.** Lane 0 clears a shared flag with a plain store while another
  group's `atomic_or` of the same word is already in flight. The clear erases the
  set, the erased group reads 0 and takes its early exit with unconverged state.
- **Exclusive discovery attribution.** Growth is detected by testing whether an
  `atomic_or` actually flipped a bit (`if (old & bit) == 0`). Exactly one group
  wins that flip, so a discovery can be attributed **solely** to a group whose
  flag write is not ordered against another group's early-exit read. Redundant
  work coverage does **not** save this: coverage can be redundant while
  attribution is exclusive, because a bit is flipped once.

## The trap this created, three times

Both failure modes above are invisible to the instrument, so a site carrying them
presents as fully covered and fully green. Three successive readings of
`vyre-pass-engine/src/optimizer/dce_program.rs` went wrong at that site: two
nearly concluded it was sound because the interpreter agreed, and the third
overcorrected and fixed the wrong mechanism at a measured 4-6x cost. Only the
exclusive-attribution reading survives, and it was established by reading the IR and
then confirming the cost and the coverage on hardware, never by a green run.

A green run can also be misread in the OTHER direction, and that happened here too.
When the interpreter runs workgroup 1 first, workgroup 1 reads a `frontier_out` the
entry seed has not filled, finds an empty frontier, and contributes nothing. That
presents as a second ordering bug masking the first. It is not a bug at all: in this
kernel workgroup 0's strided lanes visit every source, so a group that contributes
nothing has lost nothing, and adding a grid-wide seed fence was measured to buy no
closure the kernel does not already reach.

What IS a live defect at the same site is the exclusive-attribution case above, and
it was found only after that fence was reverted. Coverage is redundant while
attribution is exclusive, so the one group that covers the whole domain can be
precisely the group that misses the flag.

So: when a persistent-loop kernel shares a convergence flag across workgroups,
`reference_eval` agreement is **not** evidence of soundness, and a
redundant-coverage argument is **not** either. Establish soundness by reading the
ordering AND the attribution, or on real hardware.

## What to do instead

1. **Read the ordering, then read the ATTRIBUTION.** For every write to a
   cross-group flag, name the fence that orders it against every read. A
   `MemoryOrdering::SeqCst` barrier is workgroup scope and orders nothing across
   groups. But a missing fence is not automatically a live defect: ask what the
   unfenced group would have lost, because if its work is a pure duplicate, losing
   its flag costs nothing. Then ask separately whether DISCOVERY is exclusive, since
   a flag raised only by the lane that flipped a bit can be stolen by a duplicate,
   leaving the group that matters never raising it. Asking only the first question
   produced a 4-6x pessimization for a defect that was not live; the second question
   is the one that found the real one.
2. **Route by width, or PIN the width.** Above one workgroup, either switch to a
   grid-synchronised form (in-tree precedents: `persistent_bfs`,
   `scallop_join.rs:155`, `scallop_join_wide.rs:215`, and the `persistent_fixpoint`
   callers) or dispatch the program as ONE WORKGROUP, which is far cheaper whenever
   one workgroup already covers the whole domain. `dce_program` takes the second
   route from both of its callers (`pipeline_resident.rs`, `dce_via_encoded.rs`), and
   `reduction_metrics.rs` takes it for scalar reductions that initialise their own
   output.
3. **Do not write a test that asserts the defect.** A test asserting the *absence*
   of a grid fence locks the bug in and breaks the moment somebody fixes it. Where
   a defect is real but unwitnessable, assert the true behaviour and say in the
   doc comment that a green run is not a soundness argument. See
   `dce_bfs_multi_workgroup_agreement_is_a_backend_limit_not_a_soundness_proof` in
   `vyre-pass-engine/tests/optimizer_bfs_and_softmax_parity.rs`.
4. **Do not put a `GridSync` barrier inside a `Node::Loop`.** The barrier's release
   target is computed at emit time from a static index, so a loop body emits one
   barrier with one static target: iteration 0 releases correctly and every later
   iteration finds the counter already at or past the target and falls straight
   through. It is a silent no-op, and `contains_grid_sync` recurses into loop
   bodies, so the IR permits it and the driver reports grid sync as present. Emit
   one barrier instance per wave instead.
5. **Repeat from the host, not from a device-side loop.** The CUDA dispatch path
   zeroes the module-scope `_vyre_grid_barrier` counter before each cooperative
   launch, so every launch starts each barrier instance's static target from zero.
   That is the same mechanism, read the other way, that makes a device-side loop
   unsound and host-orchestrated repetition safe.

## The grid-synced wave shape, and where to look for its fences

The sound shape for a fixpoint wave above one workgroup is:

```text
clear the flag (one lane)
GridSync
step, and atomic_or the flag on discovery
GridSync
read the flag (uniform across the grid)
```

A word whose clear is separated from every set by a **grid-wide** fence cannot lose
a set, so per-iteration flag words are one way to get this and not the only way.
`persistent_fixpoint_grid` uses one word per iteration and never clears, which suits
a caller that can absorb a wider flag buffer. Where the caller's ABI fixes the flag
at one word, the fenced single-word form above is equivalent and avoids the ABI
break.

`persistent_bfs` already implements exactly this shape, but **the first fence is not
where you would look for it.** The wave in
`vyre-primitives/src/graph/persistent_bfs/program.rs:391-456` appears to be only
"clear, step, barrier". The clear-to-set fence lives one level down, inside the step
builder, as the snapshot barrier at
`vyre-primitives/src/graph/csr_forward_or_changed/program_parallel.rs:237`: the step
snapshots the frontier and the active gate, fences the grid, and only then runs the
gated edge scan that sets the flag. A reviewer reading the wave alone concludes the
clear is unfenced and the site is racy. It is not. When auditing a grid-synced wave,
expand the child builders before judging the fences.

## The budget you cannot unroll, and why DCE never needed to

Because a `GridSync` inside a `Node::Loop` is a silent no-op, a grid-synced fixpoint
cannot loop its waves on the device; it must emit one barrier instance per wave,
which means unrolling. That is a real constraint for `persistent_fixpoint_grid` and
for `persistent_bfs`'s grid form, which unrolls its whole budget with no cap
(`vyre-primitives/src/graph/persistent_bfs/program.rs:388`).

`vyre-pass-engine`'s DCE was once recorded here as the worst case of this,
because it passes `max_iters = node_count`. It is not a case of it at all.
`build_dce_bfs_program` never unrolled: its waves live in a bounded `Node::loop_for`,
so its IR is O(1) in the budget. Converting it to host-repeated grid-synced wave
batches was measured on an RTX 5090 at 4 to 6 times the wall time, about 232 times
the launches, and FIVE times the IR entry nodes, and was reverted. The answer for
that site is to pin the dispatch to one workgroup, which both callers now do.

Host-orchestrated repetition remains the right tool where a grid-synced wave really
must repeat and the budget scales with the input, for the reason in rule 5. It wants
an idempotent program, so a re-dispatched wave must accumulate rather than
re-initialise. Capping the wave count is not a safe fallback where a caller treats
non-convergence as a hard error, which DCE does deliberately, since deleting code
against a partial liveness set removes live code.

## `Node::Return` used to be a no-op on PTX while the interpreter honoured it (FIXED)

This was the sharpest instance of this document's thesis, because the instrument and
the device disagreed about CONTROL FLOW rather than about ordering. It is recorded
here in past tense because the divergence is now closed, and because the shape of
how it hid is the transferable lesson.

The PTX emitter handled `Return` with a comment reading "Handled by
finish_with_return; per-op Return is a no-op here". Only a trailing whole-kernel
return was emitted, so a `Return` nested inside a `Node::Loop` or an `if` emitted
NOTHING. The reference interpreter did the opposite:
`vyre-reference/src/execution/hashmap/node_step.rs:114-117` clears the frames and
sets `invocation.returned`, so the exit was real there.

A `Node::Loop` whose body ended in `if converged { Return }` therefore terminated
under `reference_eval` and ran its full trip count on CUDA. Measured on an RTX
5090: the DCE analysis on a 2000-node star that reaches its fixpoint in two
iterations cost 2450 ms at `max_iters = 2000` against 13 ms at `max_iters = 8`,
183x, wall time strictly proportional to the budget. Pinning to one workgroup did
not change it (1228 ms against 1235 ms), so it was not a race. `converged` still
read 1, because the branch does execute and write its flag before falling through.
Correct answer, hundreds of times the work, and no correctness test could see it.

`Return` now lowers to `bra $L_exit`. Re-measured on the same shape and box, budget
2000 costs 2.6x budget 8 rather than 183x, with `converged = 1` and the same 2000
live nodes at both budgets, so the answer did not move. The residual multiple is
not zero and has not been attributed; treat "the exit fires" as the established
claim and the remaining ratio as open.

The emitter now also REFUSES a `Return` it cannot prove is taken uniformly across
the grid, rather than dropping it. That half matters more than the branch: an exit
taken by only some invocations lets them leave while the rest wait at the next
barrier forever, so an invisible slowdown would have been traded for an invisible
hang. Values built from literals, buffer lengths, the subgroup size, and loads from
global or constant memory at a uniform index qualify as uniform; invocation ids,
workgroup ids, subgroup ops, shared memory, atomic return values, and loops with
non-uniform bounds do not.

The durable lesson is unchanged by the fix. A dropped control-flow node produced
right answers and wrong work, and that combination is invisible to a correctness
suite by construction. Gating work on a flag while keeping every barrier
unconditional (`dce_program.rs` does exactly this) remains the robust shape,
because it does not depend on an exit being honoured at all.

## Scope

This is a statement about `reference_eval` and
`reference_eval_lane_reversed` as verification instruments. It does not describe a
defect in `vyre-reference`: executing one workgroup at a time is a legal schedule,
and the interpreter is not obliged to enumerate the others. The error to avoid is
reading its agreement as a proof it never offered.
