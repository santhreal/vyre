//! `persistent_fixpoint`  -  single-dispatch convergence on the GPU.
//!
//! Where [`bitset_fixpoint`](super::bitset_fixpoint::bitset_fixpoint)
//! ships only the comparison + flag half of the loop and leaves the
//! caller's host code to drive the iteration, `persistent_fixpoint`
//! takes the caller's transfer-step body and wraps it in a forever-
//! loop on the GPU with the comparison + ping-pong + termination
//! check inside the kernel. The host issues ONE dispatch and reads
//! the final state; convergence happens entirely on device.
//!
//! This is the substrate primitive that replaces host-driven
//! fixpoint loops in dataflow, graph reachability, and iterative
//! bitset analyses. Higher-level crates supply their transfer body
//! once; `persistent_fixpoint` provides the convergence harness.
//!
//! ## Composition contract
//!
//! Caller supplies:
//!
//! - `transfer_body`  -  `Vec<Node>` reading from `current`, writing to
//!   `next`. Free to consume + dispatch any number of nested
//!   primitives (csr_forward_traverse, bitset_or, bitset_and, …).
//! - `current` / `next`  -  ping-pong bitset names (caller-managed).
//! - `changed`  -  convergence flag name (1-word atomic ReadWrite).
//! - `words`  -  bitset element count in 32-bit words.
//! - `max_iterations`  -  hard cap. The kernel breaks out after this
//!   many iterations even if `changed` is still set, so a buggy
//!   transfer body cannot wedge the dispatcher.
//!
//! Caller receives a [`Program`] that, when dispatched once, runs the
//! transfer body until the iteration's `changed` read is 0 or
//! `max_iterations` is reached. Output is in `current` after the
//! dispatch returns; `next` is scratch.
//!
//! The exit is a FIRST-zero-read exit, not a two-consecutive-zeros
//! exit: the loop checks the flag once per iteration and leaves
//! immediately on the first 0. `passes` therefore means iterations
//! ENTERED. An earlier revision of this doc claimed "two consecutive
//! iterations" and was simply false against the code; downstream
//! pass-count bounds are denominated in iterations entered, so the
//! code is the contract and this text was corrected to match it.
//!
//! ## LEGO discipline
//!
//! This primitive composes:
//!
//! - `Node::Loop` (vyre-foundation, IR primitive)  -  the convergence
//!   loop body.
//! - `bitset_fixpoint::bitset_fixpoint` step (re-used)  -  comparison +
//!   flag-set inside the loop body.
//! - Standard ping-pong via `Node::store(current, t, next[t])`  -
//!   in-place buffer swap on the GPU.
//!
//! ## Which one to use
//!
//! Two builders live here and they are NOT interchangeable:
//!
//! - [`persistent_fixpoint_grid`]: PREFER THIS. Sound at any group
//!   count. Replaces the in-kernel loop with `max_iterations`
//!   top-level waves separated by `MemoryOrdering::GridSync` barriers,
//!   and keeps a COLLECTIVE early exit by giving `changed` one word
//!   per iteration that is never cleared, so every write to it is an
//!   `atomic_or` and nothing conflicts.
//! - [`persistent_fixpoint`]: the older in-kernel `Node::Loop` form.
//!   Correct at ONE workgroup and racy above it, so choose it only when
//!   the state fits one group. Retained because its callers' pass
//!   counts are denominated in its behavior.
//!
//! [`persistent_fixpoint`] is RACY ABOVE ONE WORKGROUP. That is a
//! scope problem, not an atomicity problem, and an earlier revision of
//! this doc conflated the two, so they are kept apart below.
//!
//! The race is the multi-workgroup one. Lane 0 clears `changed[0]`
//! while every lane sets that same word with `atomic_or`, and the clear
//! is ordered only by a `MemoryOrdering::SeqCst` barrier, which is
//! WORKGROUP scope. Above one group nothing orders one group's clear
//! against another group's set. The severe face is a lost set: the
//! clear erases a set that already happened, and the group whose set
//! was erased reads 0, takes its own `Node::Return`, and leaves its
//! slice of the state unconverged with no error reported. For a caller
//! whose convergence means "no work remains", that is a wrong answer,
//! not just wasted work. The milder face is a false verdict, where the
//! flag read back after the dispatch does not reflect the convergence
//! actually reached.
//!
//! At ONE workgroup the same code is ordered and does NOT lose a set.
//! The per-iteration sequence is clear, barrier, `atomic_or`s, barrier,
//! barrier, read, so no two conflicting accesses to `changed[0]` are
//! ever concurrent, and within a single CTA a `bar.sync` carries a
//! CTA-scope memory fence, which is sufficient. Do not read the
//! paragraph above as a single-group defect; it is not one.
//!
//! The clear used to be a plain non-atomic `Node::store` to a location
//! every other write reaches through `atomic_or`. That mixing was
//! correct only because the barriers above ordered it, which is a
//! dependency invisible at the call site: weaken or move a barrier and
//! the program breaks without anything correctness-shaped being edited.
//! It is now an atomic exchange, so EVERY write to `changed` in both
//! builders is an atomic. In the emitted PTX the clear is an
//! `atom.global.exch` and the set an `atom.global.or.b32` at the same
//! address, instead of a plain `st.global.u32` against an atomic. This
//! costs one lane one operation per iteration and changes no value and
//! no pass count. It does NOT make this builder multi-workgroup safe;
//! the race above is about barrier SCOPE, not atomicity.
//!
//! [`persistent_fixpoint_grid`] additionally has no clear at all, so it
//! cannot lose a set and needs no ordering between a clear and the
//! sets.
//!
//! Soundness: matches the host-driven loop exactly (proven by the
//! convergence-equivalence test below).

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::fixpoint::persistent_fixpoint";

/// Canonical op id of the grid-correct sibling
/// [`persistent_fixpoint_grid`].
pub const OP_ID_GRID: &str = "vyre-primitives::fixpoint::persistent_fixpoint_grid";

/// Workgroup size both builders emit.
///
/// Exported so a caller derives its routing threshold from the
/// declared geometry instead of a literal: dispatch
/// [`persistent_fixpoint`] while
/// `words <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]` and
/// [`persistent_fixpoint_grid`] above it. Mirrors
/// `PERSISTENT_BFS_WORKGROUP_SIZE` in the graph domain.
///
/// Do not widen this to raise the single-workgroup ceiling without
/// re-checking cooperative residency. [`persistent_fixpoint_grid`]
/// needs every block co-resident, and the per-SM block limit is a
/// floor division: a width that does not divide the SM's thread budget
/// evenly truncates to fewer blocks per SM and can cut total resident
/// threads well below what a narrower block reaches. Widening trades
/// grid capacity for a higher single-group ceiling, so it is a
/// tradeoff to measure on the target device, not a free win.
pub const PERSISTENT_FIXPOINT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Build a Program that runs `transfer_body` to convergence on the
/// GPU.
///
/// One dispatch from the host. The kernel:
///
/// 1. Zeros `changed[0]`.
/// 2. Runs `transfer_body` (caller-supplied  -  reads `current`, writes `next`).
/// 3. For every word `w`, sets `changed[0]=1` iff `current[w] != next[w]`.
/// 4. Copies `next[w]` into `current[w]`.
/// 5. Reads `changed[0]`. If 0, returns (FIRST-zero-read exit, not a
///    two-consecutive-zeros exit, so `passes` means iterations
///    ENTERED).
/// 6. Repeats up to `max_iterations` times.
///
/// # Use [`persistent_fixpoint_grid`] above one workgroup
///
/// SOUND when `words <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`, i.e.
/// one workgroup covers the state. RACY above that.
///
/// Step 1's clear and step 3's set both target `changed[0]`, ordered
/// only by a `MemoryOrdering::SeqCst` barrier, which is WORKGROUP
/// scope. With more than one group nothing orders one group's clear
/// against another group's set, so a clear can erase a set that already
/// happened; that group reads 0 at step 5 and returns with its slice of
/// the state unconverged. A caller whose convergence means "no work
/// remains" gets a wrong answer from that, not merely wasted work. Both
/// writes are atomic, which does not help: the race is about barrier
/// SCOPE, not atomicity.
///
/// At one workgroup the accesses within ONE iteration are clear,
/// barrier, sets, barrier, read, so those never race and a CTA-scope
/// fence is sufficient for them. An earlier revision of this doc
/// claimed the clear made the builder unsound at one workgroup too.
/// That was wrong for the intra-iteration sequence: the barriers order
/// it.
///
/// The LOOP BACK EDGE is a separate ordering obligation, and enumerating
/// one iteration hides it. The read at step 5 and the NEXT iteration's
/// clear at step 1 touch the same word, so a barrier is required
/// BETWEEN them or the warp that takes the back edge first clears the
/// flag while a sibling warp has not yet read it. The sibling then reads
/// 0 and returns while the rest keep iterating, which is a PARTIAL exit
/// and yields a partially-transferred state that reads as converged.
/// That barrier is emitted as the last node of the loop body. This
/// builder shipped without it, and the revision that removed a
/// consecutive duplicate barrier from before the read to after it is
/// what closed the gap; see the comment at the emission site.
///
/// The clear is an atomic exchange rather than a plain store. It used
/// to be a plain store, which was correct only while those barriers
/// stood, and that dependency is invisible at the call site. Making it
/// atomic costs one lane one operation per iteration, changes no value
/// and no pass count, and removes the dependency.
///
/// [`persistent_fixpoint_grid`] takes the same positional parameters,
/// buffer names, bindings, and workgroup size, and additionally has no
/// clear at all, so it cannot lose a set. The only caller-visible
/// difference is that its `changed` buffer is `max_iterations.max(1)`
/// words and must be zeroed before dispatch, with convergence decoded
/// as the first index reading 0. This builder keeps its iteration and
/// pass-count behavior, which existing callers are denominated in.
///
/// `changed` is a 1-word atomic ReadWrite buffer. `current` and
/// `next` are word-bitset ReadWrite buffers of length `words`.
///
/// The transfer body MUST NOT touch `changed`  -  the wrapper owns the
/// convergence flag exclusively.
///
/// The transfer body MUST also write EVERY word `w < words` of `next`
/// on EVERY iteration. This is load-bearing, and violating it produces a
/// WRONG ANSWER that still reports convergence, so nothing in the run
/// looks wrong. Step 4 copies `next[w]` into `current[w]` for every `w`,
/// not only for the words the body touched, so a word the body leaves
/// unwritten overwrites `current[w]` with whatever `next` happened to
/// hold. From the following iteration the two buffers agree everywhere,
/// the compare reports no change, and the loop exits converged on
/// corrupted state.
///
/// A body that naturally writes only a subset (one lane per segment,
/// a guarded range, a scatter) must therefore either widen its write to
/// cover `words`, or pass a `words` that counts only the entries it
/// does write. Copying `current[w]` into `next[w]` for the untouched
/// entries is the cheap fix, and it is what makes the compare mean
/// "the transfer changed nothing" instead of "some entry is stale".
///
/// # Parameters
///
/// - `transfer_body`: caller-provided IR body that performs ONE step
///   of the transfer function. Reads `current`, writes `next`.
/// - `current` / `next`: bitset buffer names (ReadWrite).
/// - `changed`: 1-word convergence-flag buffer name (ReadWrite atomic).
/// - `words`: bitset element count.
/// - `max_iterations`: hard upper bound on iterations.
#[must_use]
pub fn persistent_fixpoint(
    transfer_body: Vec<Node>,
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    max_iterations: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };

    // Per-iteration body composed of:
    //   (a) zero `changed[0]` so this iteration's compare starts clean.
    //   (b) caller's transfer step (reads current → writes next).
    //   (c) convergence step + ping-pong: per word, set changed=1 if
    //       differ + copy next→current.
    let mut iter_body: Vec<Node> = Vec::new();
    // The clear is an ATOMIC exchange, not a plain store, even though only
    // lane 0 performs it and a barrier separates it from the `atomic_or`s
    // below. Every other write to this word is an `atomic_or`, and mixing a
    // non-atomic write with atomics on one location is only correct while
    // that barrier stands: it is invisible at the call site, so a later edit
    // that weakens or moves the barrier would break correctness without
    // touching anything that looks like correctness. An atomic write costs
    // one lane one operation per iteration and removes the dependency.
    iter_body.push(Node::if_then(
        Expr::eq(t.clone(), Expr::u32(0)),
        vec![Node::let_bind(
            "_pf_clear",
            Expr::atomic_exchange(changed, Expr::u32(0), Expr::u32(0)),
        )],
    ));
    iter_body.push(Node::Barrier {
        ordering: vyre_foundation::MemoryOrdering::SeqCst,
    });
    iter_body.extend(transfer_body);
    iter_body.push(compare_and_copy_node(
        t.clone(),
        current,
        next,
        changed,
        Expr::u32(0),
        "_pf_set",
        words,
    ));
    iter_body.push(Node::Barrier {
        ordering: vyre_foundation::MemoryOrdering::SeqCst,
    });
    // Termination reads `changed[0]` after the barrier above, so this
    // iteration's sets are visible to every invocation.
    //
    // The barrier that USED to sit here was a second, immediately
    // consecutive `SeqCst` barrier: a no-op, since the one above had
    // already synchronized every invocation. It is moved BELOW the
    // termination read, where it is load-bearing, so the barrier count
    // per iteration is unchanged.
    let outer = vec![Node::loop_for(
        "__pf_iter__",
        Expr::u32(0),
        Expr::u32(max_iterations),
        {
            let mut body = iter_body;
            body.push(Node::if_then(
                Expr::eq(Expr::load(changed, Expr::u32(0)), Expr::u32(0)),
                vec![Node::Return],
            ));
            // Guards the LOOP BACK EDGE, and it is required for
            // correctness at ANY group count including one.
            //
            // Without it, the read above and the NEXT iteration's clear
            // of the same word are unordered. Nothing stops the warp
            // holding invocation 0 from taking the back edge and
            // clearing `changed[0]` while another warp of the same
            // workgroup has not yet executed the read. That warp then
            // reads 0, takes the `Return`, and leaves the kernel while
            // the rest keep iterating: a PARTIAL exit. The invocations
            // that left stop contributing to `transfer_body`, so the
            // words they own freeze mid-transfer and the dispatch
            // returns a partially-transferred state that no caller can
            // distinguish from a converged one by looking at `changed`.
            //
            // `bar.sync` does not count invocations that already
            // returned, so the survivors are not stranded and nothing
            // hangs. That is exactly why this was invisible: the defect
            // costs answers, never liveness.
            //
            // This is also what makes the emitter's uniformity proof for
            // the exit condition TRUE rather than merely syntactic.
            // `vyre-emit-ptx` classifies a `LoadGlobal` at a uniform
            // index as grid-uniform, and its own note on that
            // classification requires that a value steering control flow
            // is not concurrently written without synchronization. This
            // barrier is what discharges that requirement here; the
            // grid form has no clear at all and so needs nothing.
            body.push(Node::Barrier {
                ordering: vyre_foundation::MemoryOrdering::SeqCst,
            });
            body
        },
    )];

    build_fixpoint_program(current, next, changed, words, 1, OP_ID, outer)
}

/// Build a Program that runs `transfer_body` to convergence on the GPU
/// with GRID-WIDE synchronization, for state that does not fit one
/// workgroup.
///
/// Same parameter meaning, positional order, return type, buffer
/// names, bindings, and workgroup size as [`persistent_fixpoint`], so
/// a caller selects between them on group count alone. The ONE ABI
/// difference: `changed` is `max_iterations` words wide here, not 1.
///
/// One dispatch from the host. Instead of an in-kernel `Node::Loop`,
/// the kernel body is `max_iterations` top-level WAVES separated by
/// `MemoryOrdering::GridSync` barriers, the same shape
/// `persistent_bfs_grid_sync_parallel` uses for the same reason. Wave
/// `i` emits exactly five top-level nodes:
///
/// 1. `transfer_body` (caller-supplied, reads `current`, writes
///    `next`), wrapped in a `Node::Block`.
/// 2. A `GridSync` barrier, so every group's `next` writes are
///    globally visible before anyone compares.
/// 3. For every word `w`, `atomic_or(changed, i, 1)` iff
///    `current[w] != next[w]`, then copy `next[w]` into `current[w]`.
/// 4. A `GridSync` barrier, so every group's `changed[i]` contribution
///    and `current` writes are globally visible.
/// 5. `if changed[i] == 0 { Node::Return }`.
///
/// # Why the early exit is safe here and the original's is not
///
/// A `Node::Return` under a grid-barrier protocol is normally fatal:
/// one group leaving strands every other group at a barrier that will
/// never be reached. That hazard is absent ONLY because the exit
/// decision is UNIFORM across the grid, and it is uniform because of
/// two deliberate choices that must be kept together:
///
/// - `changed` has ONE WORD PER ITERATION and is NEVER CLEARED. Wave
///   `i` only ever `atomic_or`s word `i`, and there is no clear at all.
///   A word that is never cleared cannot lose a set, which is what
///   kills the multi-group lost-set race that
///   [`persistent_fixpoint`] has: there, a clear and the sets target
///   one word and only a workgroup-scope barrier orders them, so above
///   one group a clear can erase a set. Both builders write `changed`
///   exclusively with atomics, so neither mixes a plain store with an
///   atomic on one location; what distinguishes this one is having no
///   clear to order in the first place.
/// - The read of `changed[i]` happens AFTER a `GridSync` barrier, so
///   every group observes every other group's contribution and all
///   groups compute the SAME verdict. Either the whole grid returns at
///   wave `i` or none of it does, so no group is ever stranded.
///
/// Do NOT "simplify" the per-iteration word back to a single cleared
/// word. That reintroduces the lost-set race and turns the collective
/// return into a stranding hazard in one edit.
///
/// ## The exit is honored on the PTX path
///
/// Measured, not assumed. Lowering this program and emitting PTX for a
/// three-wave build produces three unpredicated `bra $L_exit`
/// instructions, one per wave, and [`persistent_fixpoint`] produces one
/// for its in-kernel loop.
///
/// This was NOT always true, and the history is worth keeping because
/// the failure was invisible. `vyre-emit-ptx` used to handle `Return`
/// with a comment and no instruction, so a `Return` nested in an `If`
/// emitted nothing and fell through, and every emitted wave ran no
/// matter how early the grid converged. Answers stayed correct, because
/// a converged wave recomputes the same `next`, sets no flag word, and
/// copies idempotently, so only the work was wrong and no correctness
/// test in the tree noticed.
///
/// The exit now also carries a compile-time guarantee that it is safe.
/// The emitter proves the exit condition is uniform across the grid
/// before lowering the branch, and REFUSES the program otherwise,
/// because an exit taken by only some invocations would leave the rest
/// waiting at the next barrier forever. This primitive satisfies that
/// proof by construction: `changed[i]` is read from global memory at a
/// literal index, which is grid-uniform, and it is read after a grid
/// fence. Gating the exit on anything derived from an invocation id
/// would be refused at emit time rather than silently accepted.
///
/// ## The exit saves launches ONLY under a cooperative launch
///
/// This bounds the guarantee above, and the bound is easy to miss
/// because nothing fails when it applies. `MemoryOrdering::GridSync`
/// lowers either to a native cooperative grid barrier or to a KERNEL
/// SPLIT. Under the split each wave becomes its own kernel launch, and
/// `vyre_driver::grid_sync` dispatches every segment in order: a
/// `Node::Return` inside segment `N` returns from THAT launch only and
/// cannot stop the host from launching segment `N + 1`. A run that
/// converges at wave 2 of a 16-wave budget therefore still issues all
/// `2 * max_iterations + 1` segments.
///
/// Nothing about the ANSWER changes on that path. A converged wave
/// recomputes the same `next`, sets no flag word, and copies
/// idempotently, so the state and the `changed` decoding are exactly
/// what a cooperative launch produces. What disappears is the saved
/// work, and a device-side pass counter reads the full budget rather
/// than the convergence depth, which looks like a cap-out and is not
/// one. A downstream caller measured precisely that. Read the
/// convergence depth from `changed`, which is authoritative on both
/// paths, and never from a pass or launch count, which is authoritative
/// only under a native cooperative launch.
///
/// # Buffer contract
///
/// - `current` (binding 0) / `next` (binding 1): ReadWrite, `words`
///   u32 elements. Output is in `current` after the dispatch.
/// - `changed` (binding 2): ReadWrite, `max_iterations` u32 elements
///   (floored at 1 so `max_iterations == 0` still declares a valid
///   buffer). The caller MUST supply it ZERO-FILLED; this primitive
///   never clears it.
///
/// `changed` is also the pass-count readback. `changed[i] == 1` iff
/// wave `i` changed the state, and the kernel returns at the first
/// zero, so the array is a run of ones followed by zeros:
/// `passes_entered` is the index of the first zero plus one, or
/// `max_iterations` when no word is zero (budget exhausted). Unlike
/// [`persistent_fixpoint`]'s single word this verdict is trustworthy,
/// because each word is read after a grid-wide barrier and never
/// cleared.
///
/// The transfer body MUST NOT touch `changed`; the wrapper owns the
/// convergence flag exclusively.
///
/// The transfer body MUST also write EVERY word `w < words` of `next`
/// on EVERY wave, for the same reason and with the same failure mode as
/// [`persistent_fixpoint`]: the copy step writes `current[w] = next[w]`
/// for every `w`, so a word the body never wrote overwrites `current[w]`
/// with a stale `next`, the buffers then agree, and the run reports
/// convergence on corrupted state. A body that writes only a subset must
/// widen its write to cover `words`, copy `current[w]` into `next[w]`
/// for the entries it skips, or pass a `words` that counts only the
/// entries it does write.
///
/// # Grid-size ceiling
///
/// `GridSync` lowers either to a native cooperative grid barrier or to
/// a host-orchestrated kernel split, so this program carries a
/// cooperative-residency ceiling that [`persistent_fixpoint`] does
/// not: a cooperative launch needs every block co-resident. This
/// builder emits IR and cannot see the launch geometry, so the check
/// belongs to the dispatch path, which MUST refuse loudly, naming the
/// block count and the device limit, rather than quietly rerouting.
/// `VyreBackend::cooperative_grid_sync_fits` is the preflight and
/// `VyreBackend::allows_host_grid_sync_split` says whether the split
/// fallback is even permitted; a backend that answers `false` to the
/// latter has no escape hatch and a silent degrade there would be a
/// correctness failure, not a performance one.
///
/// # Parameters
///
/// - `transfer_body`: caller-provided IR body that performs ONE step
///   of the transfer function. Reads `current`, writes `next`. Cloned
///   once per wave, so it is emitted `max_iterations` times.
/// - `current` / `next`: bitset buffer names (ReadWrite).
/// - `changed`: per-iteration convergence-flag buffer name (ReadWrite
///   atomic, `max_iterations` words, zero-filled by the caller).
/// - `words`: bitset element count.
/// - `max_iterations`: wave count, and the hard upper bound on
///   iterations.
#[must_use]
pub fn persistent_fixpoint_grid(
    transfer_body: Vec<Node>,
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    max_iterations: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut entry: Vec<Node> = Vec::with_capacity(max_iterations as usize * 5);
    for iter in 0..max_iterations {
        // The caller's body is spliced once per wave. Flat splicing
        // would make its top-level `let`s duplicate siblings in one
        // region (V032), so each copy gets its own Block scope. Same
        // reason `single_word_lineage_grid_sync_body` blocks its
        // phases. A Block with no GridSync inside is preserved by the
        // interpreter's scope flattening, so the scope survives.
        entry.push(Node::block(transfer_body.clone()));
        entry.push(grid_sync_barrier());
        // Per-word compare + ping-pong, byte-identical to
        // `persistent_fixpoint`'s step except that the flag index is
        // the wave number instead of the single shared word 0. The
        // `let`s are scoped by this `If`, so repeating the node per
        // wave is not a sibling collision.
        entry.push(compare_and_copy_node(
            t.clone(),
            current,
            next,
            changed,
            Expr::u32(iter),
            "_pfg_set",
            words,
        ));
        entry.push(grid_sync_barrier());
        // Collective exit: this read sits AFTER the barrier above, so
        // every group sees the same `changed[iter]` and reaches the
        // same verdict. The whole grid leaves together or none of it
        // does, which is what keeps a `Return` legal between grid
        // barriers.
        entry.push(Node::if_then(
            Expr::eq(Expr::load(changed, Expr::u32(iter)), Expr::u32(0)),
            vec![Node::Return],
        ));
    }

    build_fixpoint_program(
        current,
        next,
        changed,
        words,
        max_iterations.max(1),
        OP_ID_GRID,
        entry,
    )
}

/// The ping-pong state and iteration budget a convergence harness is built over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixpointState<'a> {
    /// Ping-pong buffer holding the state read by the transfer body, and the
    /// output after the dispatch returns.
    pub current: &'a str,
    /// Ping-pong buffer the transfer body writes.
    pub next: &'a str,
    /// Convergence-flag buffer. Its declared width is
    /// [`FixpointRoute::changed_words`], not a caller's choice.
    pub changed: &'a str,
    /// Element count of `current` and `next`, and the bound the compare and
    /// copy steps are gated on.
    pub words: u32,
    /// Hard upper bound on iterations, and the wave count of the grid form.
    pub max_iterations: u32,
}

/// Which convergence harness a launch span requires, and the `changed` width
/// that harness indexes.
///
/// Selecting the harness and sizing the flag are ONE decision, not two.
/// [`persistent_fixpoint_grid`] indexes `changed[iteration]`, so a caller that
/// routes to it and keeps a one-word flag writes out of bounds on iteration 1,
/// and a caller that stays on [`persistent_fixpoint`] but declares
/// `max_iterations` words has a flag whose tail is never read. Returning both
/// together is what stops a caller taking one half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixpointRoute {
    /// Whether the span needs [`persistent_fixpoint_grid`] and its
    /// `MemoryOrdering::GridSync` waves.
    pub needs_grid_sync: bool,
    /// Words the `changed` buffer must declare for the chosen harness.
    pub changed_words: u32,
}

/// Route a launch of `dispatch_span` lanes to a convergence harness.
///
/// `dispatch_span` is the LAUNCH width, which is not the same number as
/// [`FixpointState::words`]. `dispatch_element_count_for_program`
/// (`vyre-driver/src/program_walks/dispatch_params.rs:19`) sizes an
/// atomic-carrying program's launch from its WIDEST declared buffer, and both
/// harnesses carry an `atomic_or`, so an op that declares buffers wider than its
/// ping-pong state is launched over those wider buffers. A kernel matrix of
/// `m * n` cells or an edge list of `n_edges` entries therefore makes the
/// dispatch multi-workgroup while the state still fits one group, and routing on
/// the state width leaves such a launch on the racing single-word flag. Passing
/// the span separately is what keeps the two numbers from being confused.
#[must_use]
pub fn fixpoint_route(dispatch_span: u32, max_iterations: u32) -> FixpointRoute {
    let needs_grid_sync = dispatch_span > PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
    FixpointRoute {
        needs_grid_sync,
        changed_words: if needs_grid_sync {
            max_iterations.max(1)
        } else {
            1
        },
    }
}

/// Build the convergence harness a launch of `dispatch_span` lanes requires,
/// with the route it selected.
///
/// The only correct way to choose between the two harnesses. A caller that
/// re-derives the comparison, or the flag width that goes with it, owns a second
/// copy of a decision whose two halves must agree; see [`FixpointRoute`].
#[must_use]
pub fn routed_persistent_fixpoint(
    transfer_body: Vec<Node>,
    state: FixpointState<'_>,
    dispatch_span: u32,
) -> (Program, FixpointRoute) {
    let route = fixpoint_route(dispatch_span, state.max_iterations);
    let build = if route.needs_grid_sync {
        persistent_fixpoint_grid
    } else {
        persistent_fixpoint
    };
    let program = build(
        transfer_body,
        state.current,
        state.next,
        state.changed,
        state.words,
        state.max_iterations,
    );
    (program, route)
}

/// The grid-wide fence separating two waves.
///
/// `MemoryOrdering::GridSync` is the ordering the driver lowers either
/// to a native cooperative grid barrier or to a kernel split;
/// `MemoryOrdering::SeqCst` is workgroup scope and would order nothing
/// between groups, which is the whole defect this builder exists to
/// avoid.
///
/// Public because [`count_grid_sync`] is the matching reader and every op that
/// emits one of these fences pins its wave structure by counting them. An op
/// that builds the barrier itself and an assertion that recognises a different
/// ordering are the same drift in two places.
#[must_use]
pub fn grid_sync_barrier() -> Node {
    Node::barrier_with_ordering(vyre_foundation::MemoryOrdering::GridSync)
}

/// Grid-wide fences in `nodes`, counted through every nesting construct.
///
/// This is the ONE reader of [`grid_sync_barrier`]. Nesting comes from
/// `vyre_foundation::transform::visit::child_bodies`, the workspace's single
/// exhaustive owner of "which node variants contain other nodes", so a new
/// nesting variant fails to compile there rather than being counted as a leaf
/// here.
///
/// Every wave-structure assertion in the workspace used to restate this walk
/// with its own `match node` ending in `_ => 0`, which classifies an
/// unrecognised nesting variant as containing no fences. A fence hidden inside
/// such a variant makes an under-fenced program's structure test pass.
#[must_use]
pub fn count_grid_sync(nodes: &[Node]) -> usize {
    let mut total = 0;
    let mut stack: Vec<&Node> = nodes.iter().collect();
    while let Some(node) = stack.pop() {
        if matches!(
            node,
            Node::Barrier {
                ordering: vyre_foundation::MemoryOrdering::GridSync
            }
        ) {
            total += 1;
        }
        for body in vyre_foundation::transform::visit::child_bodies(node) {
            stack.extend(body);
        }
    }
    total
}

/// The dispatch span [`fixpoint_route`] keys on, read back from a built program.
///
/// Every harness this module emits carries an `atomic_or` on its convergence
/// flag, and for an atomic-carrying program `vyre-driver`'s
/// `dispatch_element_count_for_program` spans the LARGEST declared buffer rather
/// than just the output. So the span a caller must pass to `fixpoint_route` is
/// recoverable from the program's own declarations, which is what lets a test
/// confirm the routing decision against the emission instead of against a
/// restatement of the rule.
#[must_use]
pub fn declared_dispatch_span(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .map(BufferDecl::count)
        .max()
        .unwrap_or(1)
}

/// Workgroups a host must launch to cover `program`.
///
/// [`declared_dispatch_span`] over the program's own declared workgroup width, so
/// neither half can be pinned to a stale constant.
#[must_use]
pub fn required_workgroups(program: &Program) -> u32 {
    declared_dispatch_span(program).div_ceil(program.workgroup_size()[0])
}

/// Declared word count of the buffer named `buffer`.
///
/// The convergence-flag width is the contract a caller has to satisfy when it
/// uploads that buffer, and it differs by route: one shared word below the
/// routing threshold, one word per iteration above it. Panics when the name is
/// absent, because a program that does not declare the buffer a test is asking
/// about is a defect in the emission rather than a zero-width buffer.
#[must_use]
pub fn declared_words(program: &Program, buffer: &str) -> u32 {
    program
        .buffers()
        .iter()
        .find(|declared| declared.name() == buffer)
        .unwrap_or_else(|| {
            panic!("Fix: the program must declare a buffer named `{buffer}`.");
        })
        .count()
}

fn compare_and_copy_node(
    invocation: Expr,
    current: &str,
    next: &str,
    changed: &str,
    changed_index: Expr,
    changed_binding: &str,
    words: u32,
) -> Node {
    Node::if_then(
        Expr::lt(invocation.clone(), Expr::u32(words)),
        vec![
            Node::let_bind("c", Expr::load(current, invocation.clone())),
            Node::let_bind("n", Expr::load(next, invocation.clone())),
            Node::if_then(
                Expr::ne(Expr::var("c"), Expr::var("n")),
                vec![Node::let_bind(
                    changed_binding,
                    Expr::atomic_or(changed, changed_index, Expr::u32(1)),
                )],
            ),
            Node::store(current, invocation, Expr::var("n")),
        ],
    )
}

fn build_fixpoint_program(
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    changed_words: u32,
    op_id: &str,
    entry: Vec<Node>,
) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(current, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage(next, 1, BufferAccess::ReadWrite, DataType::U32).with_count(words),
            BufferDecl::storage(changed, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(changed_words),
        ],
        PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(entry),
        }],
    )
}

/// CPU oracle. Iterates `transfer_step` (a closure that takes
/// `current` and writes `next`) until the two arrays match or
/// `max_iterations` is hit. Returns the final `current` state and the
/// number of iterations actually executed.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref<F>(seed: &[u32], max_iterations: u32, mut transfer_step: F) -> (Vec<u32>, u32)
where
    F: FnMut(&[u32], &mut [u32]),
{
    let mut current = Vec::new();
    let mut next = Vec::new();
    let iters = try_cpu_ref_into(
        seed,
        max_iterations,
        &mut transfer_step,
        &mut current,
        &mut next,
    )
    .expect("Fix: caller must size scratch for node_count; use try_cpu_ref on hostile layouts");
    (current, iters)
}

/// CPU oracle using caller-owned buffers.
///
/// `current` receives the final fixpoint state and `next` is retained as
/// ping-pong scratch for subsequent calls.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_into<F>(
    seed: &[u32],
    max_iterations: u32,
    transfer_step: &mut F,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> u32
where
    F: FnMut(&[u32], &mut [u32]),
{
    try_cpu_ref_into(seed, max_iterations, transfer_step, current, next).expect(
        "Fix: caller must size scratch for node_count; use try_cpu_ref_into on hostile layouts",
    )
}

/// Fallible CPU oracle using caller-owned ping-pong buffers.
///
/// The output buffers are not mutated until both have enough capacity
/// for the seed length.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into<F>(
    seed: &[u32],
    max_iterations: u32,
    transfer_step: &mut F,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<u32, String>
where
    F: FnMut(&[u32], &mut [u32]),
{
    let additional_current = seed.len().saturating_sub(current.capacity());
    let additional_next = seed.len().saturating_sub(next.capacity());
    current
        .try_reserve_exact(additional_current)
        .map_err(|err| format!("failed to reserve current fixpoint buffer: {err}"))?;
    next.try_reserve_exact(additional_next)
        .map_err(|err| format!("failed to reserve next fixpoint buffer: {err}"))?;
    current.clear();
    current.extend_from_slice(seed);
    next.clear();
    next.resize(seed.len(), 0);
    for iter in 0..max_iterations {
        next.fill(0);
        transfer_step(current, next);
        if next == current {
            return Ok(iter + 1);
        }
        std::mem::swap(current, next);
    }
    Ok(max_iterations)
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || {
            persistent_fixpoint(
                vec![Node::store(
                    "next",
                    Expr::u32(0),
                    Expr::load("current", Expr::u32(0)),
                )],
                "current",
                "next",
                "changed",
                1,
                4,
            )
        },
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[7]), to_bytes(&[0]), to_bytes(&[0])]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[7]), to_bytes(&[7]), to_bytes(&[0])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_converges_when_step_is_idempotent() {
        // Identity transfer: next = current. Should converge in 1 step.
        let seed = vec![0b1010, 0b0101];
        let (out, iters) = cpu_ref(&seed, 100, |cur, next| next.copy_from_slice(cur));
        assert_eq!(out, seed);
        assert_eq!(iters, 1);
    }

    #[test]
    fn cpu_ref_converges_on_or_to_fixed_point() {
        // Transfer: next = current | constant. Reaches fixed point
        // when constant's bits are all set in current.
        let seed = vec![0u32];
        let (out, iters) = cpu_ref(&seed, 100, |cur, next| {
            next[0] = cur[0] | 0b1010;
        });
        assert_eq!(out, vec![0b1010]);
        assert!(iters < 5, "OR-with-const converges in 1 step + 1 confirm");
    }

    #[test]
    fn cpu_ref_caps_at_max_iterations() {
        // Diverging transfer: next = current + 1 (per word). Never
        // reaches fixed point; cpu_ref returns at max_iterations.
        let seed = vec![0u32];
        let max = 16;
        let (_, iters) = cpu_ref(&seed, max, |cur, next| {
            next[0] = cur[0].wrapping_add(1);
        });
        assert_eq!(iters, max);
    }

    #[test]
    fn cpu_ref_into_reuses_ping_pong_buffers() {
        let seed = vec![0u32];
        let mut current = Vec::with_capacity(16);
        let mut next = Vec::with_capacity(16);
        let current_ptr = current.as_ptr();
        let next_ptr = next.as_ptr();
        let mut transfer = |cur: &[u32], out: &mut [u32]| {
            out[0] = cur[0] | 0b1010;
        };
        let iters = cpu_ref_into(&seed, 16, &mut transfer, &mut current, &mut next);
        assert!(iters < 5);
        assert_eq!(current, vec![0b1010]);
        assert!(current.as_ptr() == current_ptr || current.as_ptr() == next_ptr);
        assert!(next.as_ptr() == current_ptr || next.as_ptr() == next_ptr);
        assert_ne!(current.as_ptr(), next.as_ptr());
    }

    #[test]
    fn try_cpu_ref_into_reuses_buffers_and_clears_stale_tail() {
        let mut current = Vec::with_capacity(8);
        let mut next = Vec::with_capacity(8);
        current.extend([u32::MAX; 8]);
        next.extend([u32::MAX; 8]);
        let current_ptr = current.as_ptr();
        let next_ptr = next.as_ptr();

        let mut transfer = |cur: &[u32], out: &mut [u32]| out.copy_from_slice(cur);
        let iters = try_cpu_ref_into(&[1, 2], 4, &mut transfer, &mut current, &mut next).unwrap();
        assert_eq!(iters, 1);
        assert_eq!(current, vec![1, 2]);
        assert_eq!(next, vec![1, 2]);
        assert!(current.as_ptr() == current_ptr || current.as_ptr() == next_ptr);
        assert!(next.as_ptr() == current_ptr || next.as_ptr() == next_ptr);

        let iters = try_cpu_ref_into(&[], 4, &mut transfer, &mut current, &mut next).unwrap();
        assert_eq!(iters, 1);
        assert!(current.is_empty());
        assert!(next.is_empty());
    }

    #[test]
    fn program_shape_matches_contract() {
        let body = vec![Node::store("next", Expr::u32(0), Expr::u32(0))];
        let program = persistent_fixpoint(body, "current", "next", "changed", 16, 64);
        assert!(
            program.buffers.iter().any(|b| b.name() == "current"),
            "current buffer must be declared"
        );
        assert!(
            program.buffers.iter().any(|b| b.name() == "next"),
            "next buffer must be declared"
        );
        assert!(
            program.buffers.iter().any(|b| b.name() == "changed"),
            "changed buffer must be declared"
        );
    }
}
