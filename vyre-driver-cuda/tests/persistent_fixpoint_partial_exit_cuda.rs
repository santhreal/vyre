//! Hardware detector for a PARTIAL EXIT from
//! `fixpoint::persistent_fixpoint`'s in-kernel convergence loop.
//!
//! # What is being detected, and why not by comparing the answer
//!
//! The builder's loop body ends with `if changed[0] == 0 { Return }` and the next
//! iteration begins with invocation 0 clearing `changed[0]`. Unless a barrier
//! separates those two, the warp that takes the back edge first can clear the
//! flag while a sibling warp of the same workgroup has not yet executed the
//! read. The sibling reads 0, takes the `Return`, and leaves the kernel while the
//! rest keep iterating.
//!
//! Comparing the final state is a WEAK detector for that, because a transfer
//! that is monotone and saturating still ends up correct for the lanes that
//! stayed, and a lane that leaves one iteration early may already have reached
//! its fixed point. So this test does not compare the answer. It counts, per
//! lane, how many iterations that lane actually entered, and requires every lane
//! to report the SAME count. A partial exit is then visible directly and
//! immediately, whatever the state ends up being.
//!
//! That is the whole reason this file exists next to the structural tests in
//! `vyre-primitives/tests/persistent_fixpoint_loop_contracts.rs`. Those prove the
//! barrier is in the IR. This one proves the barrier is doing something real on
//! the device, so the structural tests cannot degrade into pinning a node nobody
//! needs.
//!
//! # Shape
//!
//! `words` is one full workgroup (256), which is EIGHT warps, so the loop body
//! has seven sibling warps that can lag the one holding invocation 0. The
//! transfer is a saturating increment, so convergence takes a known number of
//! iterations and every iteration before that genuinely sets the flag. Long
//! enough that the back edge is crossed many times per dispatch, since each
//! crossing is an independent chance for the race.

#![cfg(test)]

mod common;

use common::{bytes_u32, u32_bytes, with_live_backend};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};
use vyre_primitives::fixpoint::persistent_fixpoint::{
    persistent_fixpoint, PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};

/// One full workgroup of state: eight warps, so seven can lag invocation 0.
const WORDS: u32 = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

/// Iterations of real work before the transfer reaches its fixed point.
const LIMIT: u32 = 120;

/// Pass budget. Two past `LIMIT` so a correct run always converges by the flag
/// rather than by exhausting the budget, and a lane that stops early is
/// therefore never explained by the cap.
const MAX_ITERATIONS: u32 = LIMIT + 2;

/// Binding index for the per-lane iteration counter.
const VISITS_BINDING: u32 = 3;

const CURRENT: &str = "current";
const NEXT: &str = "next";
const CHANGED: &str = "changed";
const VISITS: &str = "visits";

/// A saturating increment plus a per-lane visit tally.
///
/// `next[t] = min(current[t] + 1, LIMIT)`, expressed with `select` so no lane
/// branches around its write: the builder requires EVERY word of `next` to be
/// written on EVERY iteration, and a lane that skipped its write would make the
/// compare report a stale difference instead of a real one.
fn transfer_body() -> Vec<Node> {
    let t = Expr::InvocationId { axis: 0 };
    let current = Expr::load(CURRENT, t.clone());
    vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(WORDS)),
        vec![
            // Counted at the top, so the tally includes the final iteration that
            // observes no change. Every lane that ENTERS an iteration bumps its
            // own slot, so equal tallies mean the lanes ran in lockstep and
            // unequal tallies mean some lane left the kernel early.
            Node::store(
                VISITS,
                t.clone(),
                Expr::add(Expr::load(VISITS, t.clone()), Expr::u32(1)),
            ),
            Node::store(
                NEXT,
                t.clone(),
                Expr::select(
                    Expr::lt(current.clone(), Expr::u32(LIMIT)),
                    Expr::add(current.clone(), Expr::u32(1)),
                    current,
                ),
            ),
        ],
    )]
}

/// Run the loop once and return the per-lane iteration tallies.
fn run_visits(backend: &CudaBackend) -> Vec<u32> {
    let program = persistent_fixpoint(
        transfer_body(),
        CURRENT,
        NEXT,
        CHANGED,
        WORDS,
        MAX_ITERATIONS,
    );
    // `persistent_fixpoint` declares only its own three buffers, so the tally
    // buffer is appended the same way a composing caller appends its own.
    let mut buffers = program.buffers().to_vec();
    buffers.push(
        BufferDecl::storage(
            VISITS,
            VISITS_BINDING,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(WORDS),
    );
    let program = program.with_rewritten_buffers(buffers);

    let words = WORDS as usize;
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(&vec![0u32; words]),
        u32_bytes(&vec![0u32; words]),
        u32_bytes(&[0u32]),
        u32_bytes(&vec![0u32; words]),
    ];
    let mut config = DispatchConfig::default();
    // One workgroup exactly: this is the configuration the builder documents as
    // sound, so a failure here cannot be blamed on multi-group use.
    config.grid_override = Some([1, 1, 1]);
    config.workgroup_override = Some(PERSISTENT_FIXPOINT_WORKGROUP_SIZE);
    config.dispatch_elements = Some(WORDS);

    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("the one-workgroup fixpoint dispatch must succeed");
    let mut visits = bytes_u32(&outputs[VISITS_BINDING as usize]);
    visits.truncate(words);
    visits
}

/// Verify one dispatch's tallies, returning the lanes that disagree.
fn ragged_lanes(visits: &[u32]) -> Vec<(usize, u32)> {
    // Every lane converges at the same iteration, so a correct run gives every
    // lane the same tally. Compared against an exact value and not merely
    // "all equal", so a run where the WHOLE workgroup left early (uniform, so
    // not a partial exit, but still wrong) also fails.
    let expected = LIMIT + 1;
    visits
        .iter()
        .enumerate()
        .filter(|(_, count)| **count != expected)
        .map(|(lane, count)| (lane, *count))
        .collect()
}

/// Locks out a partial exit: every lane of the workgroup must enter exactly the
/// same number of loop iterations.
///
/// If this fails, `persistent_fixpoint` is letting some invocations leave the
/// kernel while their siblings keep iterating. The consequence is a
/// partially-transferred state that reports convergence, with no error and no
/// hang, because `bar.sync` does not count invocations that already returned.
///
/// # The concurrency is the instrument, not incidental
///
/// A single one-workgroup launch is the WORST possible detector for this, and
/// running one taught that the hard way: it passes even with the guarding barrier
/// removed. With `grid_override = [1, 1, 1]` there is exactly one CTA resident on
/// the whole device, its eight warps are released from each `bar.sync` together
/// and stay in lockstep, and the warp holding invocation 0 cannot reach the clear
/// until its own read of `changed[0]` has returned from memory, by which time
/// every sibling's read has been issued too. The window is closed by having
/// nothing else to schedule.
///
/// So the launches are issued from many host threads at once. That puts many
/// independent CTAs on the SMs, the scheduler interleaves them, and a warp can
/// now be descheduled across the barrier release for long enough that its read
/// lands after another warp's clear. This is the same condition under which the
/// defect was seen from a consumer: `exatok`'s determinism suite failed only
/// when its fourteen tests ran in parallel, and never when the failing tests were
/// run alone.
#[test]
fn cuda_persistent_fixpoint_never_exits_partially_under_load() {
    with_live_backend(
        "cuda_persistent_fixpoint_never_exits_partially_under_load",
        |backend| {
            let threads = 12_usize;
            let repeats = 40_usize;

            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|index| {
                        scope.spawn(move || {
                            for repeat in 0..repeats {
                                let visits = run_visits(backend);
                                assert_eq!(
                                    visits.len(),
                                    WORDS as usize,
                                    "thread {index} repeat {repeat}: the tally buffer must \
                                     come back at its declared width"
                                );
                                let ragged = ragged_lanes(&visits);
                                assert!(
                                    ragged.is_empty(),
                                    "thread {index} repeat {repeat}: every lane must enter \
                                     exactly {} iterations, but {} lane(s) did not: {:?}. \
                                     Unequal tallies are a PARTIAL EXIT: the loop's \
                                     termination read of `changed[0]` raced the next \
                                     iteration's clear of the same word, so some warps read \
                                     0 and returned while their siblings kept iterating. The \
                                     lanes that left stop running the transfer body, which \
                                     freezes the words they own partway through and reports \
                                     convergence on a partially-transferred state. Fix: keep \
                                     the barrier that separates the termination read from the \
                                     loop back edge.",
                                    LIMIT + 1,
                                    ragged.len(),
                                    &ragged[..ragged.len().min(12)]
                                );
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().expect(
                        "every lane of every dispatch must enter the same number of iterations",
                    );
                }
            });
        },
    );
}
